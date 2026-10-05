"""Weekly release archives with explicit conservative availability and exclusions."""

from __future__ import annotations
import argparse, csv, io, json, re, time, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from .extension_sources import fetch as original_fetch
import requests, hashlib
from datetime import datetime, timezone


def fetch(url, path, params=None, attempts=1):
    path = Path(path)
    if path.exists() and path.with_suffix(path.suffix + ".json").exists():
        return original_fetch(url, path, params=params)
    response = requests.get(url, params=params, timeout=12)
    response.raise_for_status()
    info = {
        "requested_url": response.request.url,
        "final_url": response.url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(response.content).hexdigest(),
        "last_modified": response.headers.get("Last-Modified"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(info, indent=2))
    return response.content, info


EIA = "https://www.eia.gov/petroleum/supply/weekly/archive/"
CFTC = "https://www.cftc.gov/MarketReports/CommitmentsofTraders/"
# Exclude entire report-date intervals until normal publication is safely resumed.
CFTC_EXCLUSIONS = [
    ("2011-11-01", "2011-11-15", "MF Global classification uncertainty"),
    ("2012-11-27", "2012-11-27", "Broad classification correction"),
    ("2013-10-01", "2013-11-05", "Shutdown and catch-up"),
    ("2018-12-24", "2019-03-05", "Shutdown and catch-up"),
    ("2023-01-31", "2023-03-14", "ION outage and catch-up"),
    ("2025-09-30", "2025-12-31", "Shutdown and changing catch-up schedule"),
]


def excluded_cftc(date):
    return next(
        (
            why
            for start, end, why in CFTC_EXCLUSIONS
            if pd.Timestamp(start) <= date <= pd.Timestamp(end)
        ),
        None,
    )


def number(v):
    try:
        return float(str(v).replace(",", "").strip())
    except ValueError:
        return np.nan


def parse_eia(content, release, table):
    rows = list(csv.reader(io.StringIO(content.decode("cp1252"))))
    if not rows or not rows[0][0].startswith("STUB"):
        raise ValueError("Not an EIA table")
    date_col = 1 if rows[0][1] != "STUB_2" else 2
    date_text = rows[0][date_col].strip()
    observation = pd.to_datetime(
        date_text, format="%m/%d/%Y" if len(date_text.split("/")[-1]) == 4 else "%m/%d/%y"
    )
    release = pd.Timestamp(release)
    if not 0 < (release - observation).days < 30:
        raise ValueError("Invalid observation/release timing")
    out = {
        "observation_date": str(observation.date()),
        "release_date": str(release.date()),
        "available_at": str(
            (release + pd.Timedelta(days=1)).tz_localize("America/New_York").tz_convert("UTC")
        ),
    }
    if table == 1:
        stocks = {
            "Commercial (Excluding SPR)": "crude",
            "Total Motor Gasoline": "gasoline",
            "Distillate Fuel Oil": "distillate",
        }
        for row in rows:
            if not row:
                continue
            if row[0].strip() in stocks:
                key = stocks[row[0].strip()]
                out[key + "_stocks"] = number(row[1])
                out[key + "_change"] = number(row[1]) - number(row[2])
            if len(row) > 3 and row[0].strip() == "Crude Oil Supply":
                label = re.sub(r"^\(\d+\)\s*", "", row[1]).strip()
                key = {
                    "Imports": "imports",
                    "Exports": "exports",
                    "Crude Oil Input to Refineries": "refinery_inputs",
                    "Domestic Production": "production",
                }.get(label)
                if key:
                    out[key] = number(row[2])
        if not all(k + "_stocks" in out for k in stocks.values()):
            raise ValueError("Missing inventory rows")
    else:
        for row in rows:
            if len(row) > 3 and row[0].strip() == "Refiner Inputs and Utilization":
                if "Percent Utilization" in row[1]:
                    out["utilization"] = number(row[2])
                if row[1].strip() == "Crude Oil Inputs":
                    out["refinery_inputs"] = number(row[2])
    return out


def eia(root):
    base = root / "data/weekly/eia"
    base.mkdir(parents=True, exist_ok=True)
    content, _ = fetch(EIA, base / "index.html")
    releases = {}
    for a in BeautifulSoup(content, "html.parser").find_all("a", href=True):
        m = re.search(r"/archive/(20\d\d)/(\d{4}_\d{2}_\d{2})/wpsr_", a["href"])
        if m and 2011 <= int(m[1]) <= 2025:
            releases[m[2]] = urljoin(EIA, a["href"])
    records = []
    failures = []

    def one(item):
        date, url = item
        folder = base / date
        try:
            page, _ = fetch(url, folder / "release.html", attempts=2)
            soup = BeautifulSoup(page, "html.parser")
            links = {
                n: next(
                    (
                        urljoin(url, a["href"])
                        for a in soup.find_all("a", href=True)
                        if re.search(rf"(?:^|/)table{n}\.csv$", a["href"])
                    ),
                    None,
                )
                for n in [1, 2]
            }
            # Corrections are handled by the release-specific errata workbook below.
            errata = next(
                (
                    urljoin(url, a["href"])
                    for a in soup.find_all("a", href=True)
                    if "Errata" in a["href"] and a["href"].endswith(".xlsx")
                ),
                None,
            )
            result = {}
            for n in [1, 2]:
                if not links[n]:
                    raise ValueError(f"No archived table {n} CSV")
                raw, info = fetch(links[n], folder / f"table{n}.csv", attempts=2)
                part = parse_eia(raw, date.replace("_", "-"), n)
                if result and part["observation_date"] != result["observation_date"]:
                    raise ValueError("Table date mismatch")
                result.update(part)
                result[f"table{n}_sha256"] = info["sha256"]
            result["source_url"] = url
            result["errata_url"] = errata or ""
            return result, None
        except Exception as exc:
            return None, {"release": date, "url": url, "error": str(exc)}

    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, (row, err) in enumerate(pool.map(one, sorted(releases.items())), 1):
            if row:
                records.append(row)
            if err:
                failures.append(err)
            if i % 20 == 0:
                print(
                    f"EIA {i}/{len(releases)}: {len(records)} usable, {len(failures)} failures",
                    flush=True,
                )
    pd.DataFrame(records).sort_values("release_date").to_csv(
        base.parent / "eia_weekly_unfiltered.csv", index=False
    )
    (base / "failures.json").write_text(json.dumps(failures, indent=2))
    print(
        f"EIA retrieval complete: {len(records)} releases. Errata review required before use.",
        flush=True,
    )


def cftc(root):
    base = root / "data/weekly/cftc"
    base.mkdir(parents=True, exist_ok=True)
    for name, url in [
        ("special", CFTC + "HistoricalSpecialAnnouncements/index.htm"),
        ("shutdown2019", "https://www.cftc.gov/PressRoom/PressReleases/7864-19"),
        ("shutdown2013", "https://www.cftc.gov/PressRoom/PressReleases/6745-13"),
    ]:
        fetch(url, base / (name + ".html"))
    records = []
    failures = []
    exclusions = []
    for year in range(2010, 2026):
        url = f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
        try:
            raw, info = fetch(url, base / f"{year}.zip")
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                name = next(n for n in z.namelist() if n.lower().endswith(".txt"))
                data = pd.read_csv(z.open(name), dtype=str)
            data.columns = data.columns.str.strip()
            key = next(c for c in data if c.lower() == "cftc_contract_market_code")
            data = data[data[key].str.strip() == "067651"]
            datecol = next(
                c
                for c in data
                if c.lower() in ["report_date_as_yyyy-mm-dd", "report_date_as_mm_dd_yyyy"]
            )
            for _, r in data.iterrows():
                d = pd.Timestamp(r[datecol])
                why = excluded_cftc(d)
                if why:
                    exclusions.append({"observation_date": str(d.date()), "reason": why})
                    continue
                row = {
                    "observation_date": str(d.date()),
                    "available_at": str(
                        (d + pd.Timedelta(days=10))
                        .tz_localize("America/New_York")
                        .tz_convert("UTC")
                    ),
                    "source_url": url,
                    "sha256": info["sha256"],
                    "timing_basis": "conservative 10-day lag; outage/correction quarantine; historical revisions not certified",
                }
                for source, target in [
                    ("Open_Interest_All", "open_interest"),
                    ("M_Money_Positions_Long_All", "money_long"),
                    ("M_Money_Positions_Short_All", "money_short"),
                    ("Prod_Merc_Positions_Long_All", "producer_long"),
                    ("Prod_Merc_Positions_Short_All", "producer_short"),
                ]:
                    row[target] = number(r[source])
                if (
                    row["open_interest"] <= 0
                    or not np.isfinite(
                        [
                            row[k]
                            for k in [
                                "open_interest",
                                "money_long",
                                "money_short",
                                "producer_long",
                                "producer_short",
                            ]
                        ]
                    ).all()
                ):
                    raise ValueError("Invalid positions")
                records.append(row)
            print(f"CFTC {year}: {len(data)} WTI reports", flush=True)
        except Exception as exc:
            failures.append({"year": year, "url": url, "error": str(exc)})
            print(f"CFTC {year}: {exc}", flush=True)
    pd.DataFrame(records).sort_values("observation_date").to_csv(
        base.parent / "cftc_weekly.csv", index=False
    )
    (base / "failures.json").write_text(json.dumps(failures, indent=2))
    (base / "exclusions.json").write_text(json.dumps(exclusions, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", choices=["eia", "cftc"])
    a = p.parse_args()
    globals()[a.source](Path.cwd())
