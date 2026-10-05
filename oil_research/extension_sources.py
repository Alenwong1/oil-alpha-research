"""Public historical releases and archive captures. No current-vintage substitution."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import time
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from .data import sha256

EIA_INDEX = "https://www.eia.gov/outlooks/steo/outlook.php"
ALFRED = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"
CDX = "https://web.archive.org/cdx/search/cdx"


def fetch(url, path, params=None, attempts=3):
    path = Path(path)
    if path.exists() and path.with_suffix(path.suffix + ".json").exists():
        info = json.loads(path.with_suffix(path.suffix + ".json").read_text())
        if sha256(path) != info["sha256"]:
            raise ValueError(f"Cache checksum mismatch: {path}")
        return path.read_bytes(), info
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(attempts):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=45,
                headers={"User-Agent": "OilResearch/0.2 personal academic research"},
            )
            if response.status_code == 429:
                # Respect rate limiting; never work around a denied endpoint.
                time.sleep(min(30, 5 * (attempt + 1)))
                continue
            response.raise_for_status()
            info = {
                "requested_url": response.request.url,
                "final_url": response.url,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": hashlib.sha256(response.content).hexdigest(),
                "last_modified": response.headers.get("Last-Modified"),
            }
            path.write_bytes(response.content)
            path.with_suffix(path.suffix + ".json").write_text(json.dumps(info, indent=2))
            return response.content, info
        except requests.RequestException:
            if attempt + 1 == attempts:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Source rate limited after {attempts} attempts: {url}")


def eia_releases(content, start_year, end_year):
    soup = BeautifulSoup(content, "html.parser")
    releases, exclusions = [], []
    for tr in soup.find_all("tr"):
        a = tr.find("a", href=lambda h: h and re.search(r"base\.xlsx?$", h, re.I))
        if not a:
            continue
        text = tr.get_text(" ", strip=True)
        match = re.search(r"\b(\d{1,2}/\d{1,2}/\d{4})\b", text)
        if not match:
            continue
        release = pd.Timestamp(datetime.strptime(match[1], "%m/%d/%Y"))
        if not start_year <= release.year <= end_year:
            continue
        row = {
            "release_date": str(release.date()),
            "url": urljoin(EIA_INDEX, a["href"]),
            "notice": "notice" in text.lower(),
        }
        # Errata often change the workbook after its nominal release. Quarantine them.
        if row["notice"]:
            exclusions.append(
                {
                    **row,
                    "reason": "Release has a correction notice; original unamended vintage not established",
                }
            )
        else:
            releases.append(row)
    return sorted(releases, key=lambda r: r["release_date"]), exclusions


SERIES = {
    "copr_opec": "opec_crude_production",
    "copc_opec": "opec_capacity",
    "cops_opec": "opec_spare_capacity",
    "papr_world": "world_liquids_production",
    "patc_world": "world_liquids_consumption",
    "papr_nonopec": "non_opec_production",
    "pasc_oecd_t3": "oecd_inventory",
    "coprpus": "us_crude_production",
    "coscpus": "us_commercial_crude_inventory",
    "coripus": "us_refinery_crude_input",
    "pasc_us": "us_commercial_liquids_inventory",
    "pasxpus": "us_total_commercial_inventory",
    "cosx_draw": "us_commercial_crude_withdrawals",
    "prod_draw": "us_product_inventory_withdrawals",
    "wtipuus": "wti_price",
    "brepuus": "brent_price",
    "orcucus": "refinery_utilization",
    "orutcus": "refinery_utilization",
}


def parse_eia_workbook(content, release):
    file = pd.ExcelFile(io.BytesIO(content))
    release_date = pd.Timestamp(release["release_date"])
    month = release_date.to_period("M")
    records = []
    for sheet in file.sheet_names:
        if sheet.lower() not in ["2tab", "3atab", "3ctab", "4atab", "4btab", "3tab"]:
            continue
        frame = pd.read_excel(file, sheet_name=sheet, header=None)
        # These workbooks use year on row 2 and month on row 3; assert rather than guess.
        if len(frame) < 5:
            continue
        month_columns = {}
        year = None
        for col in range(2, len(frame.columns)):
            y = frame.iloc[2, col]
            if isinstance(y, (int, float)) and pd.notna(y) and 1990 <= y <= 2100:
                year = int(y)
            label = str(frame.iloc[3, col]).strip()[:3]
            if year and label in [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]:
                period = pd.Period(f"{year}-{datetime.strptime(label, '%b').month:02d}", freq="M")
                month_columns[period] = col
        for _, row in frame.iterrows():
            code = str(row.iloc[0]).strip().lower()
            if code not in SERIES:
                continue
            for offset, role in [
                (-1, "last_month_estimate"),
                (-2, "previous_month_estimate"),
                (0, "current_month_forecast"),
                (3, "three_month_forecast"),
            ]:
                observed = month + offset
                col = month_columns.get(observed)
                if col is None:
                    continue
                value = pd.to_numeric(row.iloc[col], errors="coerce")
                if pd.isna(value):
                    continue
                records.append(
                    {
                        "series": SERIES[code],
                        "role": role,
                        "observation_period": str(observed),
                        "value": float(value),
                        "release_date": release["release_date"],
                        # No release-time assumption: full extra day after release.
                        "available_at": (release_date + pd.Timedelta(days=2))
                        .tz_localize("UTC")
                        .isoformat(),
                        "source_url": release["url"],
                        "sheet": sheet,
                        "source_code": code,
                    }
                )
    return records


def download_eia(root, config):
    cache = root / "data/extension/eia"
    content, _ = fetch(EIA_INDEX, cache / "index.html")
    releases, excluded = eia_releases(
        content, config["start_year"], config["last_development_year"]
    )
    records, failures = [], []

    def work(release):
        try:
            content, info = fetch(release["url"], cache / Path(urlparse(release["url"]).path).name)
            rows = parse_eia_workbook(content, release)
            if not any(r["series"] == "opec_crude_production" for r in rows):
                return [], {
                    **release,
                    "reason": "No parsed OPEC series; workbook layout or coverage differs",
                }
            return [{**r, "source_sha256": info["sha256"]} for r in rows], None
        except Exception as exc:
            return [], {**release, "reason": str(exc)}

    with ThreadPoolExecutor(max_workers=3) as pool:
        for i, (rows, failure) in enumerate(pool.map(work, releases), 1):
            records.extend(rows)
            if failure:
                failures.append(failure)
            if i % 12 == 0:
                print(f"EIA: {i}/{len(releases)} releases, {len(records)} values", flush=True)
    pd.DataFrame(records).drop_duplicates(["series", "role", "release_date"]).to_csv(
        root / "data/extension/eia_releases.csv", index=False
    )
    (cache / "exclusions.json").write_text(json.dumps(excluded + failures, indent=2))
    print(
        f"EIA complete: {len(records)} values, {len(excluded)} correction notices quarantined, {len(failures)} parse/download failures",
        flush=True,
    )


def validate_vintage_csv(content, series, vintage):
    frame = pd.read_csv(io.BytesIO(content), index_col=0, parse_dates=True)
    expected = [f"{s}_{pd.Timestamp(vintage):%Y%m%d}" for s in series]
    if list(frame.columns) != expected:
        raise ValueError(
            f"ALFRED vintage mismatch: expected {expected}, received {list(frame.columns)}"
        )
    if (frame.index > pd.Timestamp(vintage)).any():
        raise ValueError("ALFRED returned observations after the requested vintage")
    frame.columns = series
    return frame.apply(pd.to_numeric, errors="coerce")


def download_curve(root, config):
    cache = root / "data/extension/alfred"
    series = config["curve_series"]
    months = pd.date_range(
        f"{config['start_year']}-01-01", f"{config['last_development_year']}-12-01", freq="MS"
    )
    records, failures = [], []

    def work(vintage):
        try:
            begin, end = vintage - pd.Timedelta(days=100), vintage - pd.Timedelta(days=1)
            params = {
                "id": ",".join(series),
                "cosd": ",".join([str(begin.date())] * len(series)),
                "coed": ",".join([str(end.date())] * len(series)),
                "vintage_date": ",".join([str(vintage.date())] * len(series)),
            }
            content, info = fetch(ALFRED, cache / f"{vintage:%Y-%m}.csv", params=params)
            frame = validate_vintage_csv(content, series, vintage)
            rows = []
            for s in series:
                vals = frame[s].dropna()
                if len(vals) < 22:
                    raise ValueError(f"Insufficient historical values for {s}")
                for role, value in [
                    ("level", vals.iloc[-1]),
                    ("change_21_observations", vals.iloc[-1] - vals.iloc[-22]),
                ]:
                    rows.append(
                        {
                            "series": s,
                            "role": role,
                            "observation_date": str(vals.index[-1].date()),
                            "vintage_date": str(vintage.date()),
                            "available_at": (vintage + pd.Timedelta(days=1))
                            .tz_localize("UTC")
                            .isoformat(),
                            "value": float(value),
                            "source_url": info["requested_url"],
                            "source_sha256": info["sha256"],
                        }
                    )
            time.sleep(0.5)
            return rows, None
        except Exception as exc:
            return [], {"vintage": str(vintage.date()), "reason": str(exc)}

    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, (rows, error) in enumerate(pool.map(work, months), 1):
            records.extend(rows)
            if error:
                failures.append(error)
            if i % 12 == 0:
                print(f"ALFRED: {i}/{len(months)} monthly vintages", flush=True)
    pd.DataFrame(records).to_csv(root / "data/extension/curve_releases.csv", index=False)
    (cache / "failures.json").write_text(json.dumps(failures, indent=2))
    print(f"ALFRED complete: {len(records)} values, {len(failures)} failures", flush=True)


def parse_archive_headlines(content, captured_at):
    soup = BeautifulSoup(content, "html.parser")
    rows = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not re.search(
            r"/(?:Latest-Energy-News/World-News|Energy/Crude-Oil|Commodities/Oil-Prices)/[^/]+\.html",
            href,
        ):
            continue
        title = a.get_text(" ", strip=True)
        if len(title.split()) < 4 or title in seen:
            continue
        # Title comes only from the captured HTML, never from today's article page.
        seen.add(title)
        rows.append(
            {
                "title": title,
                "article_url": href,
                "captured_at": captured_at.isoformat(),
                "available_at": (captured_at + pd.Timedelta(days=1)).isoformat(),
            }
        )
    return rows


def download_news(root, config):
    cache = root / "data/extension/oilprice_archive"
    params = {
        "url": "oilprice.com/Latest-Energy-News/World-News/",
        "output": "json",
        "filter": "statuscode:200",
        "collapse": "timestamp:6",
        "from": str(max(2010, config["start_year"])),
        "to": str(config["last_development_year"]),
    }
    content, _ = fetch(CDX, cache / "capture_index.json", params=params)
    table = json.loads(content)
    captures = [dict(zip(table[0], row)) for row in table[1:]]
    records, coverage, failures = [], [], []
    # Sequential, courteous archive access. There is one earliest capture per month.
    for i, capture in enumerate(captures, 1):
        timestamp = capture["timestamp"]
        url = f"https://web.archive.org/web/{timestamp}id_/{capture['original']}"
        try:
            content, info = fetch(url, cache / f"{timestamp}.html", attempts=2)
            actual = re.search(r"/web/(\d{14})", info["final_url"])
            if not actual:
                raise ValueError("Archive redirected outside a timestamped capture")
            observed = pd.Timestamp(datetime.strptime(actual[1], "%Y%m%d%H%M%S"), tz="UTC")
            if observed.year > config["last_development_year"]:
                raise ValueError("Capture resolved into reserved future period")
            rows = parse_archive_headlines(content, observed)
            if not rows:
                raise ValueError("No static historical headlines parsed; no current-page fallback")
            records.extend(
                [
                    {**r, "archive_url": info["final_url"], "source_sha256": info["sha256"]}
                    for r in rows
                ]
            )
            coverage.append(
                {
                    "captured_at": observed.isoformat(),
                    "available_at": (observed + pd.Timedelta(days=1)).isoformat(),
                    "headlines": len(rows),
                    "source_url": info["final_url"],
                }
            )
            time.sleep(0.5)
        except Exception as exc:
            failures.append({"timestamp": timestamp, "reason": str(exc)})
        if i % 12 == 0:
            print(
                f"OilPrice archive: {i}/{len(captures)} captures; {len(records)} headline records",
                flush=True,
            )
    if records:
        frame = (
            pd.DataFrame(records).sort_values("captured_at").drop_duplicates("title", keep="first")
        )
        frame.to_csv(root / "data/extension/headlines.csv", index=False)
    pd.DataFrame(coverage).to_csv(root / "data/extension/news_coverage.csv", index=False)
    (cache / "failures.json").write_text(json.dumps(failures, indent=2))
    print(
        f"OilPrice complete: {len(records)} captured titles, {len(failures)} unavailable captures",
        flush=True,
    )


def download_markets(root, config):
    import yfinance as yf

    cache = root / "data/extension/markets"
    cache.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache / ".yf-cache"))
    manifest = {}
    for ticker in config["additional_tickers"]:
        path = cache / f"{ticker.replace('^','INDEX_')}.csv"
        if path.exists():
            frame = pd.read_csv(path, index_col="Date", parse_dates=True)
        else:
            frame = yf.Ticker(ticker).history(
                start=f"{config['start_year']}-01-01",
                end=f"{config['last_development_year']+1}-01-01",
                auto_adjust=True,
                repair=False,
                raise_errors=True,
            )
            if frame.empty:
                raise ValueError(f"No market data for {ticker}")
            frame.index = frame.index.tz_localize(None).normalize()
            frame.index.name = "Date"
            frame.to_csv(path)
        manifest[ticker] = {
            "sha256": sha256(path),
            "rows": len(frame),
            "first": str(frame.index.min().date()),
            "last": str(frame.index.max().date()),
            "path": path.name,
        }
        print(f"Additional market data: {ticker}, {len(frame)} rows", flush=True)
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=["eia", "curve", "news", "markets"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / "extension_config.json").read_text())
    (root / "data/extension").mkdir(parents=True, exist_ok=True)
    {
        "eia": download_eia,
        "curve": download_curve,
        "news": download_news,
        "markets": download_markets,
    }[args.source](root, config)


if __name__ == "__main__":
    main()
