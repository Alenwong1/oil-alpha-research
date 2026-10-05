"""Review release-specific EIA errata; conservatively quarantine affected weeks."""

from pathlib import Path
import io, json, re, hashlib
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from .weekly_sources import fetch


def run(root):
    base = root / "data/weekly"
    sources = {}
    unknown = []
    for p in sorted((base / "eia").glob("20*/release.html")):
        soup = BeautifulSoup(p.read_bytes(), "html.parser")
        for a in soup.find_all("a", href=True):
            if "errata" not in a["href"].lower() or not re.search(r"\.xlsx?$", a["href"], re.I):
                continue
            text = (
                a.find_parent("tr").get_text(" ", strip=True)
                if a.find_parent("tr")
                else a.parent.get_text(" ", strip=True)
            )
            label = re.search(r"Errata\s+as\s+of\s+(.+?)(?:\s+XLS|\s+PDF|$)", text, re.I)
            key = label[1].strip() if label else text
            info = json.loads(p.with_suffix(".html.json").read_text())
            sources.setdefault(key, urljoin(info["final_url"], a["href"]))
    events = []
    dates = set()
    failures = []
    for i, (label, url) in enumerate(sources.items(), 1):
        try:
            key = hashlib.sha256(url.encode()).hexdigest()[:16]
            raw, info = fetch(url, base / "errata" / f"{key}.xlsx")
            book = pd.ExcelFile(io.BytesIO(raw))
            found = []
            for sheet in book.sheet_names:
                frame = pd.read_excel(book, sheet, header=None)
                # Errata tables identify affected week-ending dates. Exclude all products,
                # even if only a product outside this strategy was corrected.
                for cell in frame.to_numpy().ravel():
                    if (
                        isinstance(cell, (pd.Timestamp,))
                        or hasattr(cell, "year")
                        and hasattr(cell, "month")
                    ):
                        d = pd.Timestamp(cell)
                        if 2010 <= d.year <= 2025:
                            dates.add(d.normalize())
                            found.append(str(d.date()))
                    elif isinstance(cell, str) and re.fullmatch(
                        r"\d{1,2}/\d{1,2}/\d{2,4}", cell.strip()
                    ):
                        d = pd.to_datetime(cell)
                        if 2010 <= d.year <= 2025:
                            dates.add(d.normalize())
                            found.append(str(d.date()))
            if not found:
                failures.append(
                    {
                        "label": label,
                        "url": url,
                        "reason": "No machine-readable affected dates; manual review required",
                    }
                )
            events.append(
                {
                    "label": label,
                    "url": url,
                    "sha256": info["sha256"],
                    "affected_dates": sorted(set(found)),
                }
            )
        except Exception as exc:
            failures.append({"label": label, "url": url, "reason": str(exc)})
        print(f"Errata {i}/{len(sources)}: {label}", flush=True)
    (base / "errata_review.json").write_text(
        json.dumps({"sources": events, "failures": failures}, indent=2)
    )
    if failures:
        raise ValueError(
            "Errata review incomplete; inspect failures before creating filtered dataset"
        )
    data = pd.read_csv(base / "eia_weekly_unfiltered.csv")
    obs = pd.to_datetime(data.observation_date)
    bad = obs.isin(dates) | obs.sub(pd.Timedelta(days=7)).isin(dates)
    data.loc[bad].to_csv(base / "eia_quarantined.csv", index=False)
    clean = (
        data.loc[~bad]
        .sort_values(["release_date", "observation_date"])
        .drop_duplicates(["release_date", "observation_date"])
    )
    clean.to_csv(base / "eia_weekly.csv", index=False)
    print(
        f"EIA errata-reviewed releases: {len(clean)} retained; {bad.sum()} quarantined", flush=True
    )


if __name__ == "__main__":
    run(Path.cwd())
