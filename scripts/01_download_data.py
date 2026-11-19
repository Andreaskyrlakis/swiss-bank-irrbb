"""Download the public inputs again (no account needed).

* BCV reports (bcv.ch): Pillar 3 disclosure at 31.12.2025 and 31.12.2024, annual report 2025.
* FINMA: consultation draft of Circular 2019/2 (4 Nov 2026), which shows the shock table in force.
* SNB data portal (data.snb.ch): SARON and policy rate, money-market rates, Confederation bond
  yields (individual issues and the discontinued fitted series), client rates on new business.

The tables of the BCV PDFs were transcribed by hand into data/processed/ (page numbers in the files).
"""

import urllib.request
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "raw" / "pdf"
SNB_DIR = ROOT / "data" / "raw" / "snb"
USER_AGENT = "Mozilla/5.0 (research script)"

BCV = "https://www.bcv.ch/content/dam/bcv/fichiers/publications/publications-institutionnelles/rapports/ra-et-rse/"
PDFS = {
    "bcv_pillar3_2025-12-31_fr.pdf": BCV + "bale-iii/2026/BCV-26-02-09-Bale-III-Final-Pilier-3-au-31-12-2025.pdf",
    "bcv_pillar3_2024-12-31_fr.pdf": BCV + "bale-iii/2025/13022025-bcv-bale-iii-pilier-3-au-31-12-2024-fr.pdf",
    "bcv_annual_report_2025_fr.pdf": BCV + "rapports-annuels-et-intermediaires/2025/2025_rapport-annuel-bcv_fr_web.pdf",
    "finma_rs19-02_draft_2026-11_fr.pdf": (
        "https://www.finma.ch/fr/~/media/finma/dokumente/dokumentencenter/anhoerungen/"
        "laufende-anhoerungen/20261104-rundschreiben-19-02/20261104_rs19_02-ent.pdf"),
}
SNB_CUBES = ["snbgwdzid", "zimoma", "rendoeid", "rendoblid", "rendoblim", "zikrepro"]


def fetch(url: str, target: Path) -> None:
    request = urllib.request.Request(quote(url, safe=":/?=&~%"), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        target.write_bytes(response.read())
    print(f"saved {target.relative_to(ROOT)} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    SNB_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in PDFS.items():
        fetch(url, PDF_DIR / name)
    for cube in SNB_CUBES:
        fetch(f"https://data.snb.ch/api/cube/{cube}/data/csv/en", SNB_DIR / f"{cube}.csv")
        fetch(f"https://data.snb.ch/api/cube/{cube}/dimensions/en", SNB_DIR / f"{cube}_dims.json")
