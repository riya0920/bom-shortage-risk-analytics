"""Download the two public datasets into data/raw/ (no login needed).

    python download_data.py

1. Willems (2008), "Real-world multiechelon supply chains used for inventory
   optimization", Manufacturing & Service Operations Management 10(1):19-23.
   The data set is open to researchers who cite the paper.
2. USAID Supply Chain Management System (SCMS) Delivery History, US-government
   open data (catalog.data.gov). Mirrored as a plain CSV on GitHub.

data/raw/ is gitignored; nothing downloaded here is committed.
"""
from __future__ import annotations

import io
import pathlib
import urllib.request
import zipfile

RAW = pathlib.Path(__file__).resolve().parent / "data" / "raw"

WILLEMS_URL = ("https://seanwillems.com/wp-content/uploads/2020/11/"
               "MSOM_Data_Set_Willems_InExcel.zip")
WILLEMS_XLS = "MSOM-06-038-R2 Data Set in Excel.xls"
SCMS_URL = ("https://raw.githubusercontent.com/ColbyRobinson/"
            "Supply-Chain-Shipment-Delay-Prediction/HEAD/data/"
            "SCMS_Delivery_History_Dataset_20150929.csv")
SCMS_CSV = "SCMS_Delivery_History_Dataset_20150929.csv"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    if not (RAW / WILLEMS_XLS).exists():
        print("downloading Willems (2008) supply-chain workbook ...")
        with zipfile.ZipFile(io.BytesIO(_get(WILLEMS_URL))) as z:
            (RAW / WILLEMS_XLS).write_bytes(z.read(WILLEMS_XLS))
    if not (RAW / SCMS_CSV).exists():
        print("downloading USAID SCMS delivery history ...")
        (RAW / SCMS_CSV).write_bytes(_get(SCMS_URL))
    for f in (WILLEMS_XLS, SCMS_CSV):
        print(f"  data/raw/{f}  {(RAW / f).stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
