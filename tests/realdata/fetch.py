"""
Download the public datasets used by tests/realdata (about 12 MB, not committed).

    python tests/realdata/fetch.py

Each is a well-studied dataset with documented findings, so the analyst is
checked against data and conclusions it was not built around.
"""

import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
SOURCES = {
    # IBM sample data: telecom churn (7,043 customers)
    "Telco-Customer-Churn.csv":
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    # IBM HR Analytics attrition (1,470 employees)
    "emp_attrition.csv":
        "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv",
    # Tableau "Superstore Sales" (Canada, 8,399 order lines; old Mac \r line endings)
    "superstoreSales.csv":
        "https://raw.githubusercontent.com/curran/data/gh-pages/superstoreSales/superstoreSales.csv",
    # Women's E-Commerce Clothing Reviews (23,486 reviews with free text)
    "womens_clothing_reviews.csv":
        "https://raw.githubusercontent.com/AFAgarap/ecommerce-reviews-analysis/master/"
        "Womens%20Clothing%20E-Commerce%20Reviews.csv",
}


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    failed = 0
    for name, url in SOURCES.items():
        dest = DATA / name
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"ok      {name}")
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "analyst-tests"})
            with urllib.request.urlopen(req, timeout=120) as r:
                dest.write_bytes(r.read())
            print(f"fetched {name} ({dest.stat().st_size // 1024} KB)")
        except Exception as e:
            failed += 1
            print(f"FAILED  {name}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
