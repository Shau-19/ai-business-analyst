"""
Generate a messy 1,000-row retail orders CSV plus the ground truth computed
from the clean values it was rendered from.
"""
import json, random
from datetime import date, timedelta
import pandas as pd

random.seed(42)
N = 1000
OUT = __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0]

REGIONS = ["North", "South", "East", "West", "Central"]
CATS = ["Electronics", "Furniture", "Office Supplies"]
REPS = ["Asha Verma", "Rahul Mehta", "Chloé Martin", "Diego Ramírez", "Priya Nair", "Tom O'Brien", "Mei Lin", "Sara Khan"]
PRICES = {"Electronics": (150, 1800), "Furniture": (80, 950), "Office Supplies": (3, 60)}

GOOD_NOTES = [
    "Delivered on time, customer happy.", "Repeat buyer — asked about bulk pricing.",
    "Smooth order, no issues.", "Customer praised the packaging.", "Upsold an extended warranty.",
]
BAD_NOTES = [
    "Item arrived damaged; customer requested a replacement.",
    "Late delivery (5 days), customer complained to support.",
    "Wrong colour shipped, \"very disappointed\" per the customer.",
    "Customer says the chair wobbles, wants a refund.",
    "Box was crushed in transit, screen cracked 😞",
]
NEUTRAL = ["Paid via UPI ₹ transfer.", "Invoice re-issued with GST number.", "Called twice to confirm address, café delivery.", ""]

def messy_region(r):
    k = random.random()
    if k < 0.70: return r
    if k < 0.80: return r.lower()
    if k < 0.87: return r.upper()
    if k < 0.94: return f" {r} "
    if k < 0.97: return {"North": "Nrth", "South": "Sout", "East": "Est", "West": "Wst", "Central": "Centrl"}[r]
    return ""

def messy_date(d):
    k = random.random()
    if k < 0.40: return d.strftime("%Y-%m-%d")
    if k < 0.60: return d.strftime("%d/%m/%Y")
    if k < 0.75: return d.strftime("%b %d, %Y")
    if k < 0.88: return d.strftime("%Y/%m/%d")
    if k < 0.97: return d.strftime("%d-%b-%Y")
    return ""

def messy_units(u):
    k = random.random()
    if k < 0.85: return str(u)
    if k < 0.93: return f"{u} units"
    if k < 0.97: return f"{u}.0"
    return ""

def messy_price(p):
    k = random.random()
    if k < 0.45: return f"${p:,.2f}"
    if k < 0.75: return f"{p:.2f}"
    if k < 0.90: return f"{p:,.2f}"
    if k < 0.97: return f"USD {p:.0f}" if p == int(p) else f"{p}"
    return "N/A"

def messy_discount(d):
    if d == 0: return random.choice(["0", "none", "", "0%"])
    k = random.random()
    if k < 0.5: return f"{round(d * 100)}%"
    if k < 0.85: return f"{d:.2f}"
    return f"{round(d * 100)} %"

def messy_rating(r):
    k = random.random()
    if r is None: return random.choice(["", "n/a", "-"])
    if k < 0.80: return str(r)
    if k < 0.92: return f"{r}/5"
    return f"{r}.0"

rows, clean = [], []
start = date(2023, 1, 1)
for i in range(N):
    region = random.choices(REGIONS, weights=[25, 20, 20, 25, 10])[0]
    cat = random.choices(CATS, weights=[35, 25, 40])[0]
    d = start + timedelta(days=int(random.triangular(0, 729, 600)))
    units = random.randint(1, 25)
    if random.random() < 0.04: units = -random.randint(1, 5)            # returns
    if random.random() < 0.004: units = random.choice([5000, 9999])     # data-entry outliers
    lo, hi = PRICES[cat]
    price = round(random.uniform(lo, hi), 2)
    disc = random.choice([0, 0, 0, 0.05, 0.1, 0.15, 0.2])
    bad = random.random() < (0.35 if region == "West" else 0.12)       # West has more problems
    rating = random.choice([1, 2, 2, 3]) if bad else random.choice([4, 4, 5, 5, 3])
    if random.random() < 0.05: rating = None
    note = random.choice(BAD_NOTES if bad else GOOD_NOTES)
    if random.random() < 0.3: note += " " + random.choice(NEUTRAL)
    if random.random() < 0.03: note = (note + " ") * 12                  # very long
    if random.random() < 0.03: note = note.replace(". ", ".\n")          # embedded newline
    if random.random() < 0.06: note = ""
    rep = random.choice(REPS) if random.random() > 0.04 else ""

    rec = {
        "Order ID": f"ORD-{i + 1:05d}",
        "Order Date": messy_date(d),
        "Region": messy_region(region),
        "Product Category": random.choice([cat, cat, cat, cat.lower(), cat.upper(), f"{cat} "]),
        "Sales Rep": rep,
        "Units": messy_units(units),
        "Unit Price ($)": messy_price(price),
        "Discount": messy_discount(disc),
        "Customer Rating": messy_rating(rating),
        "Notes": note,
    }
    rows.append(rec)
    clean.append({"id": rec["Order ID"], "date": d, "region": region, "cat": cat, "rep": rep or None,
                  "units": units, "price": price, "disc": disc, "rating": rating, "bad": bad})

# ~2% exact duplicate rows (double-exported)
for j in random.sample(range(N), 20):
    rows.append(dict(rows[j])); clean.append(dict(clean[j]))
random.shuffle(idx := list(range(len(rows))))
rows = [rows[k] for k in idx]; clean = [clean[k] for k in idx]

pd.DataFrame(rows).to_csv(f"{OUT}/messy_orders.csv", index=False, encoding="utf-8")

# Ground truth from clean values, de-duplicated by Order ID (what an analyst means)
c = pd.DataFrame(clean).drop_duplicates("id")
c["revenue"] = c["units"] * c["price"] * (1 - c["disc"])
sane = c[(c["units"] > 0) & (c["units"] < 1000)]
truth = {
    "rows_in_file": len(rows), "unique_orders": int(c["id"].nunique()), "duplicate_rows": len(rows) - int(c["id"].nunique()),
    "orders_by_region": c["region"].value_counts().to_dict(),
    "avg_rating_by_region": c.groupby("region")["rating"].mean().round(2).to_dict(),
    "worst_rated_region": c.groupby("region")["rating"].mean().idxmin(),
    "revenue_total_all_rows": round(float(c["revenue"].sum()), 2),
    "revenue_total_excl_outliers_and_returns": round(float(sane["revenue"].sum()), 2),
    "revenue_by_category": sane.groupby("cat")["revenue"].sum().round(2).to_dict(),
    "returns_count": int((c["units"] < 0).sum()),
    "outlier_rows": int((c["units"] >= 1000).sum()),
    "orders_per_month_2024_12": int(((c["date"].map(lambda d: (d.year, d.month))) == (2024, 12)).sum()),
    "top_rep_by_orders": c["rep"].value_counts().idxmax(),
    "damaged_mentions": int(sum("damaged" in r["Notes"].lower() for r in rows if r["Order ID"] in set(c["id"]))),
    "west_bad_share": round(float(c[c.region == "West"]["bad"].mean()), 3),
    "other_bad_share": round(float(c[c.region != "West"]["bad"].mean()), 3),
}
json.dump(truth, open(f"{OUT}/truth.json", "w"), indent=1, default=str)
print(json.dumps(truth, indent=1, default=str))
