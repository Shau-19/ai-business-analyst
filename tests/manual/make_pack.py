"""
Build the manual test pack for the web frontend: one file per supported
format, a story that links them, edge-case files, and TEST_PLAN.md with
answers computed from the generated data.

    python tests/manual/make_pack.py

The story: in June 2024 the West region switched courier to SwiftShip
Logistics. Damage and late-delivery complaints rose, West ratings fell, and
the Q3 review (PDF) and courier meeting notes (DOCX) say so. Marketing saw
Paid Social conversions fall in August; Support worked heavy overtime.
"""

import random
import shutil
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "pack"
EDGE = OUT / "edge_cases"
sys.path.insert(0, str(ROOT))

rng = np.random.default_rng(2024)
random.seed(2024)


def fresh():
    if OUT.exists():
        shutil.rmtree(OUT)
    EDGE.mkdir(parents=True)


# ── 1. retail orders (messy CSV) ──────────────────────────────────────────────

def orders():
    src = ROOT / "tests" / "fixtures" / "messy_orders.csv"
    shutil.copy(src, OUT / "01_orders_messy.csv")


# ── 2. marketing (Excel, two sheets) ──────────────────────────────────────────

def marketing():
    rows = []
    base = {"Search": (42000, 0.031, 0.18), "Paid Social": (38000, 0.026, 0.12),
            "Email": (9000, 0.055, 0.22), "Events": (21000, 0.012, 0.30)}
    for m in range(1, 13):
        for ch, (spend, lead_rate, conv_rate) in base.items():
            s = spend * (1 + 0.04 * m) * rng.uniform(0.92, 1.08)
            leads = s * lead_rate * rng.uniform(0.9, 1.1)
            conv = conv_rate
            if ch == "Paid Social" and m in (8, 9):          # iOS tracking change
                leads *= 0.55
                conv *= 0.6
            conversions = leads * conv
            rows.append({"Month": date(2024, m, 1).isoformat(), "Channel": ch, "Spend": round(s, 2),
                         "Leads": int(leads), "Conversions": int(conversions),
                         "Revenue": round(conversions * rng.uniform(1900, 2300), 2)})
    df = pd.DataFrame(rows)
    notes = pd.DataFrame({"Note": ["Only the first sheet is imported.",
                                   "Paid Social lookalike campaigns paused 15 Sep 2024."]})
    with pd.ExcelWriter(OUT / "02_marketing_2024.xlsx") as xw:
        df.to_excel(xw, sheet_name="Monthly", index=False)
        notes.to_excel(xw, sheet_name="Notes", index=False)
    return df


# ── 3. employees (European CSV: ';', cp1252, dd/mm/yyyy, typos) ───────────────

def employees():
    first = ["Chloé", "José", "Asha", "Rahul", "Priya", "Tom", "Mei", "Sara", "Léa", "Jürgen", "Ana", "Omar",
             "Zoë", "Ravi", "Inés", "Kenji"]
    last = ["Martin", "Ramírez", "Verma", "Mehta", "Nair", "O'Brien", "Lin", "Khan", "Dubois", "Müller"]
    depts = ["Sales", "Engineering", "Support", "Finance", "Marketing"]
    rows = []
    for i in range(300):
        d = rng.choice(depts, p=[0.25, 0.3, 0.2, 0.1, 0.15])
        overtime = rng.normal(14 if d == "Support" else 5, 3)
        remote = rng.random() < (0.3 if d == "Support" else 0.6)
        sat = 8.2 - 0.25 * max(overtime, 0) + (0.6 if remote else 0) + rng.normal(0, 0.8)
        sat = int(min(10, max(1, round(sat))))
        left = rng.random() < (0.28 if sat <= 5 else 0.07)
        hire = date(2016, 1, 1) + timedelta(days=int(rng.integers(0, 3100)))
        salary = {"Sales": 58000, "Engineering": 82000, "Support": 46000, "Finance": 70000,
                  "Marketing": 61000}[d] * rng.uniform(0.85, 1.25)
        dept_written = d
        r = rng.random()
        if r < 0.05:
            dept_written = d.lower() + " "
        elif r < 0.08:
            dept_written = {"Engineering": "Enginering", "Marketing": "Marketng", "Support": "Suport",
                            "Finance": "Finanse", "Sales": "Saless"}[d]
        rows.append({
            "Employee ID": f"E{1000 + i:04d}",
            "Name": f"{random.choice(first)} {random.choice(last)}",
            "Department": dept_written,
            "Hire Date": hire.strftime("%d/%m/%Y"),
            "Salary": f"€{salary:,.0f}",
            "Overtime Hours": round(max(overtime, 0), 1),
            "Satisfaction Score": f"{sat}/10" if rng.random() < 0.2 else str(sat),
            "Remote": "Yes" if remote else "No",
            "Left Company": "Yes" if left else "No",
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "03_employees_eu.csv", sep=";", index=False, encoding="cp1252")
    return df


# ── 4. support tickets (tab-separated, UTF-8 BOM, long text) ──────────────────

def tickets():
    cats = ["Billing", "Technical", "Shipping", "Account"]
    texts = {
        "Billing": ["Charged twice for order, asked for refund.", "Invoice shows wrong GST number.",
                    "Refund still not received after two weeks, escalated to finance.",
                    "Promo code not applied at checkout."],
        "Technical": ["App crashes when uploading a photo.", "Cannot reset password, link expired.",
                      "Dashboard loads slowly on mobile.", "Export to Excel fails for large orders."],
        "Shipping": ["Parcel delivered to wrong address.", "Courier marked delivered but nothing arrived.",
                     "Box arrived damaged, SwiftShip driver refused to wait.", "Delivery late by 6 days."],
        "Account": ["Wants to change registered email.", "Asked to close account.",
                    "Two accounts merged by mistake.", "Needs GST invoice for company account."],
    }
    agents = ["Ana", "Kenji", "Léa", "Omar", "Ravi", "Zoë"]
    rows, start = [], datetime(2024, 7, 1, 9, 0)
    for i in range(400):
        c = rng.choice(cats, p=[0.3, 0.3, 0.25, 0.15])
        desc = random.choice(texts[c])
        escalated = "escalated" in desc
        hours = rng.gamma(2, 6) * (2.5 if escalated else 1)
        csat = int(min(5, max(1, round(rng.normal(2.2 if escalated else 4.1, 0.8)))))
        rows.append({"Ticket ID": f"T-{5000 + i}",
                     "Opened": (start + timedelta(hours=int(rng.integers(0, 24 * 120)))).strftime("%Y-%m-%d %H:%M"),
                     "Category": c, "Priority": rng.choice(["Low", "Medium", "High", "Urgent"], p=[.3, .4, .2, .1]),
                     "Resolution Hours": round(hours, 1), "CSAT": csat, "Agent": random.choice(agents),
                     "Description": desc})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "04_support_tickets.csv", sep="\t", index=False, encoding="utf-8-sig")
    return df


# ── 5. Q3 business review (PDF) ───────────────────────────────────────────────

def review_pdf():
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    st = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(OUT / "05_Q3_2024_Business_Review.pdf"), pagesize=A4, title="Q3 2024 Business Review")
    P = lambda t, s="BodyText": Paragraph(t, st[s])  # noqa: E731
    story = [
        P("Q3 2024 Business Review", "Title"),
        P("Prepared by the Strategy Office, 7 October 2024", "Italic"),
        Spacer(1, 12),
        P("Executive summary", "Heading2"),
        P("Q3 revenue reached $5.8M, 4% below plan. Two issues drove most of the shortfall: delivery "
          "quality in the West region and a drop in Paid Social conversions in August and September."),
        P("Regional performance", "Heading2"),
        P("In June 2024 the West region moved its last-mile deliveries to a new courier, SwiftShip "
          "Logistics, to save 11% on shipping costs. Since the switch, damage claims in the West have "
          "roughly tripled and late deliveries doubled. West customer ratings fell to about 3.5 out of 5, "
          "the lowest of any region, while the other regions held near 3.9. Customer notes frequently "
          "mention damaged items, wobbly furniture and late delivery."),
        P("Marketing", "Heading2"),
        P("Paid Social cost per lead rose about 60% in Q3 after Apple's tracking changes reduced audience "
          "match rates. Conversions from Paid Social fell sharply in August. The lookalike campaigns were "
          "paused on 15 September 2024 and budget moved to Search."),
        P("People", "Heading2"),
        P("The Support team averaged 14 hours of overtime per week in Q3, against 5 hours for the rest of "
          "the company, and its satisfaction score is the lowest of any department. Twelve new support "
          "agents will be hired in Q4."),
        P("Q4 targets", "Heading2"),
    ]
    t = Table([["Metric", "Q3 actual", "Q4 target"],
               ["Revenue", "$5.8M", "$6.5M"],
               ["West avg rating", "3.5", "4.0"],
               ["Paid Social CPL", "$41", "$28"],
               ["Support overtime / week", "14 h", "6 h"]])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
    story += [t, Spacer(1, 12),
              P("Owner for the courier review: Priya Nair (Operations). Decision due 31 October 2024.")]
    doc.build(story)


# ── 6. courier meeting notes (DOCX) ───────────────────────────────────────────

def meeting_docx():
    from docx import Document
    d = Document()
    d.add_heading("Courier Review Meeting — West Region", 0)
    d.add_paragraph("Date: 14 October 2024. Attendees: Priya Nair (Operations), Rahul Mehta (Sales), "
                    "Chloé Martin (Customer Support), Diego Ramírez (Finance).")
    d.add_heading("Background", 1)
    d.add_paragraph("SwiftShip Logistics has handled West deliveries since June 2024. Damage claims rose from "
                    "about 12 a month to over 35 a month, and late deliveries doubled.")
    d.add_heading("Decisions", 1)
    for t in ["Terminate the SwiftShip Logistics contract for the West region on 31 December 2024.",
              "Move West deliveries to BlueDart Express from 1 January 2025.",
              "Approve a transition budget of ₹18 lakh, including double-boxing for furniture.",
              "Offer affected West customers a 10% voucher on their next order."]:
        d.add_paragraph(t, style="List Bullet")
    d.add_heading("Action items", 1)
    table = d.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    for cell, h in zip(table.rows[0].cells, ["Action", "Owner", "Due"]):
        cell.text = h
    for row in [("Send termination notice to SwiftShip", "Diego Ramírez", "1 Nov 2024"),
                ("Sign BlueDart Express contract", "Priya Nair", "15 Nov 2024"),
                ("Email vouchers to affected customers", "Chloé Martin", "30 Nov 2024"),
                ("Brief West sales team", "Rahul Mehta", "20 Oct 2024")]:
        cells = table.add_row().cells
        for cell, v in zip(cells, row):
            cell.text = v
    d.add_heading("Damage claims per month (West)", 1)
    t2 = d.add_table(rows=1, cols=2)
    t2.style = "Table Grid"
    t2.rows[0].cells[0].text, t2.rows[0].cells[1].text = "Month", "Claims"
    for m, c in [("May 2024", 11), ("June 2024", 19), ("July 2024", 31), ("August 2024", 36), ("September 2024", 38)]:
        r = t2.add_row().cells
        r[0].text, r[1].text = m, str(c)
    d.save(OUT / "06_Courier_Review_Meeting.docx")


# ── 7. returns policy (TXT) ───────────────────────────────────────────────────

def policy_txt():
    (OUT / "07_Returns_and_Warranty_Policy.txt").write_text("""RETURNS AND WARRANTY POLICY (effective 1 April 2024)

Returns
Most items can be returned within 30 days of delivery. Electronics must be returned within 15 days and
must include all accessories and the original packaging. Custom furniture cannot be returned unless it
arrives damaged.

Refunds
Refunds are issued to the original payment method within 7 business days of the returned item reaching
our warehouse. Cash-on-delivery orders are refunded by bank transfer.

Damaged or late deliveries
If an item arrives damaged, report it within 48 hours with photos. We send a replacement or a full refund,
customer's choice. If a delivery is more than 5 days late, the customer may cancel for a full refund.

Warranty
Electronics carry a 1-year manufacturer warranty. Furniture carries a 2-year warranty against structural
defects, including wobbling caused by faulty joints. Office supplies have no warranty.

Contact
Support is available Monday to Saturday, 9am to 7pm IST, at support@example.com.
""", encoding="utf-8")


# ── 8. edge cases ─────────────────────────────────────────────────────────────

def edge_cases():
    (EDGE / "empty.csv").write_bytes(b"")
    (EDGE / "binary_garbage.csv").write_bytes(bytes(range(256)) * 4)
    (EDGE / "headers_only.csv").write_text("Region,Revenue,Units\n", encoding="utf-8")
    (EDGE / "single_column.csv").write_text("Revenue\n" + "\n".join(str(i * 100) for i in range(1, 31)), encoding="utf-8")
    (EDGE / "slides.pptx").write_bytes(b"PK\x03\x04 not really a presentation")
    pd.DataFrame({"Région": ["Nord", "Sud", "Est"], "Ventes (€)": [1200.5, 980.25, 1430.0]}).to_excel(
        EDGE / "UPPERCASE_NAME_ünïcode.XLSX", index=False)
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(EDGE / "scanned_no_text.pdf"), pagesize=A4)   # shapes only, no text layer
    for i in range(12):
        c.rect(60 + i * 35, 400, 25, 100 + (i % 5) * 40, fill=1)
    c.save()
    # a valid CSV with a misleading extension case and spaces in the name
    (EDGE / "Sales Q1 (final) v2.CSV").write_text("Month,Sales\nJan,100\nFeb,120\nMar,90\n", encoding="utf-8")


# ── expected answers → TEST_PLAN.md ───────────────────────────────────────────

def expected(mk, emp, tk):
    from tools.data_clean import clean_frame
    o, orep = clean_frame(pd.read_csv(OUT / "01_orders_messy.csv", dtype=str, keep_default_na=False))
    rating = o.groupby("Region")["Customer Rating"].mean().round(2)
    m = mk.copy()
    m["Month"] = pd.to_datetime(m["Month"])
    conv = m.groupby("Month")["Conversions"].sum()
    aug, jul = conv[pd.Timestamp("2024-08-01")], conv[pd.Timestamp("2024-07-01")]
    by_ch = m[m.Month.isin([pd.Timestamp("2024-07-01"), pd.Timestamp("2024-08-01")])].pivot(
        index="Channel", columns="Month", values="Conversions")
    ps_drop = by_ch.loc["Paid Social"].iloc[1] - by_ch.loc["Paid Social"].iloc[0]
    e, erep = clean_frame(pd.read_csv(OUT / "03_employees_eu.csv", sep=";", encoding="cp1252", dtype=str,
                                      keep_default_na=False))
    sat = e.groupby("Department")["Satisfaction Score"].mean().round(2)
    ot = e.groupby("Department")["Overtime Hours"].mean().round(1)
    roi = (mk.groupby("Channel")["Revenue"].sum() / mk.groupby("Channel")["Spend"].sum()).round(2)
    return {
        "orders_rows": len(o), "orders_dupes": orep["duplicates_removed"],
        "west_rating": rating["West"], "other_rating": round(o[o.Region != "West"]["Customer Rating"].mean(), 2),
        "dec_2024": int(((pd.to_datetime(o["Order Date"]).dt.strftime("%Y-%m")) == "2024-12").sum()),
        "returns": int((o["Units"] < 0).sum()),
        "mk_rows": len(mk), "aug_conv": int(aug), "jul_conv": int(jul),
        "aug_change": round((aug / jul - 1) * 100, 1), "ps_drop": int(ps_drop),
        "top_roi": roi.idxmax(), "top_roi_val": roi.max(),
        "total_spend": round(mk.Spend.sum() / 1e6, 2),
        "emp_rows": len(e), "emp_depts": e["Department"].nunique(),
        "support_sat": sat["Support"], "other_sat": round(e[e.Department != "Support"]["Satisfaction Score"].mean(), 2),
        "support_ot": ot["Support"], "other_ot": round(e[e.Department != "Support"]["Overtime Hours"].mean(), 1),
        "left_pct": round((e["Left Company"] == "Yes").mean() * 100, 1),
        "tk_rows": len(tk), "esc_pct": round(tk.Description.str.contains("escalated").mean() * 100, 1),
        "billing_csat": round(tk[tk.Category == "Billing"].CSAT.mean(), 2),
        "other_csat": round(tk[tk.Category != "Billing"].CSAT.mean(), 2),
        "avg_res": round(tk["Resolution Hours"].mean(), 1),
    }


PLAN = """# Frontend test plan

Generated by `python tests/manual/make_pack.py`. Every number below was computed from these
exact files, so a mismatch in the app is a real bug.

Open http://localhost:5173/#/app. Start a **new conversation** for each session.

---

## Session 1: messy orders (CSV cleaning, overview, "why" analysis)
Upload `01_orders_messy.csv`.

**Overview**
- [ ] KPI tiles appear within a few seconds. No "NaT" anywhere.
- [ ] Data health shows 20 duplicate rows removed (1,020 → 1,000), "Nrth → North" style typo fixes,
      and a warning on Units about 3 extreme values (9,999).
- [ ] Chart subtitles say "Excludes 3 extreme values (9,999)". Region bars are roughly even (no 22K spike).
- [ ] Suggested questions include "Why is customer rating lower in West?".

**Data tab**
- [ ] 10 columns. Units, Unit Price ($), Discount and Customer Rating are shown as Measure. Order Date is shown as Date.
- [ ] The Cleaning column lists what changed for each column.

**Ask**
| Question | Expected |
|---|---|
| How many orders are there? | {orders_rows:,} |
| How many orders were placed in December 2024? | {dec_2024} |
| How many orders have negative units (returns)? | {returns} |
| Which region has the lowest average customer rating? | West ({west_rating}) |
| Why are customer ratings lower in the West region? | "Why analysis" chip. West {west_rating} vs {other_rating}; West orders mention damage (“damaged, cracked”) and complaints (“complained, wrong”) about twice as often as other regions; the gap is mostly explained. The chart puts West first. |
| Why are Electronics ratings so low? | Says they're about the same as the rest, with no invented cause |
| Show orders per region as a pie chart | Pie with 5 slices |
| कितने ऑर्डर हैं? | {orders_rows:,}, answered in Hindi |

**Dashboards and forecast**
- [ ] Pin 2 charts. Reorder them, toggle wide, refresh the page: the layout persists.
- [ ] Forecast tab: choose Units and run it. The chart shows history plus a forecast line joined to the last actual point.

---

## Session 2: the whole story (CSV + PDF + DOCX + TXT)
Upload `01_orders_messy.csv`, `05_Q3_2024_Business_Review.pdf`, `06_Courier_Review_Meeting.docx` and
`07_Returns_and_Warranty_Policy.txt` together.

| Question | Expected |
|---|---|
| Why are customer ratings lower in the West region? | Same analysis as Session 1, plus a "From your documents" paragraph mentioning **SwiftShip Logistics** (June 2024 courier switch) |
| Which courier handles West deliveries? | SwiftShip Logistics (moving to BlueDart Express from 1 Jan 2025) |
| What is the Q4 revenue target? | $6.5M |
| Who owns the BlueDart contract signing and when is it due? | Priya Nair, 15 Nov 2024 |
| What budget was approved for the courier transition? | ₹18 lakh |
| How many damage claims were there in August 2024? | 36 |
| How long do customers have to return electronics? | 15 days |
| How quickly are refunds issued? | Within 7 business days |
| What warranty do furniture items have? | 2 years against structural defects |
| What is the CEO's phone number? | Refuses ("not available"), with no source cited |
| Compare West ratings with what the Q3 review says | Ratings from the data plus the review's explanation |

---

## Session 3: marketing (Excel, multiple sheets)
Upload `02_marketing_2024.xlsx`. Only the first sheet ("Monthly") is imported: {mk_rows} rows.

| Question | Expected |
|---|---|
| What was total marketing spend in 2024? | ${total_spend}M |
| Which channel has the best revenue to spend ratio? | {top_roi} ({top_roi_val}×) |
| Why did conversions drop in August 2024? | Down {aug_change}% ({jul_conv} → {aug_conv}); Paid Social is the main contributor ({ps_drop}) |
| Show conversions by month as a line chart | 12 points, a dip in Aug–Sep |
| Forecast revenue for the next 6 periods | Forecast chart |

---

## Session 4: European HR export (`;` separator, cp1252, dd/mm/yyyy, typos)
Upload `03_employees_eu.csv`.

- [ ] Data tab: names show accents correctly (Chloé, José, Jürgen, Zoë). No garbled characters.
- [ ] 9 columns, not 1. If you see one long column, the separator wasn't detected.
- [ ] Department has {emp_depts} values. The typos (Enginering, Suport, "sales ") were merged, and Data health lists them.
- [ ] Salary is a Measure (the € signs were stripped), and Hire Date is a Date.

| Question | Expected |
|---|---|
| How many employees are there? | {emp_rows} |
| What percentage of employees left the company? | about {left_pct}% |
| Which department has the lowest average satisfaction score? | Support ({support_sat}) |
| Why is satisfaction score lower in Support? | Support {support_sat} vs {other_sat}. **Overtime Hours** averages about {support_ot} vs {other_ot}, plus possibly Remote = "No" |

---

## Session 5: support tickets (tab-separated, BOM, long text)
Upload `04_support_tickets.csv`.

| Question | Expected |
|---|---|
| How many tickets are there? | {tk_rows} |
| What is the average resolution time? | about {avg_res} hours |
| Why is CSAT lower for Billing? | Billing {billing_csat} vs {other_csat}; the description word "escalated" (refund delays) drives it |
| How many tickets mention refund? | Counts rows whose Description contains "refund" |

---

## Session 6: edge cases (error handling)
Upload everything in `edge_cases/` at once.

- [ ] The app keeps working. The toast lists errors for `empty.csv`, `binary_garbage.csv`, `headers_only.csv`,
      `slides.pptx` and `scanned_no_text.pdf`, each with a readable reason.
- [ ] `single_column.csv`, `UPPERCASE_NAME_ünïcode.XLSX` and `Sales Q1 (final) v2.CSV` import fine (Data tab).
- [ ] Ask "What were total sales?" → 310 (from the Q1 file).

## Session 7: housekeeping
- [ ] Rename a conversation from the sidebar, then refresh: the new name persists.
- [ ] Delete a conversation: it disappears, and its uploads are removed from disk.
- [ ] Theme toggle (system / light / dark): charts stay readable in both.
- [ ] Mobile width (DevTools 390px): the sidebar collapses and nothing scrolls sideways.
- [ ] Wrong access key: the app asks again and doesn't crash.
"""


def main():
    fresh()
    orders()
    mk = marketing()
    emp = employees()
    tk = tickets()
    review_pdf()
    meeting_docx()
    policy_txt()
    edge_cases()
    exp = expected(mk, emp, tk)
    (OUT / "TEST_PLAN.md").write_text(PLAN.format(**exp), encoding="utf-8")
    print(f"Wrote {len(list(OUT.rglob('*')))} files to {OUT}")
    for k, v in exp.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
