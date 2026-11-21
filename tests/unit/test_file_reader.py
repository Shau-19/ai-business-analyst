"""CSV/Excel files as people actually export them."""

import pandas as pd
import pytest

from parsers.hybrid_csv_processor import HybridCSVProcessor

read = HybridCSVProcessor._read_frame
ROWS = "Name,City,Sales\nChloé,Paris,10\nJosé,Madrid,20\nAsha,Pune,30\n"


@pytest.mark.parametrize("label, data", [
    ("utf-8", ROWS.encode("utf-8")),
    ("utf-8 with BOM", ROWS.encode("utf-8-sig")),
    ("Excel cp1252", ROWS.encode("cp1252")),
    ("semicolon", ROWS.replace(",", ";").encode()),
    ("tab", ROWS.replace(",", "\t").encode()),
    ("pipe", ROWS.replace(",", "|").encode()),
    ("CRLF", ROWS.replace("\n", "\r\n").encode()),
])
def test_csv_variants_read_as_the_same_table(tmp_path, label, data):
    p = tmp_path / "f.csv"
    p.write_bytes(data)
    df = read(str(p))
    assert list(df.columns) == ["Name", "City", "Sales"], label
    assert df["Name"].tolist() == ["Chloé", "José", "Asha"], label
    assert df["Sales"].tolist() == [10, 20, 30], label


def test_semicolon_csv_with_decimal_commas_keeps_columns(tmp_path):
    p = tmp_path / "eu.csv"
    p.write_text('Product;Price\n"Desk";"1.299,50"\n"Lamp";"45,00"\n"Pen";"2,10"\n', encoding="utf-8")
    df = read(str(p))
    assert list(df.columns) == ["Product", "Price"]


def test_quoted_commas_stay_in_one_cell(tmp_path):
    p = tmp_path / "q.csv"
    p.write_text('id,note\n1,"late, but fine"\n2,"ok"\n3,"damaged, refunded"\n', encoding="utf-8")
    assert read(str(p))["note"].tolist() == ["late, but fine", "ok", "damaged, refunded"]


def test_single_column_csv(tmp_path):
    p = tmp_path / "one.csv"
    p.write_text("value\n1\n2\n3\n", encoding="utf-8")
    assert read(str(p))["value"].tolist() == [1, 2, 3]


def test_uppercase_extension(tmp_path):
    p = tmp_path / "DATA.CSV"
    p.write_text(ROWS, encoding="utf-8")
    assert len(read(str(p))) == 3


@pytest.mark.parametrize("data, match", [
    (b"", "empty"),
    (b"   \n\n", "empty"),
    (b"\x00\x01\x02garbage\x00", "binary"),
])
def test_bad_files_fail_with_a_readable_reason(tmp_path, data, match):
    p = tmp_path / "bad.csv"
    p.write_bytes(data)
    with pytest.raises(ValueError, match=match):
        read(str(p))


def test_excel_file(tmp_path):
    p = tmp_path / "f.xlsx"
    pd.DataFrame({"Name": ["Chloé", "Asha"], "Sales": [10, 20]}).to_excel(p, index=False)
    assert read(str(p))["Sales"].tolist() == [10, 20]
