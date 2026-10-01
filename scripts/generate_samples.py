"""Create the committed synthetic receipt PDFs."""

from __future__ import annotations

from pathlib import Path

from app.infrastructure.pdf.renderer import pdf_from_text_pages

OK = """Oak & Ember Cafe
STORE: Oak & Ember Cafe
DATE: 2026-03-15
ITEM: Latte | QTY: 2 | UNIT: 4.50 | AMT: 9.00
ITEM: Croissant | QTY: 1 | UNIT: 3.50 | AMT: 3.50
SUBTOTAL: 12.50
TAX: 1.00
TOTAL: 13.50
"""

MISMATCH = """Harbor Books
STORE: Harbor Books
DATE: 2026-04-02
ITEM: Notebook | QTY: 1 | UNIT: 6.00 | AMT: 6.00
SUBTOTAL: 10.00
TAX: 1.00
TOTAL: 11.00
"""

PAGE1 = """North Pier Market
STORE: North Pier Market
DATE: 2026-05-20
ITEM: Apples | QTY: 3 | UNIT: 1.20 | AMT: 3.60
ITEM: Bread | QTY: 1 | UNIT: 2.40 | AMT: 2.40
"""

PAGE2 = """North Pier Market
ITEM: Milk | QTY: 1 | UNIT: 1.80 | AMT: 1.80
SUBTOTAL: 7.80
TAX: 0.40
TOTAL: 8.20
"""


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "samples"
    root.mkdir(parents=True, exist_ok=True)
    (root / "receipt_ok.pdf").write_bytes(pdf_from_text_pages([OK]))
    (root / "receipt_mismatch.pdf").write_bytes(pdf_from_text_pages([MISMATCH]))
    (root / "receipt_multipage.pdf").write_bytes(pdf_from_text_pages([PAGE1, PAGE2]))
    print(f"wrote PDFs in {root}")


if __name__ == "__main__":
    main()
