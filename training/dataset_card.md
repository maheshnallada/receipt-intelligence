# CORD v2 mapping

Source: `naver-clova-ix/cord-v2` (official train / validation / test).

CORD does not label store name or date. Those target fields are `null` in
training labels. Synthetic sample PDFs in `samples/` cover those fields for
the API and UI.

| CORD | Assignment |
|---|---|
| `menu.nm` | `line_items[].name` |
| `menu.cnt` | `line_items[].qty` |
| `menu.unitprice` | `line_items[].unit_price` |
| `menu.price` | `line_items[].amount` |
| `sub_total.subtotal_price` | `subtotal` |
| `sub_total.tax_price` | `tax` |
| `total.total_price` | `total` |

Ignored: void menu, cash/change, service charge, discounts.

Split ids are frozen in `data/splits.json`. The test split is never used for
training or prompt edits.
