from training.prepare_cord import cord_to_schema


def test_cord_to_schema_flattens_nested_menu_items():
    payload = {
        "menu": {
            "nm": "[REG] BLACK SAKURA",
            "cnt": "1",
            "price": "45,455",
            "sub": [
                {"nm": "COOKIE DOH SAUCES", "cnt": "1", "price": "0"},
                {"nm": "NATA DE COCO", "cnt": "1", "price": "0"},
            ],
        },
        "sub_total": {"subtotal_price": "45,455", "tax_price": "4,545"},
        "total": {"total_price": "50,000"},
    }

    result = cord_to_schema(payload)

    assert [item["name"] for item in result["line_items"]] == [
        "[REG] BLACK SAKURA",
        "COOKIE DOH SAUCES",
        "NATA DE COCO",
    ]
    assert result["line_items"][0]["amount"] == 45455.0
    assert result["line_items"][1]["amount"] == 0.0
    assert result["total"] == 50000.0


def test_cord_to_schema_parses_at_prefixed_unit_price():
    result = cord_to_schema(
        {
            "menu": {"nm": "BLACK SAKURA", "cnt": "1", "unitprice": "@120,000", "price": "120,000"},
            "sub_total": {},
            "total": {},
        }
    )

    assert result["line_items"][0]["unit_price"] == 120000.0
