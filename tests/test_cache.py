from app.infrastructure.cache.memory import MemoryCache


def test_second_identical_key_hits() -> None:
    cache = MemoryCache()
    payload = {"store_name": "Oak", "date": None, "line_items": [], "subtotal": None, "tax": None, "total": 1.0}
    key = "abc:base:1.0.0:1.0.0"
    cache.set(key, payload, 60)
    hit = cache.get(key)
    assert hit is not None
    assert hit["store_name"] == "Oak"
    assert cache.get(key) is not None


def test_different_schema_version_misses() -> None:
    cache = MemoryCache()
    cache.set("abc:base:1.0.0:1.0.0", {"total": 1}, 60)
    assert cache.get("abc:base:2.0.0:1.0.0") is None


def test_force_refresh_is_a_read_bypass() -> None:
    """force_refresh is applied by the use case; the adapter still writes."""
    cache = MemoryCache()
    cache.set("k", {"total": 1}, 60)
    assert cache.get("k") is not None
    cache.set("k", {"total": 2}, 60)
    assert cache.get("k")["total"] == 2
