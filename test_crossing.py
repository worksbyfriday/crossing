"""Tests for crossing — verifying it catches known boundary issues."""

from crossing import (
    Crossing, CrossingReport, Loss, cross, _compare,
    json_crossing, json_crossing_strict, pickle_crossing,
    string_truncation_crossing,
)


def test_tuple_becomes_list_in_json():
    """JSON silently converts tuples to lists."""
    c = json_crossing_strict()
    report = cross(Crossing(
        encode=lambda d: __import__('json').dumps(d),
        decode=lambda s: __import__('json').loads(s),
        name="tuple test",
    ), samples=1, seed=None)
    # Direct test
    losses = _compare((1, 2, 3), [1, 2, 3])
    assert any(l.loss_type == "type_change" for l in losses)


def test_int_key_becomes_string_in_json():
    """JSON converts int dict keys to strings."""
    import json
    original = {42: "hello"}
    encoded = json.dumps(original)
    decoded = json.loads(encoded)
    losses = _compare(original, decoded)
    assert any(l.loss_type == "missing_key" for l in losses)
    assert any(l.loss_type == "added_key" for l in losses)


def test_bool_key_becomes_string_in_json():
    """JSON converts bool dict keys to lowercase strings."""
    import json
    original = {True: "yes", False: "no"}
    encoded = json.dumps(original)
    decoded = json.loads(encoded)
    losses = _compare(original, decoded)
    assert any(l.loss_type == "missing_key" for l in losses)


def test_none_key_becomes_string_in_json():
    """JSON converts None dict key to 'null' string."""
    import json
    original = {None: "value"}
    encoded = json.dumps(original)
    decoded = json.loads(encoded)
    losses = _compare(original, decoded)
    assert any(l.loss_type == "missing_key" for l in losses)


def test_nan_preserved():
    """NaN == NaN should not be reported as a loss."""
    losses = _compare(float('nan'), float('nan'))
    assert len(losses) == 0


def test_nan_to_value_is_loss():
    """NaN becoming a value should be reported."""
    losses = _compare(float('nan'), 0.0)
    assert any(l.loss_type == "value_change" for l in losses)


def test_identical_values_no_loss():
    """Identical structures should produce no losses."""
    cases = [
        42,
        "hello",
        [1, 2, 3],
        {"a": 1, "b": [True, None]},
        None,
        True,
        0.5,
    ]
    for case in cases:
        assert _compare(case, case) == [], f"False loss for {case!r}"


def test_int_float_coercion():
    """int -> float is a type change, not silent."""
    losses = _compare(42, 42.0)
    assert any(l.loss_type == "type_change" for l in losses)


def test_string_truncation():
    """Truncated strings should be caught."""
    losses = _compare("hello world", "hello")
    assert any(l.loss_type == "truncation" for l in losses)


def test_missing_dict_key():
    """Missing dict keys should be caught."""
    losses = _compare({"a": 1, "b": 2}, {"a": 1})
    assert any(l.loss_type == "missing_key" and l.path == "$.b" for l in losses)


def test_added_dict_key():
    """Added dict keys should be caught."""
    losses = _compare({"a": 1}, {"a": 1, "b": 2})
    assert any(l.loss_type == "added_key" and l.path == "$.b" for l in losses)


def test_list_length_change():
    """Changed list length should be caught."""
    losses = _compare([1, 2, 3], [1, 2])
    assert any(l.loss_type == "length_change" for l in losses)


def test_nested_loss_paths():
    """Loss paths should correctly trace nested structures."""
    losses = _compare({"a": {"b": (1,)}}, {"a": {"b": [1]}})
    assert any("a" in l.path and "b" in l.path for l in losses)


def test_pickle_is_lossless():
    """Pickle should preserve all Python types."""
    report = cross(pickle_crossing(), samples=100, seed=42)
    assert report.lossy_count == 0
    assert report.error_count == 0


def test_json_strict_catches_non_native():
    """Strict JSON should crash on non-JSON-native types."""
    report = cross(json_crossing_strict(), samples=200, seed=42)
    assert report.error_count > 0


def test_string_truncation_crossing():
    """String truncation crossing should catch long strings."""
    report = cross(string_truncation_crossing(10), samples=200, seed=42)
    assert report.lossy_count > 0
    assert all(l.loss_type == "truncation" for l in report.all_losses)


def test_cross_report_math():
    """Report counts should add up."""
    report = cross(json_crossing(), samples=100, seed=42)
    assert report.clean_count + report.lossy_count + report.error_count == report.total_samples


def test_cross_with_seed_is_deterministic():
    """Same seed should produce same results."""
    r1 = cross(json_crossing(), samples=50, seed=123)
    r2 = cross(json_crossing(), samples=50, seed=123)
    assert r1.total_loss_events == r2.total_loss_events
    assert r1.lossy_count == r2.lossy_count
    assert r1.error_count == r2.error_count


def test_custom_crossing():
    """Users should be able to define custom crossings."""
    # Simulate a boundary that lowercases all string values
    def lower_encode(d):
        if isinstance(d, str):
            return d.lower()
        return d

    c = Crossing(encode=lower_encode, decode=lambda d: d, name="lowercase")
    losses = _compare("Hello World", lower_encode("Hello World"))
    assert any(l.loss_type == "value_change" for l in losses)


if __name__ == "__main__":
    import sys
    passed = 0
    failed = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"  PASS {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL {name}: {e}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
