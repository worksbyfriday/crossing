"""Tests for crossing — verifying it catches known boundary issues."""

import pytest
from crossing import (
    Crossing, CrossingReport, Loss, cross, _compare, compose, diff,
    DiffReport,
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


def test_compose_chains_crossings():
    """Composed crossings should reveal cumulative losses."""
    pipeline = compose(
        json_crossing("step 1"),
        string_truncation_crossing(50, "step 2"),
    )
    assert "step 1" in pipeline.name
    assert "step 2" in pipeline.name
    report = cross(pipeline, samples=100, seed=42)
    # Should have some losses (type changes from JSON + truncation)
    assert report.lossy_count + report.error_count > 0


def test_compose_preserves_identity():
    """Composing a lossless crossing with itself should still be lossless."""
    pipeline = compose(pickle_crossing(), pickle_crossing())
    report = cross(pipeline, samples=50, seed=42)
    assert report.lossy_count == 0
    assert report.error_count == 0


def test_yaml_loses_types():
    """YAML should demonstrate type loss (tuples→lists, etc)."""
    pytest.importorskip("yaml")
    from crossing import yaml_crossing
    report = cross(yaml_crossing(), samples=200, seed=42)
    # YAML should have losses and/or crashes for non-native types
    assert report.lossy_count + report.error_count > 0


def test_yaml_tuple_becomes_list():
    """YAML safe_load can't reconstruct Python tuples — they crash or become lists."""
    yaml = pytest.importorskip("yaml")
    # With safe_load, python/tuple tags are rejected (ConstructorError)
    # This means tuples cause crashes in YAML round-trips, which crossing detects
    original = (1, 2, 3)
    encoded = yaml.dump(original, default_flow_style=False)
    try:
        decoded = yaml.safe_load(encoded)
        # If it doesn't crash, check for type loss
        losses = _compare(original, decoded)
        assert any(l.loss_type == "type_change" for l in losses)
    except yaml.constructor.ConstructorError:
        pass  # expected — safe_load rejects python tags


def test_toml_crashes_on_none():
    """TOML can't represent None — should crash."""
    pytest.importorskip("tomllib" if __import__("sys").version_info >= (3, 11) else "tomli")
    from crossing import toml_crossing
    c = toml_crossing()
    try:
        c.encode({"key": None})
        assert False, "TOML should crash on None"
    except (TypeError, AttributeError):
        pass  # expected


def test_csv_everything_becomes_string():
    """CSV loses all type information."""
    from crossing import csv_crossing
    c = csv_crossing()
    encoded = c.encode({"count": 42, "active": True})
    decoded = c.decode(encoded)
    # Everything should be a string now
    assert isinstance(decoded["count"], str)
    assert isinstance(decoded["active"], str)


def test_env_file_flattens():
    """Env file format flattens everything to strings."""
    from crossing import env_file_crossing
    c = env_file_crossing()
    encoded = c.encode({"PORT": 8080, "DEBUG": True})
    decoded = c.decode(encoded)
    assert isinstance(decoded["PORT"], str)
    assert isinstance(decoded["DEBUG"], str)


def test_yaml_bool_coercion():
    """YAML 1.1 coerced 'yes'/'no' to booleans; YAML 1.2 / PyYAML 6.0+ doesn't.

    This test documents the behavior rather than asserting a specific version.
    Either outcome (coerced or preserved) is informative about the crossing.
    """
    yaml = pytest.importorskip("yaml")
    original = {"answer": "yes", "switch": "off", "name": "no"}
    encoded = yaml.dump(original, default_flow_style=False)
    decoded = yaml.safe_load(encoded)
    losses = _compare(original, decoded)
    # In YAML 1.1 (PyYAML < 6.0): losses > 0 (strings coerced to bools)
    # In YAML 1.2 (PyYAML >= 6.0): losses == 0 (strings preserved)
    # Both are valid — the test just ensures crossing can measure either
    if losses:
        assert any(l.loss_type == "type_change" for l in losses)
    # If no losses, that's fine — modern PyYAML preserves these strings


def test_custom_inputs():
    """Users can provide specific inputs alongside random samples."""
    my_data = [
        {"user": "alice", "age": 30, "tags": ["admin", "user"]},
        {"config": {"nested": True, "count": 0}},
        [1, None, "three"],
    ]
    report = cross(json_crossing(), samples=10, inputs=my_data)
    # Should test at least the 3 custom inputs
    assert report.total_samples >= 3
    # Custom inputs + random should equal samples or len(inputs) if larger
    assert report.total_samples == max(10, len(my_data))


def test_custom_inputs_larger_than_samples():
    """When inputs > samples, all inputs are still tested."""
    inputs = [{"k": i} for i in range(20)]
    report = cross(json_crossing(), samples=5, inputs=inputs)
    # All 20 custom inputs should be tested, no random added
    assert report.total_samples == 20


def test_diff_identical_crossings():
    """Diffing a crossing against itself should show no divergence."""
    report = diff(json_crossing("A"), json_crossing("B"), samples=50, seed=42)
    assert report.divergent_count == 0
    assert report.equivalent_count == report.total_samples


def test_diff_finds_divergence():
    """Diffing JSON strict vs lenient should find divergence on non-native types."""
    report = diff(
        json_crossing("lenient"),
        json_crossing_strict("strict"),
        samples=200, seed=42,
    )
    # Lenient uses default=str, strict crashes — should have divergences
    assert report.divergent_count > 0


def test_diff_with_custom_inputs():
    """Diff supports custom inputs."""
    inputs = [(1, 2, 3), {"key": None}]
    report = diff(
        json_crossing("A"),
        pickle_crossing("B"),
        samples=5, inputs=inputs,
    )
    assert report.total_samples >= 2


def test_diff_report_counts():
    """DiffReport counts should be consistent."""
    report = diff(json_crossing("A"), pickle_crossing("B"), samples=50, seed=42)
    assert report.equivalent_count + report.divergent_count == report.total_samples


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
