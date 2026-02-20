# crossing

Detect silent information loss at system boundaries.

## The Problem

Most bugs aren't crashes. They're silent successes where data enters a transformation
and something doesn't come out the other side. No error is raised. The system continues
with incomplete data. The failure shows up downstream, far from where it occurred.

```python
import json

data = {"created": datetime(2024, 1, 1), "tags": ("python", "testing"), "count": 42}
result = json.loads(json.dumps(data, default=str))
# No error. But:
# - datetime became a string
# - tuple became a list
# - The system continues as if nothing happened
```

Every system has boundaries — serialization, API calls, database writes, configuration
loading, format conversions. At each boundary, data changes form. And at each boundary,
information can silently disappear.

## What This Does

Two things:

1. **Test boundaries** — generate structured random inputs, push them through an
   encode/decode boundary, report what didn't survive.
2. **Find boundaries** — scan a codebase for encode/decode pairs and show you where
   data crosses between formats.

### Testing boundaries

```python
from crossing import cross, json_crossing

report = cross(json_crossing(), samples=1000, seed=42)
report.print()
```

Output:
```
============================================================
Crossing Report: JSON round-trip
============================================================
Samples tested:    1000
Clean passages:    420 (42%)
Lossy passages:    238 (24%)
Crashes:           342 (34%)
Total loss events: 812

Loss types:
  type_change: 412
  missing_key: 200
  added_key: 200
============================================================
```

### Scanning codebases

```bash
python3 scan.py /path/to/your/project
```

Output:
```
============================================================
Crossing Scan: /path/to/your/project
============================================================
Files scanned:     413
Boundary calls:    91
Crossing types:    7
  Paired (encode+decode): 5
  Encode-only:            1
  Decode-only:            1

Paired crossings (data enters AND exits):
  JSON: 25 encode, 11 decode (paired)
  pickle: 19 encode, 20 decode (paired)
  base64: 3 encode, 1 decode (paired)
============================================================
```

This isn't fuzzing for crashes. It's fuzzing for **silent data loss**.

## Usage

### Test a built-in crossing

```python
from crossing import cross, json_crossing, json_crossing_strict

# JSON with default=str — silently converts everything to strings
report = cross(json_crossing(), samples=500)

# JSON strict — crashes on non-native types (more honest)
report = cross(json_crossing_strict(), samples=500)

# Pickle — should be lossless (baseline)
from crossing import pickle_crossing
report = cross(pickle_crossing(), samples=500)

# YAML — crashes on Python types, honest about limitations
from crossing import yaml_crossing
report = cross(yaml_crossing(), samples=500)

# CSV — everything becomes strings
from crossing import csv_crossing
report = cross(csv_crossing(), samples=200)
```

### Define a custom crossing

A crossing is any pair of functions where data goes in one side and comes back the other:

```python
from crossing import Crossing, cross

# Test your API serialization
c = Crossing(
    encode=lambda d: my_api_serialize(d),
    decode=lambda s: my_api_deserialize(s),
    name="My API boundary",
)
report = cross(c, samples=1000)
report.print()
```

### Compose crossings into pipelines

Real data crosses multiple boundaries. `compose` chains crossings to reveal
cumulative information loss:

```python
from crossing import compose, json_crossing, string_truncation_crossing, cross

# Simulate: serialize → store in VARCHAR(100) → deserialize
pipeline = compose(
    json_crossing(),
    string_truncation_crossing(100),
)
report = cross(pipeline, samples=500)
# Reveals: truncated JSON strings often produce invalid JSON (43% crashes)
# AND valid-but-incomplete data (17% silent loss)
```

### Scan a codebase for boundaries

```bash
# Find all boundary crossings
python3 scan.py /path/to/project

# Find boundaries and generate test snippets
python3 scan.py /path/to/project --generate-tests
```

The scanner finds encode/decode pairs for: JSON, YAML, pickle, TOML, base64,
URL encoding, CSV, struct, zlib, and gzip.

### Inspect individual results

```python
report = cross(json_crossing(), samples=100, seed=42)

for result in report.results:
    if result.lossy:
        print(f"Input: {result.input_value}")
        print(f"Output: {result.output_value}")
        for loss in result.losses:
            print(f"  {loss}")
```

## Built-in Crossings

| Crossing | What it tests | Typical loss rate |
|----------|---------------|-------------------|
| `json_crossing()` | JSON with `default=str` | ~24% lossy, 34% crashes |
| `json_crossing_strict()` | JSON without fallback | ~6% lossy, 52% crashes |
| `pickle_crossing()` | Python pickle | 0% (lossless baseline) |
| `yaml_crossing()` | YAML safe_load | ~0% lossy, 49% crashes |
| `toml_crossing()` | TOML via tomllib/tomli_w | varies |
| `csv_crossing()` | CSV (everything→strings) | ~82% lossy |
| `env_file_crossing()` | .env files (KEY=VALUE) | ~83% lossy |
| `url_query_crossing()` | URL query string encoding | ~80% lossy |
| `string_truncation_crossing(n)` | Simulates VARCHAR(n) columns | varies by n |
| `str_crossing()` | `repr()`/`eval()` round-trip | low (but uses eval) |

## What It Finds

The generator produces values that commonly reveal boundary issues:

- **Type coercion**: tuples → lists, datetimes → strings, Decimals → strings
- **Key type loss**: int/bool/None dict keys → string keys
- **Precision loss**: float imprecision, large integers beyond safe ranges
- **Truncation**: long strings, deeply nested structures
- **Encoding issues**: null bytes, emoji, non-UTF8 bytes
- **Edge cases**: NaN, Infinity, -0.0, empty containers

## Install

```
pip install crossing
```

Or copy `crossing.py` — it's a single file with no dependencies (scan.py is optional).

## License

MIT
