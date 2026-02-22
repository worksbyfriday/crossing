# Crossing

Detect silent information loss at system boundaries in Python codebases.

## Two Tools

### 1. Semantic Scanner — Exception Pattern Analysis

Find where the same exception type carries different meanings depending on the code path, but handlers can't distinguish them.

```bash
# Basic scan
crossing-semantic /path/to/project

# With implicit raises (dict access, getattr, etc.)
crossing-semantic --implicit /path/to/project

# JSON output for tooling
crossing-semantic --format json /path/to/project

# CI mode: fail if elevated/high risk crossings found
crossing-semantic --ci --min-risk elevated /path/to/project
```

Example: a `KeyError` that means "config key missing" and a `KeyError` that means "factor-filtered to empty" arrive at the same `except KeyError` handler. The handler assumes one meaning. The bug is silent.

### 2. Data Loss Fuzzer — Round-Trip Testing

Test whether information survives boundary crossings: serialization, API calls, database writes, format conversions.

```python
from crossing import Crossing, cross

c = Crossing(
    encode=lambda d: json.dumps(d),
    decode=lambda s: json.loads(s),
)

report = cross(c, samples=1000)
report.print()  # shows what was lost, where, and how
```

This isn't fuzzing for crashes. It's fuzzing for **silent data loss** — the operation succeeds but the output is missing something the input had.

---

## Semantic Scanner

### What It Finds

- **Polymorphic exceptions**: Multiple `raise` sites for the same exception type, caught by handlers that don't distinguish between them
- **Cross-function crossings**: Exceptions raised in called functions, caught by handlers in the caller
- **Cross-file crossings**: Same pattern across module boundaries via import resolution
- **Implicit raises**: `dict[key]` -> `KeyError`, `getattr(obj, name)` -> `AttributeError`, `int(x)` -> `ValueError`
- **Inheritance crossings**: `except ValueError` catching subclass raises like `ValidationError`
- **Scope analysis**: Whether handlers catch exceptions from direct raises or from called functions
- **Message differentiation**: Risk downgraded when all raise sites pass distinct string messages

### Risk Levels

| Level | Meaning |
|-------|---------|
| **low** | Single raise site, or polymorphic with matching handler strategies |
| **medium** | Multiple raise sites with uniform handler treatment |
| **elevated** | Scope mismatches or cross-function reachability |
| **high** | Many raise sites, few handlers, mixed implicit/explicit |

### CLI Options

```
crossing-semantic [OPTIONS] PATH

Options:
  --implicit          Detect implicit raises (dict access, getattr, etc.)
  --format FORMAT     Output format: text (default), json, markdown
  --min-risk LEVEL    Minimum risk to report: low, medium, elevated, high
  --exclude PATTERN   Exclude directories (repeatable)
  --ci                Exit code 1 if elevated/high risk crossings found
```

### Example Output

```
============================================================
Semantic Crossing Scan: /path/to/tox
============================================================
Files scanned:        42
Exception raises:     87 (58 explicit, 29 implicit)
Exception handlers:   34
Semantic crossings:   12
  Polymorphic (multi-raise):  8
  Elevated risk:              3

--- KeyError: 3 raise sites, 14 handlers --- high risk ---
  3 raise sites across different loaders (API, TOML, INI),
  14 handlers catching without distinguishing source
============================================================
```

### Real Bugs Found

The semantic scanner has identified real bugs in production codebases:

- **tox #3809**: `KeyError` meaning "factor-filtered to empty" caught by handler expecting "key doesn't exist"
- **Rich #3960**: Exception `__notes__` leaking across chained exceptions
- **pytest #14214**: Verbosity config not propagated across internal call boundary

---

## Data Loss Fuzzer

### Built-in Crossings

| Crossing | What it tests | Typical loss rate |
|----------|---------------|-------------------|
| `json_crossing()` | JSON with `default=str` | ~24% lossy, 34% crashes |
| `json_crossing_strict()` | JSON without fallback | ~6% lossy, 52% crashes |
| `pickle_crossing()` | Python pickle | 0% (lossless baseline) |
| `yaml_crossing()` | YAML safe_load | ~0% lossy, 49% crashes |
| `toml_crossing()` | TOML via tomllib/tomli_w | varies |
| `csv_crossing()` | CSV (everything becomes strings) | ~82% lossy |
| `env_file_crossing()` | .env files (KEY=VALUE) | ~83% lossy |
| `url_query_crossing()` | URL query string encoding | ~80% lossy |

### Custom Crossings

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

### Compose Pipelines

```python
from crossing import compose, json_crossing, string_truncation_crossing, cross

# Simulate: serialize -> store in VARCHAR(100) -> deserialize
pipeline = compose(
    json_crossing(),
    string_truncation_crossing(100),
)
report = cross(pipeline, samples=500)
```

### Codebase Scanning

```bash
python3 scan.py /path/to/project
```

Finds encode/decode pairs for: JSON, YAML, pickle, TOML, base64, URL encoding, CSV, struct, zlib, gzip.

---

## GitHub Action

Add Crossing to your CI pipeline:

```yaml
# .github/workflows/crossing.yml
name: Exception Analysis
on: [pull_request]

jobs:
  crossing:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: Fridayai700/crossing@main
        with:
          path: 'src/'
          fail-on-risk: 'elevated'
```

Inputs: `path`, `min-risk`, `format`, `implicit`, `exclude`, `fail-on-risk`.

---

## Benchmarks

Scanned 7 popular Python projects (Feb 2026):

| Project | Files | Crossings | High Risk |
|---|---|---|---|
| celery | 161 | 12 | 3 |
| flask | 24 | 6 | 2 |
| requests | 18 | 5 | 2 |
| rich | 100 | 5 | 1 |
| astroid | 96 | 5 | 0 |
| httpx | 23 | 3 | 0 |
| **fastapi** | **47** | **0** | **0** |

FastAPI scoring clean validates the tool. Sample audit reports: [Celery](examples/audit-celery.md), [Flask](examples/audit-flask.md), [Requests](examples/audit-requests.md).

---

## Install

```
pip install crossing
```

Or copy the files directly — no external dependencies. Python 3.10+.

## License

MIT
