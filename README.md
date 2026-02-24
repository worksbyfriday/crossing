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

### Information-Theoretic Scoring

Each crossing reports quantitative metrics based on Shannon entropy:

| Metric | What it measures |
|--------|-----------------|
| **Semantic entropy** | Bits of information carried by the exception type at raise sites (log2 of distinct origins) |
| **Handler discrimination** | Bits preserved by handlers (re-raise = full, return/pass = zero) |
| **Information loss** | Bits destroyed: entropy minus discrimination |
| **Collapse ratio** | Normalized loss: 0% (no collapse) to 100% (total meaning erasure) |

```
--- AttributeError: 4 raise sites, 3 handlers — high risk ---
  Information: 2.0 bits entropy, 0.3 bits lost, 83% collapse
```

In JSON output, each crossing includes an `information_theory` object, and the summary includes `total_information_loss_bits` and `mean_collapse_ratio` across all crossings.

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
      - uses: worksbyfriday/crossing@main
        with:
          path: 'src/'
          fail-on-risk: 'elevated'
```

Inputs: `path`, `min-risk`, `format`, `implicit`, `exclude`, `fail-on-risk`.

---

## Benchmarks

Scanned 11 popular Python projects (Feb 2026):

| Project | Files | Crossings | High Risk | Info Loss |
|---|---|---|---|---|
| **pydantic** | **402** | **119** | **12** | **22.9 bits** |
| **sqlalchemy** | **661** | **103** | **16** | **79.8 bits** |
| django | 902 | 80 | 6 | — |
| aiohttp | 166 | 53 | 11 | 25.5 bits |
| click | 62 | 14 | 5 | 7.4 bits |
| celery | 161 | 12 | 3 | — |
| flask | 24 | 6 | 2 | — |
| requests | 18 | 5 | 2 | — |
| rich | 100 | 5 | 1 | — |
| astroid | 96 | 5 | 0 | — |
| **fastapi** | **47** | **0** | **0** | **0 bits** |

FastAPI scoring clean validates the tool. Sample audit reports: [SQLAlchemy](examples/audit-sqlalchemy.md), [Django](examples/audit-django.md), [Celery](examples/audit-celery.md), [Flask](examples/audit-flask.md), [Requests](examples/audit-requests.md).

---

## API

Scan any installed Python package via HTTP:

```bash
curl https://api.fridayops.xyz/crossing/package/flask
```

Returns JSON with full crossing analysis, information theory metrics, and risk levels.

**Badge** — embed in your README:

```markdown
![crossing](https://api.fridayops.xyz/crossing/badge/flask)
```

![crossing](https://api.fridayops.xyz/crossing/badge/flask)

Other endpoints:
- `POST /crossing` — scan raw Python source
- `GET /crossing/example` — demo snippet
- `GET /crossing/packages` — list of example packages
- `GET /crossing/badge/{name}` — SVG badge

---

## Install

```
pip install crossing
```

Or copy the files directly — no external dependencies. Python 3.10+.

## License

MIT
