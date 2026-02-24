# Crossing Audit Report: HTTPX

**Project:** HTTPX (encode/httpx)
**Scanned:** 2026-02-24
**Tool:** Crossing Semantic Scanner v0.9

---

## Executive Summary

HTTPX has **10 semantic boundary crossings**, including **3 medium-risk** findings. For a 23-file codebase with 96 raise sites and 56 handlers, this gives a crossing density of 0.43 per file.

**Risk Level:** Medium.

---

## Scan Summary

| Metric | Value |
|--------|-------|
| Files scanned | 23 |
| Raise sites | 96 |
| Exception handlers | 56 |
| Total crossings | 10 |
| High risk | 0 |
| Elevated risk | 0 |
| Medium risk | 3 |
| Low risk | 7 |
| Mean collapse ratio | 31% |

---

## Findings

### MEDIUM RISK: `ValueError` — 6 raise sites, 6 handlers

**Files:** `_config.py`, `_main.py`, `_models.py`, `_status_codes.py`, `_transports/default.py`, `_urlparse.py`, `_utils.py`
**Impact:** `ValueError` is raised at 6 sites across 4 files (_config.py, _models.py, _utils.py, default.py), in 6 different functions. 6 handlers (1 re-raise, 2 return, 2 assign default). Information collapse: 83% of semantic information is lost (2.1 bits destroyed).

**Raise sites:**
- `_utils.py:166` raise `ValueError` in `URLPattern.__init__` — if pattern and ":" not in pattern → raise in __init__
- `_config.py:123` raise `ValueError` in `Timeout.__init__` — if isinstance(timeout, UnsetType) → raise in __init__ (`"httpx.Timeout must either include a default, or set all four...`)
- `_config.py:214` raise `ValueError` in `Proxy.__init__` — if url.scheme not in ("http", "https", "socks5", "socks5h") → raise in __init__
- `_models.py:696` raise `ValueError` in `Response.encoding` — if hasattr(self, "_text") → raise in encoding (`"Setting encoding after `text` has been accessed is not allow...`)
- `_transports/default.py:212` raise `ValueError` in `HTTPTransport.__init__` — in __init__
- `_transports/default.py:356` raise `ValueError` in `AsyncHTTPTransport.__init__` — in __init__

**Handlers:**
- `_urlparse.py:410` — except `ValueError` in `normalize_port` (re-raises)
- `_main.py:178` — except `ValueError` in `print_response` (assigns default)
- `_models.py:130` — except `ValueError` in `_parse_header_links` (assigns default)
- `_models.py:136` — except `ValueError` in `_parse_header_links` (handles)
- `_models.py:372` — except `ValueError` in `Headers.__eq__` (returns)
- ... and 1 more

**Information theory:** 2.6 bits entropy, 2.1 bits lost, 83% collapse

**Recommendation:** `ValueError` is a broad built-in type carrying 6 different meanings here. Consider defining project-specific exception subclasses, or narrowing handler try-blocks to minimize the catch surface.

### MEDIUM RISK: `ImportError` — 6 raise sites, 9 handlers

**Files:** `__init__.py`, `_client.py`, `_decoders.py`, `_transports/asgi.py`, `_transports/default.py`
**Impact:** `ImportError` is raised at 6 sites across 3 files (_client.py, _decoders.py, default.py), in 6 different functions. 9 handlers (4 re-raise, 2 assign default). Information collapse: 56% of semantic information is lost (1.4 bits destroyed).

**Raise sites:**
- `_decoders.py:120` raise `ImportError` in `BrotliDecoder.__init__` — if brotli is None:  # pragma: no cover → raise in __init__ (`"Using 'BrotliDecoder', but neither of the 'brotlicffi' or 'b...`)
- `_decoders.py:172` raise `ImportError` in `ZStandardDecoder.__init__` — if zstandard is None:  # pragma: no cover → raise in __init__ (`"Using 'ZStandardDecoder', ...Make sure to install httpx usin...`)
- `_client.py:680` raise `ImportError` in `Client.__init__` — in __init__ (`"Using http2=True, but the 'h2' package is not installed. Mak...`)
- `_client.py:1394` raise `ImportError` in `AsyncClient.__init__` — in __init__ (`"Using http2=True, but the 'h2' package is not installed. Mak...`)
- `_transports/default.py:191` raise `ImportError` in `HTTPTransport.__init__` — in __init__ (`"Using SOCKS proxy, but the 'socksio' package is not installe...`)
- `_transports/default.py:335` raise `ImportError` in `AsyncHTTPTransport.__init__` — in __init__ (`"Using SOCKS proxy, but the 'socksio' package is not installe...`)

**Handlers:**
- `_decoders.py:20` — except `ImportError` in `<module>` (handles)
- `_decoders.py:25` — except `ImportError` in `<module>` (assigns default)
- `_decoders.py:32` — except `ImportError` in `<module>` (assigns default)
- `__init__.py:16` — except `ImportError` in `<module>` (handles)
- `_client.py:679` — except `ImportError` in `Client.__init__` (re-raises)
- ... and 4 more

**Information theory:** 2.6 bits entropy, 1.4 bits lost, 56% collapse

**Recommendation:** Multiple handlers exist, which may provide adequate discrimination. Verify that each handler's try-block scope only exposes the expected raise sites.

### MEDIUM RISK: `KeyError` — 3 raise sites, 5 handlers

**Files:** `_auth.py`, `_models.py`
**Impact:** `KeyError` is raised at 3 sites in 3 different functions. 5 handlers (1 re-raise, 3 return, 1 assign default). Information collapse: 80% of semantic information is lost (1.3 bits destroyed).

**Raise sites:**
- `_models.py:315` raise `KeyError` in `Headers.__getitem__` — if items → raise in __getitem__
- `_models.py:354` raise `KeyError` in `Headers.__delitem__` — if not pop_indexes → raise in __delitem__
- `_models.py:1229` raise `KeyError` in `Cookies.__getitem__` — if value is None → raise in __getitem__

**Handlers:**
- `_auth.py:251` — except `KeyError` in `DigestAuth._parse_challenge` (re-raises)
- `_models.py:262` — except `KeyError` in `Headers.get` (returns)
- `_models.py:627` — except `KeyError` in `Response.http_version` (returns)
- `_models.py:636` — except `KeyError` in `Response.reason_phrase` (returns)
- `_models.py:725` — except `KeyError` in `Response._get_content_decoder` (handles)

**Information theory:** 1.6 bits entropy, 1.3 bits lost, 80% collapse

**Recommendation:** Multiple handlers exist, which may provide adequate discrimination. Verify that each handler's try-block scope only exposes the expected raise sites.

---

## Benchmark Context

| Project | Files | Crossings | Elevated+ | Density |
|---------|-------|-----------|-----------|---------|
| **HTTPX** | **23** | **10** | **0** | **0.43** |
| click | 17 | 11 | 4 | 0.65 |
| requests | 18 | 5 | 2 | 0.28 |
| hypothesis | 103 | 29 | 7 | 0.28 |
| invoke | 47 | 12 | 3 | 0.26 |
| flask | 24 | 6 | 2 | 0.25 |
| tqdm | 31 | 7 | 3 | 0.23 |
| scrapy | 113 | 23 | 8 | 0.20 |
| uvicorn | 40 | 7 | 3 | 0.18 |
| colorama | 7 | 1 | 0 | 0.14 |
| pytest | 71 | 9 | 9 | 0.13 |
| celery | 161 | 12 | 3 | 0.07 |
| rich | 100 | 5 | 1 | 0.05 |
| fastapi | 47 | 0 | 0 | 0.00 |

HTTPX's crossing density (0.43) is significantly above the benchmark average (0.20).

---

## Methodology

Crossing performs static AST analysis on Python source files. It maps every `raise` statement to every `except` handler that could catch it, then identifies **semantic boundary crossings** — places where the same exception type is raised with different meanings in different contexts. No code is executed; no network calls are made; no dependencies are required.

Risk levels:
- **Low:** Single raise site or uniform semantics
- **Medium:** Multiple raise sites in different functions — handler may not distinguish
- **Elevated:** Many divergent raise sites — high chance of incorrect handling
- **High:** Handler collapse — many raise sites, very few handlers, ambiguous behavior

---

*Report generated by [Crossing](https://fridayops.xyz/crossing/) v0.9*  
*Scan performed 2026-02-24*
