# Crossing Audit Report: attrs

**Project:** attrs (python-attrs/attrs)
**Scanned:** 2026-02-24
**Tool:** Crossing Semantic Scanner v0.9

---

## Executive Summary

attrs has **6 semantic boundary crossings**, including **3 medium-risk** findings. For a 19-file codebase with 54 raise sites and 14 handlers, this gives a crossing density of 0.32 per file.

**Risk Level:** Medium.

---

## Scan Summary

| Metric | Value |
|--------|-------|
| Files scanned | 19 |
| Raise sites | 54 |
| Exception handlers | 14 |
| Total crossings | 6 |
| High risk | 0 |
| Elevated risk | 0 |
| Medium risk | 3 |
| Low risk | 3 |
| Mean collapse ratio | 50% |

---

## Findings

### MEDIUM RISK: `NotImplementedError` — 2 raise sites, 2 handlers

**File:** `attr/_version_info.py`
**Impact:** `NotImplementedError` is raised at 2 sites in 1 different functions. 2 handlers (2 return).

**Raise sites:**
- `attr/_version_info.py:63` raise `NotImplementedError` in `VersionInfo._ensure_tuple` — if not isinstance(other, tuple) → raise in _ensure_tuple
- `attr/_version_info.py:66` raise `NotImplementedError` in `VersionInfo._ensure_tuple` — if not (1 <= len(other) <= 4) → raise in _ensure_tuple

**Handlers:**
- `attr/_version_info.py:73` — except `NotImplementedError` in `VersionInfo.__eq__` (returns)
- `attr/_version_info.py:81` — except `NotImplementedError` in `VersionInfo.__lt__` (returns)

**Recommendation:** Multiple handlers exist, which may provide adequate discrimination. Verify that each handler's try-block scope only exposes the expected raise sites.

### MEDIUM RISK: `TypeError` — 15 raise sites, 4 handlers

**Files:** `attr/_config.py`, `attr/_funcs.py`, `attr/_make.py`, `attr/converters.py`, `attr/validators.py`
**Impact:** `TypeError` is raised at 15 sites across 4 files (_config.py, _make.py, converters.py, validators.py), in 12 different functions. 4 handlers (3 assign default). Information collapse: 100% of semantic information is lost (3.6 bits destroyed).

**Raise sites:**
- `attr/validators.py:100` raise `TypeError` in `_InstanceOfValidator.__call__` — if not isinstance(value, self.type) → raise in __call__
- `attr/validators.py:183` raise `TypeError` in `matches_re` — if flags → raise in matches_re
- `attr/validators.py:604` raise `TypeError` in `_SubclassOfValidator.__call__` — if not issubclass(value, self.type) → raise in __call__
- `attr/_make.py:170` raise `TypeError` in `attrib` — if hash is not None and hash is not True and hash is not False → raise in attrib
- `attr/_make.py:622` raise `TypeError` in `evolve` — in evolve
- `attr/_make.py:1510` raise `TypeError` in `wrap` — elif eq is False → raise in wrap
- `attr/_make.py:1555` raise `TypeError` in `wrap` — if not props.is_hashable and cache_hash → raise in wrap
- `attr/_make.py:1590` raise `TypeError` in `wrap` — if cache_hash → raise in wrap
- ... and 7 more

**Handlers:**
- `attr/validators.py:241` — except `TypeError` in `_InValidator.__call__` (assigns default)
- `attr/validators.py:700` — except `TypeError` in `not_` (assigns default)
- `attr/_funcs.py:118` — except `TypeError` in `asdict` (assigns default)
- `attr/_funcs.py:313` — except `TypeError` in `astuple` (handles)

**Information theory:** 3.6 bits entropy, 3.6 bits lost, 100% collapse

**Recommendation:** `TypeError` is a broad built-in type carrying 15 different meanings here. Consider defining project-specific exception subclasses, or narrowing handler try-blocks to minimize the catch surface.

### MEDIUM RISK: `ValueError` — 30 raise sites, 2 handlers

**Files:** `attr/_cmp.py`, `attr/_funcs.py`, `attr/_make.py`, `attr/_next_gen.py`, `attr/converters.py`, `attr/validators.py`
**Impact:** `ValueError` is raised at 30 sites across 6 files (_cmp.py, _funcs.py, _make.py, _next_gen.py, converters.py, validators.py), in 25 different functions. 2 handlers (1 re-raise, 1 assign default).

**Raise sites:**
- `attr/validators.py:139` raise `ValueError` in `_MatchesReValidator.__call__` — if not self.match_func(value) → raise in __call__
- `attr/validators.py:178` raise `ValueError` in `matches_re` — if func not in valid_funcs → raise in matches_re
- `attr/validators.py:246` raise `ValueError` in `_InValidator.__call__` — if not in_options → raise in __call__
- `attr/validators.py:446` raise `ValueError` in `deep_mapping` — if key_validator is None and value_validator is None → raise in deep_mapping
- `attr/validators.py:470` raise `ValueError` in `_NumberValidator.__call__` — if not self.compare_func(value, self.bound) → raise in __call__
- `attr/validators.py:546` raise `ValueError` in `_MaxLengthValidator.__call__` — if len(value) > self.max_length → raise in __call__
- `attr/validators.py:575` raise `ValueError` in `_MinLengthValidator.__call__` — if len(value) < self.min_length → raise in __call__
- `attr/validators.py:654` raise `ValueError` in `_NotValidator.__call__` — in __call__
- ... and 22 more

**Handlers:**
- `attr/_make.py:618` — except `ValueError` in `evolve` (re-raises)
- `attr/_make.py:1018` — except `ValueError` in `_ClassBuilder._create_slots_class` (handles)

**Information theory:** 4.6 bits entropy, 2.3 bits lost, 50% collapse

**Recommendation:** `ValueError` is a broad built-in type carrying 30 different meanings here. Consider defining project-specific exception subclasses, or narrowing handler try-blocks to minimize the catch surface.

---

## Benchmark Context

| Project | Files | Crossings | Elevated+ | Density |
|---------|-------|-----------|-----------|---------|
| **attrs** | **19** | **6** | **0** | **0.32** |
| click | 17 | 11 | 4 | 0.65 |
| requests | 18 | 5 | 2 | 0.28 |
| hypothesis | 103 | 29 | 7 | 0.28 |
| invoke | 47 | 12 | 3 | 0.26 |
| flask | 24 | 6 | 2 | 0.25 |
| tqdm | 31 | 7 | 3 | 0.23 |
| scrapy | 113 | 23 | 8 | 0.20 |
| uvicorn | 40 | 7 | 3 | 0.18 |
| colorama | 7 | 1 | 0 | 0.14 |
| httpx | 23 | 3 | 0 | 0.13 |
| pytest | 71 | 9 | 9 | 0.13 |
| celery | 161 | 12 | 3 | 0.07 |
| rich | 100 | 5 | 1 | 0.05 |
| fastapi | 47 | 0 | 0 | 0.00 |

attrs's crossing density (0.32) is significantly above the benchmark average (0.20).

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
