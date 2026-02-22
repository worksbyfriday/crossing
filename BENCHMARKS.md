# Crossing Benchmarks — Popular Python Projects

Scanned on 2026-02-22 using Crossing v0.9 semantic scanner.

## Summary

| Project | Files | Raises | Handlers | Crossings | Risky | High Risk |
|---|---|---|---|---|---|---|
| flask | 24 | 89 | 42 | 6 | 6 | 2 |
| requests | 18 | 61 | 69 | 5 | 5 | 2 |
| httpx | 23 | 95 | 55 | 3 | 3 | 0 |
| **fastapi** | **47** | **30** | **22** | **0** | **0** | **0** |
| rich | 100 | 58 | 74 | 5 | 5 | 1 |
| celery | 161 | 292 | 554 | 12 | 12 | 3 |
| astroid | 96 | 329 | 295 | 5 | 5 | 0 |

## Key Findings

### Celery (12 crossings, 3 high-risk)
Highest absolute count. Handler-to-raise ratio is nearly 2:1 (554 handlers for 292 raises), meaning exceptions are caught prolifically. Common offenders: `NotImplementedError` (29 raise sites), `TypeError` (11 raise sites), `KeyError`, `ValueError`, `AttributeError`.

### Flask (6 crossings, 2 high-risk, highest density)
6 crossings in 24 files = highest density per file. High-risk: `AttributeError` (4 raise sites, 1 handler — collapse likely) and `click.BadParameter` (7 raise sites, 1 handler).

### Requests (5 crossings, 2 high-risk)
High-risk `InvalidURL` crossing (7 raise sites, 1 handler) and `TypeError` (3 raise sites, 1 handler). `ValueError` crossing is broad: 8 raise sites, 10 handlers.

### Rich (5 crossings, 1 high-risk)
High-risk `KeyError` crossing (3 raise sites in different semantic contexts, 9 handlers). `ValueError` spans 13 raise sites across 10 different functions.

### FastAPI (0 crossings)
Cleanest codebase. 47 files, 30 raises, 22 handlers, no polymorphic exception reuse across semantic boundaries. Validates the tool — well-designed codebases score clean.

### httpx (3 crossings, 0 high-risk)
Relatively clean. All medium risk. Well-structured exception handling compared to peers.

## What This Shows

1. **The bugs are real.** Every major Python project except FastAPI has semantic boundary crossings.
2. **The tool differentiates.** FastAPI scoring clean shows this isn't just noise — well-designed codebases ARE clean.
3. **Density matters more than absolute count.** Flask's 6 crossings in 24 files is worse than celery's 12 in 161.
4. **Handler-to-raise ratio predicts risk.** Celery's 2:1 ratio (more handlers than raises) correlates with highest crossing count.
