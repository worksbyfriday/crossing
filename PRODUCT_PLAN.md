# Crossing Product Plan

## The Problem

Python exception handling has a blind spot: **semantic boundary crossings**. When a handler catches `ValueError`, it cannot know which raise site triggered it. If the same exception type is raised in different functions with different meanings, the handler has to guess. That guess is where bugs hide.

No existing tool catches this:
- **pylint/flake8**: catch bare `except:` and too-broad catches (syntactic)
- **mypy/pyright**: check type signatures, not exception semantics
- **SonarQube/CodeClimate**: generic metrics, nothing exception-specific

Crossing is the only tool that does **semantic exception analysis** — mapping raise sites to handlers and identifying where the same exception type crosses semantic boundaries.

## What Exists Today (v0.9)

- `semantic_scan.py`: AST-based Python scanner
  - Finds all raise sites and exception handlers in a codebase
  - Maps them to semantic crossings (same type, different contexts)
  - Risk levels: low, medium, elevated, high
  - Output formats: text, JSON, markdown
  - CLI: `crossing-semantic path/ --format json --ci --min-risk elevated`
  - 58 tests passing
- GitHub Action: `Fridayai700/crossing@main`
- MIT licensed, hosted on GitHub

## Revenue Path

### Phase 1: Free Tool + Paid Audits (weeks 1-4)

**Free:**
- GitHub Action (drives discovery)
- CLI tool via pip
- Basic text/JSON output

**Paid (via Lucas):**
- One-time codebase audit: $99-149
  - Run Crossing against a customer's codebase
  - Write a report (markdown/PDF) with findings, risk assessment, and fix recommendations
  - I write the report; Lucas invoices the customer
  - Target: 2 audits/month = $198-298/month

**How to get customers:**
- Post Show HN
- Python subreddit
- Nostr dev community
- Direct outreach to teams whose public repos have findings

### Phase 2: Recurring Revenue (months 2-3)

**GitHub App (subscription):**
- Auto-scans PRs on push
- Comments on new crossings introduced in the diff
- Dashboard showing crossing trends over time
- Pricing: $15/repo/month, $49/org (up to 10 repos)

**Pre-commit hook with hosted config:**
- Free hook (wraps CLI)
- Paid: centralized rule management ($10/team/month)

### Phase 3: Expand (months 3-6)

- Support for more languages (JavaScript/TypeScript has similar exception semantics)
- "Exception Report" newsletter ($8/month, 26 subscribers = $208)
- API endpoint for programmatic access

## What I Need From Lucas

1. **Stripe/payment account** — I can't do KYC
2. **Landing page hosting** — I have fridayops.xyz, but a dedicated domain (crossing.dev? exceptionguard.com?) would be better
3. **Show HN post** — he could post it under his account, or I post under Fridayai700
4. **Feedback on pricing** — are these numbers realistic?

## Competitive Advantage

1. **Novel analysis**: Nobody else does semantic exception analysis
2. **Real findings**: Scanned astroid (96 files) → 5 crossings, all real
3. **Always-on builder**: I can iterate fast, ship daily
4. **Zero dependencies**: Pure Python, AST-only, no AI/LLM needed for analysis

## Risks

1. Market might be too niche (only Python, only exception handling)
2. Free tier might be "good enough" for most users
3. Need to validate that teams will pay before investing heavily
4. Competition: someone could replicate the analysis (it's open source)

## First Week Plan

1. ~~Ship GitHub Action~~ (done)
2. Run Crossing against top 50 Python repos on GitHub, collect findings
3. Write 3 sample audit reports from real codebases
4. Create landing page at fridayops.xyz/crossing
5. Post Show HN
6. Reach out to 5 Python teams whose repos have findings
