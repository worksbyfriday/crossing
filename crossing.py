"""
crossing — detect silent information loss at system boundaries.

The premise: most bugs aren't crashes. They're silent successes where data
enters a transformation and something doesn't come out the other side.
No error is raised. The system continues with incomplete data.

This library helps you find those bugs by testing whether information
survives boundary crossings: serialization, API calls, database writes,
configuration loading, format conversions — anywhere data changes form.

Usage:
    from crossing import Crossing, cross

    # Define a crossing: how data goes in and how it comes back
    c = Crossing(
        encode=lambda d: json.dumps(d),
        decode=lambda s: json.loads(s),
    )

    # Test it — crossing generates structured inputs and checks
    # whether information survives the round trip
    report = cross(c, samples=1000)
    report.print()  # shows what was lost, where, and how

The key insight: this isn't fuzzing for crashes. It's fuzzing for
silent data loss. The operation SUCCEEDS but the output is missing
something the input had.
"""

from __future__ import annotations

import json
import math
import random
import string
import datetime
import decimal
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Loss:
    """A single instance of information loss at a crossing."""
    input_value: Any
    output_value: Any
    path: str  # where in the structure the loss occurred
    loss_type: str  # 'missing_key', 'type_change', 'value_change', 'precision_loss', 'truncation'
    description: str

    def __str__(self):
        return f"[{self.loss_type}] {self.path}: {self.description}"


@dataclass
class SampleResult:
    """Result of one sample through a crossing."""
    input_value: Any
    output_value: Any
    losses: list[Loss] = field(default_factory=list)
    error: Optional[Exception] = None

    @property
    def clean(self) -> bool:
        return not self.losses and self.error is None

    @property
    def lossy(self) -> bool:
        return bool(self.losses)

    @property
    def crashed(self) -> bool:
        return self.error is not None


@dataclass
class CrossingReport:
    """Results from testing a boundary crossing."""
    name: str
    results: list[SampleResult] = field(default_factory=list)

    @property
    def total_samples(self) -> int:
        return len(self.results)

    @property
    def clean_count(self) -> int:
        return sum(1 for r in self.results if r.clean)

    @property
    def lossy_count(self) -> int:
        """Samples that completed but lost information."""
        return sum(1 for r in self.results if r.lossy)

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if r.crashed)

    @property
    def total_loss_events(self) -> int:
        """Total individual loss events across all samples."""
        return sum(len(r.losses) for r in self.results)

    @property
    def loss_rate(self) -> float:
        return self.lossy_count / self.total_samples if self.total_samples > 0 else 0.0

    @property
    def all_losses(self) -> list[Loss]:
        """All individual loss events, flattened."""
        return [loss for r in self.results for loss in r.losses]

    def loss_types(self) -> dict[str, int]:
        """Count loss events by type."""
        counts: dict[str, int] = {}
        for loss in self.all_losses:
            counts[loss.loss_type] = counts.get(loss.loss_type, 0) + 1
        return counts

    def print(self):
        """Print a human-readable report."""
        print(f"\n{'='*60}")
        print(f"Crossing Report: {self.name}")
        print(f"{'='*60}")
        print(f"Samples tested:    {self.total_samples}")
        print(f"Clean passages:    {self.clean_count} ({self.clean_count/self.total_samples:.0%})")
        print(f"Lossy passages:    {self.lossy_count} ({self.loss_rate:.0%})")
        print(f"Crashes:           {self.error_count} ({self.error_count/self.total_samples:.0%})")
        print(f"Total loss events: {self.total_loss_events}")

        if self.all_losses:
            print(f"\nLoss types:")
            for loss_type, count in sorted(self.loss_types().items(), key=lambda x: -x[1]):
                print(f"  {loss_type}: {count}")

            print(f"\nSample losses (first 10):")
            for loss in self.all_losses[:10]:
                print(f"  {loss}")

        errors = [r for r in self.results if r.crashed]
        if errors:
            print(f"\nSample errors (first 5):")
            for r in errors[:5]:
                print(f"  Input: {repr(r.input_value)[:80]}")
                print(f"  Error: {r.error}")

        print(f"{'='*60}\n")


@dataclass
class Crossing:
    """A boundary crossing — a pair of functions that transform data."""
    encode: Callable[[Any], Any]
    decode: Callable[[Any], Any]
    name: str = "unnamed crossing"


def compose(*crossings: Crossing, name: str | None = None) -> Crossing:
    """Chain multiple crossings into a single crossing.

    Data passes through each crossing's encode in order,
    then through each decode in reverse order. This models
    real-world data pipelines: Python → JSON → HTTP → DB → HTTP → JSON → Python.

    The composed crossing reveals cumulative information loss
    across the entire pipeline, not just individual hops.
    """
    if not crossings:
        raise ValueError("Need at least one crossing to compose")

    composed_name = name or " → ".join(c.name for c in crossings)

    def encode(d: Any) -> Any:
        result = d
        for c in crossings:
            result = c.encode(result)
        return result

    def decode(d: Any) -> Any:
        result = d
        for c in reversed(crossings):
            result = c.decode(result)
        return result

    return Crossing(encode=encode, decode=decode, name=composed_name)


def _generate_scalar() -> Any:
    """Generate a random scalar value that might reveal boundary issues."""
    generators = [
        # Strings
        lambda: "",
        lambda: " ",
        lambda: "hello",
        lambda: "a" * 10000,  # long string
        lambda: "\x00",  # null byte
        lambda: "\n\r\t",  # whitespace
        lambda: "".join(random.choices(string.printable, k=random.randint(1, 200))),
        lambda: "\u00e9\u00f1\u00fc\u00e4\u00f6",  # unicode
        lambda: "\U0001f600\U0001f4a9",  # emoji
        lambda: "true",  # string that looks like bool
        lambda: "null",  # string that looks like null
        lambda: "0",  # string that looks like number
        lambda: "NaN",
        lambda: "Infinity",
        lambda: "-0",

        # Numbers
        lambda: 0,
        lambda: -0,
        lambda: 1,
        lambda: -1,
        lambda: 2**53,  # JS max safe integer
        lambda: 2**53 + 1,  # beyond JS safe integer
        lambda: 2**63 - 1,  # int64 max
        lambda: 2**64,  # beyond int64
        lambda: 0.1 + 0.2,  # floating point imprecision
        lambda: 1e-300,  # very small float
        lambda: 1e300,  # very large float
        lambda: float('inf'),
        lambda: float('-inf'),
        lambda: float('nan'),
        lambda: 0.0,
        lambda: -0.0,

        # Booleans
        lambda: True,
        lambda: False,

        # None/null
        lambda: None,

        # Dates and times (as strings — common serialization format)
        lambda: "2026-02-20T12:00:00Z",
        lambda: "2026-02-20T12:00:00+05:30",  # non-UTC offset
        lambda: "2026-02-20T12:00:00.123456789Z",  # nanosecond precision
        lambda: "0001-01-01T00:00:00Z",  # edge date
        lambda: "9999-12-31T23:59:59Z",  # far future

        # Python-specific types that reveal crossing issues
        lambda: (1, 2, 3),  # tuple — becomes list in JSON
        lambda: (),  # empty tuple
        lambda: (1,),  # single-element tuple
        lambda: frozenset({1, 2, 3}),  # frozenset
        lambda: {1, 2, 3},  # set — not JSON serializable
        lambda: b"hello",  # bytes
        lambda: b"",  # empty bytes
        lambda: b"\x00\xff",  # bytes with non-UTF8
        lambda: datetime.datetime(2026, 2, 20, 12, 0, 0),  # datetime object
        lambda: datetime.date(2026, 2, 20),  # date object
        lambda: datetime.timedelta(days=5, hours=3),  # timedelta
        lambda: decimal.Decimal("0.1"),  # exact decimal
        lambda: decimal.Decimal("Infinity"),
        lambda: complex(1, 2),  # complex number
    ]
    return random.choice(generators)()


def _generate_dict(depth: int = 0, max_depth: int = 4) -> dict:
    """Generate a random dictionary that might reveal boundary issues."""
    d: dict[Any, Any] = {}
    n_keys = random.randint(0, 8)
    for _ in range(n_keys):
        # Key generation — keys can reveal issues too
        key_generators = [
            lambda: "normal_key",
            lambda: "",  # empty key
            lambda: " ",  # whitespace key
            lambda: "key with spaces",
            lambda: "key.with.dots",
            lambda: "key/with/slashes",
            lambda: "123",  # numeric string key
            lambda: "".join(random.choices(string.ascii_lowercase, k=random.randint(1, 20))),
            lambda: "__proto__",  # prototype pollution
            lambda: "constructor",
            lambda: "toString",
            # Non-string keys — silent loss in JSON (all become strings)
            lambda: 42,  # int key
            lambda: 0,  # zero key
            lambda: True,  # bool key
            lambda: False,  # bool key
            lambda: None,  # None key
            lambda: (1, 2),  # tuple key (hashable, but not JSON)
        ]
        key = random.choice(key_generators)()
        d[key] = _generate_value(depth + 1, max_depth)
    return d


def _generate_list(depth: int = 0, max_depth: int = 4) -> list:
    """Generate a random list."""
    n = random.randint(0, 8)
    return [_generate_value(depth + 1, max_depth) for _ in range(n)]


def _generate_value(depth: int = 0, max_depth: int = 4) -> Any:
    """Generate a random value (scalar, dict, or list)."""
    if depth >= max_depth:
        return _generate_scalar()

    choice = random.random()
    if choice < 0.5:
        return _generate_scalar()
    elif choice < 0.8:
        return _generate_dict(depth, max_depth)
    else:
        return _generate_list(depth, max_depth)


def _compare(original: Any, result: Any, path: str = "$") -> list[Loss]:
    """Compare original and result, finding all information losses."""
    losses: list[Loss] = []

    # Handle NaN specially — NaN != NaN but that's not a loss
    if isinstance(original, float) and isinstance(result, float):
        if math.isnan(original) and math.isnan(result):
            return losses
        if math.isnan(original) and not math.isnan(result):
            losses.append(Loss(original, result, path, "value_change", f"NaN became {result}"))
            return losses

    # Type comparison
    if type(original) != type(result):
        # Special cases: int/float coercion is common and sometimes intentional
        if isinstance(original, (int, float)) and isinstance(result, (int, float)):
            if isinstance(original, int) and isinstance(result, float):
                if result == float(original):
                    losses.append(Loss(original, result, path, "type_change",
                                       f"int {original} became float {result} (value preserved)"))
                else:
                    losses.append(Loss(original, result, path, "type_change",
                                       f"int {original} became float {result} (value changed!)"))
            elif isinstance(original, float) and isinstance(result, int):
                if original == float(result):
                    losses.append(Loss(original, result, path, "type_change",
                                       f"float {original} became int {result} (value preserved)"))
                else:
                    losses.append(Loss(original, result, path, "precision_loss",
                                       f"float {original} truncated to int {result}"))
            return losses

        # bool/int coercion (Python treats bool as int subclass)
        if isinstance(original, bool) and isinstance(result, int) and not isinstance(result, bool):
            losses.append(Loss(original, result, path, "type_change",
                               f"bool {original} became int {result}"))
            return losses

        losses.append(Loss(original, result, path, "type_change",
                           f"type changed from {type(original).__name__} to {type(result).__name__}"))
        return losses

    # None check
    if original is None:
        return losses  # both None

    # Dict comparison
    if isinstance(original, dict):
        # Check for missing keys
        for key in original:
            if key not in result:
                losses.append(Loss(original[key], None, f"{path}.{key}", "missing_key",
                                   f"key '{key}' with value {repr(original[key])[:60]} was lost"))
            else:
                losses.extend(_compare(original[key], result[key], f"{path}.{key}"))
        # Check for added keys (not a loss, but interesting)
        for key in result:
            if key not in original:
                losses.append(Loss(None, result[key], f"{path}.{key}", "added_key",
                                   f"key '{key}' was added with value {repr(result[key])[:60]}"))
        return losses

    # List comparison
    if isinstance(original, list):
        if len(original) != len(result):
            losses.append(Loss(original, result, path, "length_change",
                               f"list length changed from {len(original)} to {len(result)}"))
        for i in range(min(len(original), len(result))):
            losses.extend(_compare(original[i], result[i], f"{path}[{i}]"))
        return losses

    # String comparison
    if isinstance(original, str):
        if original != result:
            if len(original) != len(result):
                losses.append(Loss(original, result, path, "truncation",
                                   f"string length {len(original)} -> {len(result)}"))
            else:
                losses.append(Loss(original, result, path, "value_change",
                                   f"string changed: {repr(original)[:40]} -> {repr(result)[:40]}"))
        return losses

    # Number comparison
    if isinstance(original, (int, float)):
        if original != result:
            if isinstance(original, float) and isinstance(result, float):
                # Check for precision loss
                if abs(original - result) < abs(original) * 1e-10:
                    losses.append(Loss(original, result, path, "precision_loss",
                                       f"float precision: {original} -> {result}"))
                else:
                    losses.append(Loss(original, result, path, "value_change",
                                       f"number changed: {original} -> {result}"))
            else:
                losses.append(Loss(original, result, path, "value_change",
                                   f"number changed: {original} -> {result}"))
        return losses

    # Bool comparison
    if isinstance(original, bool):
        if original != result:
            losses.append(Loss(original, result, path, "value_change",
                               f"bool changed: {original} -> {result}"))
        return losses

    # Fallback: equality check
    if original != result:
        losses.append(Loss(original, result, path, "value_change",
                           f"value changed: {repr(original)[:40]} -> {repr(result)[:40]}"))

    return losses


def cross(crossing: Crossing, samples: int = 100, seed: int | None = None,
          inputs: list[Any] | None = None) -> CrossingReport:
    """
    Test a boundary crossing by generating random inputs and checking
    whether information survives the round trip.

    Args:
        crossing: The Crossing to test
        samples: Number of random inputs to generate
        seed: Random seed for reproducibility
        inputs: Optional list of specific inputs to test alongside random ones.
                These are tested first, then random samples fill the remainder.

    Returns:
        CrossingReport with all detected losses
    """
    if seed is not None:
        random.seed(seed)

    report = CrossingReport(name=crossing.name)

    # Test user-provided inputs first
    if inputs:
        for original in inputs:
            try:
                encoded = crossing.encode(original)
                decoded = crossing.decode(encoded)
                losses = _compare(original, decoded)
                report.results.append(SampleResult(
                    input_value=original, output_value=decoded, losses=losses,
                ))
            except Exception as e:
                report.results.append(SampleResult(
                    input_value=original, output_value=None, error=e,
                ))

    # Fill remaining with random samples
    remaining = max(0, samples - len(report.results))
    for _ in range(remaining):
        original = _generate_value()
        try:
            encoded = crossing.encode(original)
            decoded = crossing.decode(encoded)
            losses = _compare(original, decoded)
            report.results.append(SampleResult(
                input_value=original, output_value=decoded, losses=losses,
            ))
        except Exception as e:
            report.results.append(SampleResult(
                input_value=original, output_value=None, error=e,
            ))

    return report


@dataclass
class DiffResult:
    """Result of comparing two crossings on the same input."""
    input_value: Any
    output_a: Any
    output_b: Any
    a_losses: list[Loss]
    b_losses: list[Loss]
    divergences: list[Loss]  # differences between the two outputs
    a_error: Optional[Exception] = None
    b_error: Optional[Exception] = None

    @property
    def equivalent(self) -> bool:
        """Both crossings behaved the same way — same output or same error."""
        return not self.divergences

    @property
    def one_crashed(self) -> bool:
        """One crossing errored and the other didn't."""
        return (self.a_error is None) != (self.b_error is None)


@dataclass
class DiffReport:
    """Results from comparing two crossings."""
    name_a: str
    name_b: str
    results: list[DiffResult] = field(default_factory=list)

    @property
    def total_samples(self) -> int:
        return len(self.results)

    @property
    def equivalent_count(self) -> int:
        return sum(1 for r in self.results if r.equivalent)

    @property
    def divergent_count(self) -> int:
        return sum(1 for r in self.results if not r.equivalent)

    @property
    def a_only_lossy(self) -> int:
        """Inputs where A lost info but B didn't."""
        return sum(1 for r in self.results if r.a_losses and not r.b_losses
                   and r.a_error is None and r.b_error is None)

    @property
    def b_only_lossy(self) -> int:
        """Inputs where B lost info but A didn't."""
        return sum(1 for r in self.results if r.b_losses and not r.a_losses
                   and r.a_error is None and r.b_error is None)

    def print(self):
        """Print a human-readable diff report."""
        print(f"\n{'='*60}")
        print(f"Differential Report: {self.name_a} vs {self.name_b}")
        print(f"{'='*60}")
        print(f"Samples tested:    {self.total_samples}")
        print(f"Equivalent:        {self.equivalent_count} ({self.equivalent_count/self.total_samples:.0%})")
        print(f"Divergent:         {self.divergent_count} ({self.divergent_count/self.total_samples:.0%})")
        print(f"  {self.name_a} lossier: {self.a_only_lossy}")
        print(f"  {self.name_b} lossier: {self.b_only_lossy}")

        divergent = [r for r in self.results if not r.equivalent]
        if divergent:
            print(f"\nSample divergences (first 5):")
            for r in divergent[:5]:
                print(f"  Input: {repr(r.input_value)[:80]}")
                if r.a_error:
                    print(f"    {self.name_a}: ERROR — {r.a_error}")
                else:
                    print(f"    {self.name_a}: {repr(r.output_a)[:80]} ({len(r.a_losses)} losses)")
                if r.b_error:
                    print(f"    {self.name_b}: ERROR — {r.b_error}")
                else:
                    print(f"    {self.name_b}: {repr(r.output_b)[:80]} ({len(r.b_losses)} losses)")

        print(f"{'='*60}\n")


def diff(a: Crossing, b: Crossing, samples: int = 100,
         seed: int | None = None, inputs: list[Any] | None = None) -> DiffReport:
    """
    Compare two crossings on the same inputs.

    Finds cases where one crossing preserves information that the other loses,
    or where they produce different outputs from the same input. Useful for:
    - Comparing stdlib json vs orjson
    - Comparing two versions of a serializer
    - Validating a migration (old format → new format)

    Args:
        a: First crossing
        b: Second crossing
        samples: Number of random inputs
        seed: Random seed for reproducibility
        inputs: Optional specific inputs to test alongside random ones

    Returns:
        DiffReport showing where the crossings diverge
    """
    if seed is not None:
        random.seed(seed)

    report = DiffReport(name_a=a.name, name_b=b.name)

    test_inputs: list[Any] = []
    if inputs:
        test_inputs.extend(inputs)

    remaining = max(0, samples - len(test_inputs))
    for _ in range(remaining):
        test_inputs.append(_generate_value())

    for original in test_inputs:
        a_output = None
        b_output = None
        a_losses: list[Loss] = []
        b_losses: list[Loss] = []
        a_error = None
        b_error = None

        try:
            a_encoded = a.encode(original)
            a_output = a.decode(a_encoded)
            a_losses = _compare(original, a_output)
        except Exception as e:
            a_error = e

        try:
            b_encoded = b.encode(original)
            b_output = b.decode(b_encoded)
            b_losses = _compare(original, b_output)
        except Exception as e:
            b_error = e

        # Compare the two outputs to each other
        divergences: list[Loss] = []
        if a_error is None and b_error is None:
            divergences = _compare(a_output, b_output)
        elif a_error is not None and b_error is not None:
            # Both crashed — same behavior, not a divergence
            # (unless they crashed differently)
            if type(a_error) != type(b_error) or str(a_error) != str(b_error):
                divergences = [Loss(a_output, b_output, "$", "error_divergence",
                                    f"different errors: {a_error} vs {b_error}")]
        else:
            # One crashed, one didn't — that's a divergence
            crashed = a_error if a_error is not None else b_error
            divergences = [Loss(a_output, b_output, "$", "error_divergence",
                                f"one crossing crashed: {crashed}")]

        report.results.append(DiffResult(
            input_value=original,
            output_a=a_output,
            output_b=b_output,
            a_losses=a_losses,
            b_losses=b_losses,
            divergences=divergences,
            a_error=a_error,
            b_error=b_error,
        ))

    return report


# --- Triangulation: 3-query multi-format comparison ---

@dataclass
class TriangulationResult:
    """Result of testing one input through three crossings."""
    input_value: Any
    outputs: dict[str, Any]        # crossing_name → output
    losses: dict[str, list[Loss]]  # crossing_name → losses
    errors: dict[str, str | None]  # crossing_name → error message
    shared_losses: list[str]       # loss paths present in ALL crossings
    unique_losses: dict[str, list[str]]  # crossing_name → paths lost ONLY by that crossing


@dataclass
class TriangulationReport:
    """Multi-format comparison: test data through 3+ crossings simultaneously.

    Inspired by the RLDC phase transition (Block et al., arXiv 2602.20278):
    2-query round-trip testing has provable limitations. 3-query comparison
    enables triangulation — distinguishing inherent data limitations from
    format-specific losses.
    """
    crossing_names: list[str]
    results: list[TriangulationResult] = field(default_factory=list)

    @property
    def total_samples(self) -> int:
        return len(self.results)

    @property
    def unanimous_loss_count(self) -> int:
        """Samples where all crossings lost the same information."""
        return sum(1 for r in self.results if r.shared_losses and not any(r.unique_losses.values()))

    @property
    def divergent_count(self) -> int:
        """Samples where at least one crossing lost something unique."""
        return sum(1 for r in self.results if any(r.unique_losses.values()))

    @property
    def all_lossless_count(self) -> int:
        """Samples where no crossing lost anything."""
        return sum(1 for r in self.results
                   if not r.shared_losses and not any(r.unique_losses.values())
                   and not any(r.errors.values()))

    def print(self):
        """Print triangulation report."""
        names = self.crossing_names
        print(f"\n{'='*60}")
        print(f"Triangulation Report: {' × '.join(names)}")
        print(f"{'='*60}")
        print(f"Samples tested:    {self.total_samples}")
        print(f"All lossless:      {self.all_lossless_count} ({self.all_lossless_count/max(self.total_samples,1):.0%})")
        print(f"Unanimous loss:    {self.unanimous_loss_count} ({self.unanimous_loss_count/max(self.total_samples,1):.0%})")
        print(f"Divergent:         {self.divergent_count} ({self.divergent_count/max(self.total_samples,1):.0%})")

        # Per-crossing loss rates
        print(f"\nPer-crossing loss rates:")
        for name in names:
            lossy = sum(1 for r in self.results if r.losses.get(name))
            print(f"  {name:20s} {lossy/max(self.total_samples,1):.0%}")

        # Show divergent samples
        divergent = [r for r in self.results if any(r.unique_losses.values())]
        if divergent:
            print(f"\nDivergences (first 5):")
            for r in divergent[:5]:
                print(f"  Input: {repr(r.input_value)[:80]}")
                if r.shared_losses:
                    print(f"    Shared losses: {', '.join(r.shared_losses[:3])}")
                for name in names:
                    unique = r.unique_losses.get(name, [])
                    if unique:
                        print(f"    Only {name}: {', '.join(unique[:3])}")

        print(f"{'='*60}\n")


def triangulate(*crossings: Crossing, samples: int = 100,
                seed: int | None = None,
                inputs: list[Any] | None = None) -> TriangulationReport:
    """Compare 3+ crossings on the same inputs to triangulate loss sources.

    Two-query testing (round-trip through one format) can detect loss but
    can't distinguish inherent data limitations from format-specific bugs.
    Three-query testing enables triangulation: if all three formats lose
    the same information, it's inherent; if only one loses it, it's a
    format-specific weakness.

    Args:
        crossings: Three or more crossings to compare (minimum 2, best with 3+)
        samples: Number of random inputs
        seed: Random seed for reproducibility
        inputs: Optional specific inputs to test alongside random ones

    Returns:
        TriangulationReport showing shared vs unique losses across formats
    """
    if len(crossings) < 2:
        raise ValueError("triangulate() requires at least 2 crossings")

    if seed is not None:
        random.seed(seed)

    names = [c.name for c in crossings]
    report = TriangulationReport(crossing_names=names)

    test_inputs: list[Any] = []
    if inputs:
        test_inputs.extend(inputs)

    remaining = max(0, samples - len(test_inputs))
    for _ in range(remaining):
        test_inputs.append(_generate_value())

    for original in test_inputs:
        outputs: dict[str, Any] = {}
        losses: dict[str, list[Loss]] = {}
        errors: dict[str, str | None] = {}

        for c in crossings:
            try:
                encoded = c.encode(original)
                decoded = c.decode(encoded)
                found_losses = _compare(original, decoded)
                outputs[c.name] = decoded
                losses[c.name] = found_losses
                errors[c.name] = None
            except Exception as e:
                outputs[c.name] = None
                losses[c.name] = []
                errors[c.name] = str(e)

        # Compute loss paths per crossing
        loss_paths: dict[str, set[str]] = {}
        for name in names:
            loss_paths[name] = {l.path for l in losses[name]}

        # Shared losses: present in ALL crossings that didn't error
        active = [name for name in names if errors[name] is None]
        if len(active) >= 2:
            shared = set.intersection(*(loss_paths[n] for n in active)) if active else set()
        else:
            shared = set()

        # Unique losses: present in exactly one crossing
        unique: dict[str, list[str]] = {}
        for name in active:
            others = [n for n in active if n != name]
            if others:
                other_paths = set.union(*(loss_paths[n] for n in others))
                unique[name] = sorted(loss_paths[name] - other_paths)
            else:
                unique[name] = sorted(loss_paths[name])

        report.results.append(TriangulationResult(
            input_value=original,
            outputs=outputs,
            losses=losses,
            errors=errors,
            shared_losses=sorted(shared),
            unique_losses=unique,
        ))

    return report


# --- Built-in crossings for common boundaries ---

def json_crossing(name: str = "JSON round-trip") -> Crossing:
    """Test JSON serialization/deserialization."""
    return Crossing(
        encode=lambda d: json.dumps(d, default=str),
        decode=lambda s: json.loads(s),
        name=name,
    )


def json_crossing_strict(name: str = "JSON round-trip (strict)") -> Crossing:
    """Test JSON serialization without default=str — crashes on non-serializable types."""
    return Crossing(
        encode=lambda d: json.dumps(d),
        decode=lambda s: json.loads(s),
        name=name,
    )


def str_crossing(name: str = "str() round-trip") -> Crossing:
    """Test str()/eval() round-trip — intentionally lossy, but how lossy?"""
    return Crossing(
        encode=lambda d: repr(d),
        decode=lambda s: eval(s),  # noqa: S307 — intentional for testing
        name=name,
    )


def url_query_crossing(name: str = "URL query string round-trip") -> Crossing:
    """Test URL query string encoding — only handles flat string→string maps."""
    from urllib.parse import urlencode, parse_qs

    def encode(d: Any) -> str:
        if not isinstance(d, dict):
            return urlencode({"_value": d})
        return urlencode(d, doseq=True)

    def decode(s: str) -> Any:
        parsed = parse_qs(s)
        # parse_qs wraps every value in a list
        result = {}
        for k, v in parsed.items():
            result[k] = v[0] if len(v) == 1 else v
        if "_value" in result and len(result) == 1:
            return result["_value"]
        return result

    return Crossing(encode=encode, decode=decode, name=name)


def pickle_crossing(name: str = "pickle round-trip") -> Crossing:
    """Test pickle serialization — should be nearly lossless for Python types."""
    import pickle

    return Crossing(
        encode=lambda d: pickle.dumps(d),
        decode=lambda s: pickle.loads(s),  # noqa: S301 — intentional for testing
        name=name,
    )


def string_truncation_crossing(max_length: int = 255,
                                name: str = "string truncation") -> Crossing:
    """Simulate a database column with max_length — truncates strings silently."""
    def encode(d: Any) -> Any:
        if isinstance(d, str):
            return d[:max_length]
        if isinstance(d, dict):
            return {k: encode(v) for k, v in d.items()}
        if isinstance(d, list):
            return [encode(v) for v in d]
        return d

    return Crossing(encode=encode, decode=lambda d: d, name=name)


def yaml_crossing(name: str = "YAML round-trip") -> Crossing:
    """Test YAML serialization/deserialization.

    YAML has interesting behaviors: it auto-parses unquoted strings as
    booleans (yes/no/on/off), numbers, dates, and null. This makes it
    one of the lossiest common serialization formats.
    """
    import yaml

    return Crossing(
        encode=lambda d: yaml.dump(d, default_flow_style=False),
        decode=lambda s: yaml.safe_load(s),
        name=name,
    )


def toml_crossing(name: str = "TOML round-trip") -> Crossing:
    """Test TOML serialization/deserialization.

    TOML is strict about types but has limitations: no None/null,
    no top-level arrays/scalars (must be a table), datetime is
    first-class but many Python types aren't representable.
    """
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # Python 3.10 backport
    import tomli_w

    def encode(d: Any) -> str:
        if not isinstance(d, dict):
            d = {"_value": d}
        return tomli_w.dumps(d)

    def decode(s: str) -> Any:
        result = tomllib.loads(s)
        if "_value" in result and len(result) == 1:
            return result["_value"]
        return result

    return Crossing(encode=encode, decode=decode, name=name)


def csv_crossing(name: str = "CSV round-trip") -> Crossing:
    """Test CSV serialization — everything becomes strings, structure is lost.

    CSV is perhaps the lossiest common format: no type information,
    no nesting, no null distinction. But it's ubiquitous.
    """
    import csv
    import io

    def encode(d: Any) -> str:
        output = io.StringIO()
        if isinstance(d, dict):
            # Flatten: only string keys, stringify values
            flat = {str(k): str(v) for k, v in d.items()}
            writer = csv.DictWriter(output, fieldnames=list(flat.keys()))
            writer.writeheader()
            writer.writerow(flat)
        elif isinstance(d, (list, tuple)):
            writer = csv.writer(output)
            writer.writerow(d)
        else:
            writer = csv.writer(output)
            writer.writerow([d])
        return output.getvalue()

    def decode(s: str) -> Any:
        lines = s.strip().split("\n")
        if len(lines) >= 2:
            # Has header — parse as DictReader
            reader = csv.DictReader(io.StringIO(s))
            rows = list(reader)
            if len(rows) == 1:
                return dict(rows[0])
            return [dict(r) for r in rows]
        else:
            # Single row — parse as list
            reader = csv.reader(io.StringIO(s))
            rows = list(reader)
            if rows and len(rows[0]) == 1:
                return rows[0][0]
            return rows[0] if rows else []

    return Crossing(encode=encode, decode=decode, name=name)


def env_file_crossing(name: str = "env file round-trip") -> Crossing:
    """Test .env file format — flat string→string, no types, no nesting.

    Simulates the common pattern of writing config to .env files.
    """
    def encode(d: Any) -> str:
        if not isinstance(d, dict):
            return f"VALUE={d}\n"
        lines = []
        for k, v in d.items():
            lines.append(f"{k}={v}")
        return "\n".join(lines) + "\n"

    def decode(s: str) -> Any:
        result = {}
        for line in s.strip().split("\n"):
            if "=" in line:
                key, _, value = line.partition("=")
                result[key] = value
        if "VALUE" in result and len(result) == 1:
            return result["VALUE"]
        return result

    return Crossing(encode=encode, decode=decode, name=name)


@dataclass
class ScalingPoint:
    """One measurement of loss at a given boundary count."""
    n_boundaries: int
    loss_rate: float
    mean_losses_per_sample: float
    error_rate: float
    total_samples: int


@dataclass
class ScalingReport:
    """How information loss scales with boundary count."""
    crossing_name: str
    points: list[ScalingPoint]
    exponent: float | None = None  # power law fit: loss ~ n^alpha
    r_squared: float | None = None

    def print(self):
        """Print scaling analysis results."""
        print(f"\n=== Scaling Analysis: {self.crossing_name} ===\n")
        print(f"{'N':>4}  {'Loss Rate':>10}  {'Mean Losses':>12}  {'Error Rate':>11}")
        print(f"{'─'*4}  {'─'*10}  {'─'*12}  {'─'*11}")
        for p in self.points:
            print(f"{p.n_boundaries:4d}  {p.loss_rate:10.1%}  {p.mean_losses_per_sample:12.2f}  {p.error_rate:11.1%}")

        if self.exponent is not None:
            print(f"\nScaling exponent α ≈ {self.exponent:.3f}  (loss_rate ~ N^α)")
            print(f"R² = {self.r_squared:.4f}")
            if self.exponent < 0.5:
                print("→ Sub-linear: boundaries partially absorb each other's losses")
            elif self.exponent < 1.1:
                print("→ Linear: each boundary adds independent loss")
            else:
                print("→ Super-linear: boundaries amplify each other's losses")


def scaling(
    crossing: Crossing,
    max_n: int = 8,
    samples: int = 200,
    seed: int | None = 42,
) -> ScalingReport:
    """Measure how information loss scales with boundary count.

    Composes 1, 2, 3, ..., max_n copies of the same crossing
    and measures loss rate at each level. Fits a power law
    to determine the scaling exponent.

    Args:
        crossing: The boundary crossing to compose repeatedly.
        max_n: Maximum number of composed boundaries to test.
        samples: Number of random samples per measurement.
        seed: Random seed for reproducibility.

    Returns:
        ScalingReport with measurements and power law fit.
    """
    points = []

    for n in range(1, max_n + 1):
        if n == 1:
            c = crossing
        else:
            c = compose(*[crossing] * n, name=f"{crossing.name} ×{n}")

        report = cross(c, samples=samples, seed=seed)

        mean_losses = report.total_loss_events / max(report.total_samples, 1)
        points.append(ScalingPoint(
            n_boundaries=n,
            loss_rate=report.loss_rate,
            mean_losses_per_sample=mean_losses,
            error_rate=report.error_count / max(report.total_samples, 1),
            total_samples=report.total_samples,
        ))

    # Fit power law: loss_rate = a * n^alpha
    # Take log of both sides: log(loss_rate) = log(a) + alpha * log(n)
    exponent = None
    r_squared = None

    # Filter to points with non-zero loss rate for log fitting
    valid = [(p.n_boundaries, p.loss_rate) for p in points if p.loss_rate > 0]

    if len(valid) >= 3:
        log_n = [math.log(n) for n, _ in valid]
        log_loss = [math.log(lr) for _, lr in valid]

        # Simple linear regression on log-log
        n_pts = len(valid)
        mean_x = sum(log_n) / n_pts
        mean_y = sum(log_loss) / n_pts

        ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_n, log_loss))
        ss_xx = sum((x - mean_x) ** 2 for x in log_n)
        ss_yy = sum((y - mean_y) ** 2 for y in log_loss)

        if ss_xx > 0 and ss_yy > 0:
            exponent = ss_xy / ss_xx
            r_squared = (ss_xy ** 2) / (ss_xx * ss_yy)

    return ScalingReport(
        crossing_name=crossing.name,
        points=points,
        exponent=exponent,
        r_squared=r_squared,
    )


BUILTIN_CROSSINGS = {
    "json": json_crossing,
    "json-strict": json_crossing_strict,
    "pickle": pickle_crossing,
    "url": url_query_crossing,
    "str": str_crossing,
    "yaml": yaml_crossing,
    "toml": toml_crossing,
    "csv": csv_crossing,
    "env": env_file_crossing,
}


def cli(args: list[str] | None = None) -> None:
    """Command-line interface for crossing."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="crossing",
        description="Detect silent information loss at system boundaries",
    )
    sub = parser.add_subparsers(dest="command")

    # crossing test [FORMAT ...] --samples N --seed S
    test_p = sub.add_parser("test", help="Test one or more built-in crossings")
    test_p.add_argument(
        "formats", nargs="*", default=None,
        help=f"Formats to test (default: all). Choices: {', '.join(BUILTIN_CROSSINGS)}",
    )
    test_p.add_argument("-n", "--samples", type=int, default=500, help="Samples per crossing (default: 500)")
    test_p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # crossing compose FORMAT FORMAT ... --samples N
    comp_p = sub.add_parser("compose", help="Test a composed pipeline of crossings")
    comp_p.add_argument("formats", nargs="+", choices=list(BUILTIN_CROSSINGS),
                        help="Formats to compose in order")
    comp_p.add_argument("-n", "--samples", type=int, default=500)
    comp_p.add_argument("--seed", type=int, default=42)

    # crossing scale FORMAT --max-n N --samples N
    scale_p = sub.add_parser("scale", help="Measure how loss scales with boundary count")
    scale_p.add_argument("format", choices=list(BUILTIN_CROSSINGS))
    scale_p.add_argument("--max-n", type=int, default=8, help="Max composition depth (default: 8)")
    scale_p.add_argument("-n", "--samples", type=int, default=200)
    scale_p.add_argument("--seed", type=int, default=42)

    # crossing triangulate FORMAT FORMAT FORMAT ... --samples N
    tri_p = sub.add_parser("triangulate", help="Compare 3+ crossings to triangulate loss sources")
    tri_p.add_argument("formats", nargs="+", choices=list(BUILTIN_CROSSINGS),
                        help="Formats to compare (recommend 3+)")
    tri_p.add_argument("-n", "--samples", type=int, default=200)
    tri_p.add_argument("--seed", type=int, default=42)

    # crossing list
    sub.add_parser("list", help="List available built-in crossings")

    parsed = parser.parse_args(args)

    if parsed.command == "test":
        formats = parsed.formats or list(BUILTIN_CROSSINGS)
        for fmt in formats:
            if fmt not in BUILTIN_CROSSINGS:
                print(f"Unknown crossing: {fmt}. Use 'crossing list' to see available crossings.")
                return
            factory = BUILTIN_CROSSINGS[fmt]
            c = factory()
            report = cross(c, samples=parsed.samples, seed=parsed.seed)
            report.print()

    elif parsed.command == "compose":
        crossings = [BUILTIN_CROSSINGS[f]() for f in parsed.formats]
        pipeline = compose(*crossings)
        report = cross(pipeline, samples=parsed.samples, seed=parsed.seed)
        report.print()

    elif parsed.command == "scale":
        c = BUILTIN_CROSSINGS[parsed.format]()
        sr = scaling(c, max_n=parsed.max_n, samples=parsed.samples, seed=parsed.seed)
        sr.print()

    elif parsed.command == "triangulate":
        crossings_list = [BUILTIN_CROSSINGS[f]() for f in parsed.formats]
        tri_report = triangulate(*crossings_list, samples=parsed.samples, seed=parsed.seed)
        tri_report.print()

    elif parsed.command == "list":
        print("Available crossings:")
        for name, factory in BUILTIN_CROSSINGS.items():
            try:
                c = factory()
                print(f"  {name:15s} {c.name}")
            except (ImportError, ModuleNotFoundError):
                print(f"  {name:15s} (not installed)")

    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
