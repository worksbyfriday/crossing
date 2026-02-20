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


def cross(crossing: Crossing, samples: int = 100, seed: int | None = None) -> CrossingReport:
    """
    Test a boundary crossing by generating random inputs and checking
    whether information survives the round trip.

    Args:
        crossing: The Crossing to test
        samples: Number of random inputs to generate
        seed: Random seed for reproducibility

    Returns:
        CrossingReport with all detected losses
    """
    if seed is not None:
        random.seed(seed)

    report = CrossingReport(name=crossing.name)

    for _ in range(samples):
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


if __name__ == "__main__":
    print("crossing — detect silent information loss at system boundaries\n")

    # Test JSON round-trip (lenient)
    report = cross(json_crossing(), samples=500, seed=42)
    report.print()

    # Test JSON round-trip (strict)
    report = cross(json_crossing_strict(), samples=500, seed=42)
    report.print()

    # Test pickle (should be nearly lossless)
    report = cross(pickle_crossing(), samples=500, seed=42)
    report.print()

    # Test URL query strings
    report = cross(url_query_crossing(), samples=200, seed=42)
    report.print()

    # Test string truncation (simulates DB column)
    report = cross(string_truncation_crossing(255), samples=500, seed=42)
    report.print()

    # Test composed pipeline: JSON → truncation → JSON
    # (simulates: serialize, store in DB varchar, deserialize)
    pipeline = compose(
        json_crossing("JSON serialize"),
        string_truncation_crossing(100, "DB varchar(100)"),
    )
    report = cross(pipeline, samples=500, seed=42)
    report.print()
