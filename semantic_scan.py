"""
crossing.semantic_scan — find semantic boundary crossings in codebases.

While scan.py finds format crossings (where data changes representation),
semantic_scan.py finds semantic crossings (where meaning changes while
representation stays the same).

Primary pattern: Exception Polymorphism
  A single exception type (e.g., KeyError) is raised at multiple sites
  with different intended semantics, then caught by a handler that
  assumes a single meaning. The same signal carries different information
  depending on the code path, but the handler can't distinguish.

  Example (tox #3809):
    - KeyError raised in process_raw() meaning "factor-filtered to empty"
    - KeyError raised in SectionProxy.__getitem__ meaning "key doesn't exist"
    - Single except KeyError handler in ReplaceReferenceIni treats both
      as "unresolved reference" — wrong for the first case.

Usage:
    python3 semantic_scan.py /path/to/codebase
"""

from __future__ import annotations

import ast
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ImportRecord:
    """A tracked import statement."""
    file: str           # file containing the import
    line: int
    module: str         # dotted module path (e.g., "os.path")
    name: str           # imported name (e.g., "join") or "" for plain import
    alias: str          # local alias, or same as name


@dataclass
class CallEdge:
    """A function call edge in the call graph."""
    caller: str  # fully qualified name (class.func or func)
    callee: str  # name of the called function
    file: str
    line: int


class CallGraph:
    """Simple intra-file call graph for reachability queries."""

    def __init__(self, edges: list[CallEdge]):
        # Build adjacency list: caller -> set of callees
        self._adj: dict[str, set[str]] = {}
        for edge in edges:
            self._adj.setdefault(edge.caller, set()).add(edge.callee)

    def callees(self, func: str) -> set[str]:
        """Direct callees of a function."""
        return self._adj.get(func, set())

    def reachable(self, func: str, max_depth: int = 10) -> set[str]:
        """All functions reachable from func through the call graph."""
        visited: set[str] = set()
        stack = [func]
        depth = 0
        while stack and depth < max_depth:
            next_stack = []
            for f in stack:
                if f in visited:
                    continue
                visited.add(f)
                for callee in self._adj.get(f, set()):
                    if callee not in visited:
                        next_stack.append(callee)
            stack = next_stack
            depth += 1
        visited.discard(func)  # Don't include the starting function
        return visited

    def can_reach(self, caller: str, callee: str, max_depth: int = 10) -> bool:
        """Check if caller can reach callee through the call chain."""
        return callee in self.reachable(caller, max_depth)

    @property
    def edge_count(self) -> int:
        return sum(len(v) for v in self._adj.values())

    @property
    def node_count(self) -> int:
        nodes = set(self._adj.keys())
        for callees in self._adj.values():
            nodes.update(callees)
        return len(nodes)


@dataclass
class ExceptionRaise:
    """A site where an exception is raised."""
    file: str
    line: int
    exception_type: str
    in_function: str  # enclosing function name
    in_class: str  # enclosing class name, or ""
    source_line: str
    context: str  # brief description of what's happening
    implicit: bool = False  # True if from dict/list access, not explicit raise
    try_scope_id: int | None = None  # ID of enclosing try block, if any
    message_arg: str | None = None  # string literal argument to exception, if any

    def __str__(self):
        loc = f"{self.in_class}.{self.in_function}" if self.in_class else self.in_function
        kind = "implicit" if self.implicit else "raise"
        return f"{self.file}:{self.line} — {kind} {self.exception_type} in {loc}"


@dataclass
class ExceptionHandler:
    """A site where an exception is caught."""
    file: str
    line: int
    exception_type: str
    in_function: str
    in_class: str
    handler_body_summary: str  # first meaningful statement in handler
    source_line: str
    re_raises: bool  # does the handler re-raise?
    returns_value: bool  # does the handler return a value?
    assigns_default: bool  # does the handler assign a default value?
    try_scope_id: int | None = None  # ID of the try block this handler belongs to
    direct_raises_in_scope: int = 0  # explicit raises of this type in the try body
    uses_exception: bool = False  # does the handler use the bound exception variable?
    binds_exception: bool = False  # does the handler bind the exception to a name?

    def __str__(self):
        loc = f"{self.in_class}.{self.in_function}" if self.in_class else self.in_function
        action = "re-raises" if self.re_raises else ("returns" if self.returns_value else "handles")
        return f"{self.file}:{self.line} — except {self.exception_type} in {loc} ({action})"


@dataclass
class SemanticCrossing:
    """A detected semantic boundary crossing.

    Multiple raise sites for the same exception type, caught by
    handler(s) that may not distinguish between the different meanings.
    """
    exception_type: str
    raise_sites: list[ExceptionRaise] = field(default_factory=list)
    handler_sites: list[ExceptionHandler] = field(default_factory=list)
    risk_level: str = "low"  # low, medium, high
    description: str = ""

    @property
    def is_polymorphic(self) -> bool:
        """True if the same exception type has multiple raise sites."""
        return len(self.raise_sites) > 1

    @property
    def has_uniform_handler(self) -> bool:
        """True if handlers treat all cases identically (potential bug)."""
        if not self.handler_sites:
            return False
        # All handlers do the same thing (all return, all re-raise, etc.)
        actions = set()
        for h in self.handler_sites:
            if h.re_raises:
                actions.add("re-raise")
            elif h.returns_value:
                actions.add("return")
            elif h.assigns_default:
                actions.add("default")
            else:
                actions.add("other")
        return len(actions) == 1

    @property
    def semantic_entropy(self) -> float:
        """Shannon entropy of the raise-site distribution (bits).

        Measures how much semantic information the exception type carries
        at the point of raising. Each unique (function, context) pair is
        a distinct micro-state. Uniform distribution assumed (worst case).

        H = log2(N) where N = number of distinct semantic origins.
        Returns 0.0 for single-origin exceptions.
        """
        if len(self.raise_sites) <= 1:
            return 0.0
        # Count distinct semantic origins: (class.function) pairs
        origins = set()
        for r in self.raise_sites:
            loc = f"{r.in_class}.{r.in_function}" if r.in_class else r.in_function
            origins.add(loc)
        n = len(origins)
        return math.log2(n) if n > 1 else 0.0

    @property
    def handler_discrimination(self) -> float:
        """How much semantic information handlers preserve (bits).

        A handler that re-raises preserves all information (passes the
        exception with its full context to the next handler). A handler
        that returns a default or silently passes destroys all information.
        A handler that branches preserves some.

        No handlers means the exception propagates uncaught — full
        preservation (discrimination = semantic_entropy).

        Computed as: fraction of handlers that re-raise × semantic_entropy.
        This is a lower bound — actual discrimination from isinstance checks
        or message inspection would add capacity, but we can't detect that
        statically without deeper analysis.
        """
        if self.semantic_entropy == 0.0:
            return 0.0
        if not self.handler_sites:
            # No handler = exception propagates with full semantics
            return self.semantic_entropy
        # Score each handler's discrimination capacity
        total_capacity = 0.0
        for h in self.handler_sites:
            if h.re_raises:
                total_capacity += 1.0  # Full preservation
            elif h.uses_exception:
                # Handler inspects the exception — partial preservation
                total_capacity += 0.7
            elif h.direct_raises_in_scope > 0:
                # Handler is near the raise — likely knows the context
                total_capacity += 0.5
            else:
                total_capacity += 0.0  # Default/return/pass = total collapse
        avg_capacity = total_capacity / len(self.handler_sites)
        return avg_capacity * self.semantic_entropy

    @property
    def information_loss(self) -> float:
        """Bits of semantic information destroyed at this crossing.

        ΔH = H_raise - H_handle. Higher values mean more meaning is lost
        when exceptions flow from raise sites through handlers.
        """
        return self.semantic_entropy - self.handler_discrimination

    @property
    def collapse_ratio(self) -> float:
        """Normalized information loss: 0.0 (no collapse) to 1.0 (total).

        The fraction of semantic information destroyed by the handler.
        A collapse_ratio of 1.0 means the handler erases all distinction
        between the different raise sites — the canonical "crossing."
        """
        if self.semantic_entropy == 0.0:
            return 0.0
        return self.information_loss / self.semantic_entropy

    def __str__(self):
        return (
            f"{self.exception_type}: {len(self.raise_sites)} raise sites, "
            f"{len(self.handler_sites)} handlers — {self.risk_level} risk"
        )


@dataclass
class SemanticScanReport:
    """Results from scanning for semantic crossings."""
    root: str
    files_scanned: int = 0
    parse_errors: int = 0
    raises: list[ExceptionRaise] = field(default_factory=list)
    handlers: list[ExceptionHandler] = field(default_factory=list)
    crossings: list[SemanticCrossing] = field(default_factory=list)

    @property
    def polymorphic_crossings(self) -> list[SemanticCrossing]:
        return [c for c in self.crossings if c.is_polymorphic]

    @property
    def risky_crossings(self) -> list[SemanticCrossing]:
        return [c for c in self.crossings if c.risk_level in ("medium", "high")]

    @property
    def total_information_loss(self) -> float:
        """Total bits of semantic information lost across all crossings."""
        return sum(c.information_loss for c in self.crossings)

    @property
    def mean_collapse_ratio(self) -> float:
        """Average collapse ratio across crossings with non-zero entropy."""
        relevant = [c for c in self.crossings if c.semantic_entropy > 0]
        if not relevant:
            return 0.0
        return sum(c.collapse_ratio for c in relevant) / len(relevant)

    def filter(self, min_risk: str = "low") -> "SemanticScanReport":
        """Return a new report filtered by minimum risk level."""
        risk_order = {"low": 0, "medium": 1, "elevated": 1, "high": 2}
        threshold = risk_order.get(min_risk, 0)
        filtered = [c for c in self.crossings
                     if risk_order.get(c.risk_level, 0) >= threshold]
        report = SemanticScanReport(root=self.root)
        report.files_scanned = self.files_scanned
        report.parse_errors = self.parse_errors
        report.raises = self.raises
        report.handlers = self.handlers
        report.crossings = filtered
        return report

    def to_json(self) -> str:
        """Serialize report to JSON."""
        import json as json_mod
        explicit = [r for r in self.raises if not r.implicit]
        implicit = [r for r in self.raises if r.implicit]
        data = {
            "root": self.root,
            "summary": {
                "files_scanned": self.files_scanned,
                "parse_errors": self.parse_errors,
                "total_raises": len(self.raises),
                "explicit_raises": len(explicit),
                "implicit_raises": len(implicit),
                "total_handlers": len(self.handlers),
                "total_crossings": len(self.crossings),
                "polymorphic_crossings": len(self.polymorphic_crossings),
                "risky_crossings": len(self.risky_crossings),
                "total_information_loss_bits": round(self.total_information_loss, 2),
                "mean_collapse_ratio": round(self.mean_collapse_ratio, 2),
            },
            "crossings": [
                {
                    "exception_type": c.exception_type,
                    "risk_level": c.risk_level,
                    "description": c.description,
                    "is_polymorphic": c.is_polymorphic,
                    "raise_sites": [
                        {"file": r.file, "line": r.line,
                         "exception_type": r.exception_type,
                         "function": f"{r.in_class}.{r.in_function}" if r.in_class else r.in_function,
                         "implicit": r.implicit,
                         "context": r.context,
                         "message": r.message_arg}
                        for r in c.raise_sites
                    ],
                    "handler_sites": [
                        {"file": h.file, "line": h.line,
                         "exception_type": h.exception_type,
                         "function": f"{h.in_class}.{h.in_function}" if h.in_class else h.in_function,
                         "re_raises": h.re_raises,
                         "returns_value": h.returns_value,
                         "assigns_default": h.assigns_default,
                         "direct_raises_in_scope": h.direct_raises_in_scope}
                        for h in c.handler_sites
                    ],
                    "information_theory": {
                        "semantic_entropy_bits": round(c.semantic_entropy, 2),
                        "handler_discrimination_bits": round(c.handler_discrimination, 2),
                        "information_loss_bits": round(c.information_loss, 2),
                        "collapse_ratio": round(c.collapse_ratio, 2),
                    },
                }
                for c in self.crossings
            ],
        }
        return json_mod.dumps(data, indent=2)

    def to_markdown(self) -> str:
        """Serialize report to markdown."""
        explicit = [r for r in self.raises if not r.implicit]
        implicit = [r for r in self.raises if r.implicit]
        lines = [
            f"# Crossing Scan: `{self.root}`\n",
            f"| Metric | Count |",
            f"|--------|-------|",
            f"| Files scanned | {self.files_scanned} |",
            f"| Parse errors | {self.parse_errors} |",
            f"| Exception raises | {len(self.raises)} ({len(explicit)} explicit, {len(implicit)} implicit) |",
            f"| Exception handlers | {len(self.handlers)} |",
            f"| Semantic crossings | {len(self.crossings)} |",
            f"| Polymorphic | {len(self.polymorphic_crossings)} |",
            f"| Elevated/high risk | {len(self.risky_crossings)} |",
            "",
        ]
        if self.risky_crossings:
            lines.append("## Risky Crossings\n")
            for crossing in self.risky_crossings:
                lines.append(f"### {crossing.exception_type} ({crossing.risk_level} risk)\n")
                lines.append(f"{crossing.description}\n")
                lines.append("**Raise sites:**\n")
                for r in crossing.raise_sites[:10]:
                    lines.append(f"- `{r.file}:{r.line}` — {'implicit' if r.implicit else 'raise'} "
                                 f"{r.exception_type} in {r.in_class + '.' if r.in_class else ''}{r.in_function}")
                if len(crossing.raise_sites) > 10:
                    lines.append(f"- ... and {len(crossing.raise_sites) - 10} more")
                lines.append("\n**Handlers:**\n")
                for h in crossing.handler_sites[:10]:
                    action = "re-raises" if h.re_raises else ("returns" if h.returns_value else "handles")
                    lines.append(f"- `{h.file}:{h.line}` — except {h.exception_type} "
                                 f"in {h.in_class + '.' if h.in_class else ''}{h.in_function} ({action})")
                if len(crossing.handler_sites) > 10:
                    lines.append(f"- ... and {len(crossing.handler_sites) - 10} more")
                if crossing.semantic_entropy > 0:
                    lines.append(f"\n**Information theory:** {crossing.semantic_entropy:.1f} bits entropy, "
                                 f"{crossing.information_loss:.1f} bits lost, "
                                 f"{crossing.collapse_ratio:.0%} collapse")
                lines.append("")
        elif self.polymorphic_crossings:
            lines.append("## Polymorphic Crossings (low risk)\n")
            for crossing in self.polymorphic_crossings[:5]:
                lines.append(f"- {crossing}")
            lines.append("")
        return "\n".join(lines)

    def print(self):
        print(f"\n{'='*60}")
        print(f"Semantic Crossing Scan: {self.root}")
        print(f"{'='*60}")
        explicit = [r for r in self.raises if not r.implicit]
        implicit = [r for r in self.raises if r.implicit]
        print(f"Files scanned:        {self.files_scanned}")
        print(f"Parse errors:         {self.parse_errors}")
        print(f"Exception raises:     {len(self.raises)} ({len(explicit)} explicit, {len(implicit)} implicit)")
        print(f"Exception handlers:   {len(self.handlers)}")
        print(f"Semantic crossings:   {len(self.crossings)}")
        print(f"  Polymorphic (multi-raise):  {len(self.polymorphic_crossings)}")
        print(f"  Elevated risk:              {len(self.risky_crossings)}")
        if self.total_information_loss > 0:
            print(f"  Total info loss:            {self.total_information_loss:.1f} bits")
            print(f"  Mean collapse ratio:        {self.mean_collapse_ratio:.0%}")

        for crossing in self.risky_crossings:
            print(f"\n--- {crossing} ---")
            print(f"  {crossing.description}")
            if crossing.semantic_entropy > 0:
                print(f"  Information: {crossing.semantic_entropy:.1f} bits entropy, "
                      f"{crossing.information_loss:.1f} bits lost, "
                      f"{crossing.collapse_ratio:.0%} collapse")
            print(f"  Raise sites:")
            for r in crossing.raise_sites[:5]:
                print(f"    {r}")
            if len(crossing.raise_sites) > 5:
                print(f"    ... and {len(crossing.raise_sites) - 5} more")
            print(f"  Handlers:")
            for h in crossing.handler_sites[:5]:
                usage = ""
                if h.binds_exception and h.uses_exception:
                    usage = " [inspects exc]"
                elif h.binds_exception and not h.uses_exception:
                    usage = " [binds but ignores exc]"
                print(f"    {h}{usage}")
            if len(crossing.handler_sites) > 5:
                print(f"    ... and {len(crossing.handler_sites) - 5} more")

        if not self.risky_crossings and self.polymorphic_crossings:
            print(f"\nPolymorphic crossings (low risk):")
            for crossing in self.polymorphic_crossings[:5]:
                print(f"  {crossing}")

        print(f"{'='*60}\n")


class SemanticVisitor(ast.NodeVisitor):
    """AST visitor that finds exception raise/handle patterns."""

    def __init__(self, filename: str, source_lines: list[str], detect_implicit: bool = False):
        self.filename = filename
        self.source_lines = source_lines
        self.raises: list[ExceptionRaise] = []
        self.handlers: list[ExceptionHandler] = []
        self.call_edges: list[CallEdge] = []
        self.imports: list[ImportRecord] = []
        self.exception_parents: dict[str, str] = {}  # child -> parent exception class
        self._scope_stack: list[tuple[str, str]] = []  # (class_name, func_name)
        self._detect_implicit = detect_implicit
        self._try_scope_counter = 0
        self._try_scope_stack: list[int] = []  # stack of try scope IDs

    @property
    def _current_class(self) -> str:
        for cls, _func in reversed(self._scope_stack):
            if cls:
                return cls
        return ""

    @property
    def _current_function(self) -> str:
        for _cls, func in reversed(self._scope_stack):
            if func:
                return func
        return "<module>"

    def visit_ClassDef(self, node: ast.ClassDef):
        # Record exception inheritance: class CustomError(ValueError) -> parent=ValueError
        for base in node.bases:
            base_name = self._get_name(base)
            if base_name and self._looks_like_exception(base_name):
                self.exception_parents[node.name] = base_name
                break  # only record first exception parent
        self._scope_stack.append((node.name, ""))
        self.generic_visit(node)
        self._scope_stack.pop()

    @staticmethod
    def _looks_like_exception(name: str) -> bool:
        """Heuristic: does this class name look like an exception?"""
        return (name.endswith("Error") or name.endswith("Exception")
                or name.endswith("Warning") or name == "BaseException"
                or name == "Exception")

    @staticmethod
    def _get_name(node: ast.expr) -> str:
        """Get simple name from an AST expression."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def visit_Import(self, node: ast.Import):
        """Track `import X` statements."""
        for alias in node.names:
            self.imports.append(ImportRecord(
                file=self.filename,
                line=node.lineno,
                module=alias.name,
                name="",
                alias=alias.asname or alias.name,
            ))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Track `from X import Y` statements."""
        module = node.module or ""
        for alias in node.names:
            self.imports.append(ImportRecord(
                file=self.filename,
                line=node.lineno,
                module=module,
                name=alias.name,
                alias=alias.asname or alias.name,
            ))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._scope_stack.append(("", node.name))
        self.generic_visit(node)
        self._scope_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    @property
    def _current_try_scope(self) -> int | None:
        return self._try_scope_stack[-1] if self._try_scope_stack else None

    def visit_Try(self, node: ast.Try):
        """Track try scopes for scope-aware analysis."""
        self._try_scope_counter += 1
        scope_id = self._try_scope_counter

        # Visit the try body with the scope ID on the stack
        self._try_scope_stack.append(scope_id)
        for child in node.body:
            self.visit(child)
        self._try_scope_stack.pop()

        # Visit handlers — they belong to this try scope but are NOT in the try body
        for handler in node.handlers:
            # Tag the handler with this scope ID before visiting
            self._current_handler_scope_id = scope_id
            self.visit(handler)
        self._current_handler_scope_id = None

        # Visit else/finally outside the try scope
        for child in node.orelse:
            self.visit(child)
        for child in node.finalbody:
            self.visit(child)

    # Python 3.11+ uses TryStar for except*
    visit_TryStar = visit_Try

    def visit_Raise(self, node: ast.Raise):
        if node.exc is None:
            # bare raise — re-raise, skip
            self.generic_visit(node)
            return

        exc_type = self._get_exception_type(node.exc)
        if exc_type:
            source = self._get_source(node.lineno)
            message_arg = self._extract_message_arg(node.exc)
            self.raises.append(ExceptionRaise(
                file=self.filename,
                line=node.lineno,
                exception_type=exc_type,
                in_function=self._current_function,
                in_class=self._current_class,
                source_line=source,
                context=self._infer_context(node),
                try_scope_id=self._current_try_scope,
                message_arg=message_arg,
            ))
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        """Detect implicit KeyError/IndexError from subscript access.

        d[key] can raise KeyError (dict) or IndexError (list/tuple).
        We record both possibilities — the analysis phase determines
        whether these interact with explicit raises or handlers.
        """
        if self._detect_implicit and isinstance(node.ctx, ast.Load):
            source = self._get_source(node.lineno)
            # Record as potential KeyError (most common implicit source)
            self.raises.append(ExceptionRaise(
                file=self.filename,
                line=node.lineno,
                exception_type="KeyError",
                in_function=self._current_function,
                in_class=self._current_class,
                source_line=source,
                context=f"subscript access in {self._current_function}",
                implicit=True,
                try_scope_id=self._current_try_scope,
            ))
        self.generic_visit(node)

    # Builtins that raise ValueError on bad input
    _VALUE_ERROR_BUILTINS = frozenset({"int", "float", "complex", "bool"})

    # Methods that raise ValueError when element not found
    _VALUE_ERROR_METHODS = frozenset({"index", "remove"})

    def _visit_next_call(self, node: ast.Call):
        """Detect implicit StopIteration from next() calls without default."""
        if (self._detect_implicit
                and isinstance(node.func, ast.Name)
                and node.func.id == "next"
                and len(node.args) == 1):  # next(it) without default
            source = self._get_source(node.lineno)
            self.raises.append(ExceptionRaise(
                file=self.filename,
                line=node.lineno,
                exception_type="StopIteration",
                in_function=self._current_function,
                in_class=self._current_class,
                source_line=source,
                context=f"next() without default in {self._current_function}",
                implicit=True,
                try_scope_id=self._current_try_scope,
            ))

    def _visit_conversion_call(self, node: ast.Call):
        """Detect implicit ValueError from int(), float(), etc."""
        if (self._detect_implicit
                and isinstance(node.func, ast.Name)
                and node.func.id in self._VALUE_ERROR_BUILTINS
                and node.args):  # int(x) with an argument
            source = self._get_source(node.lineno)
            self.raises.append(ExceptionRaise(
                file=self.filename,
                line=node.lineno,
                exception_type="ValueError",
                in_function=self._current_function,
                in_class=self._current_class,
                source_line=source,
                context=f"{node.func.id}() conversion in {self._current_function}",
                implicit=True,
                try_scope_id=self._current_try_scope,
            ))

    def _visit_getattr_call(self, node: ast.Call):
        """Detect implicit AttributeError from getattr() without default."""
        if (self._detect_implicit
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) == 2):  # getattr(obj, name) without default
            source = self._get_source(node.lineno)
            self.raises.append(ExceptionRaise(
                file=self.filename,
                line=node.lineno,
                exception_type="AttributeError",
                in_function=self._current_function,
                in_class=self._current_class,
                source_line=source,
                context=f"getattr() without default in {self._current_function}",
                implicit=True,
                try_scope_id=self._current_try_scope,
            ))

    def _visit_method_call(self, node: ast.Call):
        """Detect implicit ValueError from .index(), .remove()."""
        if (self._detect_implicit
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in self._VALUE_ERROR_METHODS):
            source = self._get_source(node.lineno)
            self.raises.append(ExceptionRaise(
                file=self.filename,
                line=node.lineno,
                exception_type="ValueError",
                in_function=self._current_function,
                in_class=self._current_class,
                source_line=source,
                context=f".{node.func.attr}() call in {self._current_function}",
                implicit=True,
                try_scope_id=self._current_try_scope,
            ))

    def visit_Call(self, node: ast.Call):
        """Check calls for implicit raise patterns and record call edges."""
        # Record call edge for call graph
        callee_name = self._get_name(node.func)
        if callee_name:
            caller = self._current_function
            if self._current_class:
                caller = f"{self._current_class}.{caller}"
            self.call_edges.append(CallEdge(
                caller=caller,
                callee=callee_name,
                file=self.filename,
                line=node.lineno,
            ))

        self._visit_next_call(node)
        self._visit_conversion_call(node)
        self._visit_method_call(node)
        self._visit_getattr_call(node)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        if node.type is None:
            # bare except — catches everything
            exc_type = "BaseException"
        else:
            exc_type = self._get_name(node.type) or "unknown"

        source = self._get_source(node.lineno)

        re_raises = False
        returns_value = False
        assigns_default = False

        for stmt in node.body:
            if isinstance(stmt, ast.Raise):
                re_raises = True
            elif isinstance(stmt, ast.Return):
                returns_value = True
            elif isinstance(stmt, ast.Assign):
                assigns_default = True

        # Check if handler binds and uses the exception variable
        binds_exception = node.name is not None
        uses_exception = False
        if binds_exception:
            uses_exception = self._name_used_in(node.name, node.body)

        body_summary = self._summarize_handler_body(node.body)

        # Count direct raises of this type in the corresponding try body
        scope_id = getattr(self, "_current_handler_scope_id", None)
        direct_raises = 0
        if scope_id is not None:
            for r in self.raises:
                if (r.try_scope_id == scope_id
                        and r.exception_type == exc_type
                        and not r.implicit):
                    direct_raises += 1

        self.handlers.append(ExceptionHandler(
            file=self.filename,
            line=node.lineno,
            exception_type=exc_type,
            in_function=self._current_function,
            in_class=self._current_class,
            handler_body_summary=body_summary,
            source_line=source,
            re_raises=re_raises,
            returns_value=returns_value,
            assigns_default=assigns_default,
            try_scope_id=scope_id,
            direct_raises_in_scope=direct_raises,
            uses_exception=uses_exception,
            binds_exception=binds_exception,
        ))
        self.generic_visit(node)

    def _get_exception_type(self, node: ast.expr) -> Optional[str]:
        """Extract exception type from a raise expression."""
        if isinstance(node, ast.Call):
            return self._get_name(node.func)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_name(node)
        return None

    def _get_name(self, node: ast.expr) -> Optional[str]:
        """Get dotted name from an expression."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            base = self._get_name(node.value)
            if base:
                return f"{base}.{node.attr}"
        return None

    def _get_source(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].strip()
        return ""

    def _extract_message_arg(self, exc_node: ast.expr) -> str | None:
        """Extract string literal argument from exception constructor, if any.

        For `raise ValueError("some message")`, returns "some message".
        Returns None if the argument isn't a simple string literal.
        """
        if isinstance(exc_node, ast.Call) and exc_node.args:
            first_arg = exc_node.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                return first_arg.value
        return None

    def _infer_context(self, node: ast.Raise) -> str:
        """Try to describe what's happening around this raise.

        Scans backwards from the raise line to find the nearest enclosing
        control structure (if/elif/for/while) and includes its condition.
        """
        lineno = node.lineno
        for i in range(lineno - 2, max(lineno - 20, -1), -1):
            if 0 <= i < len(self.source_lines):
                line = self.source_lines[i].strip()
                if line.startswith(("if ", "elif ")):
                    cond = line.rstrip(":")
                    return f"{cond} → raise in {self._current_function}"
                elif line.startswith(("for ", "while ")):
                    loop = line.rstrip(":")
                    return f"inside {loop} in {self._current_function}"
                elif line.startswith(("def ", "class ", "except ")):
                    break  # hit scope boundary, stop
        return f"in {self._current_function}"

    def _name_used_in(self, name: str, body: list[ast.stmt]) -> bool:
        """Check if a variable name is referenced in any of the given statements."""
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(node, ast.Name) and node.id == name:
                return True
        return False

    def _summarize_handler_body(self, body: list[ast.stmt]) -> str:
        """Summarize what the handler does."""
        if not body:
            return "pass"
        first = body[0]
        if isinstance(first, ast.Pass):
            return "pass (silently swallowed)"
        elif isinstance(first, ast.Raise):
            return "re-raise"
        elif isinstance(first, ast.Return):
            return "return"
        elif isinstance(first, ast.Assign):
            return "assign default"
        elif isinstance(first, ast.Expr) and isinstance(first.value, ast.Call):
            func = self._get_name(first.value.func) if isinstance(first.value.func, (ast.Name, ast.Attribute)) else None
            return f"call {func}" if func else "function call"
        return "complex handler"


def scan_source(source: str, filename: str = "<input>") -> SemanticScanReport:
    """Scan a source string for semantic crossings. Returns a full report."""
    report = SemanticScanReport(root=filename)
    report.files_scanned = 1
    try:
        source_lines = source.split("\n")
        tree = ast.parse(source, filename=filename)
        visitor = SemanticVisitor(filename, source_lines)
        visitor.visit(tree)
        report.raises = visitor.raises
        report.handlers = visitor.handlers
        call_graph = CallGraph(visitor.call_edges) if visitor.call_edges else None
        report.crossings = analyze_crossings(
            report.raises, report.handlers, call_graph,
            exception_parents=visitor.exception_parents or None,
        )
    except SyntaxError as e:
        report.parse_errors = 1
    return report


def scan_file(filepath: str, detect_implicit: bool = False) -> tuple[list[ExceptionRaise], list[ExceptionHandler], list[CallEdge], dict[str, str], list[ImportRecord]]:
    """Scan a single file for exception raise/handle patterns, call edges, exception hierarchy, and imports."""
    try:
        with open(filepath) as f:
            source = f.read()
        source_lines = source.split("\n")
        tree = ast.parse(source, filename=filepath)
        visitor = SemanticVisitor(filepath, source_lines, detect_implicit=detect_implicit)
        visitor.visit(tree)
        return visitor.raises, visitor.handlers, visitor.call_edges, visitor.exception_parents, visitor.imports
    except SyntaxError:
        return [], [], [], {}, []
    except Exception:
        return [], [], [], {}, []


def _build_ancestor_map(exception_parents: dict[str, str]) -> dict[str, set[str]]:
    """Build a map from each exception type to all its ancestors.

    If CustomError -> ValueError -> Exception, then
    ancestors["CustomError"] = {"ValueError", "Exception"}.
    """
    ancestors: dict[str, set[str]] = {}
    for child in exception_parents:
        chain: set[str] = set()
        current = child
        while current in exception_parents:
            parent = exception_parents[current]
            if parent in chain:
                break  # cycle guard
            chain.add(parent)
            current = parent
        ancestors[child] = chain
    return ancestors


def _build_descendant_map(exception_parents: dict[str, str]) -> dict[str, set[str]]:
    """Build a map from each exception type to all its descendants."""
    descendants: dict[str, set[str]] = {}
    for child, parent in exception_parents.items():
        current = parent
        visited: set[str] = set()
        while current and current not in visited:
            descendants.setdefault(current, set()).add(child)
            visited.add(current)
            current = exception_parents.get(current)
    return descendants


def analyze_crossings(
    raises: list[ExceptionRaise],
    handlers: list[ExceptionHandler],
    call_graph: CallGraph | None = None,
    exception_parents: dict[str, str] | None = None,
) -> list[SemanticCrossing]:
    """Analyze raises and handlers to find semantic crossings.

    A semantic crossing exists when:
    1. The same exception type is raised at multiple sites (polymorphic)
    2. The raise sites have different semantic contexts
    3. Handler(s) treat them uniformly (potential meaning collapse)

    If a call_graph is provided, uses reachability analysis to determine
    whether a raise site can actually reach a handler through the call chain.

    If exception_parents is provided, detects inheritance crossings where
    a handler for a base class catches subclass exceptions with different
    semantics.
    """
    # Build inheritance maps
    descendants = _build_descendant_map(exception_parents) if exception_parents else {}
    ancestors = _build_ancestor_map(exception_parents) if exception_parents else {}

    # Group raises by exception type, including inheritance
    raise_groups: dict[str, list[ExceptionRaise]] = {}
    for r in raises:
        raise_groups.setdefault(r.exception_type, []).append(r)

    # Group handlers by exception type
    handler_groups: dict[str, list[ExceptionHandler]] = {}
    for h in handlers:
        handler_groups.setdefault(h.exception_type, []).append(h)

    crossings = []
    # Pre-compute which types will be absorbed by an ancestor's crossing:
    # if a type has an ancestor that has handlers, skip it (it will be
    # merged into the ancestor's crossing).
    absorbed_types: set[str] = set()
    for exc_type in raise_groups:
        for ancestor in ancestors.get(exc_type, set()):
            if ancestor in handler_groups or ancestor in raise_groups:
                absorbed_types.add(exc_type)
                break

    # Track which exception types we've already processed to avoid duplicates
    processed_types: set[str] = set()

    for exc_type, raise_sites in raise_groups.items():
        if exc_type in processed_types or exc_type in absorbed_types:
            continue
        processed_types.add(exc_type)

        handler_sites = handler_groups.get(exc_type, [])

        # Inheritance-aware: if this type has descendants, include their
        # raise sites too (because a handler for this type catches them).
        # Also include handlers from ancestor types that would catch this type.
        inherited_raise_sites = list(raise_sites)
        if descendants.get(exc_type):
            for desc_type in descendants[exc_type]:
                if desc_type in raise_groups:
                    inherited_raise_sites.extend(raise_groups[desc_type])
                    processed_types.add(desc_type)

        # Use the expanded raise sites for analysis
        has_inheritance = len(inherited_raise_sites) > len(raise_sites)
        if has_inheritance:
            raise_sites = inherited_raise_sites

        crossing = SemanticCrossing(
            exception_type=exc_type,
            raise_sites=raise_sites,
            handler_sites=handler_sites,
        )

        # Add inheritance annotation if descendants contributed raise sites
        if has_inheritance:
            child_types = set(r.exception_type for r in raise_sites) - {exc_type}
            crossing.description = (
                f"Handler for {exc_type} also catches subclass(es): "
                f"{', '.join(sorted(child_types))}. "
            )

        # Classify raise sites
        explicit = [r for r in raise_sites if not r.implicit]
        implicit = [r for r in raise_sites if r.implicit]

        # Determine risk level
        if len(raise_sites) == 1:
            crossing.risk_level = "low"
            crossing.description = "Single raise site — no polymorphism."
        elif not handler_sites:
            crossing.risk_level = "low"
            crossing.description = "Multiple raise sites but no local handlers — propagates to caller."
        elif explicit and implicit and handler_sites:
            # MIXED: explicit raises + implicit raises + handlers
            # This is the tox #3809 pattern — handlers designed for explicit
            # raises also catch incidental implicit ones with different meanings
            crossing.risk_level = "high"
            crossing.description = (
                f"{len(explicit)} explicit + {len(implicit)} implicit raise sites "
                f"for {exc_type}, {len(handler_sites)} handler(s) — "
                f"handlers designed for explicit raises may mishandle implicit ones."
            )

        elif len(handler_sites) == 1 and len(raise_sites) > 2:
            # Many raise sites, one handler — highest risk of meaning collapse
            crossing.risk_level = "high"
            crossing.description = (
                f"{len(raise_sites)} different raise sites for {exc_type}, "
                f"but only 1 handler — meaning collapse likely."
            )
        elif crossing.has_uniform_handler and len(raise_sites) > 1:
            crossing.risk_level = "medium"
            crossing.description = (
                f"Multiple raise sites with uniform handler behavior — "
                f"handler may not distinguish between different raise semantics."
            )
        elif len(raise_sites) > 1 and handler_sites:
            # Multiple raises, multiple handlers — check if handlers differentiate
            raise_locations = set()
            for r in raise_sites:
                loc = f"{r.in_class}.{r.in_function}" if r.in_class else r.in_function
                raise_locations.add(loc)

            if len(raise_locations) > 1:
                crossing.risk_level = "medium"
                crossing.description = (
                    f"{exc_type} raised in {len(raise_locations)} different "
                    f"functions/methods — different semantic contexts likely."
                )
            else:
                crossing.risk_level = "low"
                crossing.description = "Multiple raises in same function — likely same semantics."

        # Scope-aware refinement: handlers with zero direct raises in their
        # try body catch exceptions from called functions — higher crossing risk.
        # This can upgrade a "low" to "medium" or annotate existing descriptions.
        if handler_sites and len(raise_sites) > 1:
            scope_mismatched = [
                h for h in handler_sites
                if h.try_scope_id is not None and h.direct_raises_in_scope == 0
            ]
            if scope_mismatched:
                if crossing.risk_level == "low":
                    crossing.risk_level = "medium"
                crossing.description += (
                    f" [{len(scope_mismatched)} handler(s) have no direct "
                    f"{exc_type} raises in their try body — catching from "
                    f"called functions only.]"
                )

        # Call-graph-aware refinement: check which raise sites are reachable
        # from each handler's function through the call chain.
        if call_graph and handler_sites and len(raise_sites) > 1:
            for h in handler_sites:
                handler_func = f"{h.in_class}.{h.in_function}" if h.in_class else h.in_function
                reachable_funcs = call_graph.reachable(handler_func)

                reachable_raises = []
                unreachable_raises = []
                for r in raise_sites:
                    raise_func = f"{r.in_class}.{r.in_function}" if r.in_class else r.in_function
                    if raise_func == handler_func or raise_func in reachable_funcs:
                        reachable_raises.append(r)
                    else:
                        unreachable_raises.append(r)

                if reachable_raises and len(reachable_raises) > 1:
                    raise_funcs = set()
                    for r in reachable_raises:
                        rf = f"{r.in_class}.{r.in_function}" if r.in_class else r.in_function
                        raise_funcs.add(rf)
                    if len(raise_funcs) > 1:
                        if crossing.risk_level != "high":
                            crossing.risk_level = "high"
                        crossing.description += (
                            f" [Call graph: handler in {handler_func} can reach "
                            f"{len(reachable_raises)} raise sites across "
                            f"{len(raise_funcs)} functions via call chain.]"
                        )

        # Message-differentiation heuristic: if all explicit raise sites
        # pass distinct string literals to the exception constructor, the
        # crossing is likely intentional — the messages carry the semantic
        # differentiation that the exception type alone cannot.
        # Only applied when there are multiple handlers (suggesting the
        # codebase actively catches this type in multiple places and may
        # rely on message content). With a single handler, distinct messages
        # don't prevent meaning collapse in the handler's code path.
        explicit_raises = [r for r in raise_sites if not r.implicit]
        if (len(explicit_raises) > 1 and len(handler_sites) > 1
                and crossing.risk_level in ("medium", "high")):
            messages = [r.message_arg for r in explicit_raises]
            if all(m is not None for m in messages) and len(set(messages)) == len(messages):
                # All raises have distinct string literal messages
                if crossing.risk_level == "high":
                    crossing.risk_level = "medium"
                elif crossing.risk_level == "medium":
                    crossing.risk_level = "low"
                crossing.description += (
                    f" [Downgraded: all {len(explicit_raises)} raise sites "
                    f"pass distinct string messages — likely intentional differentiation.]"
                )

        crossings.append(crossing)

    return crossings


def _resolve_module_to_file(module: str, root: str) -> str | None:
    """Try to resolve a dotted module path to a file path within root.

    Handles:
      - "package.module" -> root/package/module.py
      - "package" -> root/package/__init__.py
      - relative modules within the project
    """
    parts = module.split(".")
    # Try as a module file first
    candidate = os.path.join(root, *parts) + ".py"
    if os.path.isfile(candidate):
        return candidate
    # Try as a package (__init__.py)
    candidate = os.path.join(root, *parts, "__init__.py")
    if os.path.isfile(candidate):
        return candidate
    return None



def scan_directory(root: str, exclude_dirs: set[str] | None = None, detect_implicit: bool = False) -> SemanticScanReport:
    """Scan a directory tree for semantic crossings.

    Cross-file analysis: resolves `from X import Y` to connect call graphs
    across file boundaries. When file A imports function f from file B,
    and A calls f inside a try block, the raises in B's f are connected
    to A's handler through the cross-file call graph.
    """
    if exclude_dirs is None:
        exclude_dirs = {
            ".git", "__pycache__", ".venv", "venv", "node_modules",
            ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
            ".eggs", "*.egg-info",
        }

    report = SemanticScanReport(root=root)
    all_call_edges: list[CallEdge] = []
    all_exception_parents: dict[str, str] = {}
    all_imports: list[ImportRecord] = []
    # Track which top-level names each file defines (functions, classes)
    file_definitions: dict[str, set[str]] = {}
    # Track per-file raises keyed by function name for cross-file linking
    file_raises: dict[str, dict[str, list[ExceptionRaise]]] = {}

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs
                       and not d.endswith(".egg-info")]

        for filename in filenames:
            if not filename.endswith(".py"):
                continue

            filepath = os.path.join(dirpath, filename)
            report.files_scanned += 1

            try:
                raises, handlers, call_edges, exc_parents, imports = scan_file(filepath, detect_implicit=detect_implicit)
                report.raises.extend(raises)
                report.handlers.extend(handlers)
                all_call_edges.extend(call_edges)
                all_exception_parents.update(exc_parents)
                all_imports.extend(imports)

                # Extract top-level definitions from this file
                defs: set[str] = set()
                func_raises: dict[str, list[ExceptionRaise]] = {}
                for r in raises:
                    if r.file == filepath:
                        defs.add(r.in_function)
                        func_raises.setdefault(r.in_function, []).append(r)
                for h in handlers:
                    if h.file == filepath:
                        defs.add(h.in_function)
                for edge in call_edges:
                    if edge.file == filepath:
                        # caller name without class prefix for matching
                        name = edge.caller.split(".")[-1] if "." in edge.caller else edge.caller
                        defs.add(name)
                file_definitions[filepath] = defs
                file_raises[filepath] = func_raises
            except Exception:
                report.parse_errors += 1

    # Build cross-file call edges from import resolution
    # Collect new edges separately to avoid modifying list while iterating
    cross_file_edges: list[CallEdge] = []
    for imp in all_imports:
        if imp.name:
            # `from X import Y` — resolve Y to a definition in X's file
            source_file = _resolve_module_to_file(imp.module, root)
            if source_file and source_file in file_definitions:
                if imp.name in file_definitions[source_file]:
                    for edge in all_call_edges:
                        if edge.file == imp.file and edge.callee == imp.alias:
                            cross_file_edges.append(CallEdge(
                                caller=edge.caller,
                                callee=imp.name,
                                file=f"{imp.file}->{source_file}",
                                line=edge.line,
                            ))
        else:
            # `import X` — look for `X.func()` calls (dotted attribute access)
            source_file = _resolve_module_to_file(imp.module, root)
            if source_file and source_file in file_definitions:
                prefix = imp.alias + "."
                for edge in all_call_edges:
                    if (edge.file == imp.file
                            and edge.callee.startswith(prefix)):
                        func_name = edge.callee[len(prefix):]
                        if func_name in file_definitions[source_file]:
                            cross_file_edges.append(CallEdge(
                                caller=edge.caller,
                                callee=func_name,
                                file=f"{imp.file}->{source_file}",
                                line=edge.line,
                            ))
    all_call_edges.extend(cross_file_edges)

    call_graph = CallGraph(all_call_edges) if all_call_edges else None
    report.crossings = analyze_crossings(
        report.raises, report.handlers, call_graph,
        exception_parents=all_exception_parents or None,
    )
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="crossing",
        description="Detect semantic boundary crossings in Python codebases — "
                    "where the same exception type carries different meanings.",
    )
    parser.add_argument("path", help="Directory to scan")
    parser.add_argument("--implicit", action="store_true",
                        help="Also detect implicit raises (dict access → KeyError, etc.)")
    parser.add_argument("--format", choices=["text", "json", "markdown", "report"],
                        default="text", help="Output format (default: text). 'report' generates full audit report.")
    parser.add_argument("--min-risk", choices=["low", "medium", "elevated", "high"],
                        default="low", help="Minimum risk level to report (default: low)")
    parser.add_argument("--exclude", action="append", default=[],
                        help="Glob patterns to exclude (can repeat)")
    parser.add_argument("--ci", action="store_true",
                        help="CI mode: exit code 1 if any elevated/high risk crossings found")
    parser.add_argument("--name", default="",
                        help="Project name (for --format report)")
    parser.add_argument("--repo", default="",
                        help="Repository identifier e.g. org/project (for --format report)")

    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print(f"Error: {args.path} is not a directory", file=sys.stderr)
        sys.exit(2)

    exclude_dirs = None
    if args.exclude:
        # Start with defaults and add user patterns
        exclude_dirs = {
            ".git", "__pycache__", ".venv", "venv", "node_modules",
            ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
            ".eggs",
        }
        exclude_dirs.update(args.exclude)

    report = scan_directory(args.path, exclude_dirs=exclude_dirs,
                            detect_implicit=args.implicit)

    # Apply min-risk filter
    if args.min_risk != "low":
        report = report.filter(args.min_risk)

    # Output
    if args.format == "json":
        print(report.to_json())
    elif args.format == "markdown":
        print(report.to_markdown())
    elif args.format == "report":
        import json as json_mod
        from report import generate_report
        scan_data = json_mod.loads(report.to_json())
        project_name = args.name or os.path.basename(os.path.normpath(args.path))
        print(generate_report(scan_data, project_name=project_name, repo=args.repo))
    else:
        report.print()

    # CI exit code
    if args.ci and report.risky_crossings:
        sys.exit(1)


if __name__ == "__main__":
    main()
