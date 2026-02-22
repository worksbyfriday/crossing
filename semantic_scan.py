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
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


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

        for crossing in self.risky_crossings:
            print(f"\n--- {crossing} ---")
            print(f"  {crossing.description}")
            print(f"  Raise sites:")
            for r in crossing.raise_sites[:5]:
                print(f"    {r}")
            if len(crossing.raise_sites) > 5:
                print(f"    ... and {len(crossing.raise_sites) - 5} more")
            print(f"  Handlers:")
            for h in crossing.handler_sites[:5]:
                print(f"    {h}")
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
        self._scope_stack.append((node.name, ""))
        self.generic_visit(node)
        self._scope_stack.pop()

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
            self.raises.append(ExceptionRaise(
                file=self.filename,
                line=node.lineno,
                exception_type=exc_type,
                in_function=self._current_function,
                in_class=self._current_class,
                source_line=source,
                context=self._infer_context(node),
                try_scope_id=self._current_try_scope,
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

    def _infer_context(self, node: ast.Raise) -> str:
        """Try to describe what's happening around this raise."""
        # Look at the enclosing if/for/try to understand context
        return f"in {self._current_function}"

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


def scan_file(filepath: str, detect_implicit: bool = False) -> tuple[list[ExceptionRaise], list[ExceptionHandler], list[CallEdge]]:
    """Scan a single file for exception raise/handle patterns and call edges."""
    try:
        with open(filepath) as f:
            source = f.read()
        source_lines = source.split("\n")
        tree = ast.parse(source, filename=filepath)
        visitor = SemanticVisitor(filepath, source_lines, detect_implicit=detect_implicit)
        visitor.visit(tree)
        return visitor.raises, visitor.handlers, visitor.call_edges
    except SyntaxError:
        return [], [], []
    except Exception:
        return [], [], []


def analyze_crossings(
    raises: list[ExceptionRaise],
    handlers: list[ExceptionHandler],
    call_graph: CallGraph | None = None,
) -> list[SemanticCrossing]:
    """Analyze raises and handlers to find semantic crossings.

    A semantic crossing exists when:
    1. The same exception type is raised at multiple sites (polymorphic)
    2. The raise sites have different semantic contexts
    3. Handler(s) treat them uniformly (potential meaning collapse)

    If a call_graph is provided, uses reachability analysis to determine
    whether a raise site can actually reach a handler through the call chain.
    """
    # Group raises by exception type
    raise_groups: dict[str, list[ExceptionRaise]] = {}
    for r in raises:
        raise_groups.setdefault(r.exception_type, []).append(r)

    # Group handlers by exception type
    handler_groups: dict[str, list[ExceptionHandler]] = {}
    for h in handlers:
        handler_groups.setdefault(h.exception_type, []).append(h)

    crossings = []
    for exc_type, raise_sites in raise_groups.items():
        handler_sites = handler_groups.get(exc_type, [])

        crossing = SemanticCrossing(
            exception_type=exc_type,
            raise_sites=raise_sites,
            handler_sites=handler_sites,
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

        crossings.append(crossing)

    return crossings


def scan_directory(root: str, exclude_dirs: set[str] | None = None, detect_implicit: bool = False) -> SemanticScanReport:
    """Scan a directory tree for semantic crossings."""
    if exclude_dirs is None:
        exclude_dirs = {
            ".git", "__pycache__", ".venv", "venv", "node_modules",
            ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
            ".eggs", "*.egg-info",
        }

    report = SemanticScanReport(root=root)
    all_call_edges: list[CallEdge] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs
                       and not d.endswith(".egg-info")]

        for filename in filenames:
            if not filename.endswith(".py"):
                continue

            filepath = os.path.join(dirpath, filename)
            report.files_scanned += 1

            try:
                raises, handlers, call_edges = scan_file(filepath, detect_implicit=detect_implicit)
                report.raises.extend(raises)
                report.handlers.extend(handlers)
                all_call_edges.extend(call_edges)
            except Exception:
                report.parse_errors += 1

    call_graph = CallGraph(all_call_edges) if all_call_edges else None
    report.crossings = analyze_crossings(report.raises, report.handlers, call_graph)
    return report


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 semantic_scan.py [--implicit] /path/to/codebase")
        sys.exit(1)

    detect_implicit = "--implicit" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if not args:
        print("Usage: python3 semantic_scan.py [--implicit] /path/to/codebase")
        sys.exit(1)

    root = args[0]
    if not os.path.isdir(root):
        print(f"Error: {root} is not a directory")
        sys.exit(1)

    report = scan_directory(root, detect_implicit=detect_implicit)
    report.print()


if __name__ == "__main__":
    main()
