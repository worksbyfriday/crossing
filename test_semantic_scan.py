"""Tests for semantic_scan.py — semantic boundary crossing detection."""

import os
import tempfile
import textwrap

from semantic_scan import (
    CallEdge,
    CallGraph,
    ExceptionRaise,
    ExceptionHandler,
    SemanticCrossing,
    SemanticVisitor,
    analyze_crossings,
    scan_file,
    scan_directory,
)


def _scan_code(code: str):
    """Helper: write code to temp file, scan, return (raises, handlers)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(textwrap.dedent(code))
        f.flush()
        try:
            raises, handlers, _call_edges = scan_file(f.name)
            return raises, handlers
        finally:
            os.unlink(f.name)


def _scan_code_full(code: str):
    """Helper: write code to temp file, scan, return (raises, handlers, call_edges)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(textwrap.dedent(code))
        f.flush()
        try:
            return scan_file(f.name)
        finally:
            os.unlink(f.name)


def test_explicit_raise_detected():
    """Explicit raise statements should be detected."""
    raises, handlers = _scan_code("""
        def foo():
            raise ValueError("bad")
    """)
    assert len(raises) == 1
    assert raises[0].exception_type == "ValueError"
    assert raises[0].in_function == "foo"


def test_handler_detected():
    """except clauses should be detected."""
    raises, handlers = _scan_code("""
        def foo():
            try:
                pass
            except KeyError:
                return None
    """)
    assert len(handlers) == 1
    assert handlers[0].exception_type == "KeyError"
    assert handlers[0].returns_value is True


def test_re_raise_detected():
    """Handlers that re-raise should be flagged."""
    raises, handlers = _scan_code("""
        def foo():
            try:
                pass
            except ValueError:
                raise
    """)
    assert len(handlers) == 1
    assert handlers[0].re_raises is True


def test_class_scope():
    """Raises inside a class method should track class name."""
    raises, handlers = _scan_code("""
        class MyDict:
            def __getitem__(self, key):
                raise KeyError(key)
    """)
    assert len(raises) == 1
    assert raises[0].in_class == "MyDict"
    assert raises[0].in_function == "__getitem__"


def test_polymorphic_detection():
    """Multiple raise sites for same exception = polymorphic."""
    raises, handlers = _scan_code("""
        def parse_config():
            raise KeyError("missing section")

        def filter_values():
            raise KeyError("filtered to empty")

        def lookup():
            try:
                pass
            except KeyError:
                return "default"
    """)
    crossings = analyze_crossings(raises, handlers)
    key_crossings = [c for c in crossings if c.exception_type == "KeyError"]
    assert len(key_crossings) == 1
    assert key_crossings[0].is_polymorphic is True
    assert len(key_crossings[0].raise_sites) == 2
    assert len(key_crossings[0].handler_sites) == 1


def test_high_risk_many_raises_one_handler():
    """Many raises, one handler = high risk."""
    raises, handlers = _scan_code("""
        def a(): raise ValueError("type 1")
        def b(): raise ValueError("type 2")
        def c(): raise ValueError("type 3")
        def d(): raise ValueError("type 4")

        def main():
            try:
                pass
            except ValueError:
                return "error"
    """)
    crossings = analyze_crossings(raises, handlers)
    val = [c for c in crossings if c.exception_type == "ValueError"][0]
    assert val.risk_level == "high"
    assert len(val.raise_sites) == 4


def test_single_raise_is_low_risk():
    """Single raise site = low risk."""
    raises, handlers = _scan_code("""
        def foo():
            raise ValueError("only one")

        def bar():
            try:
                pass
            except ValueError:
                return None
    """)
    crossings = analyze_crossings(raises, handlers)
    val = [c for c in crossings if c.exception_type == "ValueError"][0]
    assert val.risk_level == "low"


def test_no_handlers_is_low_risk():
    """Raises without local handlers = low risk."""
    raises, handlers = _scan_code("""
        def foo(): raise KeyError("a")
        def bar(): raise KeyError("b")
    """)
    crossings = analyze_crossings(raises, handlers)
    key = [c for c in crossings if c.exception_type == "KeyError"][0]
    assert key.risk_level == "low"


def test_directory_scan():
    """scan_directory should aggregate across files."""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "a.py"), "w") as f:
            f.write("def foo(): raise ValueError('a')\n")
        with open(os.path.join(d, "b.py"), "w") as f:
            f.write("def bar(): raise ValueError('b')\n")
        with open(os.path.join(d, "c.py"), "w") as f:
            f.write(textwrap.dedent("""
                def handler():
                    try:
                        pass
                    except ValueError:
                        return None
            """))

        report = scan_directory(d)
        assert report.files_scanned == 3
        val_crossings = [c for c in report.crossings if c.exception_type == "ValueError"]
        assert len(val_crossings) == 1
        assert val_crossings[0].is_polymorphic is True


def test_bare_except():
    """Bare except should be tracked as BaseException."""
    raises, handlers = _scan_code("""
        def foo():
            try:
                pass
            except:
                return None
    """)
    assert len(handlers) == 1
    assert handlers[0].exception_type == "BaseException"


def test_handler_assigns_default():
    """Handler with assignment should flag assigns_default."""
    raises, handlers = _scan_code("""
        def foo():
            try:
                pass
            except KeyError:
                value = "default"
    """)
    assert handlers[0].assigns_default is True


def test_uniform_handler_detection():
    """All handlers doing the same thing = uniform."""
    raises = [
        ExceptionRaise("a.py", 1, "KeyError", "f1", "", "", ""),
        ExceptionRaise("a.py", 5, "KeyError", "f2", "", "", ""),
    ]
    handlers = [
        ExceptionHandler("a.py", 10, "KeyError", "h1", "", "", "", False, True, False),
        ExceptionHandler("a.py", 20, "KeyError", "h2", "", "", "", False, True, False),
    ]
    crossings = analyze_crossings(raises, handlers)
    key = [c for c in crossings if c.exception_type == "KeyError"][0]
    assert key.has_uniform_handler is True


def _scan_code_implicit(code: str):
    """Helper: scan code with implicit raise detection enabled."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(textwrap.dedent(code))
        f.flush()
        try:
            raises, handlers, _call_edges = scan_file(f.name, detect_implicit=True)
            return raises, handlers
        finally:
            os.unlink(f.name)


def test_implicit_keyerror_from_subscript():
    """Dict subscript access should produce implicit KeyError."""
    raises, handlers = _scan_code_implicit("""
        def lookup(d, key):
            return d[key]
    """)
    implicit = [r for r in raises if r.implicit]
    assert len(implicit) >= 1
    assert any(r.exception_type == "KeyError" for r in implicit)


def test_implicit_disabled_by_default():
    """Without detect_implicit, subscript access is not tracked."""
    raises, handlers = _scan_code("""
        def lookup(d, key):
            return d[key]
    """)
    assert len(raises) == 0


def test_implicit_next_without_default():
    """next(it) without default should produce implicit StopIteration."""
    raises, handlers = _scan_code_implicit("""
        def consume(it):
            return next(it)
    """)
    implicit = [r for r in raises if r.implicit]
    assert any(r.exception_type == "StopIteration" for r in implicit)


def test_implicit_next_with_default_not_tracked():
    """next(it, default) should NOT produce implicit StopIteration."""
    raises, handlers = _scan_code_implicit("""
        def consume(it):
            return next(it, None)
    """)
    implicit = [r for r in raises if r.implicit and r.exception_type == "StopIteration"]
    assert len(implicit) == 0


def test_implicit_mixed_with_explicit():
    """Implicit + explicit raises for same exception = polymorphic crossing."""
    raises, handlers = _scan_code_implicit("""
        def explicit_raise():
            raise KeyError("missing config")

        def dict_access(d, key):
            return d[key]

        def handler():
            try:
                pass
            except KeyError:
                return "default"
    """)
    crossings = analyze_crossings(raises, handlers)
    key_crossings = [c for c in crossings if c.exception_type == "KeyError"]
    assert len(key_crossings) == 1
    crossing = key_crossings[0]
    assert crossing.is_polymorphic is True
    explicit = [r for r in crossing.raise_sites if not r.implicit]
    implicit = [r for r in crossing.raise_sites if r.implicit]
    assert len(explicit) >= 1
    assert len(implicit) >= 1


def test_subscript_store_not_tracked():
    """d[key] = value (store context) should not produce implicit raises."""
    raises, handlers = _scan_code_implicit("""
        def store(d, key, val):
            d[key] = val
    """)
    implicit = [r for r in raises if r.implicit]
    assert len(implicit) == 0


def test_implicit_int_conversion():
    """int(x) should produce implicit ValueError."""
    raises, handlers = _scan_code_implicit("""
        def parse_port(s):
            return int(s)
    """)
    implicit = [r for r in raises if r.implicit and r.exception_type == "ValueError"]
    assert len(implicit) == 1


def test_implicit_float_conversion():
    """float(x) should produce implicit ValueError."""
    raises, handlers = _scan_code_implicit("""
        def parse_price(s):
            return float(s)
    """)
    implicit = [r for r in raises if r.implicit and r.exception_type == "ValueError"]
    assert len(implicit) == 1


def test_implicit_int_no_args_not_tracked():
    """int() without args (returns 0) should not produce implicit ValueError."""
    raises, handlers = _scan_code_implicit("""
        def zero():
            return int()
    """)
    implicit = [r for r in raises if r.implicit and r.exception_type == "ValueError"]
    assert len(implicit) == 0


def test_implicit_index_method():
    """.index() should produce implicit ValueError."""
    raises, handlers = _scan_code_implicit("""
        def find_position(lst, item):
            return lst.index(item)
    """)
    implicit = [r for r in raises if r.implicit and r.exception_type == "ValueError"]
    assert len(implicit) == 1


def test_mixed_crossing_is_high_risk():
    """Explicit + implicit raises with a handler = high risk (tox #3809 pattern)."""
    raises, handlers = _scan_code_implicit("""
        def explicit_raise():
            raise ValueError("bad config")

        def parse_number(s):
            return int(s)

        def handler():
            try:
                pass
            except ValueError:
                return "default"
    """)
    crossings = analyze_crossings(raises, handlers)
    val = [c for c in crossings if c.exception_type == "ValueError"][0]
    assert val.risk_level == "high"
    explicit = [r for r in val.raise_sites if not r.implicit]
    implicit = [r for r in val.raise_sites if r.implicit]
    assert len(explicit) >= 1
    assert len(implicit) >= 1
    assert "explicit" in val.description and "implicit" in val.description


# === v0.4: scope-aware analysis ===


def test_try_scope_id_assigned():
    """Raises inside try blocks should get a try_scope_id."""
    raises, handlers = _scan_code("""
        def foo():
            try:
                raise KeyError("x")
            except KeyError:
                pass
    """)
    assert len(raises) == 1
    assert raises[0].try_scope_id is not None


def test_raise_outside_try_has_no_scope():
    """Raises outside any try block should have try_scope_id=None."""
    raises, handlers = _scan_code("""
        def foo():
            raise KeyError("x")
    """)
    assert len(raises) == 1
    assert raises[0].try_scope_id is None


def test_handler_counts_direct_raises():
    """Handler should count explicit raises of its type in the try body."""
    raises, handlers = _scan_code("""
        def foo():
            try:
                raise KeyError("intentional")
            except KeyError:
                pass
    """)
    assert len(handlers) == 1
    assert handlers[0].direct_raises_in_scope == 1


def test_handler_zero_direct_raises():
    """Handler catching from called functions should have direct_raises=0."""
    raises, handlers = _scan_code("""
        def helper():
            raise KeyError("from helper")

        def foo():
            try:
                helper()
            except KeyError:
                pass
    """)
    # The raise is in helper(), not in the try body
    handler = [h for h in handlers if h.exception_type == "KeyError"][0]
    assert handler.direct_raises_in_scope == 0


def test_scope_mismatch_upgrades_risk():
    """Handlers with no direct raises should upgrade risk level."""
    raises, handlers = _scan_code("""
        def a():
            raise ValueError("a")

        def b():
            raise ValueError("b")

        def c():
            try:
                a()
                b()
            except ValueError:
                pass
    """)
    crossings = analyze_crossings(raises, handlers)
    ve = [c for c in crossings if c.exception_type == "ValueError"][0]
    assert ve.risk_level in ("medium", "high")
    assert "called functions" in ve.description


def test_implicit_getattr_detected():
    """getattr() without default should be detected as implicit AttributeError."""
    raises, handlers = _scan_code_implicit("""
        def foo():
            obj = object()
            getattr(obj, "bar")
    """)
    attr_raises = [r for r in raises if r.exception_type == "AttributeError"]
    assert len(attr_raises) == 1
    assert attr_raises[0].implicit is True


def test_getattr_with_default_not_tracked():
    """getattr() with default should NOT be detected (it won't raise)."""
    raises, handlers = _scan_code_implicit("""
        def foo():
            obj = object()
            getattr(obj, "bar", None)
    """)
    attr_raises = [r for r in raises if r.exception_type == "AttributeError"]
    assert len(attr_raises) == 0


def test_nested_try_scopes():
    """Nested try blocks should get different scope IDs."""
    raises, handlers = _scan_code("""
        def foo():
            try:
                try:
                    raise KeyError("inner")
                except KeyError:
                    pass
                raise ValueError("outer")
            except ValueError:
                pass
    """)
    ke = [r for r in raises if r.exception_type == "KeyError"][0]
    ve = [r for r in raises if r.exception_type == "ValueError"][0]
    # They should have different try scope IDs
    assert ke.try_scope_id != ve.try_scope_id
    assert ke.try_scope_id is not None
    assert ve.try_scope_id is not None


# --- Call Graph Tests ---


def test_call_graph_basic_reachability():
    """CallGraph should track direct and transitive callees."""
    edges = [
        CallEdge("a", "b", "test.py", 1),
        CallEdge("b", "c", "test.py", 2),
        CallEdge("c", "d", "test.py", 3),
    ]
    g = CallGraph(edges)
    assert g.can_reach("a", "b")
    assert g.can_reach("a", "c")
    assert g.can_reach("a", "d")
    assert not g.can_reach("d", "a")
    assert g.can_reach("b", "c")
    assert not g.can_reach("c", "a")


def test_call_graph_no_self_reach():
    """A function should not be listed as reachable from itself."""
    edges = [CallEdge("a", "b", "test.py", 1)]
    g = CallGraph(edges)
    assert "a" not in g.reachable("a")


def test_call_graph_cycle_handling():
    """Call graph should handle cycles without infinite loop."""
    edges = [
        CallEdge("a", "b", "test.py", 1),
        CallEdge("b", "a", "test.py", 2),
    ]
    g = CallGraph(edges)
    assert g.can_reach("a", "b")
    assert g.can_reach("b", "a")


def test_call_edges_extracted():
    """scan_file should extract call edges from function bodies."""
    raises, handlers, call_edges = _scan_code_full("""
        def helper():
            raise ValueError("oops")

        def main():
            try:
                helper()
            except ValueError:
                pass
    """)
    assert len(call_edges) > 0
    # Find the edge from main -> helper
    main_calls = [e for e in call_edges if e.caller == "main"]
    assert any(e.callee == "helper" for e in main_calls)


def test_call_graph_cross_function_raise():
    """Call graph should connect raises in called functions to handlers."""
    raises, handlers, call_edges = _scan_code_full("""
        def raiser_a():
            raise KeyError("a")

        def raiser_b():
            raise KeyError("b")

        def caller():
            try:
                raiser_a()
                raiser_b()
            except KeyError:
                pass
    """)
    cg = CallGraph(call_edges)
    crossings = analyze_crossings(raises, handlers, cg)
    ke = [c for c in crossings if c.exception_type == "KeyError"][0]
    # Should have call graph annotation since handler can reach both raise sites
    assert "Call graph" in ke.description
    assert ke.risk_level == "high"


def test_call_graph_stats():
    """CallGraph should report node and edge counts."""
    edges = [
        CallEdge("a", "b", "test.py", 1),
        CallEdge("a", "c", "test.py", 2),
        CallEdge("b", "d", "test.py", 3),
    ]
    g = CallGraph(edges)
    assert g.node_count == 4  # a, b, c, d
    assert g.edge_count == 3
