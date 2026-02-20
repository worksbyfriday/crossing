"""
crossing.scan — find boundary crossings in existing codebases.

Scans Python source files for encode/decode pairs:
- json.dumps / json.loads
- yaml.dump / yaml.safe_load
- pickle.dumps / pickle.loads
- toml.dumps / tomllib.loads
- csv.writer / csv.reader
- base64.b64encode / b64decode
- urllib.parse.urlencode / parse_qs
- .encode('utf-8') / .decode('utf-8')

Outputs a report of found crossings with file locations,
and optionally generates crossing tests.

Usage:
    python3 scan.py /path/to/codebase
    python3 scan.py /path/to/codebase --generate-tests
"""

from __future__ import annotations

import ast
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Known encode/decode pairs — (module, encode_func, decode_func, crossing_name)
KNOWN_PAIRS = [
    # JSON
    ("json", "dumps", "loads", "JSON"),
    ("json", "dump", "load", "JSON (file)"),
    # YAML
    ("yaml", "dump", "safe_load", "YAML"),
    ("yaml", "dump", "load", "YAML (unsafe)"),
    ("yaml", "dump_all", "safe_load_all", "YAML (multi-doc)"),
    # Pickle
    ("pickle", "dumps", "loads", "pickle"),
    ("pickle", "dump", "load", "pickle (file)"),
    # TOML
    ("tomli_w", "dumps", None, "TOML (write)"),  # decode is tomllib.loads
    ("tomllib", None, "loads", "TOML (read)"),
    ("toml", "dumps", "loads", "TOML (toml lib)"),
    # Base64
    ("base64", "b64encode", "b64decode", "base64"),
    ("base64", "urlsafe_b64encode", "urlsafe_b64decode", "base64 (urlsafe)"),
    # URL encoding
    ("urllib.parse", "urlencode", "parse_qs", "URL query string"),
    ("urllib.parse", "quote", "unquote", "URL percent-encoding"),
    # CSV (harder to match — writer/reader classes)
    ("csv", "writer", "reader", "CSV"),
    ("csv", "DictWriter", "DictReader", "CSV (dict)"),
    # struct (binary packing)
    ("struct", "pack", "unpack", "struct (binary)"),
    # zlib/gzip compression
    ("zlib", "compress", "decompress", "zlib"),
    ("gzip", "compress", "decompress", "gzip"),
]


@dataclass
class BoundaryCall:
    """A single encode or decode call found in source."""
    file: str
    line: int
    module: str
    function: str
    is_encode: bool
    source_line: str

    def __str__(self):
        direction = "encode" if self.is_encode else "decode"
        return f"{self.file}:{self.line} — {self.module}.{self.function} ({direction})"


@dataclass
class FoundCrossing:
    """A matched pair of encode/decode calls in the same file or codebase."""
    name: str
    module: str
    encode_func: str
    decode_func: str
    encode_calls: list[BoundaryCall] = field(default_factory=list)
    decode_calls: list[BoundaryCall] = field(default_factory=list)

    @property
    def is_paired(self) -> bool:
        return bool(self.encode_calls) and bool(self.decode_calls)

    @property
    def encode_only(self) -> bool:
        return bool(self.encode_calls) and not self.decode_calls

    @property
    def decode_only(self) -> bool:
        return bool(self.decode_calls) and not self.encode_calls

    def __str__(self):
        e = len(self.encode_calls)
        d = len(self.decode_calls)
        status = "paired" if self.is_paired else ("encode only" if self.encode_only else "decode only")
        return f"{self.name}: {e} encode, {d} decode ({status})"


@dataclass
class ScanReport:
    """Results from scanning a codebase for boundary crossings."""
    root: str
    files_scanned: int = 0
    parse_errors: int = 0
    crossings: list[FoundCrossing] = field(default_factory=list)
    all_calls: list[BoundaryCall] = field(default_factory=list)

    @property
    def paired_crossings(self) -> list[FoundCrossing]:
        return [c for c in self.crossings if c.is_paired]

    @property
    def encode_only(self) -> list[FoundCrossing]:
        return [c for c in self.crossings if c.encode_only]

    @property
    def decode_only(self) -> list[FoundCrossing]:
        return [c for c in self.crossings if c.decode_only]

    def print(self):
        print(f"\n{'='*60}")
        print(f"Crossing Scan: {self.root}")
        print(f"{'='*60}")
        print(f"Files scanned:     {self.files_scanned}")
        print(f"Parse errors:      {self.parse_errors}")
        print(f"Boundary calls:    {len(self.all_calls)}")
        print(f"Crossing types:    {len(self.crossings)}")
        print(f"  Paired (encode+decode): {len(self.paired_crossings)}")
        print(f"  Encode-only:            {len(self.encode_only)}")
        print(f"  Decode-only:            {len(self.decode_only)}")

        if self.paired_crossings:
            print(f"\nPaired crossings (data enters AND exits):")
            for c in self.paired_crossings:
                print(f"  {c}")
                for call in (c.encode_calls + c.decode_calls)[:6]:
                    print(f"    {call}")
                remaining = len(c.encode_calls) + len(c.decode_calls) - 6
                if remaining > 0:
                    print(f"    ... and {remaining} more")

        if self.encode_only:
            print(f"\nEncode-only (data goes in, never comes back — or decoded elsewhere):")
            for c in self.encode_only:
                print(f"  {c}")
                for call in c.encode_calls[:3]:
                    print(f"    {call}")

        if self.decode_only:
            print(f"\nDecode-only (data comes in, never serialized here):")
            for c in self.decode_only:
                print(f"  {c}")
                for call in c.decode_calls[:3]:
                    print(f"    {call}")

        print(f"{'='*60}\n")


class BoundaryVisitor(ast.NodeVisitor):
    """AST visitor that finds encode/decode calls."""

    def __init__(self, filename: str, source_lines: list[str]):
        self.filename = filename
        self.source_lines = source_lines
        self.calls: list[BoundaryCall] = []
        self.imports: dict[str, str] = {}  # alias → module

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            name = alias.asname or alias.name
            self.imports[name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            for alias in node.names:
                name = alias.asname or alias.name
                self.imports[name] = f"{node.module}.{alias.name}"
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func_name = self._get_func_name(node.func)
        if func_name:
            self._check_boundary_call(func_name, node)
        self.generic_visit(node)

    def _get_func_name(self, node: ast.expr) -> Optional[str]:
        """Extract the full dotted name from a call target."""
        if isinstance(node, ast.Name):
            resolved = self.imports.get(node.id, node.id)
            return resolved
        elif isinstance(node, ast.Attribute):
            value = self._get_func_name(node.value)
            if value:
                return f"{value}.{node.attr}"
        return None

    def _check_boundary_call(self, func_name: str, node: ast.Call):
        """Check if this call matches a known boundary function."""
        for module, encode, decode, name in KNOWN_PAIRS:
            if encode and self._matches(func_name, module, encode):
                line = self.source_lines[node.lineno - 1].strip() if node.lineno <= len(self.source_lines) else ""
                self.calls.append(BoundaryCall(
                    file=self.filename,
                    line=node.lineno,
                    module=module,
                    function=encode,
                    is_encode=True,
                    source_line=line,
                ))
            if decode and self._matches(func_name, module, decode):
                line = self.source_lines[node.lineno - 1].strip() if node.lineno <= len(self.source_lines) else ""
                self.calls.append(BoundaryCall(
                    file=self.filename,
                    line=node.lineno,
                    module=module,
                    function=decode,
                    is_encode=False,
                    source_line=line,
                ))

    def _matches(self, func_name: str, module: str, function: str) -> bool:
        """Check if func_name matches module.function, accounting for imports."""
        # Direct match: json.dumps
        if func_name == f"{module}.{function}":
            return True
        # Short match: dumps (if imported directly)
        if func_name == f"{module}.{function}":
            return True
        # Import-resolved match
        parts = func_name.rsplit(".", 1)
        if len(parts) == 2:
            resolved_module = self.imports.get(parts[0], parts[0])
            if resolved_module == module and parts[1] == function:
                return True
        elif len(parts) == 1:
            resolved = self.imports.get(parts[0], parts[0])
            if resolved == f"{module}.{function}":
                return True
        return False


def scan_file(filepath: str) -> list[BoundaryCall]:
    """Scan a single Python file for boundary calls."""
    try:
        with open(filepath) as f:
            source = f.read()
        source_lines = source.split("\n")
        tree = ast.parse(source, filename=filepath)
        visitor = BoundaryVisitor(filepath, source_lines)
        visitor.visit(tree)
        return visitor.calls
    except SyntaxError:
        return []
    except Exception:
        return []


def scan_directory(root: str, exclude_dirs: set[str] | None = None) -> ScanReport:
    """Scan a directory tree for boundary crossings."""
    if exclude_dirs is None:
        exclude_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules",
                        ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
                        ".eggs", "*.egg-info"}

    report = ScanReport(root=root)
    all_calls: list[BoundaryCall] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip excluded directories
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs
                       and not d.endswith(".egg-info")]

        for filename in filenames:
            if not filename.endswith(".py"):
                continue

            filepath = os.path.join(dirpath, filename)
            report.files_scanned += 1

            try:
                calls = scan_file(filepath)
                all_calls.extend(calls)
            except Exception:
                report.parse_errors += 1

    report.all_calls = all_calls

    # Group calls into crossings
    crossing_map: dict[str, FoundCrossing] = {}
    for module, encode, decode, name in KNOWN_PAIRS:
        key = f"{module}:{name}"
        if key not in crossing_map:
            crossing_map[key] = FoundCrossing(
                name=name,
                module=module,
                encode_func=encode or "",
                decode_func=decode or "",
            )

    for call in all_calls:
        for module, encode, decode, name in KNOWN_PAIRS:
            key = f"{module}:{name}"
            if call.module == module:
                if call.is_encode and call.function == encode:
                    crossing_map[key].encode_calls.append(call)
                elif not call.is_encode and call.function == decode:
                    crossing_map[key].decode_calls.append(call)

    # Only include crossings that have at least one call
    report.crossings = [c for c in crossing_map.values()
                        if c.encode_calls or c.decode_calls]

    return report


def generate_test_snippet(crossing: FoundCrossing) -> str:
    """Generate a crossing test snippet for a found crossing."""
    lines = []
    lines.append(f"# Auto-generated crossing test for {crossing.name}")
    lines.append(f"# Found {len(crossing.encode_calls)} encode and {len(crossing.decode_calls)} decode calls")
    lines.append(f"from crossing import Crossing, cross")
    lines.append("")

    if crossing.module == "json":
        lines.append(f"import json")
        lines.append(f"c = Crossing(")
        lines.append(f"    encode=lambda d: json.dumps(d, default=str),")
        lines.append(f"    decode=lambda s: json.loads(s),")
        lines.append(f"    name='{crossing.name}',")
        lines.append(f")")
    elif crossing.module == "yaml":
        lines.append(f"import yaml")
        lines.append(f"c = Crossing(")
        lines.append(f"    encode=lambda d: yaml.dump(d),")
        lines.append(f"    decode=lambda s: yaml.safe_load(s),")
        lines.append(f"    name='{crossing.name}',")
        lines.append(f")")
    elif crossing.module == "pickle":
        lines.append(f"import pickle")
        lines.append(f"c = Crossing(")
        lines.append(f"    encode=lambda d: pickle.dumps(d),")
        lines.append(f"    decode=lambda s: pickle.loads(s),")
        lines.append(f"    name='{crossing.name}',")
        lines.append(f")")
    else:
        lines.append(f"# TODO: implement crossing for {crossing.module}")
        lines.append(f"# Encode locations:")
        for call in crossing.encode_calls[:5]:
            lines.append(f"#   {call}")
        lines.append(f"# Decode locations:")
        for call in crossing.decode_calls[:5]:
            lines.append(f"#   {call}")
        return "\n".join(lines)

    lines.append("")
    lines.append(f"report = cross(c, samples=500, seed=42)")
    lines.append(f"report.print()")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scan.py <directory> [--generate-tests]")
        print()
        print("Scans Python files for boundary crossings (encode/decode pairs)")
        print("and reports what it finds.")
        sys.exit(1)

    target = sys.argv[1]
    generate = "--generate-tests" in sys.argv

    if os.path.isfile(target):
        calls = scan_file(target)
        print(f"\nBoundary calls in {target}:")
        for call in calls:
            print(f"  {call}")
        if not calls:
            print("  (none found)")
    else:
        report = scan_directory(target)
        report.print()

        if generate and report.paired_crossings:
            print("\n--- Generated Test Snippets ---\n")
            for c in report.paired_crossings:
                print(generate_test_snippet(c))
                print()
