"""Report Python source-file sizes for maintainability refactors.

This diagnostic is intentionally lightweight and dependency-free.  It is meant
to make large-file refactors measurable before they become strict CI gates.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_ROOTS = ("vmec_jax", "examples/optimization", "tests")
DEFAULT_ROOT_HELPER_PREFIXES = ("solve_", "driver_", "free_boundary_", "wout_")


@dataclass(frozen=True)
class SourceFileStat:
    """Line-count record for one Python source file."""

    path: Path
    lines: int


@dataclass(frozen=True)
class FunctionStat:
    """Line-count record for one Python function or method."""

    path: Path
    qualified_name: str
    lines: int


@dataclass(frozen=True)
class RootNamespaceStat:
    """Root-package namespace metrics for refactor maintainability gates."""

    root: Path
    python_files: int
    helper_prefix_files: tuple[Path, ...]


@dataclass(frozen=True)
class PublicDocstringStat:
    """Missing-docstring record for one public package symbol."""

    path: Path
    qualified_name: str
    lineno: int
    kind: str


def count_source_lines(path: Path) -> int:
    """Return the number of physical lines in a Python source file."""

    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def iter_python_files(roots: Iterable[Path]) -> Iterable[Path]:
    """Yield Python files below the requested roots in deterministic order."""

    for root in sorted(roots):
        if root.is_file():
            if root.suffix == ".py":
                yield root
            continue
        if root.is_dir():
            yield from sorted(root.rglob("*.py"))


def collect_source_stats(roots: Iterable[Path]) -> list[SourceFileStat]:
    """Collect source line counts sorted largest first."""

    stats = [SourceFileStat(path=path, lines=count_source_lines(path)) for path in iter_python_files(roots)]
    return sorted(stats, key=lambda item: (-item.lines, str(item.path)))


class _FunctionLineVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self._scope: list[str] = []
        self.stats: list[FunctionStat] = []

    def _record_function(self, node: ast.AsyncFunctionDef | ast.FunctionDef) -> None:
        end_lineno = getattr(node, "end_lineno", node.lineno)
        qualified_name = ".".join([*self._scope, node.name])
        self.stats.append(
            FunctionStat(
                path=self.path,
                qualified_name=qualified_name,
                lines=max(1, int(end_lineno) - int(node.lineno) + 1),
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802 - ast API
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802 - ast API
        self._record_function(node)
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802 - ast API
        self._record_function(node)
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()


def collect_function_stats(roots: Iterable[Path]) -> list[FunctionStat]:
    """Collect function and method line counts sorted largest first."""

    stats: list[FunctionStat] = []
    for path in iter_python_files(roots):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        visitor = _FunctionLineVisitor(path)
        visitor.visit(tree)
        stats.extend(visitor.stats)
    return sorted(stats, key=lambda item: (-item.lines, str(item.path), item.qualified_name))


class _PublicDocstringVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self._scope: list[str] = []
        self.missing: list[PublicDocstringStat] = []

    @staticmethod
    def _is_public_callable_name(name: str) -> bool:
        return not name.startswith("_") or name in {"__init__", "__call__", "__repr__", "__str__"}

    def _record_if_missing(self, node: ast.ClassDef | ast.AsyncFunctionDef | ast.FunctionDef) -> None:
        name = node.name
        if isinstance(node, ast.ClassDef):
            public = not name.startswith("_")
            kind = "class"
        else:
            public = self._is_public_callable_name(name)
            kind = "function"
        if public and ast.get_docstring(node) is None:
            self.missing.append(
                PublicDocstringStat(
                    path=self.path,
                    qualified_name=".".join([*self._scope, name]),
                    lineno=int(node.lineno),
                    kind=kind,
                )
            )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802 - ast API
        self._record_if_missing(node)
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802 - ast API
        self._record_if_missing(node)
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802 - ast API
        self._record_if_missing(node)
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()


def collect_missing_public_docstrings(roots: Iterable[Path]) -> list[PublicDocstringStat]:
    """Collect public package functions/classes missing docstrings."""

    missing: list[PublicDocstringStat] = []
    for path in iter_python_files(roots):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        visitor = _PublicDocstringVisitor(path)
        visitor.visit(tree)
        missing.extend(visitor.missing)
    return sorted(missing, key=lambda item: (str(item.path), item.lineno, item.qualified_name))


def function_stat_key(stat: FunctionStat) -> str:
    """Return the stable ``path:function`` key used by baseline gates."""

    return f"{stat.path}:{stat.qualified_name}"


def parse_function_line_limits(values: Iterable[str]) -> dict[str, int]:
    """Parse ``path:function=max_lines`` entries for function-size baselines."""

    limits: dict[str, int] = {}
    for raw in values:
        key, sep, limit_text = str(raw).partition("=")
        if not sep:
            raise ValueError(f"Function line limit must be 'path:function=max_lines', got {raw!r}.")
        key = key.strip()
        if not key:
            raise ValueError(f"Function line limit key is empty in {raw!r}.")
        try:
            limit = int(limit_text)
        except ValueError as exc:
            raise ValueError(f"Function line limit must be an integer in {raw!r}.") from exc
        if limit < 1:
            raise ValueError(f"Function line limit must be positive in {raw!r}.")
        limits[key] = limit
    return limits


def function_line_limit_failures(
    stats: Iterable[FunctionStat],
    limits: dict[str, int],
) -> list[tuple[str, int, int]]:
    """Return named functions whose physical line count exceeds its baseline."""

    by_key = {function_stat_key(stat): stat for stat in stats}
    failures: list[tuple[str, int, int]] = []
    for key, limit in sorted(limits.items()):
        stat = by_key.get(key)
        if stat is None:
            failures.append((key, -1, limit))
        elif stat.lines > limit:
            failures.append((key, stat.lines, limit))
    return failures


def collect_root_namespace_stat(
    root: Path,
    *,
    helper_prefixes: Iterable[str] = DEFAULT_ROOT_HELPER_PREFIXES,
) -> RootNamespaceStat:
    """Collect root-package file-count metrics for namespace-sprawl gates."""

    prefixes = tuple(str(prefix) for prefix in helper_prefixes)
    if not root.is_dir():
        return RootNamespaceStat(root=root, python_files=0, helper_prefix_files=())
    root_python_files = sorted(path for path in root.glob("*.py") if path.is_file())
    helper_prefix_files = tuple(path for path in root_python_files if path.name.startswith(prefixes))
    return RootNamespaceStat(
        root=root,
        python_files=len(root_python_files),
        helper_prefix_files=helper_prefix_files,
    )


def format_source_health_report(
    stats: Iterable[SourceFileStat],
    *,
    top: int,
    warn_lines: int,
) -> str:
    """Format a source-health report for terminals and PR comments."""

    selected = list(stats)[:top]
    if not selected:
        return "No Python files found."

    path_width = max(len(str(item.path)) for item in selected)
    lines = ["Python source-health report", f"warning threshold: {warn_lines} lines", ""]
    for item in selected:
        marker = "WARN" if item.lines >= warn_lines else "    "
        lines.append(f"{marker}  {item.lines:6d}  {str(item.path):<{path_width}}")
    return "\n".join(lines)


def format_function_health_report(
    stats: Iterable[FunctionStat],
    *,
    top: int,
    warn_lines: int,
) -> str:
    """Format a function-length report for terminals and CI logs."""

    selected = list(stats)[:top]
    if not selected:
        return "\nFunction-length report\nNo Python functions found."

    target_width = max(len(f"{item.path}:{item.qualified_name}") for item in selected)
    lines = ["", "Function-length report", f"warning threshold: {warn_lines} lines", ""]
    for item in selected:
        marker = "WARN" if item.lines >= warn_lines else "    "
        target = f"{item.path}:{item.qualified_name}"
        lines.append(f"{marker}  {item.lines:6d}  {target:<{target_width}}")
    return "\n".join(lines)


def format_root_namespace_report(stat: RootNamespaceStat, *, max_helper_prefix_files: int | None = None) -> str:
    """Format root-package namespace metrics for terminals and CI logs."""

    lines = [
        "",
        "Root namespace report",
        f"root: {stat.root}",
        f"root python files: {stat.python_files}",
        f"helper-prefix files: {len(stat.helper_prefix_files)}",
    ]
    if max_helper_prefix_files is not None:
        lines.append(f"helper-prefix file limit: {max_helper_prefix_files}")
    if stat.helper_prefix_files:
        lines.append("helper-prefix paths:")
        for path in stat.helper_prefix_files:
            lines.append(f"  {path}")
    return "\n".join(lines)


def format_public_docstring_report(missing: Iterable[PublicDocstringStat]) -> str:
    """Format public-docstring coverage for terminals and CI logs."""

    rows = list(missing)
    lines = ["", "Public docstring report", f"missing public docstrings: {len(rows)}"]
    for item in rows[:100]:
        lines.append(f"  {item.path}:{item.lineno}: {item.kind} {item.qualified_name}")
    if len(rows) > 100:
        lines.append(f"  ... {len(rows) - 100} additional missing public docstrings")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        default=list(DEFAULT_ROOTS),
        help="Files or directories to scan. Defaults to vmec_jax, examples/optimization, and tests.",
    )
    parser.add_argument("--top", type=int, default=30, help="Number of largest files to print.")
    parser.add_argument("--warn-lines", type=int, default=2000, help="Mark files at or above this line count.")
    parser.add_argument("--top-functions", type=int, default=20, help="Number of largest functions to print.")
    parser.add_argument(
        "--warn-function-lines",
        type=int,
        default=150,
        help="Mark functions or methods at or above this physical line count.",
    )
    parser.add_argument(
        "--fail-lines",
        type=int,
        default=0,
        help="Exit nonzero if any scanned file is at or above this line count. Disabled by default.",
    )
    parser.add_argument(
        "--fail-function-lines",
        type=int,
        default=0,
        help="Exit nonzero if any scanned function is at or above this line count. Disabled by default.",
    )
    parser.add_argument(
        "--max-function-lines-at",
        action="append",
        default=None,
        metavar="PATH:FUNCTION=LINES",
        help=(
            "Exit nonzero if the named function exceeds this baseline. "
            "May be repeated; use it to prevent known large functions from growing during refactors."
        ),
    )
    parser.add_argument(
        "--root-namespace",
        default="vmec_jax",
        help="Root package to inspect for namespace-sprawl metrics.",
    )
    parser.add_argument(
        "--root-helper-prefix",
        action="append",
        default=None,
        help=(
            "Root-level helper prefix to count. May be repeated. "
            f"Defaults to {', '.join(DEFAULT_ROOT_HELPER_PREFIXES)}."
        ),
    )
    parser.add_argument(
        "--max-root-helper-prefix-files",
        type=int,
        default=-1,
        help=(
            "Exit nonzero if root-level helper-prefix files exceed this count. "
            "Use the current baseline during migration to prevent new sprawl."
        ),
    )
    parser.add_argument(
        "--max-root-python-files",
        type=int,
        default=-1,
        help="Exit nonzero if root-level Python files exceed this count. Disabled by default.",
    )
    parser.add_argument(
        "--require-public-docstrings",
        action="store_true",
        help="Exit nonzero if public functions/classes below --public-docstring-root lack docstrings.",
    )
    parser.add_argument(
        "--public-docstring-root",
        action="append",
        default=None,
        help="Package roots to scan for public docstring coverage. Defaults to vmec_jax.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    roots = [Path(root) for root in args.roots]
    stats = collect_source_stats(roots)
    function_stats = collect_function_stats(roots)
    print(format_source_health_report(stats, top=args.top, warn_lines=args.warn_lines))
    print(format_function_health_report(function_stats, top=args.top_functions, warn_lines=args.warn_function_lines))
    helper_prefixes = tuple(args.root_helper_prefix or DEFAULT_ROOT_HELPER_PREFIXES)
    namespace_stat = collect_root_namespace_stat(Path(args.root_namespace), helper_prefixes=helper_prefixes)
    helper_limit = None if args.max_root_helper_prefix_files < 0 else int(args.max_root_helper_prefix_files)
    print(format_root_namespace_report(namespace_stat, max_helper_prefix_files=helper_limit))
    docstring_roots = [Path(root) for root in (args.public_docstring_root or ["vmec_jax"])]
    missing_public_docstrings = collect_missing_public_docstrings(docstring_roots)
    print(format_public_docstring_report(missing_public_docstrings))

    failed = False
    if args.fail_lines > 0 and any(item.lines >= args.fail_lines for item in stats):
        failed = True
    if args.fail_function_lines > 0 and any(item.lines >= args.fail_function_lines for item in function_stats):
        failed = True
    try:
        function_limits = parse_function_line_limits(args.max_function_lines_at or [])
    except ValueError as exc:
        print(f"source-health error: {exc}")
        return 2
    function_limit_failures = function_line_limit_failures(function_stats, function_limits)
    for key, observed, limit in function_limit_failures:
        if observed < 0:
            print(f"source-health error: {key} was not found for function-size baseline {limit}.")
        else:
            print(f"source-health error: {key} has {observed} lines, exceeding baseline {limit}.")
    failed = failed or bool(function_limit_failures)
    if args.max_root_helper_prefix_files >= 0:
        failed = failed or len(namespace_stat.helper_prefix_files) > int(args.max_root_helper_prefix_files)
    if args.max_root_python_files >= 0:
        failed = failed or namespace_stat.python_files > int(args.max_root_python_files)
    if args.require_public_docstrings:
        failed = failed or bool(missing_public_docstrings)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
