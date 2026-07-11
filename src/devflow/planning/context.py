from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

BINARY_SUFFIXES = {
    ".7z", ".avi", ".bmp", ".class", ".dll", ".exe", ".gif", ".gz",
    ".ico", ".jar", ".jpeg", ".jpg", ".mov", ".mp3", ".mp4", ".pdf",
    ".png", ".pyc", ".so", ".tar", ".webp", ".woff", ".woff2", ".zip",
}


def directory_summary(
    tracked_files: list[str],
    *,
    max_depth: int = 4,
    collapse_threshold: int = 50,
    max_lines: int = 300,
) -> str:
    tree: dict[str, Any] = {}
    for file_name in tracked_files:
        node = tree
        for part in PurePosixPath(file_name).parts:
            node = node.setdefault(part, {})

    def file_count(node: dict[str, Any]) -> int:
        if not node:
            return 1
        return sum(file_count(child) for child in node.values())

    lines: list[str] = ["./"]

    truncated = False

    def render(node: dict[str, Any], depth: int) -> None:
        nonlocal truncated
        for name, child in sorted(node.items()):
            if len(lines) >= max_lines:
                truncated = True
                return
            indent = "  " * depth
            if not child:
                lines.append(f"{indent}{name}")
                continue

            count = file_count(child)
            if depth >= max_depth or count > collapse_threshold:
                lines.append(f"{indent}{name}/ ({count} files)")
            else:
                lines.append(f"{indent}{name}/")
                render(child, depth + 1)
                if truncated:
                    return

    render(tree, 1)
    if truncated:
        lines.append("  ... (directory summary truncated)")
    return "\n".join(lines)


def _valid_requested_files(
    requested_files: list[str],
    tracked_files: set[str],
    limit: int,
) -> list[str]:
    valid: list[str] = []
    for requested in requested_files[:limit]:
        path = PurePosixPath(requested)
        normalized = path.as_posix()
        if path.is_absolute() or ".." in path.parts:
            continue
        if normalized in tracked_files and normalized not in valid:
            valid.append(normalized)
    return valid


def _read_text(
    path: Path,
    max_chars: int,
    *,
    skip_if_larger: bool = False,
) -> str | None:
    if path.suffix.casefold() in BINARY_SUFFIXES:
        return None
    try:
        if skip_if_larger and path.stat().st_size > max_chars * 4:
            return None
        with path.open("rb") as file:
            data = file.read(max_chars * 4)
    except OSError:
        return None
    if b"\x00" in data:
        return None
    return data.decode("utf-8", errors="replace")[:max_chars]


def gather_requested_context(
    repo_path: str,
    tracked_files: list[str],
    context_request: dict[str, Any],
    *,
    max_requested_files: int,
    max_searches: int,
    max_context_chars: int,
    max_search_results_chars: int,
) -> dict[str, Any]:
    repo = Path(repo_path)
    tracked_set = set(tracked_files)
    requested_files = _valid_requested_files(
        context_request.get("files", []),
        tracked_set,
        max_requested_files,
    )

    selected_files: dict[str, str] = {}
    remaining_file_chars = max_context_chars
    for file_name in requested_files:
        if remaining_file_chars <= 0:
            break
        contents = _read_text(repo / file_name, remaining_file_chars)
        if contents is not None:
            selected_files[file_name] = contents
            remaining_file_chars -= len(contents)

    searches = [
        search.strip()
        for search in context_request.get("searches", [])[:max_searches]
        if search.strip()
    ]
    search_results: dict[str, list[dict[str, Any]]] = {}
    remaining_search_chars = max_search_results_chars
    for search in searches:
        if remaining_search_chars <= 0:
            break
        matches: list[dict[str, Any]] = []
        needle = search.casefold()
        for file_name in tracked_files:
            if remaining_search_chars <= 0 or len(matches) >= 20:
                break
            contents = _read_text(
                repo / file_name,
                250_000,
                skip_if_larger=True,
            )
            if contents is None:
                continue
            for line_number, line in enumerate(contents.splitlines(), start=1):
                if needle not in line.casefold():
                    continue
                snippet = line.strip()[:300]
                cost = len(file_name) + len(snippet) + 30
                if cost > remaining_search_chars:
                    remaining_search_chars = 0
                    break
                matches.append({
                    "file": file_name,
                    "line": line_number,
                    "text": snippet,
                })
                remaining_search_chars -= cost
                if len(matches) >= 20:
                    break
        search_results[search] = matches

    return {
        "context_request": context_request,
        "selected_files": selected_files,
        "search_results": search_results,
        "limits": {
            "requested_files": max_requested_files,
            "searches": max_searches,
            "selected_file_chars": max_context_chars,
            "search_result_chars": max_search_results_chars,
        },
    }
