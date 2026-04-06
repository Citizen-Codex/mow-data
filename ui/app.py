from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path as FilePath
from urllib.parse import parse_qs, urlparse

if __package__ in (None, ""):
    sys.path.append(str(FilePath(__file__).resolve().parents[1]))

from src.grid import create_random_grid
from src.optimal_solver import SearchProgress, optimal_solver
from src.shared_types import Grid, Path as SolverPath, Point
from src.solvers import find_start, random_walk_solver, snake_solver, spiral_solver
from src.visualize import calculate_visit_counts, trace_path_points


SOLVER_OPTIONS = {"snake", "spiral", "random_walk", "optimal"}
STRATEGY_OPTIONS = {"least_overlap", "shortest"}
START_OPTIONS = {"auto", "down", "right"}
OPTIMAL_PROGRESS_INTERVAL_SECONDS = 1.0
SOLVER_LABELS = {
    "snake": "Snake",
    "spiral": "Spiral",
    "random_walk": "Random walk",
    "optimal": "Optimal",
}


@dataclass(slots=True)
class SolutionSnapshot:
    solver_key: str
    solver_label: str
    detail_label: str
    path: SolverPath
    points: list[Point]
    move_count: int
    overlap_count: int
    visited_open_cells: int
    open_cell_count: int
    coverage_complete: bool
    start: Point | None
    end: Point | None

    def to_dict(self) -> dict[str, object]:
        return {
            "solverKey": self.solver_key,
            "solverLabel": self.solver_label,
            "detailLabel": self.detail_label,
            "pathString": "".join(self.path["moves"]),
            "points": [list(point) for point in self.points],
            "moveCount": self.move_count,
            "overlapCount": self.overlap_count,
            "visitedOpenCells": self.visited_open_cells,
            "openCellCount": self.open_cell_count,
            "coverageComplete": self.coverage_complete,
            "start": list(self.start) if self.start is not None else None,
            "end": list(self.end) if self.end is not None else None,
        }


def count_open_cells(grid: Grid) -> int:
    return sum(1 for row in grid for cell in row if cell == 1)


def resolve_start_mode(start_mode: str, seed: int) -> tuple[bool | None, str]:
    if start_mode == "down":
        return True, "down-first"
    if start_mode == "right":
        return False, "right-first"
    if start_mode == "auto":
        resolved = random.Random(seed).choice((True, False))
        return resolved, "auto"
    raise ValueError(f"Unknown start mode: {start_mode}")


def solve_snapshot(
    grid: Grid,
    solver_key: str,
    move_strategy: str,
    start_mode: str,
    seed: int,
) -> SolutionSnapshot:
    start_down, start_label = resolve_start_mode(start_mode, seed)

    if solver_key == "snake":
        path = snake_solver(grid, move_strategy=move_strategy, start_down=start_down)
        detail_label = f"{move_strategy}, {start_label}"
    elif solver_key == "spiral":
        path = spiral_solver(grid, move_strategy=move_strategy, start_down=start_down)
        detail_label = f"{move_strategy}, {start_label}"
    elif solver_key == "random_walk":
        path = random_walk_solver(grid, rng=random.Random(seed))
        detail_label = f"seeded walk ({seed})"
    elif solver_key == "optimal":
        path = optimal_solver(
            grid, progress_reporter=lambda message: print(message, flush=True)
        )
        detail_label = "exact branch-and-bound"
    else:
        raise ValueError(f"Unknown solver: {solver_key}")

    return build_solution_snapshot(grid, solver_key, detail_label, path)


def build_solution_snapshot(
    grid: Grid,
    solver_key: str,
    detail_label: str,
    path: SolverPath,
) -> SolutionSnapshot:
    points = trace_path_points(path)
    visit_counts = calculate_visit_counts(points)
    overlap_count = sum(count - 1 for count in visit_counts.values() if count > 1)
    visited_open_cells = sum(
        1
        for row, col in visit_counts
        if 0 <= row < len(grid) and 0 <= col < len(grid[0]) and grid[row][col] == 1
    )
    open_cell_count = count_open_cells(grid)

    return SolutionSnapshot(
        solver_key=solver_key,
        solver_label=SOLVER_LABELS[solver_key],
        detail_label=detail_label,
        path=path,
        points=points,
        move_count=len(path["moves"]),
        overlap_count=overlap_count,
        visited_open_cells=visited_open_cells,
        open_cell_count=open_cell_count,
        coverage_complete=visited_open_cells == open_cell_count,
        start=path["start"],
        end=points[-1] if points else None,
    )


def build_progress_payload(grid: Grid, progress: SearchProgress) -> dict[str, object]:
    snapshot = build_solution_snapshot(
        grid,
        "optimal",
        "best so far",
        progress.best_path,
    )
    return {
        "type": "progress",
        "phase": progress.phase,
        "message": progress.message,
        "elapsedSeconds": progress.elapsed_seconds,
        "statesExplored": progress.states_explored,
        "statesPerSecond": progress.states_per_second,
        "bestLength": progress.best_length,
        "maxVisitedCells": progress.max_visited_cells,
        "nodeCount": progress.node_count,
        "prunedBranches": progress.pruned_branches,
        "solution": snapshot.to_dict(),
    }


def build_grid_payload(
    size: int,
    seed: int,
    removed_fraction_range: tuple[float, float],
) -> dict[str, object]:
    grid = create_random_grid(size, seed, removed_fraction_range=removed_fraction_range)
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    open_cells = count_open_cells(grid)
    removed_cells = rows * cols - open_cells
    start = find_start(grid)
    return {
        "grid": grid,
        "size": size,
        "seed": seed,
        "removedFractionRange": list(removed_fraction_range),
        "rows": rows,
        "cols": cols,
        "openCells": open_cells,
        "removedCells": removed_cells,
        "density": (open_cells / (rows * cols)) if rows and cols else 0.0,
        "start": list(start) if start is not None else None,
    }


class MowUIHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/api/health":
            self._send_json({"ok": True})
            return

        if parsed.path == "/api/grid":
            self._handle_grid(parsed.query)
            return

        if parsed.path == "/":
            self.path = "/index.html"

        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/solve":
            self._handle_solve()
            return
        if parsed.path == "/api/solve-stream":
            self._handle_solve_stream()
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Unknown API route")

    def _parse_solve_request(self, body: object) -> tuple[Grid, str, str, str, int]:
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")

        grid = body.get("grid")
        solver_key = body.get("solver")
        move_strategy = body.get("strategy")
        start_mode = body.get("startMode")
        seed = body.get("seed")

        if not isinstance(grid, list) or not grid:
            raise ValueError("grid must be a non-empty 2D array")
        if solver_key not in SOLVER_OPTIONS:
            raise ValueError(f"Unknown solver: {solver_key}")
        if move_strategy not in STRATEGY_OPTIONS:
            raise ValueError(f"Unknown strategy: {move_strategy}")
        if start_mode not in START_OPTIONS:
            raise ValueError(f"Unknown startMode: {start_mode}")
        if not isinstance(seed, int):
            raise ValueError("seed must be an integer")

        return grid, solver_key, move_strategy, start_mode, seed

    def _handle_grid(self, query: str) -> None:
        params = parse_qs(query)

        try:
            size = int(params.get("size", ["7"])[0])
            seed = int(params.get("seed", ["7"])[0])
            removed_min = float(params.get("removedMin", ["0.18"])[0])
            removed_max = float(params.get("removedMax", ["0.42"])[0])
        except ValueError as exc:
            self._send_json({"error": f"Invalid grid parameters: {exc}"}, status=400)
            return

        if size <= 0:
            self._send_json({"error": "size must be positive"}, status=400)
            return

        if not (0.0 <= removed_min <= removed_max < 1.0):
            self._send_json(
                {
                    "error": "removedMin and removedMax must satisfy 0.0 <= min <= max < 1.0"
                },
                status=400,
            )
            return

        payload = build_grid_payload(size, seed, (removed_min, removed_max))
        self._send_json(payload)

    def _handle_solve(self) -> None:
        try:
            body = self._read_json_body()
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"Invalid JSON body: {exc}"}, status=400)
            return

        try:
            grid, solver_key, move_strategy, start_mode, seed = (
                self._parse_solve_request(body)
            )
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        try:
            snapshot = solve_snapshot(grid, solver_key, move_strategy, start_mode, seed)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)
            return

        self._send_json({"solution": snapshot.to_dict()})

    def _handle_solve_stream(self) -> None:
        try:
            body = self._read_json_body()
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"Invalid JSON body: {exc}"}, status=400)
            return

        try:
            grid, solver_key, _, _, _ = self._parse_solve_request(body)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        if solver_key != "optimal":
            self._send_json(
                {"error": "Streaming solve is only available for the optimal solver"},
                status=400,
            )
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def send_event(payload: dict[str, object]) -> None:
            encoded = (json.dumps(payload) + "\n").encode("utf-8")
            self.wfile.write(encoded)
            self.wfile.flush()

        def handle_progress(progress: SearchProgress) -> None:
            if progress.phase == "finished":
                return
            send_event(build_progress_payload(grid, progress))

        try:
            path = optimal_solver(
                grid,
                progress_reporter=lambda message: print(message, flush=True),
                progress_callback=handle_progress,
                progress_interval_seconds=OPTIMAL_PROGRESS_INTERVAL_SECONDS,
            )
            snapshot = build_solution_snapshot(
                grid,
                "optimal",
                "exact branch-and-bound",
                path,
            )
            send_event(
                {
                    "type": "solution",
                    "message": "Optimal solver finished.",
                    "solution": snapshot.to_dict(),
                }
            )
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            try:
                send_event({"type": "error", "error": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _read_json_body(self) -> object:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length)
        return json.loads(raw.decode("utf-8") if raw else "{}")

    def _send_json(self, payload: object, *, status: int = 200) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)


def serve(host: str, port: int) -> None:
    ui_dir = FilePath(__file__).resolve().parent
    handler = partial(MowUIHandler, directory=str(ui_dir))
    server = ThreadingHTTPServer((host, port), handler)

    display_host = host if host not in {"0.0.0.0", "::"} else "127.0.0.1"
    print(f"Serving UI at http://{display_host}:{port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the browser UI for grid generation and solver exploration"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind")
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
