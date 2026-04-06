from __future__ import annotations

import argparse
import csv
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
from src.shared_types import MOVE_DELTAS, Grid, Path as SolverPath, Point
from src.solvers import find_start, random_walk_solver, snake_solver, spiral_solver
from src.visualize import calculate_visit_counts, trace_path_points


SOLVER_OPTIONS = {"snake", "spiral", "random_walk", "optimal"}
STRATEGY_OPTIONS = {"least_overlap", "shortest"}
START_OPTIONS = {"auto", "down", "right"}
OPTIMAL_PROGRESS_INTERVAL_SECONDS = 1.0
DEFAULT_REMOVED_FRACTION_RANGE = (0.18, 0.42)
SOLVER_LABELS = {
    "snake": "Snake",
    "spiral": "Spiral",
    "random_walk": "Random walk",
    "optimal": "Optimal",
}
HUMAN_RESULTS_PATH = (
    FilePath(__file__).resolve().parents[1] / "data" / "mow_test_rows.csv"
)
HUMAN_CONFIG_SPECS = {
    0: {"size": 7, "seed": 59},
    1: {"size": 7, "seed": 59},
    2: {"size": 10, "seed": 59},
    3: {"size": 14, "seed": 3},
}
MOVE_CODES_BY_DELTA = {delta: move for move, delta in MOVE_DELTAS.items()}


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


def build_human_config_payload(config_id: int) -> dict[str, object]:
    spec = HUMAN_CONFIG_SPECS[config_id]
    grid_payload = build_grid_payload(
        spec["size"],
        spec["seed"],
        DEFAULT_REMOVED_FRACTION_RANGE,
    )
    return {
        "id": config_id,
        "label": f"Config {config_id}",
        "size": spec["size"],
        "seed": spec["seed"],
        "grid": grid_payload,
    }


def human_points_to_path_string(points: list[Point]) -> str:
    if len(points) < 2:
        return ""

    moves: list[str] = []
    for previous, current in zip(points, points[1:]):
        delta = (current[0] - previous[0], current[1] - previous[1])
        moves.append(MOVE_CODES_BY_DELTA.get(delta, "?"))
    return "".join(moves)


def build_human_attempt_payload(
    row: dict[str, str],
    config_payload: dict[str, object],
) -> dict[str, object]:
    raw_points = json.loads(row["result"])
    points = [(int(point["y"]), int(point["x"])) for point in raw_points]
    timestamps_ms = [int(point["t"]) for point in raw_points]
    visit_counts = calculate_visit_counts(points)
    grid_payload = config_payload["grid"]
    if not isinstance(grid_payload, dict):
        raise ValueError("Config grid payload is invalid")

    grid = grid_payload["grid"]
    if not isinstance(grid, list):
        raise ValueError("Config grid is invalid")

    overlap_count = sum(count - 1 for count in visit_counts.values() if count > 1)
    visited_open_cells = sum(
        1
        for row_index, col_index in visit_counts
        if 0 <= row_index < len(grid)
        and 0 <= col_index < len(grid[0])
        and grid[row_index][col_index] == 1
    )
    open_cell_count = int(grid_payload["openCells"])
    move_count = max(0, len(points) - 1)
    duration_ms = timestamps_ms[-1] - timestamps_ms[0] if timestamps_ms else 0
    average_step_ms = (duration_ms / move_count) if move_count else 0.0

    return {
        "id": int(row["id"]),
        "createdAt": row["created_at"],
        "user": row["user"],
        "configId": int(row["config"]),
        "configLabel": config_payload["label"],
        "configSize": config_payload["size"],
        "configSeed": config_payload["seed"],
        "pathString": human_points_to_path_string(points),
        "points": [list(point) for point in points],
        "timestampsMs": timestamps_ms,
        "moveCount": move_count,
        "overlapCount": overlap_count,
        "visitedOpenCells": visited_open_cells,
        "openCellCount": open_cell_count,
        "coverageComplete": visited_open_cells == open_cell_count,
        "start": list(points[0]) if points else None,
        "end": list(points[-1]) if points else None,
        "durationMs": duration_ms,
        "averageStepMs": average_step_ms,
        "solverKey": "human",
        "solverLabel": "Human",
        "detailLabel": f"{row['user']} on {config_payload['label']}",
    }


def load_human_results_payload() -> dict[str, object]:
    if not HUMAN_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Human results file not found: {HUMAN_RESULTS_PATH}")

    config_payloads = {
        config_id: build_human_config_payload(config_id)
        for config_id in sorted(HUMAN_CONFIG_SPECS)
    }
    attempts: list[dict[str, object]] = []

    with HUMAN_RESULTS_PATH.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            config_id = int(row["config"])
            config_payload = config_payloads.get(config_id)
            if config_payload is None:
                raise ValueError(f"Unknown human-results config: {config_id}")
            attempts.append(build_human_attempt_payload(row, config_payload))

    attempts.sort(key=lambda attempt: int(attempt["id"]))
    users = sorted({str(attempt["user"]) for attempt in attempts})
    config_summaries = [
        {
            "id": config_payload["id"],
            "label": config_payload["label"],
            "size": config_payload["size"],
            "seed": config_payload["seed"],
            "grid": config_payload["grid"],
        }
        for config_payload in config_payloads.values()
    ]

    return {
        "attemptCount": len(attempts),
        "userCount": len(users),
        "users": users,
        "configs": config_summaries,
        "attempts": attempts,
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

        if parsed.path == "/api/human-results":
            self._handle_human_results()
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

    def _handle_human_results(self) -> None:
        try:
            payload = load_human_results_payload()
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, status=404)
            return
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)
            return

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
