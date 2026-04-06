from collections.abc import Callable
from typing import Any

from src.shared_types import Grid, MOVE_DELTAS, Path, Point


def trace_path_points(path: Path) -> list[Point]:
    start = path["start"]
    if start is None:
        return []

    row, col = start
    points: list[Point] = [start]
    for move in path["moves"]:
        d_row, d_col = MOVE_DELTAS[move]
        row += d_row
        col += d_col
        points.append((row, col))

    return points


def calculate_visit_counts(points: list[Point]) -> dict[Point, int]:
    counts: dict[Point, int] = {}
    for point in points:
        counts[point] = counts.get(point, 0) + 1
    return counts


def path_stats(path: Path) -> str:
    points = trace_path_points(path)
    counts = calculate_visit_counts(points)
    overlaps = sum(count - 1 for count in counts.values() if count > 1)
    return (
        f"Start: {path['start']} | Moves: {len(path['moves'])} | Overlaps: {overlaps}"
    )


def _draw_base_grid(
    canvas: Any,
    grid: Grid,
    cell_bbox: Callable[[int, int], tuple[int, int, int, int]],
) -> tuple[int, int]:
    open_count = 0
    blocked_count = 0

    rows = len(grid)
    cols = len(grid[0])
    for row in range(rows):
        for col in range(cols):
            x0, y0, x1, y1 = cell_bbox(row, col)
            if grid[row][col] == 0:
                blocked_count += 1
                canvas.create_rectangle(
                    x0, y0, x1, y1, fill="#334155", outline="#1e293b"
                )
            else:
                open_count += 1
                canvas.create_rectangle(
                    x0, y0, x1, y1, fill="#ffffff", outline="#cbd5e1"
                )

    return open_count, blocked_count


def _show_grid_window_tk(
    grid: Grid,
    *,
    title: str,
    cell_size: int,
    text_panel: int,
    draw_overlay: Callable[..., None] | None,
    legend_lines: list[str],
) -> bool:
    try:
        import tkinter as tk
    except ModuleNotFoundError:
        return False

    if not grid or not grid[0]:
        return False

    rows = len(grid)
    cols = len(grid[0])
    margin = 24
    width = cols * cell_size + margin * 2
    height = rows * cell_size + margin * 2 + text_panel

    try:
        root = tk.Tk()
    except tk.TclError:
        return False

    root.title(title)
    canvas = tk.Canvas(
        root, width=width, height=height, bg="#f7fafc", highlightthickness=0
    )
    canvas.pack()

    def cell_bbox(row: int, col: int) -> tuple[int, int, int, int]:
        x0 = margin + col * cell_size
        y0 = margin + row * cell_size
        x1 = x0 + cell_size
        y1 = y0 + cell_size
        return x0, y0, x1, y1

    _draw_base_grid(canvas, grid, cell_bbox)

    if draw_overlay is not None:
        draw_overlay(canvas, cell_bbox, tk, rows, cols)

    legend_y = margin + rows * cell_size + 20
    for idx, line in enumerate(legend_lines):
        canvas.create_text(
            margin,
            legend_y + (idx * 20),
            anchor="w",
            fill="#0f172a" if idx == 0 else "#334155",
            font=("TkDefaultFont", 10),
            text=line,
        )

    root.mainloop()
    return True


def _item_tags(tag: str | None) -> dict[str, tuple[str, ...]]:
    if tag is None:
        return {}
    return {"tags": (tag,)}


def _draw_path_overlay(
    canvas: Any,
    cell_bbox: Callable[[int, int], tuple[int, int, int, int]],
    tk: Any,
    grid: Grid,
    path: Path,
    *,
    cell_size: int,
    rows: int,
    cols: int,
    tag: str | None = None,
) -> None:
    points = trace_path_points(path)
    visit_counts = calculate_visit_counts(points)
    item_tags = _item_tags(tag)

    for i in range(1, len(points)):
        pr, pc = points[i - 1]
        cr, cc = points[i]
        if not (
            0 <= pr < rows and 0 <= pc < cols and 0 <= cr < rows and 0 <= cc < cols
        ):
            continue

        px0, py0, px1, py1 = cell_bbox(pr, pc)
        cx0, cy0, cx1, cy1 = cell_bbox(cr, cc)
        x_start = (px0 + px1) / 2
        y_start = (py0 + py1) / 2
        x_end = (cx0 + cx1) / 2
        y_end = (cy0 + cy1) / 2

        canvas.create_line(
            x_start,
            y_start,
            x_end,
            y_end,
            fill="#0f766e",
            width=3,
            arrow=tk.LAST,
            arrowshape=(10, 12, 5),
            **item_tags,
        )

    for (row, col), count in visit_counts.items():
        if (
            count <= 1
            or not (0 <= row < rows and 0 <= col < cols)
            or grid[row][col] == 0
        ):
            continue

        x0, y0, x1, y1 = cell_bbox(row, col)
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        radius = cell_size * 0.18
        canvas.create_oval(
            cx - radius,
            cy - radius,
            cx + radius,
            cy + radius,
            fill="#fb7185",
            outline="#be123c",
            width=1,
            **item_tags,
        )
        canvas.create_text(
            cx,
            cy,
            text=str(count),
            fill="#ffffff",
            font=("TkDefaultFont", 9, "bold"),
            **item_tags,
        )

    if path["start"] is not None:
        s_row, s_col = path["start"]
        if 0 <= s_row < rows and 0 <= s_col < cols and grid[s_row][s_col] == 1:
            x0, y0, x1, y1 = cell_bbox(s_row, s_col)
            canvas.create_rectangle(
                x0 + 5,
                y0 + 5,
                x1 - 5,
                y1 - 5,
                outline="#16a34a",
                width=3,
                **item_tags,
            )
            canvas.create_text(
                x0 + 12,
                y0 + 12,
                text="S",
                fill="#15803d",
                font=("TkDefaultFont", 10, "bold"),
                **item_tags,
            )

    if points:
        e_row, e_col = points[-1]
        if 0 <= e_row < rows and 0 <= e_col < cols and grid[e_row][e_col] == 1:
            x0, y0, x1, y1 = cell_bbox(e_row, e_col)
            canvas.create_rectangle(
                x0 + 5,
                y0 + 5,
                x1 - 5,
                y1 - 5,
                outline="#dc2626",
                width=3,
                **item_tags,
            )
            canvas.create_text(
                x0 + 12,
                y0 + 12,
                text="E",
                fill="#b91c1c",
                font=("TkDefaultFont", 10, "bold"),
                **item_tags,
            )


class _LivePathWindow:
    def __init__(
        self,
        grid: Grid,
        path: Path,
        *,
        title: str,
        cell_size: int,
        status_line: str,
    ) -> None:
        import tkinter as tk

        self.tk = tk
        self.grid = grid
        self.cell_size = cell_size
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows else 0
        self.margin = 24
        self.closed = False
        self.overlay_tag = "live-path-overlay"
        self.dynamic_legend_tag = "live-path-legend"

        width = self.cols * cell_size + self.margin * 2
        height = self.rows * cell_size + self.margin * 2 + 80

        self.root = tk.Tk()
        self.root.title(title)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.canvas = tk.Canvas(
            self.root,
            width=width,
            height=height,
            bg="#f7fafc",
            highlightthickness=0,
        )
        self.canvas.pack()

        _draw_base_grid(self.canvas, grid, self.cell_bbox)

        legend_y = self.margin + self.rows * cell_size + 20
        self.canvas.create_text(
            self.margin,
            legend_y,
            anchor="w",
            fill="#0f172a",
            font=("TkDefaultFont", 10),
            text="Arrows: direction | Red badges: overlap count | Green box: start | Red box: end",
        )

        self.update_path(path, status_line=status_line)

    def cell_bbox(self, row: int, col: int) -> tuple[int, int, int, int]:
        x0 = self.margin + col * self.cell_size
        y0 = self.margin + row * self.cell_size
        x1 = x0 + self.cell_size
        y1 = y0 + self.cell_size
        return x0, y0, x1, y1

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.root.destroy()
        except self.tk.TclError:
            pass

    def update_path(self, path: Path, *, status_line: str) -> bool:
        if self.closed:
            return False

        try:
            self.canvas.delete(self.overlay_tag)
            self.canvas.delete(self.dynamic_legend_tag)
            _draw_path_overlay(
                self.canvas,
                self.cell_bbox,
                self.tk,
                self.grid,
                path,
                cell_size=self.cell_size,
                rows=self.rows,
                cols=self.cols,
                tag=self.overlay_tag,
            )

            legend_y = self.margin + self.rows * self.cell_size + 40
            for idx, line in enumerate((status_line, path_stats(path))):
                self.canvas.create_text(
                    self.margin,
                    legend_y + (idx * 20),
                    anchor="w",
                    fill="#334155",
                    font=("TkDefaultFont", 10),
                    text=line,
                    tags=(self.dynamic_legend_tag,),
                )

            self.root.update_idletasks()
            self.root.update()
        except self.tk.TclError:
            self.closed = True
            return False

        return True

    def run(self) -> None:
        if self.closed:
            return
        try:
            self.root.mainloop()
        except self.tk.TclError:
            self.closed = True


def open_live_grid_path_tk(
    grid: Grid,
    path: Path,
    *,
    title: str = "Grid Path Visualizer",
    cell_size: int = 42,
    status_line: str = "Waiting for progress...",
) -> _LivePathWindow | None:
    if not grid or not grid[0]:
        return None

    try:
        import tkinter as tk
    except ModuleNotFoundError:
        return None

    try:
        return _LivePathWindow(
            grid,
            path,
            title=title,
            cell_size=cell_size,
            status_line=status_line,
        )
    except (RuntimeError, tk.TclError):
        return None


def show_grid_path_tk(
    grid: Grid,
    path: Path,
    *,
    title: str = "Grid Path Visualizer",
    cell_size: int = 42,
) -> bool:
    def draw_overlay(
        canvas: Any,
        cell_bbox: Callable[[int, int], tuple[int, int, int, int]],
        tk: Any,
        rows: int,
        cols: int,
    ) -> None:
        _draw_path_overlay(
            canvas,
            cell_bbox,
            tk,
            grid,
            path,
            cell_size=cell_size,
            rows=rows,
            cols=cols,
        )

    legend_lines = [
        "Arrows: direction | Red badges: overlap count | Green box: start | Red box: end",
        path_stats(path),
    ]

    return _show_grid_window_tk(
        grid,
        title=title,
        cell_size=cell_size,
        text_panel=60,
        draw_overlay=draw_overlay,
        legend_lines=legend_lines,
    )


def show_grid_tk(
    grid: Grid,
    *,
    title: str = "Grid Visualizer",
    cell_size: int = 42,
) -> bool:
    rows = len(grid) if grid else 0
    cols = len(grid[0]) if rows else 0
    open_count = sum(1 for row in grid for cell in row if cell == 1)
    blocked_count = rows * cols - open_count

    legend_lines = [
        "White: open cell | Dark: removed cell",
        f"Open: {open_count} | Removed: {blocked_count}",
    ]

    return _show_grid_window_tk(
        grid,
        title=title,
        cell_size=cell_size,
        text_panel=50,
        draw_overlay=None,
        legend_lines=legend_lines,
    )
