const ARROW_MOVES = {
  ArrowUp: { code: "u", label: "up", delta: [-1, 0] },
  ArrowDown: { code: "d", label: "down", delta: [1, 0] },
  ArrowLeft: { code: "l", label: "left", delta: [0, -1] },
  ArrowRight: { code: "r", label: "right", delta: [0, 1] },
};

const state = {
  gridData: null,
  displayedSolution: null,
  optimalReference: null,
  currentStep: 0,
  playTimer: null,
  manualPath: null,
  manualActive: false,
};

const elements = {
  sizeInput: document.getElementById("sizeInput"),
  seedInput: document.getElementById("seedInput"),
  removedMinInput: document.getElementById("removedMinInput"),
  removedMaxInput: document.getElementById("removedMaxInput"),
  randomizeSeedButton: document.getElementById("randomizeSeedButton"),
  generateButton: document.getElementById("generateButton"),
  solverSelect: document.getElementById("solverSelect"),
  strategySelect: document.getElementById("strategySelect"),
  startModeSelect: document.getElementById("startModeSelect"),
  solveButton: document.getElementById("solveButton"),
  optimalButton: document.getElementById("optimalButton"),
  manualToggleButton: document.getElementById("manualToggleButton"),
  resetManualButton: document.getElementById("resetManualButton"),
  undoManualButton: document.getElementById("undoManualButton"),
  showManualButton: document.getElementById("showManualButton"),
  manualModeChip: document.getElementById("manualModeChip"),
  manualStats: document.getElementById("manualStats"),
  prevButton: document.getElementById("prevButton"),
  playButton: document.getElementById("playButton"),
  nextButton: document.getElementById("nextButton"),
  fullButton: document.getElementById("fullButton"),
  stepSlider: document.getElementById("stepSlider"),
  stepLabel: document.getElementById("stepLabel"),
  statusText: document.getElementById("statusText"),
  gridStats: document.getElementById("gridStats"),
  solutionStats: document.getElementById("solutionStats"),
  referenceStats: document.getElementById("referenceStats"),
  gridCanvas: document.getElementById("gridCanvas"),
};

function isBusy() {
  return elements.generateButton.disabled;
}

function setStatus(text, tone = "info") {
  elements.statusText.textContent = text;
  elements.statusText.className = `status ${tone}`;
}

function randomizeSeed() {
  elements.seedInput.value = String(Math.floor(Math.random() * 1_000_000));
}

function readGridOptions() {
  const size = Number(elements.sizeInput.value);
  const seed = Number(elements.seedInput.value);
  const removedMin = Number(elements.removedMinInput.value);
  const removedMax = Number(elements.removedMaxInput.value);

  if (!Number.isInteger(size) || size <= 0) {
    throw new Error("Grid size must be a positive integer.");
  }
  if (!Number.isInteger(seed) || seed < 0) {
    throw new Error("Seed must be a non-negative integer.");
  }
  if (!(removedMin >= 0 && removedMin <= removedMax && removedMax < 1)) {
    throw new Error("Removed fractions must satisfy 0 <= min <= max < 1.");
  }

  return { size, seed, removedMin, removedMax };
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed with status ${response.status}`);
  }
  return payload;
}

async function readErrorMessage(response) {
  const raw = await response.text();
  if (!raw) {
    return `Request failed with status ${response.status}`;
  }

  try {
    const payload = JSON.parse(raw);
    return payload.error || `Request failed with status ${response.status}`;
  } catch {
    return raw;
  }
}

async function streamJsonLines(url, options = {}, { onMessage } = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }
  if (!response.body) {
    throw new Error("Streaming responses are not available in this browser.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    let newlineIndex = buffer.indexOf("\n");
    while (newlineIndex !== -1) {
      const line = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);
      if (line) {
        onMessage?.(JSON.parse(line));
      }
      newlineIndex = buffer.indexOf("\n");
    }

    if (done) {
      break;
    }
  }

  const trailing = buffer.trim();
  if (trailing) {
    onMessage?.(JSON.parse(trailing));
  }
}

function setBusy(busy) {
  const disabled = Boolean(busy);
  elements.generateButton.disabled = disabled;
  elements.solveButton.disabled = disabled;
  elements.optimalButton.disabled = disabled;
  syncSolverControls();
  syncManualControls();
  syncPlaybackControls();
}

function syncSolverControls() {
  const needsPathControls = ["snake", "spiral"].includes(elements.solverSelect.value);
  elements.strategySelect.disabled = isBusy() || !needsPathControls;
  elements.startModeSelect.disabled = isBusy() || !needsPathControls;
}

function syncManualControls() {
  const hasGrid = Boolean(state.gridData);
  const hasManualPath = Boolean(state.manualPath);
  elements.manualToggleButton.disabled = isBusy() || !hasGrid;
  elements.resetManualButton.disabled = isBusy() || !hasGrid;
  elements.undoManualButton.disabled =
    isBusy() || !hasManualPath || state.manualPath.moves.length === 0;
  elements.showManualButton.disabled = isBusy() || !hasManualPath;
}

function syncPlaybackControls() {
  const hasSolution = Boolean(state.displayedSolution);
  const disabled = isBusy() || !hasSolution;
  elements.prevButton.disabled = disabled;
  elements.playButton.disabled = disabled;
  elements.nextButton.disabled = disabled;
  elements.fullButton.disabled = disabled;
  elements.stepSlider.disabled = disabled;
}

async function generateGrid() {
  stopPlayback();

  let options;
  try {
    options = readGridOptions();
  } catch (error) {
    setStatus(error.message, "error");
    return;
  }

  setBusy(true);
  setStatus("Generating grid...", "info");

  try {
    const query = new URLSearchParams({
      size: String(options.size),
      seed: String(options.seed),
      removedMin: String(options.removedMin),
      removedMax: String(options.removedMax),
    });
    state.gridData = await fetchJson(`/api/grid?${query.toString()}`);
    state.displayedSolution = null;
    state.optimalReference = null;
    state.currentStep = 0;
    state.manualPath = null;
    state.manualActive = false;
    setStatus("Grid generated. Choose a solver or start a manual path.", "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
    render();
  }
}

function confirmOptimalRuntime() {
  if (!state.gridData) {
    return false;
  }
  if (state.gridData.openCells <= 48) {
    return true;
  }
  return window.confirm(
    `This grid has ${state.gridData.openCells} open cells. The exact solver is exponential and may take a while. Continue?`
  );
}

async function requestSolution(solver, options = {}) {
  if (!state.gridData) {
    throw new Error("Generate a grid first.");
  }

  const body = {
    grid: state.gridData.grid,
    solver,
    strategy: elements.strategySelect.value,
    startMode: elements.startModeSelect.value,
    seed: Number(elements.seedInput.value),
  };

  if (solver === "optimal") {
    return requestOptimalSolution(body, options);
  }

  const payload = await fetchJson("/api/solve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  return payload.solution;
}

async function requestOptimalSolution(body, { onProgress } = {}) {
  let finalSolution = null;

  await streamJsonLines(
    "/api/solve-stream",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
    {
      onMessage(payload) {
        if (payload.type === "progress") {
          onProgress?.(payload);
          return;
        }

        if (payload.type === "solution") {
          finalSolution = payload.solution;
          return;
        }

        if (payload.type === "error") {
          throw new Error(payload.error || "Optimal solve failed.");
        }
      },
    }
  );

  if (!finalSolution) {
    throw new Error("Optimal solve ended without returning a solution.");
  }

  return finalSolution;
}

function pauseManualInput() {
  if (!state.manualActive) {
    return;
  }
  state.manualActive = false;
  if (state.displayedSolution?.solverKey === "manual") {
    state.displayedSolution = buildManualSolution();
  }
}

async function solveDisplayed() {
  const solver = elements.solverSelect.value;
  if (solver === "optimal" && !confirmOptimalRuntime()) {
    return;
  }

  stopPlayback();
  pauseManualInput();
  setBusy(true);
  setStatus(`Solving ${solver}...`, "info");

  try {
    const solution = await requestSolution(solver, {
      onProgress(payload) {
        if (!payload.solution) {
          return;
        }
        state.displayedSolution = payload.solution;
        state.currentStep = payload.solution.moveCount;
        setStatus(payload.message || "Solving optimal...", "info");
        render();
      },
    });
    state.displayedSolution = solution;
    state.currentStep = solution.moveCount;
    if (solver === "optimal") {
      state.optimalReference = solution;
    }
    setStatus(`Displayed ${solution.solverLabel.toLowerCase()} solution.`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
    render();
  }
}

async function loadOptimalReference() {
  if (!confirmOptimalRuntime()) {
    return;
  }

  stopPlayback();
  pauseManualInput();
  setBusy(true);
  setStatus("Loading optimal reference...", "info");

  const previousDisplayedSolution = state.displayedSolution;
  const previousStep = state.currentStep;

  try {
    state.optimalReference = await requestSolution("optimal", {
      onProgress(payload) {
        if (!payload.solution) {
          return;
        }
        state.displayedSolution = payload.solution;
        state.currentStep = payload.solution.moveCount;
        setStatus(payload.message || "Loading optimal reference...", "info");
        render();
      },
    });
    if (previousDisplayedSolution) {
      state.displayedSolution = previousDisplayedSolution;
      state.currentStep = Math.min(previousStep, previousDisplayedSolution.moveCount);
    } else {
      state.displayedSolution = state.optimalReference;
      state.currentStep = state.optimalReference.moveCount;
    }
    setStatus("Loaded optimal reference for the current grid.", "success");
  } catch (error) {
    if (previousDisplayedSolution) {
      state.displayedSolution = previousDisplayedSolution;
      state.currentStep = Math.min(previousStep, previousDisplayedSolution.moveCount);
    } else {
      state.displayedSolution = null;
      state.currentStep = 0;
    }
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
    render();
  }
}

function createEmptyManualPath() {
  if (!state.gridData?.start) {
    throw new Error("This grid does not have a valid start cell.");
  }
  return {
    moves: [],
    points: [state.gridData.start.slice()],
  };
}

function buildManualSolution() {
  if (!state.gridData || !state.manualPath) {
    return null;
  }

  const points = state.manualPath.points.map((point) => [...point]);
  const visitCounts = countVisits(points);
  let overlapCount = 0;
  let visitedOpenCells = 0;

  for (const [key, count] of visitCounts.entries()) {
    const [row, col] = key.split(",").map(Number);
    if (state.gridData.grid[row]?.[col] === 1) {
      visitedOpenCells += 1;
    }
    if (count > 1) {
      overlapCount += count - 1;
    }
  }

  return {
    solverKey: "manual",
    solverLabel: "Manual",
    detailLabel: state.manualActive ? "arrow-key input active" : "user-created path",
    pathString: state.manualPath.moves.join(""),
    points,
    moveCount: state.manualPath.moves.length,
    overlapCount,
    visitedOpenCells,
    openCellCount: state.gridData.openCells,
    coverageComplete: visitedOpenCells === state.gridData.openCells,
    start: points[0] || null,
    end: points[points.length - 1] || null,
  };
}

function showManualPath({ keepStatus = false } = {}) {
  if (!state.manualPath) {
    setStatus("Start a manual path first.", "info");
    return;
  }

  stopPlayback();
  state.displayedSolution = buildManualSolution();
  state.currentStep = state.displayedSolution.moveCount;
  if (!keepStatus) {
    setStatus("Displayed the current manual path.", "success");
  }
  render();
}

function toggleManualInput() {
  if (!state.gridData) {
    setStatus("Generate a grid first.", "error");
    return;
  }

  stopPlayback();

  if (!state.manualActive) {
    if (!state.manualPath) {
      state.manualPath = createEmptyManualPath();
    }
    state.manualActive = true;
    state.displayedSolution = buildManualSolution();
    state.currentStep = state.displayedSolution.moveCount;
    setStatus(
      "Manual input active. Use arrow keys to move and Backspace to undo.",
      "success"
    );
  } else {
    state.manualActive = false;
    if (state.displayedSolution?.solverKey === "manual") {
      state.displayedSolution = buildManualSolution();
    }
    setStatus("Manual input paused.", "info");
  }

  render();
}

function resetManualPath() {
  if (!state.gridData) {
    setStatus("Generate a grid first.", "error");
    return;
  }

  stopPlayback();
  state.manualPath = createEmptyManualPath();
  state.manualActive = true;
  state.displayedSolution = buildManualSolution();
  state.currentStep = 0;
  setStatus("Manual path reset. Use arrow keys to move.", "success");
  render();
}

function undoManualMove({ keepStatus = false } = {}) {
  if (!state.manualPath) {
    setStatus("Start a manual path first.", "info");
    return;
  }

  if (state.manualPath.moves.length === 0) {
    if (!keepStatus) {
      setStatus("Manual path is already at the start cell.", "info");
    }
    return;
  }

  stopPlayback();
  state.manualPath.moves.pop();
  state.manualPath.points.pop();
  state.displayedSolution = buildManualSolution();
  state.currentStep = state.displayedSolution.moveCount;

  if (!keepStatus) {
    setStatus("Removed the last manual move.", "success");
  }

  render();
}

function attemptManualMove(move) {
  if (!state.manualActive || !state.gridData) {
    return;
  }

  if (!state.manualPath) {
    state.manualPath = createEmptyManualPath();
  }

  const currentPoint = state.manualPath.points[state.manualPath.points.length - 1];
  const nextRow = currentPoint[0] + move.delta[0];
  const nextCol = currentPoint[1] + move.delta[1];

  if (
    nextRow < 0 ||
    nextRow >= state.gridData.rows ||
    nextCol < 0 ||
    nextCol >= state.gridData.cols ||
    state.gridData.grid[nextRow][nextCol] !== 1
  ) {
    setStatus(`Blocked move: cannot go ${move.label} from the current cell.`, "error");
    return;
  }

  stopPlayback();
  state.manualPath.moves.push(move.code);
  state.manualPath.points.push([nextRow, nextCol]);
  state.displayedSolution = buildManualSolution();
  state.currentStep = state.displayedSolution.moveCount;

  if (state.displayedSolution.coverageComplete) {
    setStatus("Manual path now covers every open cell.", "success");
  } else {
    setStatus(
      "Manual input active. Use arrow keys to move and Backspace to undo.",
      "info"
    );
  }

  render();
}

function shouldIgnoreManualKeydown(event) {
  if (event.altKey || event.ctrlKey || event.metaKey) {
    return true;
  }

  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return false;
  }

  return (
    target.isContentEditable ||
    ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)
  );
}

function handleManualKeydown(event) {
  if (!state.manualActive || isBusy() || shouldIgnoreManualKeydown(event)) {
    return;
  }

  if (event.key === "Backspace") {
    event.preventDefault();
    undoManualMove({ keepStatus: true });
    setStatus("Manual input active. Use arrow keys to move and Backspace to undo.", "info");
    return;
  }

  const move = ARROW_MOVES[event.key];
  if (!move) {
    return;
  }

  event.preventDefault();
  attemptManualMove(move);
}

function renderStats() {
  if (!state.gridData) {
    elements.gridStats.textContent = "No grid loaded.";
  } else {
    const grid = state.gridData;
    elements.gridStats.textContent = `Grid ${grid.rows}x${grid.cols} | seed ${grid.seed} | open ${grid.openCells} | removed ${grid.removedCells} | density ${grid.density.toFixed(2)}`;
  }

  if (!state.displayedSolution) {
    elements.solutionStats.textContent = "No displayed solution yet.";
  } else {
    const solution = state.displayedSolution;
    elements.solutionStats.textContent = `Displayed: ${solution.solverLabel} (${solution.detailLabel}) | moves ${solution.moveCount} | overlaps ${solution.overlapCount} | coverage ${solution.visitedOpenCells}/${solution.openCellCount} ${solution.coverageComplete ? "(complete)" : "(partial)"}`;
  }

  if (!state.optimalReference) {
    elements.referenceStats.textContent = "Optimal reference not loaded for this grid.";
  } else if (!state.displayedSolution) {
    const reference = state.optimalReference;
    elements.referenceStats.textContent = `Optimal reference: ${reference.moveCount} moves | overlaps ${reference.overlapCount}`;
  } else {
    const reference = state.optimalReference;
    const solution = state.displayedSolution;
    const moveDelta = solution.moveCount - reference.moveCount;
    const overlapDelta = solution.overlapCount - reference.overlapCount;
    elements.referenceStats.textContent = `Optimal reference: ${reference.moveCount} moves | overlaps ${reference.overlapCount} | delta vs displayed ${formatDelta(moveDelta)} moves, ${formatDelta(overlapDelta)} overlaps`;
  }
}

function renderManualControls() {
  if (!state.gridData) {
    elements.manualModeChip.textContent = "no grid";
    elements.manualModeChip.className = "chip chip-idle";
    elements.manualStats.textContent = "Generate a grid before starting a manual path.";
  } else if (state.manualActive) {
    elements.manualModeChip.textContent = "input active";
    elements.manualModeChip.className = "chip chip-active";
    const manualSolution = buildManualSolution();
    elements.manualStats.textContent = manualSolution
      ? `Manual path: moves ${manualSolution.moveCount} | overlaps ${manualSolution.overlapCount} | coverage ${manualSolution.visitedOpenCells}/${manualSolution.openCellCount} ${manualSolution.coverageComplete ? "(complete)" : "(partial)"}`
      : "Start a manual path to use the arrow keys.";
  } else if (state.manualPath) {
    elements.manualModeChip.textContent = "paused";
    elements.manualModeChip.className = "chip chip-paused";
    const manualSolution = buildManualSolution();
    elements.manualStats.textContent = `Manual path: moves ${manualSolution.moveCount} | overlaps ${manualSolution.overlapCount} | coverage ${manualSolution.visitedOpenCells}/${manualSolution.openCellCount} ${manualSolution.coverageComplete ? "(complete)" : "(partial)"}`;
  } else {
    elements.manualModeChip.textContent = "idle";
    elements.manualModeChip.className = "chip chip-idle";
    elements.manualStats.textContent = "Manual path not started.";
  }

  elements.manualToggleButton.textContent = state.manualActive
    ? "Pause manual input"
    : state.manualPath
      ? "Resume manual input"
      : "Start manual path";

  syncManualControls();
}

function formatDelta(value) {
  return `${value >= 0 ? "+" : ""}${value}`;
}

function renderStepControls() {
  const maxStep = state.displayedSolution ? state.displayedSolution.moveCount : 0;
  elements.stepSlider.max = String(maxStep);
  elements.stepSlider.value = String(Math.min(state.currentStep, maxStep));
  elements.stepLabel.textContent = `Step ${Math.min(state.currentStep, maxStep)} / ${maxStep}`;
  syncPlaybackControls();
}

function render() {
  renderStats();
  renderManualControls();
  renderStepControls();
  drawCanvas();
  syncSolverControls();
}

function drawCanvas() {
  const canvas = elements.gridCanvas;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!state.gridData) {
    return;
  }

  const grid = state.gridData.grid;
  const rows = grid.length;
  const cols = grid[0].length;
  const maxGridPixels = 680;
  const cellSize = Math.max(
    24,
    Math.min(64, Math.floor(maxGridPixels / Math.max(rows, cols)))
  );
  const margin = 28;
  const width = cols * cellSize + margin * 2;
  const height = rows * cellSize + margin * 2;
  canvas.width = width;
  canvas.height = height;

  const bbox = (row, col) => {
    const x0 = margin + col * cellSize;
    const y0 = margin + row * cellSize;
    return [x0, y0, x0 + cellSize, y0 + cellSize];
  };

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const [x0, y0, x1, y1] = bbox(row, col);
      ctx.fillStyle = grid[row][col] === 1 ? "#ffffff" : "#334155";
      ctx.strokeStyle = grid[row][col] === 1 ? "#cbd5e1" : "#1e293b";
      ctx.lineWidth = 1;
      ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    }
  }

  const start = state.gridData.start;
  let prefixPoints = [];
  let visitCounts = new Map();
  let currentPoint = null;

  if (state.displayedSolution) {
    const maxStep = state.displayedSolution.moveCount;
    state.currentStep = Math.max(0, Math.min(state.currentStep, maxStep));
    prefixPoints = state.displayedSolution.points.slice(0, state.currentStep + 1);
    visitCounts = countVisits(prefixPoints);
    currentPoint = prefixPoints[prefixPoints.length - 1] || null;

    for (const [key, count] of visitCounts.entries()) {
      const [row, col] = key.split(",").map(Number);
      if (grid[row]?.[col] !== 1) {
        continue;
      }
      const [x0, y0, x1, y1] = bbox(row, col);
      ctx.fillStyle = count === 1 ? "#dcfce7" : "#fecaca";
      ctx.fillRect(x0 + 4, y0 + 4, x1 - x0 - 8, y1 - y0 - 8);
    }

    for (let index = 1; index < prefixPoints.length; index += 1) {
      const [prevRow, prevCol] = prefixPoints[index - 1];
      const [nextRow, nextCol] = prefixPoints[index];
      const [px0, py0, px1, py1] = bbox(prevRow, prevCol);
      const [nx0, ny0, nx1, ny1] = bbox(nextRow, nextCol);
      drawArrow(
        ctx,
        (px0 + px1) / 2,
        (py0 + py1) / 2,
        (nx0 + nx1) / 2,
        (ny0 + ny1) / 2,
        "#0f766e"
      );
    }

    for (const [key, count] of visitCounts.entries()) {
      if (count <= 1) {
        continue;
      }
      const [row, col] = key.split(",").map(Number);
      if (grid[row]?.[col] !== 1) {
        continue;
      }
      const [x0, y0, x1, y1] = bbox(row, col);
      const cx = (x0 + x1) / 2;
      const cy = (y0 + y1) / 2;
      const radius = cellSize * 0.16;
      ctx.fillStyle = "#be123c";
      ctx.strokeStyle = "#881337";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 11px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(count), cx, cy);
    }
  }

  if (start) {
    const [x0, y0, x1, y1] = bbox(start[0], start[1]);
    ctx.strokeStyle = "#16a34a";
    ctx.lineWidth = 3;
    ctx.strokeRect(x0 + 5, y0 + 5, x1 - x0 - 10, y1 - y0 - 10);
    ctx.fillStyle = "#15803d";
    ctx.font = "bold 12px Inter, system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText("S", x0 + 11, y0 + 9);
  }

  if (currentPoint) {
    const [x0, y0, x1, y1] = bbox(currentPoint[0], currentPoint[1]);
    const cx = (x0 + x1) / 2;
    const cy = (y0 + y1) / 2;
    const radius = cellSize * 0.18;
    ctx.fillStyle = "#2563eb";
    ctx.strokeStyle = "#1d4ed8";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    if (
      state.displayedSolution &&
      state.displayedSolution.moveCount > 0 &&
      state.currentStep >= state.displayedSolution.moveCount
    ) {
      ctx.strokeStyle = "#dc2626";
      ctx.lineWidth = 3;
      ctx.strokeRect(x0 + 5, y0 + 5, x1 - x0 - 10, y1 - y0 - 10);
      ctx.fillStyle = "#b91c1c";
      ctx.font = "bold 12px Inter, system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText("E", x0 + 11, y0 + 9);
    }
  }
}

function countVisits(points) {
  const counts = new Map();
  for (const point of points) {
    const key = point.join(",");
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  return counts;
}

function drawArrow(ctx, startX, startY, endX, endY, color) {
  const headLength = 10;
  const angle = Math.atan2(endY - startY, endX - startX);
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(startX, startY);
  ctx.lineTo(endX, endY);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(endX, endY);
  ctx.lineTo(
    endX - headLength * Math.cos(angle - Math.PI / 6),
    endY - headLength * Math.sin(angle - Math.PI / 6)
  );
  ctx.lineTo(
    endX - headLength * Math.cos(angle + Math.PI / 6),
    endY - headLength * Math.sin(angle + Math.PI / 6)
  );
  ctx.closePath();
  ctx.fill();
}

function stopPlayback() {
  if (state.playTimer !== null) {
    window.clearTimeout(state.playTimer);
    state.playTimer = null;
  }
  elements.playButton.textContent = "Play";
}

function playNextFrame() {
  if (!state.displayedSolution) {
    stopPlayback();
    return;
  }

  if (state.currentStep >= state.displayedSolution.moveCount) {
    stopPlayback();
    render();
    return;
  }

  state.currentStep += 1;
  render();
  state.playTimer = window.setTimeout(playNextFrame, 140);
}

function togglePlayback() {
  if (!state.displayedSolution) {
    return;
  }
  if (state.playTimer !== null) {
    stopPlayback();
    return;
  }
  if (state.currentStep >= state.displayedSolution.moveCount) {
    state.currentStep = 0;
  }
  elements.playButton.textContent = "Pause";
  playNextFrame();
}

function stepBack() {
  if (!state.displayedSolution) {
    return;
  }
  stopPlayback();
  state.currentStep = Math.max(0, state.currentStep - 1);
  render();
}

function stepForward() {
  if (!state.displayedSolution) {
    return;
  }
  stopPlayback();
  state.currentStep = Math.min(
    state.displayedSolution.moveCount,
    state.currentStep + 1
  );
  render();
}

function stepToFull() {
  if (!state.displayedSolution) {
    return;
  }
  stopPlayback();
  state.currentStep = state.displayedSolution.moveCount;
  render();
}

elements.randomizeSeedButton.addEventListener("click", randomizeSeed);
elements.generateButton.addEventListener("click", generateGrid);
elements.solveButton.addEventListener("click", solveDisplayed);
elements.optimalButton.addEventListener("click", loadOptimalReference);
elements.manualToggleButton.addEventListener("click", toggleManualInput);
elements.resetManualButton.addEventListener("click", resetManualPath);
elements.undoManualButton.addEventListener("click", () => undoManualMove());
elements.showManualButton.addEventListener("click", () => showManualPath());
elements.solverSelect.addEventListener("change", syncSolverControls);
elements.prevButton.addEventListener("click", stepBack);
elements.playButton.addEventListener("click", togglePlayback);
elements.nextButton.addEventListener("click", stepForward);
elements.fullButton.addEventListener("click", stepToFull);
elements.stepSlider.addEventListener("input", (event) => {
  if (!state.displayedSolution) {
    return;
  }
  stopPlayback();
  state.currentStep = Number(event.target.value);
  render();
});
window.addEventListener("keydown", handleManualKeydown);

generateGrid();
