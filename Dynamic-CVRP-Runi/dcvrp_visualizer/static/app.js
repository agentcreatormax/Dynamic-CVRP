(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const els = {
    status: $("statusBadge"),
    plot: $("routePlot"),
    legend: $("routeLegend"),
    selectionPill: $("selectionPill"),
    selectedTitle: $("selectedTitle"),
    selectedInfo: $("selectedInfo"),
    selectedDemand: $("selectedDemandInput"),
    updateDemandButton: $("updateDemandButton"),
    deleteButton: $("deleteButton"),
    previousRoutesButton: $("previousRoutesButton"),
    previousStateHint: $("previousStateHint"),
    addForm: $("addForm"),
    addButton: $("addButton"),
    x: $("xInput"),
    y: $("yInput"),
    demand: $("demandInput"),
    passes: $("passesInput"),
    refineK: $("refineKInput"),
    hmax: $("hmaxInput"),
    queueCap: $("queueCapInput"),
    refineButton: $("refineButton"),
    resetButton: $("resetButton"),
    cost: $("costValue"),
    costDelta: $("costDelta"),
    routes: $("routesValue"),
    customers: $("customersValue"),
    algo: $("algoValue"),
    wall: $("wallValue"),
    passesValue: $("passesValue"),
    profiler: $("profilerGrid"),
    routeTable: $("routeTable"),
    capacity: $("capacityChip"),
    lastAction: $("lastAction"),
    error: $("errorBox"),
  };

  const routeColors = [
    "#60a5fa", "#f97316", "#22c55e", "#e879f9",
    "#facc15", "#2dd4bf", "#fb7185", "#a78bfa",
    "#38bdf8", "#84cc16", "#f472b6", "#fb923c"
  ];

  let state = null;
  let selectedId = null;
  let busy = false;
  let plotTransform = null;
  let previousHover = false;
  let previousPinned = false;

  function fmt(v, digits = 3) {
    if (v === null || v === undefined || v === "") return "—";
    const n = Number(v);
    if (!Number.isFinite(n)) return String(v);
    if (Math.abs(n) >= 100000) return n.toLocaleString(undefined, { maximumFractionDigits: 1 });
    if (Math.abs(n) >= 100) return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
    return n.toLocaleString(undefined, { maximumFractionDigits: digits });
  }

  function setStatus(text, kind = "") {
    els.status.textContent = text;
    els.status.className = "status-badge" + (kind ? ` ${kind}` : "");
  }

  function setBusy(flag, text = "GPU running…") {
    busy = flag;
    els.addButton.disabled = flag;
    els.refineButton.disabled = flag;
    els.resetButton.disabled = flag;
    els.deleteButton.disabled = flag || selectedId === null;
    els.updateDemandButton.disabled = flag || selectedId === null;
    els.selectedDemand.disabled = flag || selectedId === null;
    if (flag) setStatus(text, "busy");
    else setStatus("Ready", "ok");
  }

  function showError(message) {
    if (!message) {
      els.error.hidden = true;
      els.error.textContent = "";
      return;
    }
    els.error.hidden = false;
    els.error.textContent = message;
    setStatus("Error", "bad");
  }

  function params() {
    return {
      passes: Number(els.passes.value),
      refine_k: Number(els.refineK.value),
      hmax: Number(els.hmax.value),
      queue_cap: Number(els.queueCap.value),
    };
  }

  async function api(path, body = null) {
    const options = { method: body ? "POST" : "GET", headers: {} };
    if (body) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(body);
    }

    const res = await fetch(path, options);
    let data;
    try {
      data = await res.json();
    } catch {
      throw new Error(`HTTP ${res.status}: invalid JSON response`);
    }
    if (!res.ok || !data.ok) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }
    return data;
  }

  function svgEl(name, attrs = {}) {
    const e = document.createElementNS("http://www.w3.org/2000/svg", name);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
    return e;
  }

  function starPoints(cx, cy, outer = 13, inner = 6, arms = 5) {
    const pts = [];
    for (let i = 0; i < arms * 2; i++) {
      const r = i % 2 === 0 ? outer : inner;
      const a = -Math.PI / 2 + (i * Math.PI) / arms;
      pts.push(`${cx + Math.cos(a) * r},${cy + Math.sin(a) * r}`);
    }
    return pts.join(" ");
  }

  function previousState() {
    return state?.last?.previous_state || null;
  }

  function previousVisible() {
    return Boolean(previousState()) && (previousHover || previousPinned);
  }

  function findPreviousCustomer(c) {
    const prev = previousState();
    if (!prev || !c) return null;
    const eps = 1e-9;
    return prev.customers.find(p =>
      Math.abs(Number(p.x) - Number(c.x)) <= eps &&
      Math.abs(Number(p.y) - Number(c.y)) <= eps
    ) || null;
  }

  function getBounds() {
    const xs = [state.depot.x, ...state.customers.map(c => c.x)];
    const ys = [state.depot.y, ...state.customers.map(c => c.y)];

    // Keep the plot scale stable when the previous-state overlay is shown.
    const prev = previousState();
    if (prev) {
      xs.push(prev.depot.x, ...prev.customers.map(c => c.x));
      ys.push(prev.depot.y, ...prev.customers.map(c => c.y));
    }

    let minX = Math.min(...xs), maxX = Math.max(...xs);
    let minY = Math.min(...ys), maxY = Math.max(...ys);

    if (maxX - minX < 1) { minX -= 1; maxX += 1; }
    if (maxY - minY < 1) { minY -= 1; maxY += 1; }

    const padx = (maxX - minX) * 0.12;
    const pady = (maxY - minY) * 0.12;
    minX -= padx; maxX += padx;
    minY -= pady; maxY += pady;

    return { minX, maxX, minY, maxY };
  }

  function renderPreviousControl() {
    const prev = previousState();
    const available = Boolean(prev);
    els.previousRoutesButton.disabled = busy || !available;
    els.previousRoutesButton.setAttribute("aria-pressed", previousPinned ? "true" : "false");

    if (!available) {
      els.previousRoutesButton.textContent = "Previous routes";
      els.previousStateHint.hidden = true;
      return;
    }

    els.previousRoutesButton.textContent = previousPinned
      ? "Previous routes: pinned"
      : "Hover: previous routes";

    if (previousVisible()) {
      els.previousStateHint.hidden = false;
      els.previousStateHint.textContent =
        `Previous iteration · cost ${fmt(prev.cost, 2)} · ${prev.routes.length} routes`;
    } else {
      els.previousStateHint.hidden = true;
    }
  }

  function drawPlot() {
    els.plot.replaceChildren();
    if (!state) return;

    const W = 820, H = 580, M = 34;
    const b = getBounds();

    const sx = (x) => M + ((x - b.minX) / (b.maxX - b.minX)) * (W - 2 * M);
    const sy = (y) => H - M - ((y - b.minY) / (b.maxY - b.minY)) * (H - 2 * M);
    const invX = (px) => b.minX + ((px - M) / (W - 2 * M)) * (b.maxX - b.minX);
    const invY = (py) => b.minY + ((H - M - py) / (H - 2 * M)) * (b.maxY - b.minY);

    plotTransform = { sx, sy, invX, invY, W, H, M };

    const byId = new Map(state.customers.map(c => [c.id, c]));

    // Small coordinate labels.
    const minLabel = svgEl("text", { x: M, y: H - 10, class: "axis-label" });
    minLabel.textContent = `(${fmt(b.minX, 1)}, ${fmt(b.minY, 1)})`;
    els.plot.appendChild(minLabel);

    const maxLabel = svgEl("text", {
      x: W - M,
      y: 16,
      class: "axis-label",
      "text-anchor": "end"
    });
    maxLabel.textContent = `(${fmt(b.maxX, 1)}, ${fmt(b.maxY, 1)})`;
    els.plot.appendChild(maxLabel);

    // Previous iteration: dashed neutral routes beneath the current solution.
    if (previousVisible()) {
      const prev = previousState();
      const prevById = new Map(prev.customers.map(c => [c.id, c]));

      prev.routes.forEach((route) => {
        const pts = [{ x: prev.depot.x, y: prev.depot.y }];
        route.customers.forEach(cid => {
          const c = prevById.get(cid);
          if (c) pts.push(c);
        });
        pts.push({ x: prev.depot.x, y: prev.depot.y });

        const poly = svgEl("polyline", {
          points: pts.map(p => `${sx(p.x)},${sy(p.y)}`).join(" "),
          class: "previous-route-line",
        });
        els.plot.appendChild(poly);
      });

      // Hollow previous-customer markers make additions/deletions obvious.
      prev.customers.forEach(c => {
        const old = svgEl("circle", {
          cx: sx(c.x), cy: sy(c.y), r: 8,
          class: "previous-customer-node"
        });
        const title = svgEl("title");
        title.textContent =
          `Previous customer ${c.id} | demand ${c.demand} | route ${c.route ?? "—"}`;
        old.appendChild(title);
        els.plot.appendChild(old);
      });
    }

    // Current routes.
    state.routes.forEach((route, idx) => {
      const pts = [{ x: state.depot.x, y: state.depot.y }];
      route.customers.forEach(cid => {
        const c = byId.get(cid);
        if (c) pts.push(c);
      });
      pts.push({ x: state.depot.x, y: state.depot.y });

      const poly = svgEl("polyline", {
        points: pts.map(p => `${sx(p.x)},${sy(p.y)}`).join(" "),
        class: "route-line",
        stroke: routeColors[idx % routeColors.length],
      });
      els.plot.appendChild(poly);
    });

    // Depot.
    const dx = sx(state.depot.x), dy = sy(state.depot.y);
    const star = svgEl("polygon", {
      points: starPoints(dx, dy),
      class: "depot-star"
    });
    const starTitle = svgEl("title");
    starTitle.textContent = `Depot (${state.depot.x}, ${state.depot.y})`;
    star.appendChild(starTitle);
    els.plot.appendChild(star);

    const dlabel = svgEl("text", {
      x: dx + 16,
      y: dy - 13,
      class: "depot-label"
    });
    dlabel.textContent = "Depot";
    els.plot.appendChild(dlabel);

    // Current customers.
    state.customers.forEach(c => {
      const routeIdx = c.route ? c.route - 1 : 0;
      const g = svgEl("g", {
        class: "customer-node" + (selectedId === c.id ? " selected" : ""),
        "data-id": c.id,
        transform: `translate(${sx(c.x)} ${sy(c.y)})`
      });

      const circle = svgEl("circle", {
        r: selectedId === c.id ? 13 : 11,
        fill: routeColors[routeIdx % routeColors.length]
      });
      const title = svgEl("title");
      title.textContent =
        `Customer ${c.id} | (${c.x}, ${c.y}) | demand ${c.demand}` +
        (c.route ? ` | route ${c.route}` : "");
      circle.appendChild(title);

      const text = svgEl("text");
      text.textContent = c.id;

      g.appendChild(circle);
      g.appendChild(text);

      g.addEventListener("click", (ev) => {
        ev.stopPropagation();
        selectCustomer(c.id);
      });

      els.plot.appendChild(g);
    });

    drawPreview();
    renderPreviousControl();
  }

  function drawPreview() {
    if (!plotTransform || !state) return;
    els.plot.querySelectorAll(".preview-point").forEach(n => n.remove());

    const x = Number(els.x.value);
    const y = Number(els.y.value);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;

    const { sx, sy } = plotTransform;
    const px = sx(x), py = sy(y);
    const c1 = svgEl("circle", { cx: px, cy: py, r: 9, class: "preview-point" });
    const l1 = svgEl("line", { x1: px - 15, y1: py, x2: px + 15, y2: py, class: "preview-point" });
    const l2 = svgEl("line", { x1: px, y1: py - 15, x2: px, y2: py + 15, class: "preview-point" });
    els.plot.append(c1, l1, l2);
  }

  function selectCustomer(id) {
    selectedId = selectedId === id ? null : id;
    updateSelection();
    drawPlot();
  }

  function updateSelection() {
    const c = state?.customers.find(x => x.id === selectedId);

    if (!c) {
      selectedId = null;
      els.selectionPill.textContent = "No customer selected";
      els.selectedTitle.textContent = "Customer controls";
      els.selectedInfo.textContent = "Select a point on the plot.";
      els.selectedDemand.value = "";
      els.selectedDemand.disabled = true;
      els.updateDemandButton.disabled = true;
      els.deleteButton.disabled = true;
      return;
    }

    const prev = findPreviousCustomer(c);
    let transition = "No matching customer in the previous iteration";
    if (prev) {
      const oldRoute = prev.route === null || prev.route === undefined ? "—" : `R${prev.route}`;
      const newRoute = c.route === null || c.route === undefined ? "—" : `R${c.route}`;
      const demandChanged = Number(prev.demand) !== Number(c.demand)
        ? ` · quantity ${fmt(prev.demand)} → ${fmt(c.demand)}`
        : "";
      transition = `previous ${oldRoute} → current ${newRoute}${demandChanged}`;
    }

    els.selectionPill.textContent = `Customer ${c.id} selected`;
    els.selectedTitle.textContent = `Customer ${c.id}`;
    els.selectedInfo.innerHTML =
      `x=${fmt(c.x)} · y=${fmt(c.y)} · demand=${fmt(c.demand)}<br>` +
      `route=${c.route ?? "—"} · position=${c.position ?? "—"}<br>` +
      `<span class="transition-text">${transition}</span>`;

    els.selectedDemand.value = c.demand;
    els.selectedDemand.disabled = busy;
    els.updateDemandButton.disabled = busy;
    els.deleteButton.disabled = busy;
  }

  function renderLegend() {
    els.legend.replaceChildren();
    if (!state) return;

    state.routes.forEach((r, idx) => {
      const item = document.createElement("span");
      item.className = "legend-item";

      const dot = document.createElement("span");
      dot.className = "legend-dot";
      dot.style.background = routeColors[idx % routeColors.length];

      const text = document.createElement("span");
      text.textContent = `R${r.index} · load ${fmt(r.load, 1)}/${fmt(state.capacity, 1)}`;

      item.append(dot, text);
      els.legend.appendChild(item);
    });
  }

  function renderRouteTable() {
    els.routeTable.replaceChildren();
    if (!state) return;

    state.routes.forEach((r, idx) => {
      const row = document.createElement("div");
      row.className = "route-row";

      const n = document.createElement("div");
      n.className = "route-name";
      n.style.color = routeColors[idx % routeColors.length];
      n.textContent = `R${r.index}`;

      const load = document.createElement("div");
      load.className = "route-load";
      load.textContent = `${fmt(r.load, 1)}/${fmt(state.capacity, 1)}`;

      const nodes = document.createElement("div");
      nodes.className = "route-nodes";
      nodes.textContent = r.customers.length
        ? `Depot → ${r.customers.join(" → ")} → Depot`
        : "empty";

      row.append(n, load, nodes);
      els.routeTable.appendChild(row);
    });
  }

  const profileLabels = {
    delete_total_ms: "DELETE total",
    add_total_ms: "ADD / repair total",
    scheduler_task_avg_us: "Scheduler / assignment",
    scheduler_total_ms: "Scheduler total (legacy)",
    worker_task_avg_us: "Worker / route task",
    avg_active_route_worker_warps: "Worker warps avg",
    max_active_route_worker_warps: "Worker warps max",
    avg_active_scheduler_warps: "Scheduler warps avg",
    max_active_scheduler_warps: "Scheduler warps max",
    scheduler_warps: "Scheduler width (legacy)",
    refine_total_ms: "Refinement total",
    move89_ms: "Move8/9",
    move1swap_ms: "Move1/SWAP*",
    move7_ms: "Move7",
  };

  function unitFor(key) {
    if (key.endsWith("_ms")) return " ms";
    if (key.endsWith("_us")) return " μs";
    return "";
  }

  function renderProfiler() {
    els.profiler.replaceChildren();

    const last = state?.last;
    const profile = last?.profile || {};
    els.lastAction.textContent = last?.description || "Initial state";

    const ordered = [
      "delete_total_ms",
      "add_total_ms",
      "scheduler_task_avg_us",
      "scheduler_total_ms",
      "worker_task_avg_us",
      "avg_active_route_worker_warps",
      "max_active_route_worker_warps",
      "avg_active_scheduler_warps",
      "max_active_scheduler_warps",
      "scheduler_warps",
      "refine_total_ms",
      "move89_ms",
      "move1swap_ms",
      "move7_ms",
    ];

    const shown = ordered.filter(k => Object.prototype.hasOwnProperty.call(profile, k));

    if (!shown.length) {
      const empty = document.createElement("div");
      empty.className = "selected-info";
      empty.textContent = "Run ADD, DELETE, or refinement to populate GPU profiling.";
      els.profiler.appendChild(empty);
      return;
    }

    shown.forEach(key => {
      const item = document.createElement("div");
      item.className = "profile-item";

      const label = document.createElement("span");
      label.textContent = profileLabels[key] || key;

      const value = document.createElement("strong");
      value.textContent = `${fmt(profile[key], 4)}${unitFor(key)}`;

      item.append(label, value);
      els.profiler.appendChild(item);
    });
  }

  function renderMetrics() {
    if (!state) return;

    els.cost.textContent = fmt(state.cost, 3);
    els.routes.textContent = String(state.routes.length);
    els.customers.textContent = `${state.customers.length} customers`;
    els.capacity.textContent = `Q = ${fmt(state.capacity, 2)}`;

    const last = state.last || {};
    const p = last.profile || {};

    if (last.cost_change === null || last.cost_change === undefined) {
      els.costDelta.textContent = "Current solution";
    } else {
      const d = Number(last.cost_change);
      const sign = d > 0 ? "+" : "";
      els.costDelta.textContent = `${last.action || "last"}: ${sign}${fmt(d, 3)}`;
    }

    if (p.algorithm_time_ms !== undefined) {
      els.algo.textContent = `${fmt(p.algorithm_time_ms, 4)} ms`;
    } else {
      els.algo.textContent = "—";
    }

    if (last.wall_time_ms !== undefined) {
      els.wall.textContent = `Wall ${fmt(last.wall_time_ms, 2)} ms`;
    } else {
      els.wall.textContent = "Wall —";
    }

    if (last.passes_run !== null && last.passes_run !== undefined) {
      els.passesValue.textContent =
        `${last.passes_run}/${last.passes_requested ?? "—"}`;
    } else {
      els.passesValue.textContent =
        last.passes_requested !== null && last.passes_requested !== undefined
          ? `—/${last.passes_requested}`
          : "—";
    }
  }

  function applyState(next, firstLoad = false) {
    state = next;

    if (firstLoad && next.defaults) {
      els.passes.value = next.defaults.passes;
      els.refineK.value = next.defaults.refine_k;
      els.hmax.value = next.defaults.hmax;
      els.queueCap.value = next.defaults.queue_cap;
    }

    if (selectedId !== null && !state.customers.some(c => c.id === selectedId)) {
      selectedId = null;
    }
    if (!previousState()) {
      previousPinned = false;
      previousHover = false;
    }

    renderMetrics();
    updateSelection();
    drawPlot();
    renderLegend();
    renderProfiler();
    renderRouteTable();
  }

  async function loadState() {
    showError("");
    setBusy(true, "Connecting…");
    try {
      const data = await api("/api/state");
      applyState(data, true);
      setBusy(false);
    } catch (err) {
      setBusy(false);
      showError(err.message);
    }
  }

  async function doOperation(path, body, label) {
    if (busy) return;
    showError("");
    setBusy(true, label);
    try {
      const data = await api(path, body);
      selectedId = null;
      applyState(data);
      setBusy(false);
    } catch (err) {
      setBusy(false);
      showError(err.message);
    }
  }

  els.addForm.addEventListener("submit", (ev) => {
    ev.preventDefault();
    doOperation("/api/add", {
      ...params(),
      x: Number(els.x.value),
      y: Number(els.y.value),
      demand: Number(els.demand.value),
    }, "Adding customer…");
  });

  els.deleteButton.addEventListener("click", () => {
    if (selectedId === null) return;
    doOperation("/api/delete", {
      ...params(),
      id: selectedId,
    }, `Deleting customer ${selectedId}…`);
  });

  els.updateDemandButton.addEventListener("click", () => {
    if (selectedId === null) return;
    const demand = Number(els.selectedDemand.value);
    if (!Number.isFinite(demand) || demand <= 0) {
      showError("Quantity / demand must be a positive number.");
      return;
    }
    doOperation("/api/update-demand", {
      ...params(),
      id: selectedId,
      demand,
    }, `Updating quantity for customer ${selectedId}…`);
  });

  els.refineButton.addEventListener("click", () => {
    doOperation("/api/refine", params(), "Refining routes…");
  });

  els.resetButton.addEventListener("click", () => {
    if (!window.confirm("Reset the interactive state back to the original toy benchmark?")) return;
    doOperation("/api/reset", {}, "Resetting…");
  });

  els.previousRoutesButton.addEventListener("mouseenter", () => {
    if (!previousState() || busy) return;
    previousHover = true;
    drawPlot();
  });

  els.previousRoutesButton.addEventListener("mouseleave", () => {
    previousHover = false;
    drawPlot();
  });

  els.previousRoutesButton.addEventListener("click", () => {
    if (!previousState() || busy) return;
    previousPinned = !previousPinned;
    drawPlot();
  });

  els.plot.addEventListener("click", (ev) => {
    if (!plotTransform || !state) return;

    const rect = els.plot.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width) * plotTransform.W;
    const py = ((ev.clientY - rect.top) / rect.height) * plotTransform.H;

    if (
      px < plotTransform.M || px > plotTransform.W - plotTransform.M ||
      py < plotTransform.M || py > plotTransform.H - plotTransform.M
    ) return;

    els.x.value = plotTransform.invX(px).toFixed(2);
    els.y.value = plotTransform.invY(py).toFixed(2);
    drawPreview();
  });

  els.x.addEventListener("input", drawPreview);
  els.y.addEventListener("input", drawPreview);

  loadState();
})();
