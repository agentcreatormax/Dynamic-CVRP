"""
Dash application for interactive VRP visualization and editing.
Supports both single-solution (.sol) and multi-solution files (topten.sol format).
"""

import threading
import time
from datetime import datetime

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import plotly.graph_objects as go

from parser import parse_vrp, parse_sol, parse_multi_sol, write_vrp, write_sol, compute_cost

# ── Qualitative color palette (15 distinct colors, cycled for routes) ─────────
ROUTE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]


# ── Shared state ─────────────────────────────────────────────────────────────
class AppState:
    """Thread-safe shared state between watchdog and Dash callbacks."""

    def __init__(self, vrp_path, sol_path):
        self.vrp_path = vrp_path
        self.sol_path = sol_path
        self.version = 0
        self.lock = threading.Lock()
        self.last_updated = datetime.now().strftime("%H:%M:%S")
        self.is_multi = False          # True when sol file has multiple solutions
        self.reload()

    def reload(self):
        with self.lock:
            self.vrp_data = parse_vrp(self.vrp_path)
            depot = self.vrp_data["depot"]

            # Auto-detect multi-solution file by checking for 'Solution #' headers
            raw = self._read_first_bytes(self.sol_path, 4096)
            if "Solution #" in raw or "solution #" in raw.lower():
                self.is_multi = True
                multi = parse_multi_sol(self.sol_path, id_offset=0)
                self.multi_data = multi
                # Also expose as single sol_data for backward compat (Solution #1)
                self.sol_data = multi["solutions"][0] if multi["solutions"] else {
                    "routes": [], "cost": None, "errors": []
                }
            else:
                self.is_multi = False
                self.sol_data = parse_sol(self.sol_path, depot_id=depot)
                self.multi_data = None

            self.version += 1
            self.last_updated = datetime.now().strftime("%H:%M:%S")

    def _read_first_bytes(self, path, n):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read(n)
        except Exception:
            return ""

    def bump(self):
        with self.lock:
            self.version += 1
            self.last_updated = datetime.now().strftime("%H:%M:%S")


# Module-level state, set by create_app()
_state: AppState = None  # type: ignore


# ── Figure builder ─────────────────────────────────────────────────────────
def build_figure(vrp, sol_data, max_labeled_nodes=200):
    """
    Build a Plotly figure from VRP + one solution's data.

    For large instances (many routes/nodes), labels are suppressed for performance.
    Hover tooltips are always available.
    """
    fig = go.Figure()
    coords = vrp["coords"]
    demands = vrp["demands"]
    depot = vrp["depot"]

    if not coords:
        fig.add_annotation(
            text="No node data available", showarrow=False,
            font=dict(size=20, color="red"), xref="paper", yref="paper", x=0.5, y=0.5
        )
        fig.update_layout(template="plotly_white")
        return fig

    routes = sol_data.get("routes", [])
    nodes_in_routes = set()
    large_instance = (len(routes) > 50 or len(coords) > 500)

    # ── Draw routes ───────────────────────────────────────────────────────
    for idx, route in enumerate(routes):
        color = ROUTE_COLORS[idx % len(ROUTE_COLORS)]
        route_label = f"Route #{idx + 1}"

        # Full path: depot → customers → depot
        path = [depot] + [nid for nid in route if nid in coords] + [depot]
        xs = [coords[nid][0] for nid in path]
        ys = [coords[nid][1] for nid in path]

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(color=color, width=1.5 if large_instance else 2.5),
            name=route_label,
            legendgroup=route_label,
            hoverinfo="skip",
            showlegend=(not large_instance),  # hide legend for huge instances
        ))

        # Customer dots + optional labels
        cust_nodes = [nid for nid in route if nid in coords]
        cust_xs = [coords[nid][0] for nid in cust_nodes]
        cust_ys = [coords[nid][1] for nid in cust_nodes]
        cust_hover = [
            f"Node {nid}<br>({coords[nid][0]}, {coords[nid][1]})<br>Demand: {demands.get(nid, '?')}"
            for nid in cust_nodes
        ]

        show_text = not large_instance
        if show_text:
            cust_ids = [str(nid - depot) for nid in cust_nodes]
        else:
            cust_ids = None

        fig.add_trace(go.Scatter(
            x=cust_xs, y=cust_ys,
            mode="markers+text" if show_text else "markers",
            marker=dict(size=5 if large_instance else 10,
                        color=color, line=dict(width=0.5, color="white")),
            text=cust_ids,
            textposition="top center" if show_text else None,
            textfont=dict(size=8, color="#333") if show_text else None,
            hovertext=cust_hover,
            hoverinfo="text",
            name=route_label,
            legendgroup=route_label,
            showlegend=False,
        ))

        nodes_in_routes.update(cust_nodes)

    # ── Unrouted nodes ────────────────────────────────────────────────────
    unrouted = [nid for nid in coords if nid != depot and nid not in nodes_in_routes]
    if unrouted:
        fig.add_trace(go.Scatter(
            x=[coords[nid][0] for nid in unrouted],
            y=[coords[nid][1] for nid in unrouted],
            mode="markers",
            marker=dict(size=4, color="#b0bec5", line=dict(width=0.5, color="#78909c")),
            hovertext=[
                f"Node {nid}<br>({coords[nid][0]}, {coords[nid][1]})"
                f"<br>Demand: {demands.get(nid, '?')} [UNROUTED]"
                for nid in unrouted
            ],
            hoverinfo="text",
            name="Unrouted",
            showlegend=len(unrouted) > 0,
        ))

    # ── Depot ─────────────────────────────────────────────────────────────
    if depot in coords:
        fig.add_trace(go.Scatter(
            x=[coords[depot][0]], y=[coords[depot][1]],
            mode="markers+text",
            marker=dict(size=14, color="#c62828", symbol="circle",
                        line=dict(width=2, color="#b71c1c")),
            text=["Depot"],
            textposition="bottom center",
            textfont=dict(size=10, color="#c62828", family="Arial Black"),
            hovertext=f"DEPOT (Node {depot})<br>({coords[depot][0]}, {coords[depot][1]})",
            hoverinfo="text",
            name="Depot",
        ))

    # ── Layout ────────────────────────────────────────────────────────────
    cost_val = sol_data.get("cost")
    if cost_val is not None:
        cost_str = f"{cost_val:,.0f}"
    else:
        cost_str = "N/A"

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=f"<b>{vrp.get('name', 'VRP')}</b>  |  {len(routes)} routes  |  Cost: {cost_str}",
            x=0.5, font=dict(size=15),
        ),
        xaxis=dict(showgrid=True, gridcolor="#eee", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#eee", zeroline=False,
                   scaleanchor="x", scaleratio=1),
        legend=dict(
            orientation="v", x=1.02, y=1, font=dict(size=9),
            bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc", borderwidth=1,
        ),
        margin=dict(l=40, r=180 if not large_instance else 20, t=50, b=40),
        plot_bgcolor="white",
        hovermode="closest",
        uirevision="same",  # preserve zoom/pan across data updates
    )
    return fig


# ── Layout helpers ────────────────────────────────────────────────────────────
def _sidebar_style():
    return {
        "width": "100%", "padding": "12px",
        "fontFamily": "'Segoe UI', Roboto, sans-serif",
        "fontSize": "13px", "color": "#333",
        "overflowY": "auto", "height": "100vh",
        "background": "linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%)",
        "borderLeft": "1px solid #dee2e6",
    }


def _card(title, children, color="#0d6efd"):
    return html.Div([
        html.H4(title, style={
            "margin": "0 0 8px 0", "padding": "6px 10px",
            "background": color, "color": "white",
            "borderRadius": "6px 6px 0 0", "fontSize": "13px",
        }),
        html.Div(children, style={"padding": "0 10px 10px 10px"}),
    ], style={
        "background": "white", "borderRadius": "6px",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.12)", "marginBottom": "12px",
    })


def _input_field(id_, placeholder, type_="text"):
    return dcc.Input(
        id=id_, placeholder=placeholder, type=type_,
        style={"width": "100%", "padding": "5px 7px", "marginBottom": "5px",
               "border": "1px solid #ced4da", "borderRadius": "4px", "fontSize": "12px"},
        debounce=True,
    )


def _button(id_, text, color="#0d6efd"):
    return html.Button(
        text, id=id_, n_clicks=0,
        style={"padding": "5px 11px", "background": color, "color": "white",
               "border": "none", "borderRadius": "4px", "cursor": "pointer",
               "fontSize": "12px", "marginRight": "5px", "marginTop": "3px"},
    )


def _meta_table(vrp, sol_data, last_updated):
    n_customers = max(0, len(vrp["coords"]) - (1 if vrp["depot"] in vrp["coords"] else 0))
    cost_val = sol_data.get("cost")
    cost_str = f"{cost_val:,.0f}" if cost_val is not None else "N/A"
    return [
        html.Table([
            html.Tr([html.Td("Name:", style={"fontWeight": "bold", "paddingRight": "8px"}),
                     html.Td(vrp.get("name") or "—")]),
            html.Tr([html.Td("Customers:"), html.Td(str(n_customers))]),
            html.Tr([html.Td("Vehicles:"), html.Td(str(len(sol_data.get("routes", []))))]),
            html.Tr([html.Td("Capacity:"), html.Td(str(vrp.get("capacity", "?")))]),
            html.Tr([html.Td("Cost:"), html.Td(cost_str)]),
        ], style={"width": "100%", "fontSize": "12px"}),
        html.Div(f"Last updated: {last_updated}",
                 style={"marginTop": "6px", "fontSize": "11px", "color": "#6c757d"}),
    ]


# ── Layout builders ───────────────────────────────────────────────────────────
def _build_single_layout():
    """Layout for single-solution mode (original A-n32-k5 style)."""
    return html.Div([
        dcc.Interval(id="poll-interval", interval=1500, n_intervals=0),
        dcc.Store(id="data-version", data=0),

        html.Div([
            # Graph
            html.Div([
                dcc.Graph(id="vrp-graph", style={"height": "100vh"},
                          config={"scrollZoom": True, "displayModeBar": True}),
            ], style={"flex": "3", "minWidth": "0"}),

            # Sidebar
            html.Div([
                html.Div(style=_sidebar_style(), children=[
                    _card("Instance Info", [
                        html.Div(id="meta-info"),
                    ], color="#495057"),
                    html.Div(id="status-msg", style={
                        "padding": "7px", "marginBottom": "8px", "borderRadius": "4px",
                        "fontSize": "12px", "whiteSpace": "pre-wrap",
                    }),
                    _card("Node Editor", [
                        _input_field("node-id-input", "Customer ID (as in .sol)", "number"),
                        _input_field("node-x-input", "X coordinate", "number"),
                        _input_field("node-y-input", "Y coordinate", "number"),
                        _input_field("node-demand-input", "Demand", "number"),
                        html.Div([
                            _button("add-node-btn", "Add Node", "#198754"),
                            _button("edit-node-btn", "Update Node", "#0d6efd"),
                            _button("del-node-btn", "Delete Node", "#dc3545"),
                        ]),
                    ], color="#198754"),
                    _card("Route Editor", [
                        _input_field("route-num-input", "Route # (1-based)", "number"),
                        _input_field("route-nodes-input", "Customer IDs (space-sep, as in .sol)"),
                        html.Div([
                            _button("add-route-btn", "Add Route", "#198754"),
                            _button("edit-route-btn", "Update Route", "#0d6efd"),
                            _button("del-route-btn", "Delete Route", "#dc3545"),
                        ]),
                    ], color="#0d6efd"),
                ]),
            ], style={"flex": "1", "minWidth": "280px", "maxWidth": "360px"}),
        ], style={"display": "flex", "height": "100vh"}),
    ])


def _build_multi_layout(n_solutions):
    """Layout for multi-solution mode: a tab per solution."""
    tabs = [
        dcc.Tab(label=f"Solution #{i + 1}", value=str(i),
                style={"padding": "6px 10px", "fontSize": "12px"},
                selected_style={"padding": "6px 10px", "fontSize": "12px",
                                "fontWeight": "bold", "borderTop": "3px solid #0d6efd"})
        for i in range(n_solutions)
    ]
    return html.Div([
        dcc.Interval(id="poll-interval", interval=2000, n_intervals=0),
        dcc.Store(id="data-version", data=0),

        # Header bar
        html.Div([
            html.Div(id="multi-header",
                     style={"padding": "8px 16px", "fontWeight": "bold",
                            "fontSize": "14px", "color": "#333"}),
        ], style={"background": "#f0f4f8", "borderBottom": "1px solid #dee2e6"}),

        # Tab strip
        dcc.Tabs(
            id="solution-tabs",
            value="0",
            children=tabs,
            style={"backgroundColor": "#f8f9fa"},
        ),

        # Graph area (fills remaining height)
        html.Div(id="tab-graph-container", style={"height": "calc(100vh - 90px)"}),

        # Hidden status
        html.Div(id="status-msg", style={"display": "none"}),
        html.Div(id="meta-info", style={"display": "none"}),
        # Dummy elements for single-mode callbacks (not used in multi mode)
        html.Div(id="vrp-graph", style={"display": "none"}),
        # Dummy inputs for unused callbacks
        html.Div([
            dcc.Input(id="node-id-input", style={"display": "none"}),
            dcc.Input(id="node-x-input", style={"display": "none"}),
            dcc.Input(id="node-y-input", style={"display": "none"}),
            dcc.Input(id="node-demand-input", style={"display": "none"}),
            dcc.Input(id="route-num-input", style={"display": "none"}),
            dcc.Input(id="route-nodes-input", style={"display": "none"}),
            html.Button(id="add-node-btn", style={"display": "none"}),
            html.Button(id="edit-node-btn", style={"display": "none"}),
            html.Button(id="del-node-btn", style={"display": "none"}),
            html.Button(id="add-route-btn", style={"display": "none"}),
            html.Button(id="edit-route-btn", style={"display": "none"}),
            html.Button(id="del-route-btn", style={"display": "none"}),
        ], style={"display": "none"}),
    ])


# ── Dash app factory ─────────────────────────────────────────────────────────
def create_app(state: AppState):
    global _state
    _state = state

    app = dash.Dash(
        __name__,
        title="VRP Visualizer",
        update_title=None,
        suppress_callback_exceptions=True,
    )

    if state.is_multi:
        n = len(state.multi_data["solutions"])
        app.layout = _build_multi_layout(n)
    else:
        app.layout = _build_single_layout()

    # ── Version polling ─────────────────────────────────────────────────
    @app.callback(
        Output("data-version", "data"),
        Input("poll-interval", "n_intervals"),
        State("data-version", "data"),
    )
    def check_version(n, current_ver):
        if _state.version != current_ver:
            return _state.version
        return no_update

    # ── Multi-solution: render selected tab's graph ──────────────────────
    if state.is_multi:
        @app.callback(
            Output("tab-graph-container", "children"),
            Output("multi-header", "children"),
            Input("solution-tabs", "value"),
            Input("data-version", "data"),
        )
        def update_multi_graph(tab_value, ver):
            with _state.lock:
                vrp = _state.vrp_data
                solutions = _state.multi_data.get("solutions", [])
                ts = _state.last_updated

            idx = int(tab_value) if tab_value is not None else 0
            if idx < 0 or idx >= len(solutions):
                return html.Div("No solution data.", style={"padding": "20px"}), ""

            sol = solutions[idx]
            fig = build_figure(vrp, sol)

            # Header text
            n_solutions = len(solutions)
            header = (
                f"{vrp.get('name', 'VRP')}  |  {n_solutions} solutions  |  "
                f"Capacity: {vrp.get('capacity', '?')}  |  Last updated: {ts}"
            )

            graph = dcc.Graph(
                figure=fig,
                style={"height": "100%"},
                config={"scrollZoom": True, "displayModeBar": True},
            )
            return graph, header

    # ── Single-solution: update graph + metadata ─────────────────────────
    @app.callback(
        Output("vrp-graph", "figure"),
        Output("meta-info", "children"),
        Output("status-msg", "children"),
        Output("status-msg", "style"),
        Input("data-version", "data"),
        prevent_initial_call=state.is_multi,  # skip in multi mode
    )
    def update_single_display(ver):
        if _state.is_multi:
            return no_update, no_update, no_update, no_update
        with _state.lock:
            vrp = _state.vrp_data
            sol = _state.sol_data
            ts = _state.last_updated

        fig = build_figure(vrp, sol)
        meta = _meta_table(vrp, sol, ts)

        all_errors = vrp.get("errors", []) + sol.get("errors", [])
        for route in sol["routes"]:
            for nid in route:
                if nid not in vrp["coords"]:
                    all_errors.append(f"Node {nid} in solution but not in VRP coords")
        if all_errors:
            status = html.Div([html.Div(e) for e in all_errors])
            style = {"padding": "7px", "marginBottom": "8px", "borderRadius": "4px",
                     "fontSize": "12px", "whiteSpace": "pre-wrap",
                     "background": "#fff3cd", "color": "#856404", "border": "1px solid #ffc107"}
        else:
            status, style = "", {"display": "none"}
        return fig, meta, status, style

    # ── Node editing (single-solution mode only) ─────────────────────────
    @app.callback(
        Output("data-version", "data", allow_duplicate=True),
        Output("status-msg", "children", allow_duplicate=True),
        Output("status-msg", "style", allow_duplicate=True),
        Input("add-node-btn", "n_clicks"),
        Input("edit-node-btn", "n_clicks"),
        Input("del-node-btn", "n_clicks"),
        State("node-id-input", "value"),
        State("node-x-input", "value"),
        State("node-y-input", "value"),
        State("node-demand-input", "value"),
        prevent_initial_call=True,
    )
    def handle_node_edit(add_c, edit_c, del_c, nid, x, y, demand):
        if _state.is_multi:
            return no_update, no_update, no_update
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update

        btn = ctx.triggered[0]["prop_id"].split(".")[0]
        err_style = {"padding": "7px", "marginBottom": "8px", "borderRadius": "4px",
                     "fontSize": "12px", "background": "#f8d7da", "color": "#721c24",
                     "border": "1px solid #f5c6cb"}
        ok_style = {"padding": "7px", "marginBottom": "8px", "borderRadius": "4px",
                    "fontSize": "12px", "background": "#d4edda", "color": "#155724",
                    "border": "1px solid #c3e6cb"}

        if btn == "del-node-btn":
            if nid is None:
                return no_update, "Enter a Customer ID to delete.", err_style
            cust_id = int(nid)
            with _state.lock:
                vrp = _state.vrp_data
                depot = vrp["depot"]
                vrp_nid = cust_id + depot
                if vrp_nid == depot:
                    return no_update, "Cannot delete the depot node.", err_style
                if vrp_nid not in vrp["coords"]:
                    return no_update, f"Customer {cust_id} does not exist.", err_style
                del vrp["coords"][vrp_nid]
                if vrp_nid in vrp["demands"]:
                    del vrp["demands"][vrp_nid]
                vrp["dimension"] = len(vrp["coords"])
                sol = _state.sol_data
                for route in sol["routes"]:
                    if vrp_nid in route:
                        route.remove(vrp_nid)
                err = write_vrp(_state.vrp_path, vrp)
                if err:
                    return no_update, f"File write error: {err}", err_style
                write_sol(_state.sol_path, sol["routes"], sol["cost"], depot)
                _state.bump()
            return _state.version, f"Customer {cust_id} deleted.", ok_style

        if nid is None or x is None or y is None:
            return no_update, "Customer ID, X, and Y are required.", err_style
        cust_id, x, y = int(nid), float(x), float(y)
        demand = int(demand) if demand is not None else 0

        if btn == "add-node-btn":
            with _state.lock:
                vrp = _state.vrp_data
                depot = vrp["depot"]
                vrp_nid = cust_id + depot
                if vrp_nid in vrp["coords"]:
                    return no_update, f"Customer {cust_id} already exists. Use Update.", err_style
                vrp["coords"][vrp_nid] = (x, y)
                vrp["demands"][vrp_nid] = demand
                vrp["dimension"] = len(vrp["coords"])
                err = write_vrp(_state.vrp_path, vrp)
                if err:
                    return no_update, f"File write error: {err}", err_style
                _state.bump()
            return _state.version, f"Customer {cust_id} added at ({x}, {y}).", ok_style

        if btn == "edit-node-btn":
            with _state.lock:
                vrp = _state.vrp_data
                depot = vrp["depot"]
                vrp_nid = cust_id + depot
                if vrp_nid not in vrp["coords"]:
                    return no_update, f"Customer {cust_id} does not exist. Use Add.", err_style
                vrp["coords"][vrp_nid] = (x, y)
                vrp["demands"][vrp_nid] = demand
                err = write_vrp(_state.vrp_path, vrp)
                if err:
                    return no_update, f"File write error: {err}", err_style
                _state.bump()
            return _state.version, f"Customer {cust_id} updated.", ok_style

        return no_update, no_update, no_update

    # ── Route editing (single-solution mode only) ────────────────────────
    @app.callback(
        Output("data-version", "data", allow_duplicate=True),
        Output("status-msg", "children", allow_duplicate=True),
        Output("status-msg", "style", allow_duplicate=True),
        Input("add-route-btn", "n_clicks"),
        Input("edit-route-btn", "n_clicks"),
        Input("del-route-btn", "n_clicks"),
        State("route-num-input", "value"),
        State("route-nodes-input", "value"),
        prevent_initial_call=True,
    )
    def handle_route_edit(add_c, edit_c, del_c, route_num, nodes_str):
        if _state.is_multi:
            return no_update, no_update, no_update
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update

        btn = ctx.triggered[0]["prop_id"].split(".")[0]
        err_style = {"padding": "7px", "marginBottom": "8px", "borderRadius": "4px",
                     "fontSize": "12px", "background": "#f8d7da", "color": "#721c24",
                     "border": "1px solid #f5c6cb"}
        ok_style = {"padding": "7px", "marginBottom": "8px", "borderRadius": "4px",
                    "fontSize": "12px", "background": "#d4edda", "color": "#155724",
                    "border": "1px solid #c3e6cb"}

        if btn == "del-route-btn":
            if route_num is None:
                return no_update, "Enter a route number to delete.", err_style
            ridx = int(route_num) - 1
            with _state.lock:
                sol = _state.sol_data
                if ridx < 0 or ridx >= len(sol["routes"]):
                    return no_update, f"Route #{route_num} does not exist.", err_style
                sol["routes"].pop(ridx)
                sol["cost"] = compute_cost(_state.vrp_data["coords"], sol["routes"],
                                           _state.vrp_data["depot"])
                write_sol(_state.sol_path, sol["routes"], sol["cost"],
                          _state.vrp_data["depot"])
                _state.bump()
            return _state.version, f"Route #{route_num} deleted.", ok_style

        if not nodes_str or not nodes_str.strip():
            return no_update, "Enter customer IDs (space-separated).", err_style
        try:
            cust_ids_input = [int(x) for x in nodes_str.strip().split()]
        except ValueError:
            return no_update, "Customer IDs must be integers.", err_style

        with _state.lock:
            vrp = _state.vrp_data
            sol = _state.sol_data
            depot = vrp["depot"]
            node_ids = [cid + depot for cid in cust_ids_input]

            for i, nid in enumerate(node_ids):
                if nid not in vrp["coords"]:
                    return no_update, f"Customer {cust_ids_input[i]} not found.", err_style
                if nid == depot:
                    return no_update, f"Customer {cust_ids_input[i]} maps to depot.", err_style

            route_demand = sum(vrp["demands"].get(nid, 0) for nid in node_ids)
            if vrp["capacity"] > 0 and route_demand > vrp["capacity"]:
                return (no_update,
                        f"Route demand ({route_demand}) exceeds capacity ({vrp['capacity']}).",
                        err_style)

            if btn == "add-route-btn":
                existing = {nid for r in sol["routes"] for nid in r}
                dups = set(node_ids) & existing
                if dups:
                    return no_update, f"Nodes {sorted(dups)} already in other routes.", err_style
                sol["routes"].append(node_ids)
                sol["cost"] = compute_cost(vrp["coords"], sol["routes"], depot)
                write_sol(_state.sol_path, sol["routes"], sol["cost"], depot)
                _state.bump()
                return _state.version, f"Route #{len(sol['routes'])} added.", ok_style

            if btn == "edit-route-btn":
                if route_num is None:
                    return no_update, "Enter a route number to edit.", err_style
                ridx = int(route_num) - 1
                if ridx < 0 or ridx >= len(sol["routes"]):
                    return no_update, f"Route #{route_num} does not exist.", err_style
                existing = {nid for i, r in enumerate(sol["routes"]) if i != ridx for nid in r}
                dups = set(node_ids) & existing
                if dups:
                    return no_update, f"Nodes {sorted(dups)} already in other routes.", err_style
                sol["routes"][ridx] = node_ids
                sol["cost"] = compute_cost(vrp["coords"], sol["routes"], depot)
                write_sol(_state.sol_path, sol["routes"], sol["cost"], depot)
                _state.bump()
                return _state.version, f"Route #{route_num} updated.", ok_style

        return no_update, no_update, no_update

    return app
