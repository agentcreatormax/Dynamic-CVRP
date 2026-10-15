"""
Robust parsers and writers for CVRPLIB .vrp and .sol file formats.
"""

import re
import os


def parse_vrp(filepath):
    """
    Parse a CVRPLIB/TSPLIB-style .vrp file.

    Returns a dict with keys:
        name, comment, type, dimension, capacity, edge_weight_type,
        coords  : {node_id(int): (x(float), y(float))}
        demands : {node_id(int): demand(int)}
        depot   : int (node_id of the depot)
        errors  : list of warning/error strings (empty if clean)
    """
    result = {
        "name": "",
        "comment": "",
        "type": "CVRP",
        "dimension": 0,
        "capacity": 0,
        "edge_weight_type": "EUC_2D",
        "coords": {},
        "demands": {},
        "depot": 1,
        "errors": [],
    }

    if not os.path.isfile(filepath):
        result["errors"].append(f"VRP file not found: {filepath}")
        return result

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        result["errors"].append(f"Error reading VRP file: {e}")
        return result

    section = None  # Current section being parsed

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.upper() == "EOF":
            if line.upper() == "EOF":
                break
            continue

        # Detect header key-value pairs
        header_match = re.match(
            r"^(NAME|COMMENT|TYPE|DIMENSION|CAPACITY|EDGE_WEIGHT_TYPE)\s*:\s*(.*)",
            line,
            re.IGNORECASE,
        )
        if header_match:
            key = header_match.group(1).upper()
            val = header_match.group(2).strip()
            if key == "NAME":
                result["name"] = val
            elif key == "COMMENT":
                result["comment"] = val
            elif key == "TYPE":
                result["type"] = val
            elif key == "DIMENSION":
                try:
                    result["dimension"] = int(val)
                except ValueError:
                    result["errors"].append(f"Invalid DIMENSION value: {val}")
            elif key == "CAPACITY":
                try:
                    result["capacity"] = int(val)
                except ValueError:
                    result["errors"].append(f"Invalid CAPACITY value: {val}")
            elif key == "EDGE_WEIGHT_TYPE":
                result["edge_weight_type"] = val
            section = None
            continue

        # Detect section headers
        section_match = re.match(
            r"^(NODE_COORD_SECTION|DEMAND_SECTION|DEPOT_SECTION)",
            line,
            re.IGNORECASE,
        )
        if section_match:
            section = section_match.group(1).upper()
            continue

        # Parse section data
        if section == "NODE_COORD_SECTION":
            parts = line.split()
            if len(parts) >= 3:
                try:
                    nid = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    result["coords"][nid] = (x, y)
                except ValueError:
                    result["errors"].append(f"Bad coord line: {line}")
            else:
                result["errors"].append(f"Incomplete coord line: {line}")

        elif section == "DEMAND_SECTION":
            parts = line.split()
            if len(parts) >= 2:
                try:
                    nid = int(parts[0])
                    dem = int(parts[1])
                    result["demands"][nid] = dem
                except ValueError:
                    result["errors"].append(f"Bad demand line: {line}")
            else:
                result["errors"].append(f"Incomplete demand line: {line}")

        elif section == "DEPOT_SECTION":
            parts = line.split()
            if parts:
                try:
                    val = int(parts[0])
                    if val >= 0:
                        result["depot"] = val
                except ValueError:
                    result["errors"].append(f"Bad depot line: {line}")

    # Fallback dimension
    if result["dimension"] == 0 and result["coords"]:
        result["dimension"] = len(result["coords"])

    return result


def parse_sol(filepath, depot_id=1, id_offset=None):
    """
    Parse a standard CVRPLIB .sol file (single solution).

    id_offset: how much to add to each node ID in the file to get the VRP node ID.
        - CVRPLIB format: id_offset = depot_id (customer IDs are 1..N-1, shifted by depot)
        - Raw-ID format (large instances, topten.sol etc.): id_offset = 0 (no shift)
        - Default (None): uses depot_id for backward compatibility with A-n32-k5 style files.

    Returns a dict with keys:
        routes : list of lists of VRP node IDs (ints)
        cost   : float or None
        errors : list of warning/error strings
    """
    if id_offset is None:
        id_offset = depot_id  # legacy: A-n32-k5 style

    result = {
        "routes": [],
        "cost": None,
        "errors": [],
    }

    if not os.path.isfile(filepath):
        result["errors"].append(f"SOL file not found: {filepath}")
        return result

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        result["errors"].append(f"Error reading SOL file: {e}")
        return result

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Match route lines:  Route #N: id1 id2 id3 ...
        route_match = re.match(r"^Route\s*#\d+\s*:\s*(.*)", line, re.IGNORECASE)
        if route_match:
            ids_str = route_match.group(1).strip()
            if ids_str:
                try:
                    route = [int(x) + id_offset for x in ids_str.split()]
                    result["routes"].append(route)
                except ValueError:
                    result["errors"].append(f"Bad route line: {line}")
            else:
                result["routes"].append([])
            continue

        # Match cost line:  Cost <number> (may be float like 1.12548e+06)
        cost_match = re.match(r"^Cost\s+([\d.e+\-]+)", line, re.IGNORECASE)
        if cost_match:
            try:
                result["cost"] = float(cost_match.group(1))
            except ValueError:
                pass
            continue

    return result


def parse_multi_sol(filepath, id_offset=0):
    """
    Parse a multi-solution file like topten.sol.

    Format:
        # comment line (optional, may contain the source .vrp path)
        [blank line]
        Solution #1 Cost <value>
        Route #1: id1 id2 ...
        ...
        [blank line]
        Solution #2 Cost <value>
        Route #1: id1 id2 ...
        ...

    id_offset: added to each route node ID to get the VRP node ID.
        Use 0 for large-instance raw-ID files (like topten.sol).
        Use depot_id for CVRPLIB-style customer-offset files.

    Returns a dict with keys:
        solutions : list of dicts, each with keys:
            index : int (1-based)
            cost  : float or None
            routes: list of lists of node IDs
        vrp_hint  : str or None  (path extracted from comment line, if present)
        errors    : list of warning/error strings
    """
    result = {
        "solutions": [],
        "vrp_hint": None,
        "errors": [],
    }

    if not os.path.isfile(filepath):
        result["errors"].append(f"Multi-sol file not found: {filepath}")
        return result

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        result["errors"].append(f"Error reading file: {e}")
        return result

    current_solution = None

    for raw_line in lines:
        line = raw_line.strip()

        # Comment line — may contain VRP file hint
        if line.startswith("#"):
            comment = line.lstrip("# ").strip()
            # Try to extract a .vrp path from the comment
            vrp_match = re.search(r"(\S+\.vrp)", comment, re.IGNORECASE)
            if vrp_match:
                result["vrp_hint"] = vrp_match.group(1)
            continue

        if not line:
            continue

        # Solution header:  Solution #N Cost <value>
        sol_match = re.match(
            r"^Solution\s*#(\d+)\s+Cost\s+([\d.e+\-]+)", line, re.IGNORECASE
        )
        if sol_match:
            current_solution = {
                "index": int(sol_match.group(1)),
                "cost": float(sol_match.group(2)),
                "routes": [],
            }
            result["solutions"].append(current_solution)
            continue

        # Route line
        route_match = re.match(r"^Route\s*#\d+\s*:\s*(.*)", line, re.IGNORECASE)
        if route_match and current_solution is not None:
            ids_str = route_match.group(1).strip()
            if ids_str:
                try:
                    route = [int(x) + id_offset for x in ids_str.split()]
                    current_solution["routes"].append(route)
                except ValueError:
                    result["errors"].append(f"Bad route line: {line[:60]}")
            continue

    if not result["solutions"]:
        result["errors"].append(
            "No 'Solution #N Cost X' headers found. "
            "Is this a multi-solution file? Use parse_sol() for single-solution files."
        )

    return result


def write_vrp(filepath, data):
    """
    Write VRP data back to a .vrp file in CVRPLIB format.

    data: dict with keys name, comment, type, dimension, capacity,
          edge_weight_type, coords, demands, depot
    """
    try:
        lines = []
        lines.append(f"NAME : {data.get('name', 'unknown')}")
        lines.append(f"COMMENT : {data.get('comment', '')}")
        lines.append(f"TYPE : {data.get('type', 'CVRP')}")
        lines.append(f"DIMENSION : {data.get('dimension', len(data.get('coords', {})))}")
        lines.append(f"EDGE_WEIGHT_TYPE : {data.get('edge_weight_type', 'EUC_2D')}")
        lines.append(f"CAPACITY : {data.get('capacity', 100)}")
        lines.append("NODE_COORD_SECTION")

        for nid in sorted(data["coords"].keys()):
            x, y = data["coords"][nid]
            # Use integer format if coordinates are whole numbers
            if x == int(x) and y == int(y):
                lines.append(f" {nid} {int(x)} {int(y)}")
            else:
                lines.append(f" {nid} {x} {y}")

        lines.append("DEMAND_SECTION")
        for nid in sorted(data["demands"].keys()):
            lines.append(f"{nid} {data['demands'][nid]}")

        lines.append("DEPOT_SECTION")
        lines.append(f" {data.get('depot', 1)}")
        lines.append(" -1")
        lines.append("EOF")
        lines.append("")

        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))

        return None  # Success
    except Exception as e:
        return str(e)


def write_sol(filepath, routes, cost, depot_id=1):
    """
    Write solution data back to a .sol file.

    routes: list of lists of vrp node IDs
    cost: int or None
    depot_id: the depot node ID (used to convert back to customer-only IDs)

    In the .sol file, customer IDs are vrp_node_id - depot_id.
    """
    try:
        lines = []
        for i, route in enumerate(routes, 1):
            # Convert vrp node IDs to customer-only IDs
            customer_ids = " ".join(str(nid - depot_id) for nid in route)
            lines.append(f"Route #{i}: {customer_ids}")

        if cost is not None:
            lines.append(f"Cost {cost}")

        lines.append("")

        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))

        return None  # Success
    except Exception as e:
        return str(e)


def compute_cost(coords, routes, depot_id):
    """Compute total Euclidean distance of all routes."""
    import math

    total = 0.0
    for route in routes:
        if not route:
            continue
        # depot → first customer
        full_path = [depot_id] + list(route) + [depot_id]
        for i in range(len(full_path) - 1):
            n1, n2 = full_path[i], full_path[i + 1]
            if n1 in coords and n2 in coords:
                dx = coords[n1][0] - coords[n2][0]
                dy = coords[n1][1] - coords[n2][1]
                total += math.sqrt(dx * dx + dy * dy)
    return round(total)
