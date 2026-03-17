"""
WNTR-based pipe network generation.

Generates a realistic water distribution network with spatially correlated
pipe attributes, placed over a configurable city center. Uses WNTR for
hydraulic simulation to produce realistic pressures and flows.

The network avoids the unrealistic perfect-grid look by:
- Removing random edges to create dead-ends and irregular blocks
- Varying block sizes across the network
- Adding diagonal cross-connections
- Adding larger jitter for organic street patterns
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
import wntr

from pipe_leak.config import SIM_CONFIG, SimulationConfig


def create_grid_network(
    config: SimulationConfig | None = None,
) -> wntr.network.WaterNetworkModel:
    """
    Create a synthetic water network with realistic irregular topology.

    Instead of a perfect grid, this:
    - Uses variable spacing to mimic real city blocks
    - Removes ~18% of grid edges to create dead-ends and irregular blocks
    - Adds diagonal connector pipes for realism
    - Adds larger random jitter to node positions
    """
    cfg = config or SIM_CONFIG
    rng = np.random.default_rng(cfg.seed)

    wn = wntr.network.WaterNetworkModel()

    # Grid dimensions
    grid_side = int(np.ceil(np.sqrt(cfg.num_pipes / 2)))
    grid_side = max(grid_side, 5)

    # Variable spacing: some blocks are wider, some narrower
    base_spacing = 0.002  # ~200m
    row_spacings = base_spacing * rng.uniform(0.6, 1.5, size=grid_side)
    col_spacings = base_spacing * rng.uniform(0.6, 1.5, size=grid_side)

    # Cumulative offsets for each row/col
    row_offsets = np.concatenate([[0], np.cumsum(row_spacings[:-1])])
    col_offsets = np.concatenate([[0], np.cumsum(col_spacings[:-1])])
    # Center the grid
    row_offsets -= row_offsets.mean()
    col_offsets -= col_offsets.mean()

    # Create junction nodes with larger jitter
    node_coords = {}
    for row in range(grid_side):
        for col in range(grid_side):
            node_id = f"J-{row:03d}-{col:03d}"
            lat = cfg.center_lat + row_offsets[row]
            lon = cfg.center_lon + col_offsets[col]

            # Larger random jitter for organic look (±35% of base spacing)
            lat += rng.uniform(-0.0007, 0.0007)
            lon += rng.uniform(-0.0007, 0.0007)

            # Elevation varies smoothly
            base_elev = 15.0
            elev = base_elev + row * 0.3 + col * 0.1 + rng.normal(0, 0.8)

            wn.add_junction(node_id, base_demand=0.003, elevation=elev)
            node_coords[node_id] = (lon, lat)

    # Add reservoir and tank
    res_lat = cfg.center_lat + row_offsets[0] - base_spacing * 2
    res_lon = cfg.center_lon + col_offsets[0] - base_spacing * 2
    wn.add_reservoir("R-001", base_head=60.0)
    node_coords["R-001"] = (res_lon, res_lat)

    tank_lat = cfg.center_lat + row_offsets[-1] + base_spacing * 2
    tank_lon = cfg.center_lon + col_offsets[-1] + base_spacing * 2
    wn.add_tank("T-001", elevation=30.0, init_level=5.0, max_level=10.0, diameter=15.0)
    node_coords["T-001"] = (tank_lon, tank_lat)

    # Connect reservoir and tank
    corner_00 = "J-000-000"
    corner_nn = f"J-{grid_side - 1:03d}-{grid_side - 1:03d}"
    wn.add_pipe("P-RES-IN", "R-001", corner_00, length=300, diameter=0.6, roughness=100)
    wn.add_pipe("P-TANK-IN", corner_nn, "T-001", length=300, diameter=0.5, roughness=100)

    # Collect all potential pipe edges, then selectively remove some
    pipe_edges = []

    for row in range(grid_side):
        for col in range(grid_side):
            current = f"J-{row:03d}-{col:03d}"

            # Horizontal pipe (to the right)
            if col < grid_side - 1:
                neighbor = f"J-{row:03d}-{col + 1:03d}"
                is_artery = (row % 5 == 0 or col % 5 == 0)
                pipe_edges.append((current, neighbor, is_artery, "horizontal"))

            # Vertical pipe (downward)
            if row < grid_side - 1:
                neighbor = f"J-{row + 1:03d}-{col:03d}"
                is_artery = (row % 5 == 0 or col % 5 == 0)
                pipe_edges.append((current, neighbor, is_artery, "vertical"))

    # Add diagonal connections (~5% of grid cells)
    for row in range(grid_side - 1):
        for col in range(grid_side - 1):
            if rng.random() < 0.05:
                current = f"J-{row:03d}-{col:03d}"
                diag = f"J-{row + 1:03d}-{col + 1:03d}"
                pipe_edges.append((current, diag, False, "diagonal"))

    # Remove random non-artery edges to create irregular topology
    # Never remove arteries (every 5th row/col) to keep the network connected
    # Remove ~18% of non-artery edges
    filtered_edges = []
    for start, end, is_artery, direction in pipe_edges:
        if is_artery:
            filtered_edges.append((start, end, is_artery, direction))
        else:
            if rng.random() > 0.18:
                filtered_edges.append((start, end, is_artery, direction))

    # Ensure connectivity: check that all nodes can be reached from the reservoir
    # by building an adjacency set and doing BFS
    adjacency = {}
    all_nodes = set(node_coords.keys())
    for start, end, _, _ in filtered_edges:
        adjacency.setdefault(start, set()).add(end)
        adjacency.setdefault(end, set()).add(start)
    # Add reservoir/tank connections
    adjacency.setdefault("R-001", set()).add(corner_00)
    adjacency.setdefault(corner_00, set()).add("R-001")
    adjacency.setdefault(corner_nn, set()).add("T-001")
    adjacency.setdefault("T-001", set()).add(corner_nn)

    # BFS from reservoir
    visited = set()
    queue = ["R-001"]
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        for neighbor in adjacency.get(node, []):
            if neighbor not in visited:
                queue.append(neighbor)

    # Any junction not visited needs to be reconnected
    # Add back edges to reconnect disconnected nodes
    disconnected = all_nodes - visited
    for node in disconnected:
        if not node.startswith("J-"):
            continue
        # Parse row/col
        parts = node.split("-")
        row, col = int(parts[1]), int(parts[2])
        # Try to connect to nearest visited neighbor
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < grid_side and 0 <= nc < grid_side:
                neighbor = f"J-{nr:03d}-{nc:03d}"
                if neighbor in visited:
                    is_artery = (row % 5 == 0 or col % 5 == 0)
                    filtered_edges.append((node, neighbor, is_artery, "reconnect"))
                    adjacency.setdefault(node, set()).add(neighbor)
                    adjacency.setdefault(neighbor, set()).add(node)
                    visited.add(node)
                    break

    # Now add all pipes
    pipe_count = 0
    for start, end, is_artery, direction in filtered_edges:
        pipe_id = f"P-{pipe_count:04d}"
        length = rng.uniform(120, 350)

        if is_artery:
            diameter = rng.choice([0.3, 0.4, 0.5])
        elif direction == "diagonal":
            diameter = rng.choice([0.1, 0.15])
        else:
            diameter = rng.choice([0.1, 0.15, 0.2])

        wn.add_pipe(pipe_id, start, end, length=length,
                     diameter=diameter, roughness=rng.uniform(80, 130))
        pipe_count += 1

    wn._node_coords = node_coords
    return wn


def _assign_spatially_correlated_attributes(
    pipes_df: pd.DataFrame, config: SimulationConfig, rng: np.random.Generator
) -> pd.DataFrame:
    """Assign pipe attributes with spatial correlation."""
    cfg = config
    n = len(pipes_df)

    # Divide network into spatial zones based on pipe midpoint latitude
    lat_mid = pipes_df["mid_lat"].values
    lat_min, lat_max = lat_mid.min(), lat_mid.max()
    n_zones = 5
    zone_edges = np.linspace(lat_min - 0.0001, lat_max + 0.0001, n_zones + 1)
    pipes_df["zone"] = np.digitize(lat_mid, zone_edges) - 1
    pipes_df["zone"] = pipes_df["zone"].clip(0, n_zones - 1)

    # Each zone gets a primary installation era (spatially correlated)
    era_keys = list(cfg.material_by_era.keys())
    zone_eras = rng.choice(len(era_keys), size=n_zones)

    # Assign installation year per pipe (within zone's era, with noise)
    install_years = np.zeros(n, dtype=int)
    materials = []
    for zone_id in range(n_zones):
        mask = pipes_df["zone"].values == zone_id
        era = era_keys[zone_eras[zone_id]]
        year_min, year_max = era
        zone_count = mask.sum()
        if zone_count == 0:
            continue
        install_years[mask] = rng.integers(year_min, year_max, size=zone_count)
        era_materials = cfg.material_by_era[era]
        materials.extend(rng.choice(era_materials, size=zone_count).tolist())

    pipes_df["installation_year"] = install_years
    pipes_df["material"] = materials
    pipes_df["age"] = 2026 - pipes_df["installation_year"]

    # Soil type — spatially correlated by zone
    soil_names = list(cfg.soil_zones.keys())
    zone_soils = rng.choice(soil_names, size=n_zones)
    soil_types = []
    soil_corrosivity = np.zeros(n)
    for zone_id in range(n_zones):
        mask = pipes_df["zone"].values == zone_id
        zone_count = mask.sum()
        if zone_count == 0:
            continue
        primary_soil = zone_soils[zone_id]
        soils = []
        for _ in range(zone_count):
            if rng.random() < 0.7:
                soils.append(primary_soil)
            else:
                soils.append(rng.choice(soil_names))
        soil_types.extend(soils)
        for i, idx in enumerate(np.where(mask)[0]):
            soil_corrosivity[idx] = cfg.soil_zones[soils[i]]["corrosivity"]

    pipes_df["soil_type"] = soil_types
    pipes_df["soil_corrosivity"] = soil_corrosivity

    # Diameter category
    def diameter_category(d_m):
        d_in = d_m * 39.37
        if d_in <= 6:
            return "small (2-6)"
        elif d_in <= 12:
            return "medium (8-12)"
        elif d_in <= 24:
            return "large (14-24)"
        else:
            return "very large (>24)"

    pipes_df["diameter_category"] = pipes_df["diameter_m"].apply(diameter_category)

    # Depth (feet) — correlated with diameter
    pipes_df["depth_ft"] = np.where(
        pipes_df["diameter_m"] > 0.3,
        rng.uniform(5, 8, size=n),
        rng.uniform(3, 5.5, size=n),
    )

    # Previous repairs — Poisson, rate depends on age
    pipes_df["prev_repairs"] = rng.poisson(
        np.maximum(0.1, pipes_df["age"].values / 25), size=n
    )

    # Last inspection (days ago)
    base_inspection = rng.integers(0, 365 * 3, size=n)
    pipes_df["last_inspection_days"] = np.where(
        pipes_df["age"] < 20, np.minimum(base_inspection, 365), base_inspection
    )

    # Traffic load
    pipes_df["traffic_load"] = np.where(
        pipes_df["diameter_m"] > 0.25,
        rng.integers(5, 11, size=n),
        rng.integers(1, 7, size=n),
    )

    return pipes_df


def run_hydraulic_simulation(
    wn: wntr.network.WaterNetworkModel,
) -> dict:
    """Run WNTR hydraulic simulation and extract per-pipe results."""
    wn.options.time.duration = 24 * 3600
    wn.options.time.hydraulic_timestep = 3600

    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    node_pressure = results.node["pressure"].mean()
    pipe_velocity = results.link["velocity"].mean()
    pipe_flowrate = results.link["flowrate"].mean()

    hydraulics = {}
    for pipe_name in wn.pipe_name_list:
        pipe = wn.get_link(pipe_name)
        start_node = pipe.start_node_name
        end_node = pipe.end_node_name

        p_start = node_pressure.get(start_node, 30.0)
        p_end = node_pressure.get(end_node, 30.0)

        hydraulics[pipe_name] = {
            "pressure_avg_m": float((p_start + p_end) / 2),
            "velocity_avg_ms": float(abs(pipe_velocity.get(pipe_name, 0.5))),
            "flowrate_avg_lps": float(abs(pipe_flowrate.get(pipe_name, 1.0))),
        }

    return hydraulics


def build_pipe_network(
    config: SimulationConfig | None = None,
) -> gpd.GeoDataFrame:
    """
    Build a complete pipe network GeoDataFrame with realistic attributes.

    This is the main entry point for network generation.
    """
    cfg = config or SIM_CONFIG
    rng = np.random.default_rng(cfg.seed)

    print(f"Creating realistic network (~{cfg.num_pipes} pipes)...")
    wn = create_grid_network(cfg)
    node_coords = wn._node_coords

    print("Running hydraulic simulation...")
    hydraulics = run_hydraulic_simulation(wn)

    # Build pipe records with geometry
    records = []
    for pipe_name in wn.pipe_name_list:
        pipe = wn.get_link(pipe_name)
        start = pipe.start_node_name
        end = pipe.end_node_name

        if start in node_coords and end in node_coords:
            lon1, lat1 = node_coords[start]
            lon2, lat2 = node_coords[end]
            geom = LineString([(lon1, lat1), (lon2, lat2)])
            mid_lat = (lat1 + lat2) / 2
            mid_lon = (lon1 + lon2) / 2
        else:
            mid_lat = cfg.center_lat
            mid_lon = cfg.center_lon
            geom = Point(mid_lon, mid_lat)

        hyd = hydraulics.get(pipe_name, {})
        records.append({
            "pipe_id": pipe_name,
            "start_node": start,
            "end_node": end,
            "length_m": pipe.length,
            "diameter_m": pipe.diameter,
            "roughness": pipe.roughness,
            "mid_lat": mid_lat,
            "mid_lon": mid_lon,
            "pressure_avg_m": hyd.get("pressure_avg_m", 30.0),
            "velocity_avg_ms": hyd.get("velocity_avg_ms", 0.5),
            "flowrate_avg_lps": hyd.get("flowrate_avg_lps", 1.0),
            "geometry": geom,
        })

    pipes_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")

    print("Assigning spatially correlated attributes...")
    pipes_gdf = _assign_spatially_correlated_attributes(pipes_gdf, cfg, rng)

    print(f"Network complete: {len(pipes_gdf)} pipes generated.")
    return pipes_gdf
