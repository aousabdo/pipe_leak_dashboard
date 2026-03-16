"""PyDeck map rendering helpers."""

import pandas as pd
import numpy as np
import pydeck as pdk

SEVERITY_COLOR_MAP = {
    "Minor": [56, 161, 105, 180],
    "Moderate": [221, 107, 32, 200],
    "Major": [229, 62, 62, 220],
    "Critical": [155, 44, 44, 255],
}


def pipe_network_map(
    pipes_df: pd.DataFrame,
    risk_scores: np.ndarray | None = None,
    center_lat: float | None = None,
    center_lon: float | None = None,
) -> pdk.Deck:
    """
    Create a pydeck map showing the pipe network as line segments.

    Pipes are colored by risk score (red=high, green=low) if provided,
    otherwise by material type.
    """
    if center_lat is None:
        center_lat = pipes_df["mid_lat"].mean()
    if center_lon is None:
        center_lon = pipes_df["mid_lon"].mean()

    # Build line data from pipe geometries
    lines = []
    for _, pipe in pipes_df.iterrows():
        geom = pipe.get("geometry")
        if geom is not None and hasattr(geom, "coords") and len(list(geom.coords)) >= 2:
            coords = list(geom.coords)
            start = list(coords[0])
            end = list(coords[-1])
        else:
            continue

        risk = 0.5
        if risk_scores is not None:
            idx = pipes_df.index.get_loc(pipe.name)
            if idx < len(risk_scores):
                risk = float(risk_scores[idx])

        # Color: green (low risk) -> red (high risk)
        r = int(min(255, risk * 510))
        g = int(min(255, (1 - risk) * 510))
        color = [r, g, 50, 180]

        lines.append({
            "start": start,
            "end": end,
            "color": color,
            "pipe_id": pipe.get("pipe_id", ""),
            "material": pipe.get("material", ""),
            "age": int(pipe.get("age", 0)),
            "risk": round(risk * 100, 1),
        })

    line_layer = pdk.Layer(
        "LineLayer",
        data=lines,
        get_source_position="start",
        get_target_position="end",
        get_color="color",
        get_width=3,
        pickable=True,
        auto_highlight=True,
    )

    tooltip = {
        "html": (
            "<b>Pipe:</b> {pipe_id}<br>"
            "<b>Material:</b> {material}<br>"
            "<b>Age:</b> {age} years<br>"
            "<b>Risk:</b> {risk}%"
        ),
        "style": {"backgroundColor": "#1a365d", "color": "white", "fontSize": "12px"},
    }

    view = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=13,
        pitch=0,
    )

    return pdk.Deck(
        layers=[line_layer],
        initial_view_state=view,
        tooltip=tooltip,
        map_style="light",
    )


def leak_events_map(
    events_df: pd.DataFrame,
    center_lat: float | None = None,
    center_lon: float | None = None,
) -> pdk.Deck:
    """Create a pydeck map showing leak event locations colored by severity."""
    if events_df is None or events_df.empty:
        return _empty_map(center_lat or 38.58, center_lon or -121.49)

    if center_lat is None:
        center_lat = events_df["latitude"].mean()
    if center_lon is None:
        center_lon = events_df["longitude"].mean()

    # Add color column
    df = events_df.copy()
    df["color"] = df["severity"].map(SEVERITY_COLOR_MAP)
    df["color"] = df["color"].apply(lambda x: x if isinstance(x, list) else [100, 100, 100, 180])
    df["radius"] = df["severity"].map(
        {"Minor": 40, "Moderate": 60, "Major": 90, "Critical": 130}
    ).fillna(50)

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["longitude", "latitude"],
        get_radius="radius",
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )

    # Heatmap layer
    heat_layer = pdk.Layer(
        "HeatmapLayer",
        data=df,
        get_position=["longitude", "latitude"],
        get_weight="flow_rate_gpm",
        radiusPixels=40,
        opacity=0.5,
    )

    tooltip = {
        "html": (
            "<b>Pipe:</b> {pipe_id}<br>"
            "<b>Date:</b> {date}<br>"
            "<b>Severity:</b> {severity}<br>"
            "<b>Flow:</b> {flow_rate_gpm} GPM<br>"
            "<b>Cost:</b> ${repair_cost}"
        ),
        "style": {"backgroundColor": "#1a365d", "color": "white", "fontSize": "12px"},
    }

    view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=13, pitch=0)

    return pdk.Deck(
        layers=[heat_layer, scatter_layer],
        initial_view_state=view,
        tooltip=tooltip,
        map_style="light",
    )


def risk_heatmap(
    pipes_df: pd.DataFrame,
    risk_scores: np.ndarray,
    center_lat: float | None = None,
    center_lon: float | None = None,
) -> pdk.Deck:
    """Create a heatmap of predicted leak risk across the network."""
    if center_lat is None:
        center_lat = pipes_df["mid_lat"].mean()
    if center_lon is None:
        center_lon = pipes_df["mid_lon"].mean()

    df = pipes_df[["pipe_id", "mid_lat", "mid_lon", "material", "age"]].copy()
    df["risk"] = risk_scores
    df["risk_pct"] = (risk_scores * 100).round(1)

    heat_layer = pdk.Layer(
        "HeatmapLayer",
        data=df,
        get_position=["mid_lon", "mid_lat"],
        get_weight="risk",
        radiusPixels=50,
        opacity=0.6,
    )

    # High-risk points as scatter overlay
    high_risk = df[df["risk"] > 0.5].copy()
    high_risk["color"] = [[229, 62, 62, 200]] * len(high_risk)

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=high_risk,
        get_position=["mid_lon", "mid_lat"],
        get_radius=50,
        get_fill_color="color",
        pickable=True,
    )

    tooltip = {
        "html": (
            "<b>Pipe:</b> {pipe_id}<br>"
            "<b>Material:</b> {material}<br>"
            "<b>Age:</b> {age} years<br>"
            "<b>Risk:</b> {risk_pct}%"
        ),
        "style": {"backgroundColor": "#1a365d", "color": "white", "fontSize": "12px"},
    }

    view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=13, pitch=0)

    return pdk.Deck(
        layers=[heat_layer, scatter_layer],
        initial_view_state=view,
        tooltip=tooltip,
        map_style="light",
    )


def _empty_map(lat: float, lon: float) -> pdk.Deck:
    """Return an empty map centered at the given location."""
    view = pdk.ViewState(latitude=lat, longitude=lon, zoom=12)
    return pdk.Deck(layers=[], initial_view_state=view, map_style="light")
