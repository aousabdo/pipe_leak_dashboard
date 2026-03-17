"""PyDeck map rendering helpers — polished with dark basemap and vivid layers."""

import pandas as pd
import numpy as np
import pydeck as pdk

SEVERITY_COLOR_MAP = {
    "Minor": [34, 197, 94, 200],
    "Moderate": [245, 158, 11, 220],
    "Major": [239, 68, 68, 240],
    "Critical": [153, 27, 27, 255],
}

_MAP_STYLE = "mapbox://styles/mapbox/dark-v11"

_TOOLTIP_STYLE = {
    "backgroundColor": "rgba(15, 23, 42, 0.92)",
    "color": "#f8fafc",
    "fontSize": "13px",
    "borderRadius": "8px",
    "padding": "10px 14px",
    "boxShadow": "0 4px 12px rgba(0,0,0,0.3)",
    "fontFamily": "Inter, -apple-system, sans-serif",
    "lineHeight": "1.5",
}


def pipe_network_map(
    pipes_df: pd.DataFrame,
    risk_scores: np.ndarray | None = None,
    center_lat: float | None = None,
    center_lon: float | None = None,
) -> pdk.Deck:
    """
    Pipe network as glowing line segments on dark basemap.
    Color: green (safe) -> yellow (moderate) -> red (high risk).
    """
    if center_lat is None:
        center_lat = pipes_df["mid_lat"].mean()
    if center_lon is None:
        center_lon = pipes_df["mid_lon"].mean()

    lines = []
    for i, (_, pipe) in enumerate(pipes_df.iterrows()):
        geom = pipe.get("geometry")
        if geom is None or not hasattr(geom, "coords"):
            continue
        coords = list(geom.coords)
        if len(coords) < 2:
            continue

        risk = 0.3
        if risk_scores is not None and i < len(risk_scores):
            risk = float(risk_scores[i])

        # Smooth green -> yellow -> red gradient
        if risk < 0.3:
            r, g, b = 34, 197, 94
        elif risk < 0.6:
            t = (risk - 0.3) / 0.3
            r = int(34 + (245 - 34) * t)
            g = int(197 + (158 - 197) * t)
            b = int(94 + (11 - 94) * t)
        else:
            t = (risk - 0.6) / 0.4
            r = int(245 + (239 - 245) * t)
            g = int(158 + (68 - 158) * t)
            b = int(11 + (68 - 11) * t)

        alpha = int(140 + risk * 115)
        width = 2 + risk * 4

        lines.append({
            "start": list(coords[0]),
            "end": list(coords[-1]),
            "color": [r, g, b, alpha],
            "width": width,
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
        get_width="width",
        width_min_pixels=1,
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 255, 80],
    )

    tooltip = {
        "html": (
            "<div style='margin-bottom:4px'><b style='font-size:14px'>{pipe_id}</b></div>"
            "<b>Material:</b> {material}<br>"
            "<b>Age:</b> {age} years<br>"
            "<b>Risk Score:</b> <span style='color:#ef4444;font-weight:700'>{risk}%</span>"
        ),
        "style": _TOOLTIP_STYLE,
    }

    view = pdk.ViewState(
        latitude=center_lat, longitude=center_lon,
        zoom=13.5, pitch=15, bearing=0,
    )

    return pdk.Deck(
        layers=[line_layer],
        initial_view_state=view,
        tooltip=tooltip,
        map_style=_MAP_STYLE,
    )


def leak_events_map(
    events_df: pd.DataFrame,
    center_lat: float | None = None,
    center_lon: float | None = None,
) -> pdk.Deck:
    """Leak events as pulsing scatter points with heatmap underlay."""
    if events_df is None or events_df.empty:
        return _empty_map(center_lat or 38.58, center_lon or -121.49)

    if center_lat is None:
        center_lat = events_df["latitude"].mean()
    if center_lon is None:
        center_lon = events_df["longitude"].mean()

    df = events_df.copy()
    df["color"] = df["severity"].map(SEVERITY_COLOR_MAP)
    df["color"] = df["color"].apply(lambda x: x if isinstance(x, list) else [100, 100, 100, 180])
    df["radius"] = df["severity"].map(
        {"Minor": 35, "Moderate": 55, "Major": 80, "Critical": 120}
    ).fillna(45)

    # Ensure date is string for tooltip
    df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["cost_str"] = df["repair_cost"].apply(lambda x: f"${x:,.0f}")
    df["loss_str"] = df["water_loss_gallons"].apply(lambda x: f"{x:,.0f}")

    heat_layer = pdk.Layer(
        "HeatmapLayer",
        data=df,
        get_position=["longitude", "latitude"],
        get_weight="flow_rate_gpm",
        radiusPixels=50,
        opacity=0.4,
        threshold=0.05,
    )

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["longitude", "latitude"],
        get_radius="radius",
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 255, 80],
        opacity=0.85,
    )

    tooltip = {
        "html": (
            "<div style='margin-bottom:4px'><b style='font-size:14px'>{pipe_id}</b> "
            "<span style='opacity:0.7'>| {date_str}</span></div>"
            "<b>Severity:</b> {severity}<br>"
            "<b>Flow Rate:</b> {flow_rate_gpm} GPM<br>"
            "<b>Water Loss:</b> {loss_str} gal<br>"
            "<b>Repair Cost:</b> {cost_str}"
        ),
        "style": _TOOLTIP_STYLE,
    }

    view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=13.5, pitch=15)

    return pdk.Deck(
        layers=[heat_layer, scatter_layer],
        initial_view_state=view,
        tooltip=tooltip,
        map_style=_MAP_STYLE,
    )


def risk_heatmap(
    pipes_df: pd.DataFrame,
    risk_scores: np.ndarray,
    center_lat: float | None = None,
    center_lon: float | None = None,
) -> pdk.Deck:
    """3D risk heatmap with elevated columns for high-risk areas."""
    if center_lat is None:
        center_lat = pipes_df["mid_lat"].mean()
    if center_lon is None:
        center_lon = pipes_df["mid_lon"].mean()

    df = pipes_df[["pipe_id", "mid_lat", "mid_lon", "material", "age"]].copy()
    df["risk"] = risk_scores
    df["risk_pct"] = (risk_scores * 100).round(1)

    # Color by risk: green -> red
    colors = []
    for r in risk_scores:
        if r < 0.3:
            colors.append([34, 197, 94, 160])
        elif r < 0.6:
            colors.append([245, 158, 11, 190])
        else:
            colors.append([239, 68, 68, 220])
    df["color"] = colors

    # Elevation proportional to risk
    df["elevation"] = (risk_scores * 500).astype(int)

    column_layer = pdk.Layer(
        "ColumnLayer",
        data=df,
        get_position=["mid_lon", "mid_lat"],
        get_elevation="elevation",
        elevation_scale=1,
        get_fill_color="color",
        radius=30,
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 255, 80],
    )

    heat_layer = pdk.Layer(
        "HeatmapLayer",
        data=df,
        get_position=["mid_lon", "mid_lat"],
        get_weight="risk",
        radiusPixels=60,
        opacity=0.35,
        threshold=0.03,
    )

    tooltip = {
        "html": (
            "<div style='margin-bottom:4px'><b style='font-size:14px'>{pipe_id}</b></div>"
            "<b>Material:</b> {material}<br>"
            "<b>Age:</b> {age} years<br>"
            "<b>Risk:</b> <span style='color:#ef4444;font-weight:700'>{risk_pct}%</span>"
        ),
        "style": _TOOLTIP_STYLE,
    }

    view = pdk.ViewState(
        latitude=center_lat, longitude=center_lon,
        zoom=13.5, pitch=50, bearing=-20,
    )

    return pdk.Deck(
        layers=[heat_layer, column_layer],
        initial_view_state=view,
        tooltip=tooltip,
        map_style=_MAP_STYLE,
    )


def _empty_map(lat: float, lon: float) -> pdk.Deck:
    view = pdk.ViewState(latitude=lat, longitude=lon, zoom=12, pitch=15)
    return pdk.Deck(layers=[], initial_view_state=view, map_style=_MAP_STYLE)
