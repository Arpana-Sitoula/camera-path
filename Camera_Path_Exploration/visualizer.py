"""
visualize_camera_path.py

Reads a Met3D CameraSequence XML (the file produced by exporter.py /
export_tracks_to_xml) and renders the camera path on a world map so you can
sanity-check it BEFORE loading it into Met3D.

Usage:
    python visualize_camera_path.py path/to/camera_sequence_tracked.xml
    python visualize_camera_path.py path/to/camera_sequence_tracked.xml -o my_map.html

Requires: plotly (falls back to matplotlib if plotly isn't installed).
    pip install plotly
"""

import argparse
import xml.etree.ElementTree as ET


def parse_sequence_keys(xml_path):
    """
    Parse all <SequenceKey> elements from the CameraSequence XML, in
    document order, and tag each with its timestep group index.
    Returns a list of dicts: {order, lat, lon, z, advance, timestep_group}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    keys = []
    group = 0

    for i, elem in enumerate(root.findall("SequenceKey")):
        lat = float(elem.get("lat"))
        lon = float(elem.get("lon"))
        z = float(elem.get("z"))
        advance = elem.get("advanceTimestep", "0") == "1"

        keys.append({
            "order": i,
            "lat": lat,
            "lon": lon,
            "z": z,
            "advance": advance,
            "timestep_group": group,
        })

        if advance:
            group += 1

    if not keys:
        raise ValueError(f"No <SequenceKey> elements found in {xml_path}")

    return keys


def render_with_plotly(keys, output_path):
    import plotly.graph_objects as go

    lons = [k["lon"] for k in keys]
    lats = [k["lat"] for k in keys]
    groups = [k["timestep_group"] for k in keys]
    zooms = [k["z"] for k in keys]

    z_min, z_max = min(zooms), max(zooms)

    def scaled_size(z):
        if z_max == z_min:
            return 14
        t = (z - z_min) / (z_max - z_min)
        return 20 - 10 * t  # smaller z (closer zoom, e.g. TC) -> slightly larger marker

    sizes = [scaled_size(z) for z in zooms]

    hover_text = [
        (
            f"order: {k['order']}<br>"
            f"lat: {k['lat']:.2f}, lon: {k['lon']:.2f}<br>"
            f"z (zoom): {k['z']:.1f}<br>"
            f"timestep group: {k['timestep_group']}<br>"
            f"advanceTimestep: {'yes' if k['advance'] else 'no'}"
        )
        for k in keys
    ]

    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        lon=lons, lat=lats, mode="lines",
        line=dict(width=1.5, color="rgba(120,120,120,0.6)"),
        hoverinfo="skip", showlegend=False,
    ))

    fig.add_trace(go.Scattergeo(
        lon=lons, lat=lats, mode="markers+text",
        text=[str(k["order"]) for k in keys],
        textposition="top center",
        textfont=dict(size=9, color="black"),
        marker=dict(
            size=sizes, color=groups, colorscale="Viridis",
            colorbar=dict(title="Timestep<br>group"),
            line=dict(width=0.5, color="white"),
        ),
        hovertext=hover_text, hoverinfo="text", showlegend=False,
    ))

    boundary_idx = [i for i, k in enumerate(keys) if k["advance"]]
    if boundary_idx:
        fig.add_trace(go.Scattergeo(
            lon=[lons[i] for i in boundary_idx],
            lat=[lats[i] for i in boundary_idx],
            mode="markers",
            marker=dict(size=6, color="red", symbol="x"),
            name="advanceTimestep", hoverinfo="skip",
        ))

    fig.update_geos(
        projection_type="natural earth",
        showland=True, landcolor="rgb(235,235,235)",
        showocean=True, oceancolor="rgb(245,250,255)",
        showcountries=True, countrycolor="rgb(200,200,200)",
        coastlinecolor="rgb(150,150,150)",
    )

    fig.update_layout(
        title="Camera sequence preview (order labeled, color = timestep group, size = zoom)",
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(x=0, y=0),
    )

    fig.write_html(output_path, include_plotlyjs=True)
    print(f"Saved interactive map -> {output_path}")


def render_with_matplotlib(keys, output_path):
    import matplotlib.pyplot as plt

    lons = [k["lon"] for k in keys]
    lats = [k["lat"] for k in keys]
    groups = [k["timestep_group"] for k in keys]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(lons, lats, "-", color="gray", linewidth=1, alpha=0.6, zorder=1)
    sc = ax.scatter(lons, lats, c=groups, cmap="viridis", s=60, zorder=2, edgecolor="white")

    for k in keys:
        ax.annotate(str(k["order"]), (k["lon"], k["lat"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")

    boundary = [k for k in keys if k["advance"]]
    if boundary:
        ax.scatter([k["lon"] for k in boundary], [k["lat"] for k in boundary],
                   marker="x", color="red", s=40, zorder=3, label="advanceTimestep")

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Camera sequence preview (order labeled, color = timestep group)")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Timestep group")
    ax.legend(loc="lower left")

    out_png = output_path if output_path.endswith(".png") else output_path.rsplit(".", 1)[0] + ".png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"plotly not installed — saved static map -> {out_png}")
    print("Tip: pip install plotly   for an interactive, hoverable version.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml_path", help="Path to the CameraSequence XML file")
    parser.add_argument("-o", "--output", default=None,
                         help="Output file path (.html for plotly, .png for matplotlib fallback)")
    args = parser.parse_args()

    keys = parse_sequence_keys(args.xml_path)
    print(f"Parsed {len(keys)} SequenceKey entries, "
          f"{keys[-1]['timestep_group'] + (1 if keys[-1]['advance'] else 0)} timestep group(s).")

    output_path = args.output or args.xml_path.rsplit(".", 1)[0] + "_preview.html"

    try:
        render_with_plotly(keys, output_path)
    except ImportError:
        render_with_matplotlib(keys, output_path)


if __name__ == "__main__":
    main()