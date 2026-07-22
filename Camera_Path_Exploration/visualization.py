import numpy as np
from scipy import ndimage
from skimage import measure
from extractor import load_mask, LONS, LATS, MIN_TC_PIXELS, MIN_AR_PIXELS, haversine_km


def _extract_blob_contour(nc_path, feature, target_lon, target_lat, min_size):
    """
    Reload a single timestep's mask, find the connected blob whose centroid
    is closest to (target_lon, target_lat) — i.e. the one belonging to this
    track — and return its outline(s) as lists of (lon, lat) boundary points.
    """
    mask = load_mask(nc_path, feature)
    binary = (mask > 0.5).astype(int)
    labeled, count = ndimage.label(binary)

    best_id, best_dist = None, float("inf")
    for blob_id in range(1, count + 1):
        blob = (labeled == blob_id)
        if blob.sum() < min_size:
            continue
        cy, cx = ndimage.center_of_mass(blob)
        lon = float(LONS[int(np.clip(cx, 0, len(LONS) - 1))])
        lat = float(LATS[int(np.clip(cy, 0, len(LATS) - 1))])
        d = haversine_km(target_lon, target_lat, lon, lat)
        if d < best_dist:
            best_dist, best_id = d, blob_id

    if best_id is None:
        return []

    blob_mask = (labeled == best_id)
    contours = measure.find_contours(blob_mask.astype(float), 0.5)

    rings = []
    for contour in contours:
        cols = np.clip(contour[:, 1].astype(int), 0, len(LONS) - 1)
        rows = np.clip(contour[:, 0].astype(int), 0, len(LATS) - 1)
        rings.append(list(zip(LONS[cols], LATS[rows])))
    return rings


def animate_top_tracks_with_shapes(top, nc_paths, output_path="top_tracks_shapes.gif"):
    """
    Same idea as animate_top_tracks, but fills in the REAL detected shape
    of each top feature at each timestep, instead of just a point.
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    timesteps = sorted({p["timestep"] for r in top for p in r["seq"]})
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90); ax.grid(alpha=0.3)
    title = ax.set_title("")

    history_lines = []
    for i, r in enumerate(top):
        line, = ax.plot([], [], color=colors[i], linewidth=1, alpha=0.5)
        history_lines.append(line)
    ax.legend([f"#{r['rank']} {r['feature']}-{r['id']}" for r in top], loc="lower left")

    def update(t):
        title.set_text(f"Timestep {t}")
        for patch in list(ax.patches):
            patch.remove()
        for coll in list(ax.collections):
            coll.remove()

        for i, r in enumerate(top):
            so_far = [p for p in r["seq"] if p["timestep"] <= t]
            if not so_far:
                continue
            history_lines[i].set_data([p["lon"] for p in so_far], [p["lat"] for p in so_far])

            current = [p for p in r["seq"] if p["timestep"] == t]
            if not current:
                continue
            p = current[0]
            min_size = MIN_TC_PIXELS if r["feature"] == "TC" else MIN_AR_PIXELS
            rings = _extract_blob_contour(nc_paths[t], r["feature"], p["lon"], p["lat"], min_size)
            for ring in rings:
                if len(ring) < 3:
                    continue
                xs, ys = zip(*ring)
                ax.fill(xs, ys, color=colors[i], alpha=0.6)

        return history_lines

    animation.FuncAnimation(fig, update, frames=timesteps, interval=700).save(output_path, writer="pillow")


def animate_camera_view_with_shape(track_seq, feature, z_value, nc_paths, output_path="camera_view.gif"):
    """
    Camera-perspective view that fills the REAL shape inside a zoom box
    riding along the track, instead of an empty rectangle.
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.patches as patches

    min_size = MIN_TC_PIXELS if feature == "TC" else MIN_AR_PIXELS
    half_width = max(z_value / 2 + 5, 5)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90); ax.grid(alpha=0.3)
    box = patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor="red", linewidth=2)
    ax.add_patch(box)
    label = ax.text(0, 0, "", fontsize=9, color="red")

    def update(i):
        for coll in list(ax.collections):
            coll.remove()

        p = track_seq[i]
        nc_path = nc_paths[p["timestep"]]
        rings = _extract_blob_contour(nc_path, feature, p["lon"], p["lat"], min_size)
        for ring in rings:
            if len(ring) < 3:
                continue
            xs, ys = zip(*ring)
            ax.fill(xs, ys, color="teal", alpha=0.6)

        box.set_xy((p["lon"] - half_width, p["lat"] - half_width))
        box.set_width(2 * half_width); box.set_height(2 * half_width)
        label.set_position((p["lon"] + half_width * 1.05, p["lat"]))
        label.set_text(f"t={p['timestep']}")
        return box, label

    animation.FuncAnimation(fig, update, frames=len(track_seq), interval=700).save(output_path, writer="pillow")


def plot_ranking(rows, top_n=5, output_path="ranking_overview.png"):
    import matplotlib.pyplot as plt
    fig, (ax_bar, ax_map) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 2]})

    shown = rows[:top_n]
    ax_bar.barh([f"#{r['rank']} {r['feature']}-{r['id']}" for r in reversed(shown)],
                [r["score"] for r in reversed(shown)], color="teal")
    ax_bar.set_xlabel("Interest score (0-1)")
    ax_bar.set_title(f"Top {top_n} ranked features")

    for r in rows:
        ax_map.plot([p["lon"] for p in r["seq"]], [p["lat"] for p in r["seq"]],
                     color="lightgray", linewidth=0.8, zorder=1)

    colors = plt.cm.tab10.colors
    for i, r in enumerate(shown):
        lons, lats = [p["lon"] for p in r["seq"]], [p["lat"] for p in r["seq"]]
        c = colors[i % len(colors)]
        ax_map.plot(lons, lats, linewidth=2.5, color=c, label=f"#{r['rank']} {r['feature']}-{r['id']}")
        ax_map.scatter(lons[0], lats[0], marker="o", color=c, s=60)
        ax_map.scatter(lons[-1], lats[-1], marker="s", color=c, s=60)

    ax_map.set_xlim(-180, 180); ax_map.set_ylim(-90, 90)
    ax_map.legend(loc="lower left", fontsize=8)
    ax_map.set_title("All tracks (gray), top-ranked highlighted\n(●=start, ■=end)")
    ax_map.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)

def animate_top_tracks(top_rows, output_path="top_tracks.gif"):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    timesteps = sorted({p["timestep"] for r in top_rows for p in r["seq"]})
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90); ax.grid(alpha=0.3)
    lines, points = [], []
    for i, r in enumerate(top_rows):
        line, = ax.plot([], [], color=colors[i], linewidth=2, label=f"#{r['rank']} {r['feature']}-{r['id']}")
        point, = ax.plot([], [], "o", color=colors[i], markersize=10)
        lines.append(line); points.append(point)
    ax.legend(loc="lower left")
    title = ax.set_title("")

    def update(t):
        title.set_text(f"Timestep {t}")
        for i, r in enumerate(top_rows):
            so_far = [p for p in r["seq"] if p["timestep"] <= t]
            if so_far:
                lines[i].set_data([p["lon"] for p in so_far], [p["lat"] for p in so_far])
                points[i].set_data([so_far[-1]["lon"]], [so_far[-1]["lat"]])
        return lines + points

    animation.FuncAnimation(fig, update, frames=timesteps, interval=600).save(output_path, writer="pillow")

def animate_camera_view(track_seq, z_value, output_path="camera_view.gif"):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.patches as patches

    half_width = max(z_value / 2, 5)  # crude: bigger z (AR) -> wider box than TC
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90); ax.grid(alpha=0.3)
    ax.plot([p["lon"] for p in track_seq], [p["lat"] for p in track_seq], color="lightgray", linewidth=1)
    box = patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor="red", linewidth=2)
    ax.add_patch(box)
    label = ax.text(0, 0, "", fontsize=9, color="red")

    def update(i):
        p = track_seq[i]
        box.set_xy((p["lon"] - half_width, p["lat"] - half_width))
        box.set_width(2 * half_width); box.set_height(2 * half_width)
        label.set_position((p["lon"] + half_width * 1.05, p["lat"]))
        label.set_text(f"t={p['timestep']}")
        return box, label

    animation.FuncAnimation(fig, update, frames=len(track_seq), interval=700).save(output_path, writer="pillow")