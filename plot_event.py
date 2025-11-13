import plotly.graph_objs as go
import numpy as np
import awkward as ak
import h5py
from argparse import ArgumentParser
import os
import glob
import random

"""
for modifying the plotly layout.scene.camera.eye attribute.
"""
def get_eye_xyz( camera_phi, camera_th, zoom=2 ):
    return dict(
        x = zoom * np.cos(camera_phi) * np.cos(camera_th),
        y = zoom * np.sin(camera_phi) * np.cos(camera_th),
        z = zoom * np.sin(camera_th)
    )

"""
cleaner plotly.graph_objs.Layout object for 3d plots.
"""
def get_3d_layout():
    axis_settings = dict(
        showgrid=False,
        showticklabels=False,
        backgroundcolor='whitesmoke',
        title='',
        showspikes=False
    )
    return go.Layout(
            scene=dict(
            xaxis=axis_settings,
            yaxis=axis_settings,
            zaxis=axis_settings,
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=get_eye_xyz( np.pi/3, np.pi/18, 2 )
            )
        ),
        margin=dict(l=20, r=20, t=40, b=40),
    )

"""
plot objects for the strings and the boundaries of the detector.
"""
def plot_I3det(center_offset=np.array([0,0,0])):

    I3_dom_z = np.loadtxt("./resources/I3_geo/I3_dom_pos.txt") - center_offset[2]
    I3_str_xy = np.loadtxt("./resources/I3_geo/I3_string_xypos.txt") - center_offset[:2]
    N_strings = I3_str_xy.shape[0]

    I3_strings = [ go.Scatter3d(
            x=[x,x],
            y=[y,y],
            z=[I3_dom_z[0], I3_dom_z[-1]],
            mode='lines',
            line=dict(color='lightgrey', width=1),
            showlegend=False,
            hoverinfo="skip"
        ) for (x,y) in I3_str_xy
    ]

    boundary_strs = np.array([75, 31, 1, 6, 50, 74, 72, 78, 75])
    I3_borders = [ go.Scatter3d(
            x=I3_str_xy[boundary_strs-1, 0],
            y=I3_str_xy[boundary_strs-1, 1],
            z=np.full(N_strings, z),
            mode='lines',
            line=dict(color='grey', width=1),
            showlegend=False,
            hoverinfo="skip"
        ) for z in I3_dom_z[[-1, 0]]
    ]

    return I3_strings + I3_borders

"""
visualize an event by making a scatter plot of the unique DOMs hit.
"""
def plot_first_hits( evt, center_offset=np.array([0,0,0]) ):

    # Extract hit data from the awkward array event
    x_coords = evt.pulses.sensor_pos_x
    y_coords = evt.pulses.sensor_pos_y
    z_coords = evt.pulses.sensor_pos_z
    pulse_times = evt.pulses.pulse_times
    
    # Get the first hit time for each DOM
    first_hit_times = np.array([min(times) for times in pulse_times])

    # Get total charge for each DOM to use for marker size
    pulse_charges = evt.pulses.pulse_charges
    total_charges = np.array([sum(charges) for charges in pulse_charges])
    marker_sizes = 8 * np.log10(total_charges + 1) # Scale marker size

    hits = go.Scatter3d(
            x = x_coords,
            y = y_coords,
            z = z_coords,
            customdata = first_hit_times,
            mode = 'markers',
            marker = dict(
                size = marker_sizes,
                color = first_hit_times,
                colorscale = 'Rainbow_r',
                colorbar=dict(title='Time (ns)'),
            ),
            showlegend=False,
            hoverinfo=['x','y','z','text'],
            hovertemplate="x: %{x:.2f} m<br>y: %{y:.2f} m<br>z: %{z:.2f} m<br>t: %{customdata:.2f} ns",
            name="current_evt"
    )

    return hits

def plot_direction( dir_vec, pivot_pt, color="black" ):

    pt_0 = pivot_pt - 500 * dir_vec
    pt_1 = pivot_pt + 500 * dir_vec
    arrow_vec = 20 * dir_vec

    plot_dir_line = go.Scatter3d(
            x = [ pt_0[0], pt_1[0] ],
            y = [ pt_0[1], pt_1[1] ],
            z = [ pt_0[2], pt_1[2] ],
            mode ='lines',
            line = dict( color=color, width=6 ),
            showlegend=False,
            name="arrow_line",
        )

    plot_dir_arrow = go.Cone(
        x = [ pt_1[0] ],
        y = [ pt_1[1] ],
        z = [ pt_1[2] ],
        u = [ arrow_vec[0], ],
        v = [ arrow_vec[1], ],
        w = [ arrow_vec[2], ],
        anchor="center",
        showscale=False,
        sizemode="absolute",
        sizeref=100,
        colorscale=[[0, color], [1, color]],
        name="arrow_head"
    )

    return [ plot_dir_line, plot_dir_arrow ]


def display_evt(evt, center_offset=np.array([0, 0, 0]), show_direction=True):
    """Creates and returns a plotly Figure for a single event.
       - Pulses are always drawn.
       - MC direction is drawn only if available and show_direction=True.
    """
    fig = go.Figure(data=plot_I3det(center_offset), layout=get_3d_layout())

    # Pulses (first hits etc.)
    fig.add_trace(plot_first_hits(evt, center_offset))

    # Optionally overlay primary direction if mc_truth exists
    if show_direction and ("mc_truth" in getattr(evt, "fields", [])):
        direction = None
        try:
            # support both attribute and mapping access
            mc_truth = evt["mc_truth"] if "mc_truth" in evt.fields else evt.mc_truth
            direction = np.array(
                mc_truth["primary_direction"]
                if "primary_direction" in getattr(mc_truth, "fields", [])
                else mc_truth.primary_direction
            )
        except Exception:
            direction = None

        # If we have a sane (3,) direction vector, add it
        if direction is not None and direction.size == 3 and np.all(np.isfinite(direction)):
            # Center-of-charge pivot in the *recentered* frame
            x = np.asarray(evt.pulses.sensor_pos_x) - center_offset[0]
            y = np.asarray(evt.pulses.sensor_pos_y) - center_offset[1]
            z = np.asarray(evt.pulses.sensor_pos_z) - center_offset[2]

            # pulse_charges is ragged (list of charges per sensor)
            charges_ll = evt.pulses.pulse_charges
            # sum per sensor (handles list/np.array)
            totals = np.array([float(np.sum(c)) for c in charges_ll])

            if np.sum(totals) > 0:
                pivot = np.average(np.c_[x, y, z], axis=0, weights=totals)
            else:
                pivot = np.array([0.0, 0.0, 0.0])

            # plot_direction(...) is your existing routine that returns a list of Plotly traces
            for tr in plot_direction(direction, pivot):
                fig.add_trace(tr)
        # else: silently skip when direction is missing/invalid

    return fig

def combine_images_horizontally(image_paths, output_path):
    """Combines multiple images into a single image horizontally."""
    from PIL import Image
    import io

    images = [Image.open(io.BytesIO(path)) for path in image_paths]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.save(output_path)

def load_events(path: str, fmt: str = "auto", pulses_group: str = "pulses",
                include_extras: bool = False):
    """
    Returns an ak.Array of events with per-event jagged fields:
      x, y, z, t, q  (and optionally: weight, aux, summary_stats)

    Parquet: assumes the file was written with ak.to_parquet (reads via ak.from_parquet).
    HDF5: expects root groups named 'event_<id>' with:
        <event>/pulses/{sensor_pos_x, sensor_pos_y, sensor_pos_z,
                        pulse_times, pulse_charges, aux?, summary_stats?}
        <event>/weights  (scalar)  [optional]
    """
    # --- format detection ---
    if fmt == "auto":
        ext = os.path.splitext(path)[1].lower()
        if ext in {".parquet", ".parq", ".pq"}:
            fmt = "parquet"
        elif ext in {".h5", ".hdf5", ".hdf"}:
            fmt = "hdf5"
        else:
            raise ValueError(f"Unknown file extension '{ext}'. Set fmt to 'parquet' or 'hdf5'.")

    if fmt == "parquet":
        return ak.from_parquet(path)

    # --- HDF5: one group per event, pulse arrays inside '<event>/<pulses_group>' ---

    with h5py.File(path, "r") as f:
        # collect and sort event names numerically
        ev_names = [k for k, v in f.items() if isinstance(v, h5py.Group) and re.fullmatch(r"event_\d+", k)]
        if not ev_names:
            raise RuntimeError("No event_* groups found at file root.")
        ev_names.sort(key=lambda s: int(re.findall(r"\d+", s)[-1]))

        events = []
        for ev_name in ev_names:
            g = f[ev_name]
            if pulses_group not in g:
                raise RuntimeError(f"'{ev_name}' has no '{pulses_group}' group. Found: {list(g.keys())}")
            pg = g[pulses_group]

            # required pulse arrays (exact names you reported)
            try:
                xs = np.asarray(pg["sensor_pos_x"][...])
                ys = np.asarray(pg["sensor_pos_y"][...])
                zs = np.asarray(pg["sensor_pos_z"][...])
                ts = np.asarray(pg["pulse_times"][...])
                qs = np.asarray(pg["pulse_charges"][...])
            except KeyError as e:
                raise RuntimeError(f"Missing dataset {e} in '{ev_name}/{pulses_group}'. "
                                   f"Available: {list(pg.keys())}")

            if not (len(xs) == len(ys) == len(zs) == len(ts) == len(qs)):
                raise RuntimeError(f"Length mismatch in '{ev_name}/{pulses_group}' arrays.")

            ev = {"x": xs, "y": ys, "z": zs, "t": ts, "q": qs}

            # optional event weight
            if "weights" in g:
                try:
                    w = np.asarray(g["weights"][...]).reshape(-1)[0]
                    ev["weight"] = float(w)
                except Exception:
                    pass

            # optional extras from pulses
            if include_extras:
                if "aux" in pg:
                    ev["aux"] = np.asarray(pg["aux"][...])
                if "summary_stats" in pg:
                    ev["summary_stats"] = np.asarray(pg["summary_stats"][...])  # shape (N, 9)

            events.append(ev)

        # ak.Array(list-of-dicts) -> jagged per-event structure
        return ak.Array(events)


def _ensure_float1d(a, name):
    import numpy as np
    arr = np.asarray(a)
    if arr.dtype == object:
        # should not happen for x/y/z, but be robust
        arr = np.array([float(np.asarray(x).reshape(-1)[0]) for x in arr], dtype=np.float64)
    else:
        arr = arr.astype(np.float64, copy=False)
    return arr.reshape(-1)

def _as_listoflists_float(a, name):
    """
    Convert an HDF5 vlen/object dataset into a Python list-of-lists of float.
    Also handles 2D numeric (N, M) by splitting rows, and 1D numeric by wrapping each item.
    """
    import numpy as np
    arr = np.asarray(a)
    if arr.dtype == object:
        return [np.asarray(x, dtype=np.float64).reshape(-1).tolist() for x in arr]
    if arr.ndim == 2:
        return [row.astype(np.float64).reshape(-1).tolist() for row in arr]
    if arr.ndim == 1:
        return [[float(v)] for v in arr.astype(np.float64)]
    raise TypeError(f"Dataset '{name}' has unsupported shape {arr.shape} and dtype {arr.dtype}")

def load_single_event(
    path: str,
    event_id: int | None = None,
    event_index: int | None = None,
    fmt: str = "auto",
    pulses_group: str = "pulses",
):
    """
    Return ONE event as an awkward.Record with fields x,y,z,t,q (and optional weight)
    - HDF5: loads exactly /event_<event_id>/<pulses_group>/*
    - Parquet: loads the file, then selects by event_id field (if present) or by index
    """
    # auto-detect format
    if fmt == "auto":
        ext = os.path.splitext(path)[1].lower()
        if ext in {".parquet", ".parq", ".pq"}:
            fmt = "parquet"
        elif ext in {".h5", ".hdf5", ".hdf"}:
            fmt = "hdf5"
        else:
            raise ValueError(f"Unknown file extension '{ext}'. Set --format explicitly.")

    if fmt == "hdf5":
        if event_id is None:
            raise ValueError("For HDF5 you must provide --event-id (e.g., 18520 for 'event_18520').")
        with h5py.File(path, "r") as f:
            ev_name = f"event_{event_id}"
            if ev_name not in f:
                # helpfully list a few nearby keys
                candidates = [k for k in f.keys() if k.startswith("event_")]
                raise KeyError(f"{ev_name} not found. Available (sample): {candidates[:10]}")
            g = f[ev_name]
            if pulses_group not in g:
                raise KeyError(f"Group '{pulses_group}' not in {ev_name}. Found: {list(g.keys())}")
            pg = g[pulses_group]

            # ... inside fmt == "hdf5" branch, after pg = g[pulses_group]
            xs = _ensure_float1d(pg["sensor_pos_x"][...], "sensor_pos_x")
            ys = _ensure_float1d(pg["sensor_pos_y"][...], "sensor_pos_y")
            zs = _ensure_float1d(pg["sensor_pos_z"][...], "sensor_pos_z")

            # PRESERVE ragged structure for times/charges (list-of-lists)
            times_ll   = _as_listoflists_float(pg["pulse_times"][...],   "pulse_times")
            charges_ll = _as_listoflists_float(pg["pulse_charges"][...], "pulse_charges")

            # Basic consistency checks
            n_sensors = len(xs)
            if not (len(ys) == len(zs) == n_sensors == len(times_ll) == len(charges_ll)):
                raise RuntimeError(
                    f"Length mismatch in {ev_name}/{pulses_group}: "
                    f"len(x/y/z)={len(xs)}/{len(ys)}/{len(zs)}, "
                    f"len(times)={len(times_ll)}, len(charges)={len(charges_ll)}"
                )

            pulses_rec = {
                "sensor_pos_x": xs,
                "sensor_pos_y": ys,
                "sensor_pos_z": zs,
                "pulse_times":  times_ll,    # list-of-lists
                "pulse_charges": charges_ll, # list-of-lists
            }

            # Optional extras
            if "summary_stats" in pg:
                import numpy as np
                pulses_rec["summary_stats"] = np.asarray(pg["summary_stats"][...], dtype=np.float64)

            ev = {"pulses": pulses_rec}

            # optional event-level weight
            if "weights" in g:
                import numpy as np
                try:
                    ev["weight"] = float(np.asarray(g["weights"][...]).reshape(-1)[0])
                except Exception:
                    pass

            return ak.Record(ev)

    # parquet
    data = ak.from_parquet(path)
    n = len(data)

    # Prefer matching an 'event_id' field if present
    if event_id is not None and "event_id" in getattr(data, "fields", []):
        # Find index where event_id matches
        matches = ak.where(data["event_id"] == event_id)[0]
        if len(matches) == 0:
            raise KeyError(f"No event with event_id == {event_id} in {os.path.basename(path)}")
        idx = int(ak.to_numpy(matches)[0])
        return data[idx]

    # Otherwise use explicit index
    if event_index is None:
        # if user gave an id but there's no event_id field, treat id as index (best-effort)
        if event_id is not None:
            event_index = int(event_id)
        else:
            raise ValueError("For Parquet, provide --event-index or ensure a field named 'event_id' exists.")

    if not (0 <= event_index < n):
        raise IndexError(f"--event-index {event_index} out of range [0, {n-1}]")
    return data[event_index]

def _ensure_numeric1d(a, name):
    """
    Return a 1D float64 numpy array. Tolerates object-dtype (e.g. vlen) by
    elementwise casting. Raises if we end up with NaNs or empty arrays.
    """
    import numpy as _np
    arr = _np.asarray(a)
    if arr.dtype == object:
        # common patterns: array of Python scalars, or array of tiny ndarrays
        try:
            arr = _np.array(
                [float(_np.asarray(x).reshape(-1)[0]) for x in arr],
                dtype=_np.float64,
            )
        except Exception as exc:
            raise TypeError(f"Dataset '{name}' has object entries that can't be cast to float: {exc}")
    else:
        arr = arr.astype(_np.float64, copy=False)
    arr = arr.reshape(-1)
    if arr.size == 0:
        raise ValueError(f"Dataset '{name}' is empty after conversion.")
    return arr


def main():
    parser = ArgumentParser(
        description="Visualize a specific neutrino event from a Parquet or HDF5 file."
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Input file (.parquet/.parq/.pq or .h5/.hdf5)")
    parser.add_argument("--format", choices=["auto", "parquet", "hdf5"], default="auto",
                        help="Force input format (default: auto by extension)")
    parser.add_argument("--event-id", type=int, default=None,
                        help="Event numeric id. For HDF5 use the number after 'event_'. "
                             "For Parquet, matches the 'event_id' field if present; "
                             "otherwise treated as --event-index.")
    parser.add_argument("--event-index", type=int, default=None,
                        help="0-based event index (Parquet only, unless you want to treat --event-id as index).")
    parser.add_argument("--pulses-group", default="pulses",
                        help="HDF5 pulses group name (default: 'pulses').")
    parser.add_argument("--recenter", action="store_true", default=True,
                        help="Recenter the detector at (0,0,0).")
    parser.add_argument("-o", "--output", default=None,
                        help="Output PNG file. If omitted, derives from input name and event id/index.")
    args = parser.parse_args()

    # Compute detector center if requested
    center_offset = np.array([0., 0., 0.])
    if args.recenter:
        I3_str_xy = np.loadtxt("./resources/I3_geo/I3_string_xypos.txt")
        I3_dom_z  = np.loadtxt("./resources/I3_geo/I3_dom_pos.txt")
        center_offset = np.array([I3_str_xy[:,0].mean(), I3_str_xy[:,1].mean(), I3_dom_z.mean()])

    # Load exactly one event
    event = load_single_event(
        path=args.input,
        event_id=args.event_id,
        event_index=args.event_index,
        fmt=args.format,
        pulses_group=args.pulses_group,
    )

    # Plot and save
    fig = display_evt(event, center_offset)
    # Title: prefer HDF5-style id if given, else index
    if args.event_id is not None:
        title = f"Event ID: {args.event_id}"
    elif args.event_index is not None:
        title = f"Event Index: {args.event_index}"
    else:
        title = "Event"
    fig.update_layout(title=title, width=800, height=800)

    # Derive output file name if not provided
    if args.output is None:
        stem, _ = os.path.splitext(os.path.basename(args.input))
        suffix = (f"id-{args.event_id}" if args.event_id is not None
                  else f"idx-{args.event_index if args.event_index is not None else 0}")
        args.output = f"{stem}_{suffix}.png"

    img = fig.to_image(format="png", scale=2)
    with open(args.output, "wb") as fh:
        fh.write(img)

    print(f"Saved {args.output}")

if __name__ == "__main__":
    main() 
