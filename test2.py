# -*- coding: utf-8 -*-
"""
test2.py

- Reconstruct building geometry from your coordinate schema (rooms/corridors/stairwell as boxes).
- Identify exterior faces and window faces (only used for classification; we no longer model window transmission).
- Random radiation to validate framework:
    (1) Exterior facade shortwave mesh -> equivalent exterior air gain (placeholder fraction).
    (2) Interior non-window wall shortwave mesh (from raytracing in real use; random here)
        -> absorbed radiant power -> ASHRAE Solar RTS (24-term) convolution -> delayed air gain.
- Build multi-zone thermal network:
    * Each room is one air node
    * Outdoor branch per room (To) with g_out = U_ext * A_ext
    * Inter-room branches along X within each row & floor
- Use dm4bem.tc2ss(...) for core math, then compute hourly steady-state temperatures via SS gain.

NOTE:
- Window transmitted solar is REMOVED (per your request).
  Interior non-window surface irradiance is assumed already includes projected shortwave results.
"""

import numpy as np
import pandas as pd
import dm4bem


# ---------------------------
# Utilities
# ---------------------------

def make_mesh_areas(total_area_m2: float, cell_area_m2: float) -> np.ndarray:
    n = int(np.ceil(total_area_m2 / cell_area_m2))
    return np.full(n, total_area_m2 / n, dtype=float)


def random_irradiance_Wm2(rng: np.random.Generator, n_cells: int, n_hours: int, lo: float, hi: float) -> np.ndarray:
    return rng.uniform(lo, hi, size=(n_cells, n_hours))


def extract_input_names_keep_duplicates(b: np.ndarray, f: np.ndarray) -> list[str]:
    """
    IMPORTANT for older dm4bem tc2ss():
    u contains each *occurrence* of a source in b (branch sources) then f (node sources),
    including duplicates (e.g., 'To' appearing N times in b -> N columns in u).
    """
    names: list[str] = []
    for v in b.reshape(-1):
        if isinstance(v, str) and v.strip():
            names.append(v)
    for v in f.reshape(-1):
        if isinstance(v, str) and v.strip():
            names.append(v)
    return names


def build_U(input_data_set: pd.DataFrame, input_names: list[str]) -> np.ndarray:
    ds = input_data_set.copy()
    U = np.zeros((len(ds), len(input_names)), dtype=float)
    for j, name in enumerate(input_names):
        if name in ds.columns:
            U[:, j] = ds[name].to_numpy(dtype=float)
        else:
            U[:, j] = 0.0
    return U


def rts_convolution(Q_radiant: np.ndarray, rtf_frac: np.ndarray) -> np.ndarray:
    """
    RTS: Phi(t) = sum_{k=0..23} rtf(k) * Q_radiant(t-k)
    Implemented as causal convolution with zero-padding for t<0.
    """
    full = np.convolve(Q_radiant, rtf_frac, mode="full")
    return full[: len(Q_radiant)]


# ---------------------------
# 1) Geometry reconstruction
# ---------------------------

SCHEMA = {
    "project": "24x60_Building",
    "bounds": {"x": [0, 60], "y": [0, 24], "z": [0, 8]},
    "thickness": {"exterior": 0.5, "interior_shared": 0.25},
    "stairwell": {"x": [0.5, 4.0], "y": [0.5, 23.5], "z": [0.5, 7.5]},
    "corridor": {
        "1F": {"x": [4.0, 59.5], "y": [10.5, 13.5], "z": [0.5, 3.5]},
        "2F": {"x": [4.0, 59.5], "y": [10.5, 13.5], "z": [4.5, 7.5]},
    },
    "room_logic": {
        "x_segments": [15, 30, 45, 60],
        "x_ranges": {
            1: [4.0, 14.75],
            2: [15.25, 29.75],
            3: [30.25, 44.75],
            4: [45.25, 59.5],
        },
        "y_split": {"front": [0.5, 9.5], "corridor": [10.5, 13.5], "back": [14.5, 23.5]},
        "z_split": {1: [0.5, 3.5], 2: [4.5, 7.5]},
    },
}


def build_rooms_from_schema(schema: dict) -> list[dict]:
    rooms = []
    xr = schema["room_logic"]["x_ranges"]
    y_front = schema["room_logic"]["y_split"]["front"]
    y_back = schema["room_logic"]["y_split"]["back"]
    zr = schema["room_logic"]["z_split"]

    for k in (1, 2):
        z0, z1 = zr[k]
        for j, yb in ((1, y_front), (2, y_back)):
            y0, y1 = yb
            for i in (1, 2, 3, 4):
                x0, x1 = xr[i]
                rooms.append(
                    {
                        "id": f"R{i}{'F' if j==1 else 'B'}_{k}F",  # e.g. R1F_1F
                        "i": i,
                        "j": j,
                        "k": k,
                        "x": (x0, x1),
                        "y": (y0, y1),
                        "z": (z0, z1),
                    }
                )
    return rooms


def room_faces(room: dict) -> dict:
    x0, x1 = room["x"]
    y0, y1 = room["y"]
    z0, z1 = room["z"]
    dx, dy, dz = (x1 - x0), (y1 - y0), (z1 - z0)

    A_x = dy * dz  # faces normal to x
    A_y = dx * dz  # faces normal to y
    A_z = dx * dy  # floor/ceiling

    return {
        "west": {"plane": ("x", x0), "area": A_x, "normal": (-1, 0, 0)},
        "east": {"plane": ("x", x1), "area": A_x, "normal": (+1, 0, 0)},
        "south": {"plane": ("y", y0), "area": A_y, "normal": (0, -1, 0)},
        "north": {"plane": ("y", y1), "area": A_y, "normal": (0, +1, 0)},
        "floor": {"plane": ("z", z0), "area": A_z, "normal": (0, 0, -1)},
        "ceiling": {"plane": ("z", z1), "area": A_z, "normal": (0, 0, +1)},
    }


def face_is_exterior(room: dict, face_name: str, schema: dict) -> bool:
    x0, x1 = room["x"]
    y0, y1 = room["y"]

    ext_min_x, ext_max_x = 0.5, 59.5
    ext_min_y, ext_max_y = 0.5, 23.5

    if face_name == "west" and np.isclose(x0, ext_min_x):
        return True
    if face_name == "east" and np.isclose(x1, ext_max_x):
        return True
    if face_name == "south" and np.isclose(y0, ext_min_y):
        return True
    if face_name == "north" and np.isclose(y1, ext_max_y):
        return True
    return False


def face_has_window(room: dict, face_name: str, schema: dict) -> bool:
    """
    Window logic (for classification only):
    - every room has window(s) on its exterior vertical face(s)
    - corner rooms (i=4, front/back) have two exterior faces -> both have windows
    """
    if face_name in ("floor", "ceiling"):
        return False
    if not face_is_exterior(room, face_name, schema):
        return False

    # End rooms i=4 have east exterior + (south or north) exterior => both windowed
    if room["i"] == 4:
        return True

    # Others only have y-exterior face (south for front, north for back) => that face windowed
    return True


def describe_reconstruction(rooms: list[dict]) -> str:
    n_rooms = len(rooms)
    floors = sorted(set(r["k"] for r in rooms))
    n_end = sum(1 for r in rooms if r["i"] == 4)
    return (
        f"Reconstruction result: {n_rooms} rooms (4 along X x 2 rows along Y x {len(floors)} floors). "
        f"Room IDs like R1F_1F (i=1, front, 1F). End rooms (i=4) count={n_end} (corner rooms on each floor/row).\n"
        f"Exterior envelope for room faces is checked at x in {{0.5,59.5}}, y in {{0.5,23.5}}; "
        f"rooms start at x>=4.0 because the stairwell void occupies x in [0.5,4.0]."
    )


# ---------------------------
# 2) Build adjacency & network
# ---------------------------

def build_adjacency(rooms: list[dict]) -> dict:
    """
    Adjacency between rooms through shared walls:
    - Along X within same row (front/back) and same floor: i=1-2-3-4 chain.
    - No adjacency between front/back due to corridor void.
    """
    idx = {(r["i"], r["j"], r["k"]): r["id"] for r in rooms}
    neigh = {r["id"]: [] for r in rooms}
    for r in rooms:
        i, j, k = r["i"], r["j"], r["k"]
        if i < 4:
            neigh[r["id"]].append(idx[(i + 1, j, k)])
        if i > 1:
            neigh[r["id"]].append(idx[(i - 1, j, k)])
    return neigh


def shared_wall_area(rooms: list[dict], rid_a: str, rid_b: str) -> float:
    """
    For X-adjacent rooms, shared wall is a face normal to x. Area = dy * dz.
    """
    ra = next(rr for rr in rooms if rr["id"] == rid_a)
    dy = ra["y"][1] - ra["y"][0]
    dz = ra["z"][1] - ra["z"][0]
    return dy * dz


# ---------------------------
# Main
# ---------------------------

def main():
    print("[INFO] dm4bem imported OK.")
    print(f"[INFO] dm4bem module path: {getattr(dm4bem, '__file__', 'unknown')}")

    rng = np.random.default_rng(123)

    # Step 1: reconstruct geometry
    rooms = build_rooms_from_schema(SCHEMA)
    print("\n=== Step 1: Geometry reconstruction (from coordinate schema) ===")
    print(describe_reconstruction(rooms))

    # per-room face classification
    room_meta = {}
    for r in rooms:
        faces = room_faces(r)
        ext_faces = []
        win_faces = []
        for fn in ("west", "east", "south", "north"):
            if face_is_exterior(r, fn, SCHEMA):
                ext_faces.append(fn)
            if face_has_window(r, fn, SCHEMA):
                win_faces.append(fn)
        room_meta[r["id"]] = {
            "volume": (r["x"][1] - r["x"][0]) * (r["y"][1] - r["y"][0]) * (r["z"][1] - r["z"][0]),
            "faces": faces,
            "ext_faces": ext_faces,
            "win_faces": win_faces,
        }

    print("\nSample room exterior/window faces:")
    for rid in [rooms[0]["id"], rooms[3]["id"], rooms[4]["id"], rooms[7]["id"]]:
        m = room_meta[rid]
        print(f"  {rid}: exterior={m['ext_faces']} windows={m['win_faces']}")

    # Step 2: build framework
    print("\n=== Step 2: Build dm4bem + RTS steady-state framework ===")

    adjacency = build_adjacency(rooms)

    # time axis
    n_hours = 24
    time = pd.date_range(start="2000-01-01 00:00:00", periods=n_hours, freq="1h")
    To = rng.normal(loc=0.0, scale=5.0, size=n_hours)

    # U-values (demo)
    U_ext = 0.50  # exterior wall U
    U_int = 1.50  # internal partition U

    # RTS coefficients: ASHRAE Solar RTS Table 13 example
    rtf_pct = np.array([54, 16, 8, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
    rtf = rtf_pct / 100.0

    # surface shortwave absorptivity (demo)
    alpha_int = 0.60

    # Define exterior wall absorptivity
    alpha_ext = 0.60  # 外墙吸收率

    cell_area = 1.0

    # Precompute per-room air gains Phi_room(t)
    Phi = {}  # room_id -> (n_hours,) W

    for r in rooms:
        rid = r["id"]
        meta = room_meta[rid]

        # (A) Exterior facade shortwave mesh (random demo)
        A_ext = sum(meta["faces"][fn]["area"] for fn in meta["ext_faces"]) if meta["ext_faces"] else 0.0
        ext_areas = make_mesh_areas(A_ext, cell_area) if A_ext > 0 else np.array([0.0])
        q_ext = random_irradiance_Wm2(rng, len(ext_areas), n_hours, lo=0.0, hi=300.0) if A_ext > 0 else np.zeros((1, n_hours))

        # Placeholder: convert some fraction to air gain (you may later replace with your own mapping)
        eta_ext_to_air = 0.05
        Q_ext = eta_ext_to_air * (ext_areas @ q_ext)  # (n_hours,)

        # Calculate exterior wall absorption using alpha_ext
        Q_abs_ext = alpha_ext * (ext_areas @ q_ext)  # 外墙吸收的辐射热量
        Phi_ext_to_air = rts_convolution(Q_abs_ext, rtf)  # 外墙辐射转化为空气增益

        # (B) Interior non-window surface shortwave mesh (random demo; in your real pipeline this comes from raytracing)
        A_vert_total = sum(meta["faces"][fn]["area"] for fn in ("west", "east", "south", "north"))
        A_win_faces = sum(meta["faces"][fn]["area"] for fn in meta["win_faces"]) if meta["win_faces"] else 0.0
        A_nonwin_pool = max(0.0, A_vert_total - A_win_faces)

        int_areas = make_mesh_areas(A_nonwin_pool, cell_area) if A_nonwin_pool > 0 else np.array([0.0])
        q_int = random_irradiance_Wm2(rng, len(int_areas), n_hours, lo=0.0, hi=50.0) if A_nonwin_pool > 0 else np.zeros((1, n_hours))

        # NEW LOGIC:
        # Treat interior non-window shortwave as RADIANT gain absorbed by surfaces,
        # then use RTS to convert to delayed air gain.
        if A_nonwin_pool > 0:
            Q_abs_int = alpha_int * (int_areas @ q_int)        # (n_hours,) W
            Phi_int_to_air = rts_convolution(Q_abs_int, rtf)   # (n_hours,) W
        else:
            Phi_int_to_air = np.zeros(n_hours)

        # Internal gains (demo)
        Q_internal = 80.0 * np.ones(n_hours)

        # Total air gains (NO window transmission term; NO instantaneous convective from same q_int)
        Phi[rid] = Phi_ext_to_air + Phi_int_to_air + Q_internal

    # ---------------------------
    # Build dm4bem thermal circuit for N rooms
    # ---------------------------
    room_ids = [r["id"] for r in rooms]
    nR = len(room_ids)

    # outdoor conductance per room
    g_out = np.zeros(nR)
    for idx, rid in enumerate(room_ids):
        meta = room_meta[rid]
        A_ext = sum(meta["faces"][fn]["area"] for fn in meta["ext_faces"]) if meta["ext_faces"] else 0.0
        g_out[idx] = U_ext * A_ext

    # inter-room pairs (avoid double count)
    inter_pairs = []
    seen = set()
    for rid in room_ids:
        for nb in adjacency[rid]:
            key = tuple(sorted((rid, nb)))
            if key in seen:
                continue
            seen.add(key)
            inter_pairs.append(key)

    # branches: outdoor first, then inter-room
    branches = []
    for rid in room_ids:
        branches.append(("out", rid))
    for a, b_room in inter_pairs:
        branches.append((a, b_room))

    nq = len(branches)
    ntheta = nR

    # A matrix (nq x ntheta)
    A = np.zeros((nq, ntheta), dtype=float)
    # diagonal G
    Gdiag = np.zeros(nq, dtype=float)
    # b sources
    b = np.empty(nq, dtype=object)
    b[:] = 0

    # outdoor branches
    for bi, (src, rid) in enumerate(branches[:nR]):
        j = room_ids.index(rid)
        A[bi, j] = +1.0
        Gdiag[bi] = g_out[j]
        b[bi] = "To"  # duplicate 'To' per room (needed by your dm4bem version)

    # inter-room branches
    for bi, (a, b_room) in enumerate(branches[nR:], start=nR):
        ia = room_ids.index(a)
        ib = room_ids.index(b_room)
        A[bi, ia] = -1.0
        A[bi, ib] = +1.0
        Gdiag[bi] = U_int * shared_wall_area(rooms, a, b_room)
        b[bi] = 0

    G = np.diag(Gdiag).astype(float)

    # C diagonal per room (air only)
    rho_air = 1.2
    cp_air = 1000.0
    C_air = np.array([rho_air * cp_air * room_meta[rid]["volume"] for rid in room_ids], dtype=float)
    C = np.diag(C_air).astype(float)

    # f heat sources per node
    f = np.array([f"Phi_{rid}" for rid in room_ids], dtype=object)

    # y outputs: all room temps
    y = np.ones(ntheta, dtype=float)

    # input dataframe
    input_df = pd.DataFrame(index=time)
    input_df["To"] = To
    for rid in room_ids:
        input_df[f"Phi_{rid}"] = Phi[rid]

    # core dm4bem
    As, Bs, Cs, Ds = dm4bem.tc2ss(A, G, C, b, f, y)

    # build U with duplicates preserved
    input_names = extract_input_names_keep_duplicates(b, f)
    U = build_U(input_df, input_names)

    As_np = np.asarray(As, dtype=float)
    Bs_np = np.asarray(Bs, dtype=float)
    Cs_np = np.asarray(Cs, dtype=float)
    Ds_np = np.asarray(Ds, dtype=float)

    nu_expected = Bs_np.shape[1]
    if U.shape[1] != nu_expected:
        raise ValueError(f"nu mismatch: Bs expects {nu_expected}, U has {U.shape[1]}")

    # steady-state gain
    K_ss = (-Cs_np @ np.linalg.inv(As_np) @ Bs_np + Ds_np)
    Y = (K_ss @ U.T).T  # (n_hours, nR)

    T = pd.DataFrame(Y, index=time, columns=room_ids)
    T = T - 273.15

    print("\n=== Result: Hourly steady-state room temperatures (first 3 rooms, first 6 hours) ===")
    print(T[room_ids[:3]].head(6).round(2).to_string())

    print("\n=== Result: Last hour (first 6 rooms) ===")
    print(T[room_ids[:6]].tail(1).round(2).to_string())

    print("\n[OK] All done.")


if __name__ == "__main__":
    main()
