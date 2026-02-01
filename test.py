import numpy as np
import pandas as pd
import dm4bem


def make_mesh_areas(total_area_m2: float, cell_area_m2: float) -> np.ndarray:
    """Return patch areas summing to total_area_m2, roughly cell_area_m2 each."""
    n = int(np.ceil(total_area_m2 / cell_area_m2))
    return np.full(n, total_area_m2 / n, dtype=float)


def random_irradiance_Wm2(rng: np.random.Generator, n_cells: int, n_hours: int, lo: float, hi: float) -> np.ndarray:
    """Random irradiance on each patch, W/m². Shape (n_cells, n_hours)."""
    return rng.uniform(lo, hi, size=(n_cells, n_hours))


def extract_input_names_keep_duplicates(b: np.ndarray, f: np.ndarray) -> list[str]:
    """
    IMPORTANT for older dm4bem tc2ss():
    u contains each *occurrence* of a source in b (branch sources) then f (node sources),
    including duplicates (e.g., 'To' appearing twice in b -> two columns in u).
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
    """
    Build numeric U(t) with columns ordered as input_names.
    If input_names contains duplicates, U will contain duplicated columns too.
    Missing names are filled with zeros.
    """
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


def main():
    print("[INFO] dm4bem imported OK.")
    print(f"[INFO] dm4bem module path: {getattr(dm4bem, '__file__', 'unknown')}")
    assert hasattr(dm4bem, "tc2ss"), "dm4bem.tc2ss() not found - check your clone/version."

    rng = np.random.default_rng(42)

    # -----------------------
    # 1) Geometry: two rooms
    # -----------------------
    W, D, H = 4.0, 5.0, 3.0
    V_room = W * D * H

    A_shared = D * H
    A_vert_ext = 2 * (W * H) + (D * H)  # 3 exposed sides per room
    A_roof = W * D
    A_window = 2.0

    # room1 interior receiving surfaces pool (4 walls + floor + ceiling)
    A_int_room1 = (W * H) + (W * H - A_window) + (D * H) + (D * H) + (W * D) + (W * D)

    print("\n=== Geometry ===")
    print(f"W×D×H = {W:.1f}×{D:.1f}×{H:.1f} m")
    print(f"A_shared = {A_shared:.2f} m²")
    print(f"A_vert_ext(each room, excl. shared) = {A_vert_ext:.2f} m²")
    print(f"A_roof(each room) = {A_roof:.2f} m²")
    print(f"A_window (room1) = {A_window:.2f} m²")
    print(f"A_int_room1 (for random interior mesh) = {A_int_room1:.2f} m²")

    # -----------------------------
    # 2) U-values -> conductances
    # -----------------------------
    U_wall = 0.50
    U_roof = 0.30
    U_window = 1.60
    U_shared = 1.50

    g12 = U_shared * A_shared
    g1_out = U_wall * A_vert_ext + U_roof * A_roof + U_window * A_window
    g2_out = U_wall * A_vert_ext + U_roof * A_roof

    print("\n=== Conductances (W/K) ===")
    print(f"g12 (between rooms) = {g12:.2f}")
    print(f"g1_out (room1->out) = {g1_out:.2f}")
    print(f"g2_out (room2->out) = {g2_out:.2f}")

    # -----------------------------------------
    # 3) Random hourly data (24 hours)
    # -----------------------------------------
    n_hours = 24
    time = pd.date_range(start="2000-01-01 00:00:00", periods=n_hours, freq="1h")

    To = rng.normal(loc=0.0, scale=5.0, size=n_hours)

    # Radiation meshes (1 m² per patch)
    cell_area = 1.0
    areas_ext_r1 = make_mesh_areas(A_vert_ext, cell_area)
    areas_ext_r2 = make_mesh_areas(A_vert_ext, cell_area)
    areas_int_r1 = make_mesh_areas(A_int_room1, cell_area)

    # Random irradiance on exterior surfaces (placeholder)
    q_ext_r1 = random_irradiance_Wm2(rng, len(areas_ext_r1), n_hours, lo=0.0, hi=300.0)
    q_ext_r2 = random_irradiance_Wm2(rng, len(areas_ext_r2), n_hours, lo=0.0, hi=300.0)

    # Random irradiance on interior surfaces (placeholder)
    q_int_r1 = random_irradiance_Wm2(rng, len(areas_int_r1), n_hours, lo=0.0, hi=50.0)

    # Window irradiance -> transmitted power (beam-like)
    tau_glass = 0.65
    E_win = rng.uniform(0.0, 500.0, size=n_hours)  # W/m² on window
    P_trans = tau_glass * A_window * E_win         # W entering the zone as shortwave

    # Convert exterior/interior surface SW to equivalent air gains (demo placeholders)
    eta_ext_to_air = 0.05
    eta_int_to_air = 0.10
    Q1_ext = eta_ext_to_air * (areas_ext_r1 @ q_ext_r1)  # W
    Q2_ext = eta_ext_to_air * (areas_ext_r2 @ q_ext_r2)  # W
    Q1_int = eta_int_to_air * (areas_int_r1 @ q_int_r1)  # W

    # -----------------------------
    # 3.1 RTS (Scheme A): Window beam solar -> delayed air gain
    # -----------------------------
    # Step A: distribute transmitted solar onto interior mesh patches (weights sum to 1)
    w = rng.random(len(areas_int_r1))
    w = w / w.sum()

    # Convert power to irradiance per patch:
    # P_inc_i(t) = w_i * P_trans(t)   [W]
    # E_inc_i(t) = P_inc_i(t) / A_i   [W/m²]
    P_inc = (w[:, None] * P_trans[None, :])  # (n_patches, n_hours)
    E_inc = P_inc / areas_int_r1[:, None]

    # Step B: absorption on each patch (can be per-patch alpha_i; here uniform)
    alpha_int = 0.60
    Q_abs = np.sum(alpha_int * areas_int_r1[:, None] * E_inc, axis=0)  # (n_hours,) W
    # With uniform alpha and weights summing to 1, Q_abs ≈ alpha_int * P_trans

    # Step C: Apply ASHRAE Solar RTS (Table 13):
    # Choose: Medium-weight, With Carpet, 50% Glass
    # Values are in %, sum to 100.
    rtf_pct = np.array([54, 16, 8, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
    rtf_frac = rtf_pct / 100.0

    # Window radiant gain (beam, no interior shading) is treated as 100% radiant in RTS,
    # and converted to (delayed) load-to-air using solar RTS factors.
    Phi_win_to_air = rts_convolution(Q_abs, rtf_frac)  # (n_hours,) W

    # Internal gains
    Q1_internal = 100.0 * np.ones(n_hours)
    Q2_internal = 50.0 * np.ones(n_hours)

    # Total node gains (what dm4bem sees)
    Phi_R1 = Q1_ext + Q1_int + Phi_win_to_air + Q1_internal
    Phi_R2 = Q2_ext + Q2_internal

    print("\n=== Gains sample (hour 0) ===")
    print(f"To[0]={To[0]:.2f} °C")
    print(f"P_trans[0]={P_trans[0]:.2f} W, Q_abs[0]={Q_abs[0]:.2f} W, Phi_win_to_air[0]={Phi_win_to_air[0]:.2f} W")
    print(f"Phi_R1[0]={Phi_R1[0]:.2f} W, Phi_R2[0]={Phi_R2[0]:.2f} W")

    # -----------------------------------------
    # 4) Build A, G, C, b, f, y for dm4bem.tc2ss
    # -----------------------------------------
    # Node order: [room1, room2]
    # Branch order: [out->r1, r1->r2, out->r2]
    A = np.array([
        [ +1.0,  0.0],   # out -> room1
        [ -1.0, +1.0],   # room1 -> room2
        [  0.0, +1.0],   # out -> room2
    ], dtype=float)

    G = np.diag([g1_out, g12, g2_out]).astype(float)

    rho_air = 1.2
    cp_air = 1000.0
    C_air = rho_air * cp_air * V_room
    C = np.diag([C_air, C_air]).astype(float)

    # IMPORTANT: 'To' appears twice -> u has To twice
    b = np.array(["To", 0, "To"], dtype=object)
    f = np.array(["Phi_R1", "Phi_R2"], dtype=object)
    y = np.array([1.0, 1.0], dtype=float)

    As, Bs, Cs, Ds = dm4bem.tc2ss(A, G, C, b, f, y)

    print("\n=== CORE CONFIRMATION ===")
    print("Using dm4bem.tc2ss(A, G, C, b, f, y) as core operator.")
    print("tc2ss returned 4 objects: As, Bs, Cs, Ds")

    # -----------------------------------------
    # 5) Hourly steady-state temps via SS gain
    #    y_ss = (-Cs As^{-1} Bs + Ds) u
    # -----------------------------------------
    input_data_set = pd.DataFrame({"To": To, "Phi_R1": Phi_R1, "Phi_R2": Phi_R2}, index=time)

    input_names = extract_input_names_keep_duplicates(b, f)  # e.g. ['To','To','Phi_R1','Phi_R2']
    U = build_U(input_data_set, input_names)

    As_np = np.asarray(As, dtype=float)
    Bs_np = np.asarray(Bs, dtype=float)
    Cs_np = np.asarray(Cs, dtype=float)
    Ds_np = np.asarray(Ds, dtype=float)

    nu_expected = Bs_np.shape[1]
    if U.shape[1] != nu_expected:
        raise ValueError(f"nu mismatch: Bs expects {nu_expected}, U has {U.shape[1]} ; input_names={input_names}")

    K_ss = (-Cs_np @ np.linalg.inv(As_np) @ Bs_np + Ds_np)
    Y = (K_ss @ U.T).T

    y_ss = pd.DataFrame(Y, index=time, columns=["T_room1", "T_room2"][:Y.shape[1]])

    print("\n=== Hourly steady-state room temperatures (first 6 hours) ===")
    print(y_ss.head(6).round(2).to_string())

    print("\n=== Last hour ===")
    print(y_ss.tail(1).round(2).to_string())

    print("\n[OK] Done.")


if __name__ == "__main__":
    main()
