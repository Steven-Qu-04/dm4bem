# -*- coding: utf-8 -*-
"""
check_building.py - dm4bem multi-room + RTS, driven by Radiance CSV matrices
(中文 CONFIG；室外温度 CSV：hour 按顺序取，不做对齐)

Radiance 目录结构：
  <工作目录>/results_YYYYMMDD/Room_i_j_k/YYYYMMDD_HH00/*.csv

CSV 规则：
- 每个 CSV 是 2D 矩阵 Z [W/m²]，网格 1m×1m
- Z < 无效值阈值（默认 0.0，即 -1）为 mask，不参与积分
- 面总入射功率 W = sum(valid) * 1.0（因为每格 1㎡）
- 外表面：文件名含 外表面标记（默认 "_out_"）
- 室内面：文件名不含 外表面标记，且满足 仅包含/排除关键词

To（室外温度）CSV：
- 不做时间对齐，直接按顺序取前 N 行
- 若列名不存在：优先找 temp_C，否则找第一个非 hour 的数值列
"""

import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import dm4bem


# =========================
# 中文 CONFIG（你只需要改这里）
# =========================
CONFIG = {
    "工作目录": r".",          # 【用户必填】Radiance 工程根目录（里面应有 results_YYYYMMDD）
    "目标日期": "20160101",                  # 【用户必填】或设为 None 自动选择最新 results_*
    "结果目录": None,                        # 【可选】若你想直接指定 results_YYYYMMDD 的绝对路径，写这里

    "房间文件夹前缀": "Room_",
    "小时文件夹正则": r"^\d{8}_\d{2}00$",
    "CSV后缀": "_matrix_Wm2.csv",
    "无效值阈值": 0.0,                        # Z < 0 视为无效（-1 mask）

    "外表面标记": "_out_",
    "室内面仅包含关键词": [],                 # e.g. ["wall_"]
    "室内面排除关键词": [],                   # e.g. ["floor_", "ceiling_"]

    "模拟小时数": 24,

    # 室外温度
    "室外温度来源": "CSV",                   # "随机" | "常数" | "CSV"
    "室外温度_常数_C": 0.0,
    "室外温度_随机": {"均值_C": 0.0, "标准差_C": 5.0, "随机种子": 123},
    "室外温度_CSV": {
        "路径": r"rep_temp_24h.csv",
        "列名": "To_C",  # 你的文件是 temp_C；这里即便不改也会自动兜底
    },

    # 热工参数
    "热工参数": {
        "外墙U值_W每平米K": 0.50,
        "房间间墙U值_W每平米K": 1.50,
        "空气密度_kg每立方": 1.2,
        "空气比热_J每kgK": 1000.0,
    },

    # 辐射->空气
    "辐射到空气": {
        "外表面即时到空气比例": 0.05,         # 占位
        "室内表面短波吸收率": 0.60,
        "RTS_24系数": [
            0.54, 0.16, 0.08, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00
        ],
    },

    # 室内其它得热
    "室内得热": {
        "模式": "常数",                         # "常数" | "随机" | "按房间"
        "常数_W": 0,
        "随机": {"均值_W": 80.0, "标准差_W": 20.0, "随机种子": 999},
        "按房间_W": {},
    },
}


# ---------------------------
# Building geometry schema (你的坐标重构逻辑保持不变)
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
                        "id": f"R{i}{'F' if j == 1 else 'B'}_{k}F",
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

    A_x = dy * dz
    A_y = dx * dz
    A_z = dx * dy

    return {
        "west": {"area": A_x},
        "east": {"area": A_x},
        "south": {"area": A_y},
        "north": {"area": A_y},
        "floor": {"area": A_z},
        "ceiling": {"area": A_z},
    }


def face_is_exterior(room: dict, face_name: str) -> bool:
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


def face_has_window(room: dict, face_name: str) -> bool:
    if face_name in ("floor", "ceiling"):
        return False
    if not face_is_exterior(room, face_name):
        return False
    return True


def describe_reconstruction(rooms: list[dict]) -> str:
    n_rooms = len(rooms)
    floors = sorted(set(r["k"] for r in rooms))
    n_end = sum(1 for r in rooms if r["i"] == 4)
    return (
        f"Reconstruction result: {n_rooms} rooms (4 along X x 2 rows along Y x {len(floors)} floors). "
        f"End rooms (i=4) count={n_end}.\n"
        f"Exterior envelope for room faces checked at x in {{0.5,59.5}}, y in {{0.5,23.5}}."
    )


def build_adjacency(rooms: list[dict]) -> dict:
    idx = {(r["i"], r["j"], r["k"]): r["id"] for r in rooms}
    neigh = {r["id"]: [] for r in rooms}
    for r in rooms:
        i, j, k = r["i"], r["j"], r["k"]
        if i < 4:
            neigh[r["id"]].append(idx[(i + 1, j, k)])
        if i > 1:
            neigh[r["id"]].append(idx[(i - 1, j, k)])
    return neigh


def shared_wall_area(rooms: list[dict], rid_a: str) -> float:
    ra = next(rr for rr in rooms if rr["id"] == rid_a)
    dy = ra["y"][1] - ra["y"][0]
    dz = ra["z"][1] - ra["z"][0]
    return dy * dz


# ---------------------------
# CONFIG 解析
# ---------------------------

def resolve_results_dir(config: dict) -> Path:
    if config.get("结果目录"):
        p = Path(config["结果目录"])
        if not p.exists():
            raise FileNotFoundError(f"结果目录不存在: {p}")
        return p

    workdir = Path(config["工作目录"])
    if not workdir.exists():
        raise FileNotFoundError(f"工作目录不存在: {workdir}")

    target = config.get("目标日期")
    if target:
        p = workdir / f"results_{target}"
        if not p.exists():
            raise FileNotFoundError(f"未找到目标日期结果目录: {p}")
        return p

    candidates = sorted(workdir.glob("results_????????"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"工作目录下未找到 results_YYYYMMDD: {workdir}")
    return candidates[0]


def make_outdoor_temperature(time_index: pd.DatetimeIndex, config: dict) -> np.ndarray:
    """
    关键修改：
    - CSV 模式：严格按顺序取值（hour 列忽略，不做时间对齐）
    - 列名不匹配：自动兜底 temp_C 或第一个非 hour 的数值列
    """
    src = config.get("室外温度来源", "随机")
    n = len(time_index)

    if src == "常数":
        return np.full(n, float(config.get("室外温度_常数_C", 0.0)), dtype=float)

    if src == "随机":
        p = config.get("室外温度_随机", {})
        rng = np.random.default_rng(int(p.get("随机种子", 123)))
        return rng.normal(
            loc=float(p.get("均值_C", 0.0)),
            scale=float(p.get("标准差_C", 5.0)),
            size=n
        ).astype(float)

    if src == "CSV":
        p = config.get("室外温度_CSV", {})
        csv_path = Path(p.get("路径", ""))
        col = str(p.get("列名", "")).strip()

        if not csv_path.exists():
            raise FileNotFoundError(f"室外温度 CSV 文件不存在: {csv_path}")

        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError(f"室外温度 CSV 为空: {csv_path}")

        # 1) 若指定列不存在：自动兜底 temp_C 或第一个非 hour 的数值列
        if col and col in df.columns:
            use_col = col
        else:
            # 优先 temp_C
            if "temp_C" in df.columns:
                use_col = "temp_C"
                if col and col != use_col:
                    print(f"[WARN] 未找到列 '{col}'，自动改用 '{use_col}'")
            else:
                # 找第一个非 hour 的数值列
                hour_like = {"hour", "hours", "hr"}
                candidates = []
                for c in df.columns:
                    if str(c).lower() in hour_like:
                        continue
                    s = pd.to_numeric(df[c], errors="coerce")
                    if s.notna().sum() > 0:
                        candidates.append(c)
                if not candidates:
                    raise ValueError(f"室外温度 CSV 中找不到可用的数值列。实际列={list(df.columns)}")
                use_col = candidates[0]
                if col:
                    print(f"[WARN] 未找到列 '{col}'，自动改用 '{use_col}'")
                else:
                    print(f"[WARN] 未指定列名，自动选用 '{use_col}'")

        # 2) 按顺序取前 n 行（hour 列忽略，不对齐）
        vals = pd.to_numeric(df[use_col], errors="coerce").to_numpy(dtype=float)
        if len(vals) < n:
            raise ValueError(f"室外温度 CSV 行数不足: {len(vals)} < {n} (需要 {n} 小时)")
        return vals[:n]

    raise ValueError(f"未知 室外温度来源: {src}（应为 '随机'/'常数'/'CSV'）")


def make_internal_gains(room_ids: list[str], time_index: pd.DatetimeIndex, config: dict) -> dict[str, np.ndarray]:
    gcfg = config.get("室内得热", {})
    mode = gcfg.get("模式", "常数")
    n = len(time_index)
    out: dict[str, np.ndarray] = {}

    if mode == "常数":
        val = float(gcfg.get("常数_W", 0.0))
        for rid in room_ids:
            out[rid] = np.full(n, val, dtype=float)
        return out

    if mode == "随机":
        p = gcfg.get("随机", {})
        rng = np.random.default_rng(int(p.get("随机种子", 999)))
        mean = float(p.get("均值_W", 80.0))
        std = float(p.get("标准差_W", 20.0))
        for rid in room_ids:
            out[rid] = rng.normal(loc=mean, scale=std, size=n).astype(float)
        return out

    if mode == "按房间":
        per = gcfg.get("按房间_W", {}) or {}
        for rid in room_ids:
            out[rid] = np.full(n, float(per.get(rid, 0.0)), dtype=float)
        return out

    raise ValueError(f"未知 室内得热.模式: {mode}（应为 '常数'/'随机'/'按房间'）")


# ---------------------------
# Radiance CSV 读取
# ---------------------------

def load_csv_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(str(path), delimiter=",")


def sum_valid_power_W(Z: np.ndarray, invalid_lt: float) -> float:
    valid = Z[Z >= invalid_lt]
    if valid.size == 0:
        return 0.0
    return float(np.sum(valid))  # 1m² 网格 -> sum 即 W


def interior_file_allowed(filename: str, config: dict) -> bool:
    fn = filename.lower()
    inc = [s.lower() for s in (config.get("室内面仅包含关键词") or [])]
    exc = [s.lower() for s in (config.get("室内面排除关键词") or [])]

    if inc:
        return any(k in fn for k in inc)
    if exc:
        return not any(k in fn for k in exc)
    return True


def load_room_hour_powers(hour_dir: Path, config: dict) -> tuple[float, float]:
    P_ext = 0.0
    P_int = 0.0
    if not hour_dir.exists():
        return P_ext, P_int

    suffix = config.get("CSV后缀", "_matrix_Wm2.csv")
    ext_token = str(config.get("外表面标记", "_out_"))
    invalid_lt = float(config.get("无效值阈值", 0.0))

    for fp in hour_dir.iterdir():
        if not fp.is_file():
            continue
        name = fp.name
        if not name.endswith(suffix):
            continue

        try:
            Z = load_csv_matrix(fp)
        except Exception:
            print(f"[WARN] Failed to read CSV: {fp}")
            continue

        p = sum_valid_power_W(Z, invalid_lt=invalid_lt)

        if ext_token in name:
            P_ext += p
        else:
            if interior_file_allowed(name, config):
                P_int += p

    return P_ext, P_int


def load_radiance_day(results_dir: Path, rooms: list[dict], config: dict) -> tuple[pd.DatetimeIndex, dict, dict]:
    hour_re = re.compile(config.get("小时文件夹正则", r"^\d{8}_\d{2}00$"))
    n_hours = int(config.get("模拟小时数", 24))

    base = results_dir.name
    m = re.match(r"^results_(\d{8})$", base)
    ymd = m.group(1) if m else None

    radiance_map = {f"Room_{r['i']}_{r['j']}_{r['k']}": r["id"] for r in rooms}
    P_ext_by_rid = {r["id"]: np.zeros(n_hours, dtype=float) for r in rooms}
    P_int_by_rid = {r["id"]: np.zeros(n_hours, dtype=float) for r in rooms}

    for rf in results_dir.iterdir():
        if not rf.is_dir():
            continue
        rid = radiance_map.get(rf.name)
        if not rid:
            continue

        for hf in rf.iterdir():
            if not hf.is_dir():
                continue
            if not hour_re.match(hf.name):
                continue

            ymd_hf = hf.name[:8]
            hh = int(hf.name[9:11])
            if ymd is None:
                ymd = ymd_hf

            if 0 <= hh < n_hours:
                P_ext, P_int = load_room_hour_powers(hf, config)
                P_ext_by_rid[rid][hh] = P_ext
                P_int_by_rid[rid][hh] = P_int

    if ymd is None:
        raise RuntimeError(f"无法从 {results_dir} 推断日期（未找到小时文件夹）。")

    time_index = pd.date_range(
        start=f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]} 00:00:00",
        periods=n_hours,
        freq="1h",
    )

    for rid in P_ext_by_rid:
        if np.allclose(P_ext_by_rid[rid], 0.0) and np.allclose(P_int_by_rid[rid], 0.0):
            print(f"[WARN] Room {rid}: P_ext/P_int 全为 0（可能缺 CSV 或全为 -1 mask）。")

    return time_index, P_ext_by_rid, P_int_by_rid


# ---------------------------
# dm4bem 核心辅助
# ---------------------------

def extract_input_names_keep_duplicates(b: np.ndarray, f: np.ndarray) -> list[str]:
    names: list[str] = []
    for v in b.reshape(-1):
        if isinstance(v, str) and v.strip():
            names.append(v)
    for v in f.reshape(-1):
        if isinstance(v, str) and v.strip():
            names.append(v)
    return names


def build_U(input_data_set: pd.DataFrame, input_names: list[str]) -> np.ndarray:
    U = np.zeros((len(input_data_set), len(input_names)), dtype=float)
    for j, name in enumerate(input_names):
        if name in input_data_set.columns:
            U[:, j] = input_data_set[name].to_numpy(dtype=float)
        else:
            U[:, j] = 0.0
    return U


def rts_convolution(Q_radiant: np.ndarray, rtf_frac: np.ndarray) -> np.ndarray:
    full = np.convolve(Q_radiant, rtf_frac, mode="full")
    return full[: len(Q_radiant)]


# ---------------------------
# Main
# ---------------------------

def main():
    print("[INFO] dm4bem imported OK.")
    print(f"[INFO] dm4bem module path: {getattr(dm4bem, '__file__', 'unknown')}")
    print(f"[INFO] RUNNING FILE: {os.path.abspath(__file__)}")

    rooms = build_rooms_from_schema(SCHEMA)
    print("\n=== Step 1: Geometry reconstruction ===")
    print(describe_reconstruction(rooms))

    room_meta = {}
    for r in rooms:
        faces = room_faces(r)
        ext_faces, win_faces = [], []
        for fn in ("west", "east", "south", "north"):
            if face_is_exterior(r, fn):
                ext_faces.append(fn)
            if face_has_window(r, fn):
                win_faces.append(fn)
        room_meta[r["id"]] = {
            "faces": faces,
            "ext_faces": ext_faces,
            "win_faces": win_faces,
            "volume": (r["x"][1] - r["x"][0]) * (r["y"][1] - r["y"][0]) * (r["z"][1] - r["z"][0]),
        }

    print("\nSample room exterior/window faces:")
    for rid in [rooms[0]["id"], rooms[3]["id"], rooms[4]["id"], rooms[7]["id"]]:
        m = room_meta[rid]
        print(f"  {rid}: exterior={m['ext_faces']} windows={m['win_faces']}")

    results_dir = resolve_results_dir(CONFIG)
    print("\n=== Step 2: Load Radiance CSV powers ===")
    print(f"[INFO] results_dir = {results_dir}")
    time, P_ext_by_rid, P_int_by_rid = load_radiance_day(results_dir, rooms, CONFIG)
    print(f"[INFO] Loaded day: {time[0]} to {time[-1]}")

    # 修改后的 To：CSV 按顺序取
    To = make_outdoor_temperature(time, CONFIG)

    room_ids = [r["id"] for r in rooms]
    internal_gains = make_internal_gains(room_ids, time, CONFIG)

    print("\n=== Step 3: Build dm4bem + RTS steady-state framework ===")

    th = CONFIG["热工参数"]
    U_ext = float(th["外墙U值_W每平米K"])
    U_int = float(th["房间间墙U值_W每平米K"])
    rho_air = float(th["空气密度_kg每立方"])
    cp_air = float(th["空气比热_J每kgK"])

    sa = CONFIG["辐射到空气"]
    eta_ext_to_air = float(sa["外表面即时到空气比例"])
    alpha_int = float(sa["室内表面短波吸收率"])
    rtf = np.array(sa["RTS_24系数"], dtype=float)
    if rtf.size != 24:
        raise ValueError(f"RTS_24系数 必须 24 个数，目前={rtf.size}")

    adjacency = build_adjacency(rooms)
    nR = len(room_ids)
    n_hours = len(time)

    # Room air gains
    Phi = {}
    for rid in room_ids:
        P_ext = P_ext_by_rid[rid]
        P_int = P_int_by_rid[rid]
        Q_ext_to_air = eta_ext_to_air * P_ext
        Q_abs_int = alpha_int * P_int
        Phi_int_to_air = rts_convolution(Q_abs_int, rtf)
        Phi[rid] = Q_ext_to_air + Phi_int_to_air + internal_gains[rid]

    # Outdoor conductance per room
    g_out = np.zeros(nR)
    for idx, rid in enumerate(room_ids):
        meta = room_meta[rid]
        A_ext = sum(meta["faces"][fn]["area"] for fn in meta["ext_faces"]) if meta["ext_faces"] else 0.0
        g_out[idx] = U_ext * A_ext

    # Unique inter-room pairs
    inter_pairs = []
    seen = set()
    for rid in room_ids:
        for nb in adjacency[rid]:
            key = tuple(sorted((rid, nb)))
            if key in seen:
                continue
            seen.add(key)
            inter_pairs.append(key)

    branches = []
    for rid in room_ids:
        branches.append(("out", rid))
    for a, b_room in inter_pairs:
        branches.append((a, b_room))

    nq = len(branches)
    A = np.zeros((nq, nR), dtype=float)
    Gdiag = np.zeros(nq, dtype=float)
    b = np.empty(nq, dtype=object)
    b[:] = 0

    # Outdoor branches
    for bi, (_, rid) in enumerate(branches[:nR]):
        j = room_ids.index(rid)
        A[bi, j] = +1.0
        Gdiag[bi] = g_out[j]
        b[bi] = "To"  # duplicate per room

    # Inter-room branches
    for bi, (a, b_room) in enumerate(branches[nR:], start=nR):
        ia = room_ids.index(a)
        ib = room_ids.index(b_room)
        A[bi, ia] = -1.0
        A[bi, ib] = +1.0
        Gdiag[bi] = U_int * shared_wall_area(rooms, a)
        b[bi] = 0

    G = np.diag(Gdiag).astype(float)

    # Air capacitance (for tc2ss completeness)
    C_air = np.array([rho_air * cp_air * room_meta[rid]["volume"] for rid in room_ids], dtype=float)
    C = np.diag(C_air).astype(float)

    f = np.array([f"Phi_{rid}" for rid in room_ids], dtype=object)
    y = np.ones(nR, dtype=float)

    # Inputs
    input_df = pd.DataFrame(index=time)
    input_df["To"] = To
    for rid in room_ids:
        input_df[f"Phi_{rid}"] = Phi[rid]

    # CORE OPERATOR
    As, Bs, Cs, Ds = dm4bem.tc2ss(A, G, C, b, f, y)

    input_names = extract_input_names_keep_duplicates(b, f)
    U = build_U(input_df, input_names)

    As_np = np.asarray(As, dtype=float)
    Bs_np = np.asarray(Bs, dtype=float)
    Cs_np = np.asarray(Cs, dtype=float)
    Ds_np = np.asarray(Ds, dtype=float)

    if U.shape[1] != Bs_np.shape[1]:
        raise ValueError(f"nu mismatch: Bs expects {Bs_np.shape[1]}, U has {U.shape[1]}")

    # Steady-state gain
    K_ss = (-Cs_np @ np.linalg.inv(As_np) @ Bs_np + Ds_np)
    Y = (K_ss @ U.T).T
    T = pd.DataFrame(Y, index=time, columns=room_ids)

    print("\n=== Result: first 3 rooms, first 6 hours ===")
    print(T[room_ids[:3]].head(6).round(2).to_string())

    print("\n=== Result: last hour, first 6 rooms ===")
    print(T[room_ids[:6]].tail(1).round(2).to_string())

    print("\n[OK] Done.")


if __name__ == "__main__":
    main()
