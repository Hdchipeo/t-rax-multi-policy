#!/usr/bin/env python3
"""Build modular, CAD-accurate MuJoCo STL meshes and 20-shoe track loop table from 'Trax Robot.obj' & 'Trax Robot.glb'.

Coordinate transform from Shapr3D CAD (meters in OBJ, -Y forward, +Z up) to MuJoCo robot frame (+X right, +Y forward, +Z up):
    X_mj = -X_cad
    Y_mj = -Y_cad
    Z_mj = +Z_cad
All exported STLs are written in millimeters (for standard <mesh scale="0.001 0.001 0.001"/>)
relative to their respective kinematic link / sprocket / track-shoe local pivot origins.
"""
import json
import struct
from pathlib import Path
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "trax"
OBJ_PATH = ASSETS / "Trax Robot.obj"
GLB_PATH = ASSETS / "Trax Robot.glb"

# Exact CAD-derived pivot origins in MuJoCo trax_base frame (in millimeters)
ORIGIN_BASE = np.array([0.0, 0.0, 0.0], dtype=np.float64)
ORIGIN_PAN = np.array([-0.088, -1.175, 21.413], dtype=np.float64)
ORIGIN_TILT = np.array([-0.088, -1.175, 33.275], dtype=np.float64)
ORIGIN_SP_FL = np.array([-35.000, -0.4855, 1.242], dtype=np.float64)
ORIGIN_SP_FR = np.array([+35.000, -0.4855, 1.242], dtype=np.float64)
ORIGIN_SP_RL = np.array([-35.028, -36.525, -1.925], dtype=np.float64)
ORIGIN_SP_RR = np.array([+35.028, -36.525, -1.925], dtype=np.float64)
TRACK_RIGHT_X_MM = +35.018
TRACK_LEFT_X_MM = -35.018


def load_glb_kdtree(glb_path: Path):
    data = glb_path.read_bytes()
    json_len = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20 : 20 + json_len].decode("utf-8"))
    bin_chunk = data[20 + json_len + 8 :]

    all_pts = []
    all_ids = []
    for mi, mesh in enumerate(gltf["meshes"]):
        for prim in mesh["primitives"]:
            acc = gltf["accessors"][prim["attributes"]["POSITION"]]
            bv = gltf["bufferViews"][acc["bufferView"]]
            offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
            stride = bv.get("byteStride", 12)
            count = acc["count"]
            if stride == 12:
                pts = np.frombuffer(bin_chunk, dtype=np.float32, count=count * 3, offset=offset).reshape(count, 3)
            else:
                pts = np.ndarray((count, 3), dtype=np.float32, buffer=bin_chunk, offset=offset, strides=(stride, 4))
            step = max(1, len(pts) // 4000)
            sub = pts[::step].copy()
            all_pts.append(sub)
            all_ids.append(np.full(len(sub), mi, dtype=np.int32))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(all_pts).astype(np.float64))
    return o3d.geometry.KDTreeFlann(pcd), np.concatenate(all_ids)


def parse_obj_full(obj_path: Path):
    print(f"Loading {obj_path.name} ...")
    verts = []
    groups = []
    cur = None
    with obj_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line:
                continue
            c0 = line[0]
            if c0 == "v" and line[1] == " ":
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif c0 == "g" and line[1] == " ":
                cur = {"idx": len(groups), "name": line[2:].strip(), "faces": []}
                groups.append(cur)
            elif c0 == "f" and line[1] == " ":
                if cur is None:
                    cur = {"idx": len(groups), "name": "__default__", "faces": []}
                    groups.append(cur)
                p = line.split()
                i0 = int(p[1].split("/")[0]) - 1
                i1 = int(p[2].split("/")[0]) - 1
                i2 = int(p[3].split("/")[0]) - 1
                cur["faces"].append((i0, i1, i2))

    V_m = np.asarray(verts, dtype=np.float64)
    print(f"Loaded {len(V_m):,} vertices across {len(groups)} groups.")
    return V_m, groups


def make_o3d_mesh(v_mm: np.ndarray, f_local: np.ndarray, target_tris: int = 0) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v_mm)
    mesh.triangles = o3d.utility.Vector3iVector(f_local)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if target_tris > 0 and len(mesh.triangles) > int(target_tris * 1.15):
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_tris)
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
    return mesh


def write_binary_stl(path: Path, meshes: list[o3d.geometry.TriangleMesh], origin_mm: np.ndarray):
    all_v = []
    all_f = []
    v_offset = 0
    for m in meshes:
        if len(m.triangles) == 0:
            continue
        v = np.asarray(m.vertices, dtype=np.float64) - origin_mm[None, :]
        f = np.asarray(m.triangles, dtype=np.int32) + v_offset
        all_v.append(v)
        all_f.append(f)
        v_offset += len(v)

    V = np.vstack(all_v)
    F = np.vstack(all_f)
    t0 = V[F[:, 0]]
    t1 = V[F[:, 1]]
    t2 = V[F[:, 2]]
    normals = np.cross(t1 - t0, t2 - t0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, np.maximum(norms, 1e-12))

    n_tris = len(F)
    stl_dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("v0", "<f4", (3,)),
        ("v1", "<f4", (3,)),
        ("v2", "<f4", (3,)),
        ("attr", "<u2"),
    ])
    records = np.zeros(n_tris, dtype=stl_dtype)
    records["normal"] = normals.astype(np.float32)
    records["v0"] = t0.astype(np.float32)
    records["v1"] = t1.astype(np.float32)
    records["v2"] = t2.astype(np.float32)

    header = f"T-Rax CAD Mesh ({path.name}) - mm".encode("ascii").ljust(80, b"\0")[:80]
    with path.open("wb") as fp:
        fp.write(header)
        fp.write(struct.pack("<I", n_tris))
        fp.write(records.tobytes())
    print(f"  Wrote {path.name:32s} | {n_tris:6,d} tris | {path.stat().st_size / 1024:7.1f} KB")
    return n_tris, V, F


def write_crease_obj(path: Path, meshes: list[o3d.geometry.TriangleMesh], origin_mm: np.ndarray, crease_deg: float = 35.0):
    """Write Wavefront OBJ with area-weighted crease-angle vertex normals so flat CAD plates have zero Gouraud streaking while fillets stay smooth."""
    all_v = []
    all_f = []
    v_offset = 0
    for m in meshes:
        if len(m.triangles) == 0:
            continue
        v = np.asarray(m.vertices, dtype=np.float64) - origin_mm[None, :]
        f = np.asarray(m.triangles, dtype=np.int32) + v_offset
        all_v.append(v)
        all_f.append(f)
        v_offset += len(v)

    V = np.vstack(all_v)
    F = np.vstack(all_f)
    t0, t1, t2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    raw_n = np.cross(t1 - t0, t2 - t0)
    areas2 = np.linalg.norm(raw_n, axis=1, keepdims=True)
    unit_n = np.divide(raw_n, np.maximum(areas2, 1e-12))

    # Build vertex -> incident face list
    v2f = [[] for _ in range(len(V))]
    for fi, (a, b, c) in enumerate(F):
        v2f[a].append(fi)
        v2f[b].append(fi)
        v2f[c].append(fi)

    cos_thresh = float(np.cos(np.radians(crease_deg)))
    corner_normals = []
    face_vn_idx = np.empty_like(F)
    for fi in range(len(F)):
        ni = unit_n[fi]
        for ci in range(3):
            vi = F[fi, ci]
            inc = v2f[vi]
            if len(inc) == 1:
                cn = ni
            else:
                inc_arr = np.asarray(inc, dtype=np.int32)
                dots = unit_n[inc_arr] @ ni
                keep = inc_arr[dots >= cos_thresh]
                summed = raw_n[keep].sum(axis=0)
                nrm = float(np.linalg.norm(summed))
                cn = summed / nrm if nrm > 1e-12 else ni
            corner_normals.append(cn)
            face_vn_idx[fi, ci] = len(corner_normals)

    CN = np.round(np.asarray(corner_normals, dtype=np.float64), 5)
    uniq_cn, inv_idx = np.unique(CN, axis=0, return_inverse=True)
    fn_1based = inv_idx.reshape(F.shape) + 1
    fv_1based = F + 1

    lines = [f"# T-Rax CAD Mesh ({path.name}) - mm with {crease_deg:.0f}-deg crease normals\n"]
    lines.extend(f"v {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in V)
    lines.extend(f"vn {nx:.5f} {ny:.5f} {nz:.5f}\n" for nx, ny, nz in uniq_cn)
    lines.extend(
        f"f {fv[0]}//{fn[0]} {fv[1]}//{fn[1]} {fv[2]}//{fn[2]}\n"
        for fv, fn in zip(fv_1based, fn_1based)
    )
    path.write_text("".join(lines), encoding="ascii")
    n_tris = len(F)
    print(f"  Wrote {path.name:32s} | {n_tris:6,d} tris | {path.stat().st_size / 1024:7.1f} KB")
    return n_tris, V, F


def register_2d_icp(src_yz: np.ndarray, dst_yz: np.ndarray, init_angle: float) -> tuple[float, float, float, float]:
    """Exact 2D rigid registration in (Y_mj, Z_mj) mapping canonical centered src_yz -> dst_yz."""
    pcd_dst = o3d.geometry.PointCloud()
    pts3_dst = np.column_stack([np.zeros(len(dst_yz)), dst_yz])
    pcd_dst.points = o3d.utility.Vector3dVector(pts3_dst)
    kdtree = o3d.geometry.KDTreeFlann(pcd_dst)

    c_dst = dst_yz.mean(axis=0)
    alpha = init_angle
    t = c_dst.copy()

    # Subsample 1500 evenly spaced vertices from canonical shoe for sub-micrometer 2D Procrustes ICP
    sub_src = src_yz[:: max(1, len(src_yz) // 1500)]
    for _ in range(35):
        ca, sa = np.cos(alpha), np.sin(alpha)
        R = np.array([[ca, -sa], [sa, ca]])
        transformed = sub_src @ R.T + t[None, :]
        matched = np.empty_like(transformed)
        for i, pt in enumerate(transformed):
            _, idx, _ = kdtree.search_knn_vector_3d([0.0, pt[0], pt[1]], 1)
            matched[i] = dst_yz[idx[0]]
        # 2D Kabsch / Procrustes SVD update from sub_src -> matched
        mu_s = sub_src.mean(axis=0)
        mu_m = matched.mean(axis=0)
        H = (sub_src - mu_s).T @ (matched - mu_m)
        U, _, Vt = np.linalg.svd(H)
        R_new = Vt.T @ U.T
        if np.linalg.det(R_new) < 0:
            Vt[1, :] *= -1
            R_new = Vt.T @ U.T
        alpha_new = float(np.arctan2(R_new[1, 0], R_new[0, 0]))
        t_new = mu_m - R_new @ mu_s
        if abs(alpha_new - alpha) < 1e-7 and np.linalg.norm(t_new - t) < 1e-6:
            alpha, t = alpha_new, t_new
            break
        alpha, t = alpha_new, t_new

    ca, sa = np.cos(alpha), np.sin(alpha)
    R = np.array([[ca, -sa], [sa, ca]])
    trans = sub_src @ R.T + t[None, :]
    errs = [np.sqrt(kdtree.search_knn_vector_3d([0.0, p[0], p[1]], 1)[2][0]) for p in trans]
    return float(t[0]), float(t[1]), float(alpha), float(np.mean(errs))


def main():
    kdtree, glb_ids = load_glb_kdtree(GLB_PATH)
    V_m, groups = parse_obj_full(OBJ_PATH)
    center_offset = np.array([0.0, 0.0061235, 0.029393], dtype=np.float64)

    # Convert all vertices to MuJoCo robot frame in millimeters: X_mj = -X_cad, Y_mj = -Y_cad, Z_mj = +Z_cad
    # Note det(diag(-1, -1, +1)) == +1 (proper 180-deg rotation around Z, preserves triangle winding!)
    V_mj_mm = np.column_stack([-V_m[:, 0], -V_m[:, 1], V_m[:, 2]]) * 1000.0

    buckets = {
        "chassis_black": [],
        "chassis_white": [],
        "chassis_silver": [],
        "chassis_gold": [],
        "sprocket_front_right": [],
        "sprocket_front_left": [],
        "sprocket_rear_right": [],
        "sprocket_rear_left": [],
        "track_shoes_right": [],
        "track_shoes_left": [],
        "neck_pan_bracket": [],
        "head_black": [],
        "head_white": [],
        "head_silver": [],
        "head_lens": [],
    }

    pan_ids = {134, 135, 80, 40}
    sp_fr = {117, 57, 68, 98}
    sp_fl = {75, 72, 70, 131}
    sp_rr = {94, 171}
    sp_rl = {34, 206}
    chassis_white_ids = {139, 140, 143, 8, 168}
    head_white_armor_ids = {112, 173, 69, 82, 126, 127, 28, 62, 100, 115, 141, 190, 102}

    group_meta = []
    for g in groups:
        idx = g["idx"]
        F_g = np.asarray(g["faces"], dtype=np.int32)
        vmin_idx = int(F_g.min())
        vmax_idx = int(F_g.max())
        v_cad_m = V_m[vmin_idx : vmax_idx + 1]
        v_mj_mm = V_mj_mm[vmin_idx : vmax_idx + 1].copy()
        f_local = F_g - vmin_idx

        # Match GLB material index
        step = max(1, len(v_cad_m) // 25)
        s_cad = v_cad_m[::step] - center_offset
        s_glb = np.column_stack([s_cad[:, 0], s_cad[:, 2], -s_cad[:, 1]])
        votes = {}
        for pt in s_glb:
            _, kidx, _ = kdtree.search_knn_vector_3d(pt, 1)
            mid = int(glb_ids[kidx[0]])
            votes[mid] = votes.get(mid, 0) + 1
        glb = max(votes.items(), key=lambda kv: kv[1])[0]

        zmin_cad = float(v_cad_m[:, 2].min() * 1000.0)
        xmean_cad = float(v_cad_m[:, 0].mean() * 1000.0)
        fc = len(F_g)

        info = {"idx": idx, "name": g["name"], "v_mj_mm": v_mj_mm, "f_local": f_local, "faces": fc, "glb": glb}
        group_meta.append(info)

        if fc in (60958, 60960):
            if xmean_cad < 0:
                buckets["track_shoes_right"].append(info)
            else:
                buckets["track_shoes_left"].append(info)
        elif idx in sp_fr:
            buckets["sprocket_front_right"].append(info)
        elif idx in sp_fl:
            buckets["sprocket_front_left"].append(info)
        elif idx in sp_rr:
            buckets["sprocket_rear_right"].append(info)
        elif idx in sp_rl:
            buckets["sprocket_rear_left"].append(info)
        elif idx in pan_ids:
            buckets["neck_pan_bracket"].append(info)
        elif idx == 118 or zmin_cad >= 38.0:
            if idx == 103:
                buckets["head_lens"].append(info)
            elif idx in head_white_armor_ids:
                buckets["head_white"].append(info)
            elif glb == 4:
                buckets["head_silver"].append(info)
            else:
                buckets["head_black"].append(info)
        else:
            if idx in chassis_white_ids:
                buckets["chassis_white"].append(info)
            elif glb in (11, 16):
                buckets["chassis_gold"].append(info)
            elif glb in (8, 10, 12, 14, 23, 26, 28, 31):
                buckets["chassis_silver"].append(info)
            else:
                buckets["chassis_black"].append(info)

    assert sum(len(v) for v in buckets.values()) == 235, "Incomplete group partition!"

    def build_bucket_meshes(items, per_group_cap=0):
        out = []
        for it in items:
            cap = per_group_cap if (per_group_cap > 0 and it["faces"] > per_group_cap) else 0
            out.append(make_o3d_mesh(it["v_mj_mm"], it["f_local"], target_tris=cap))
        return out

    print("\nGenerating production binary STL meshes in assets/trax/ ...")
    total_tris = 0

    # 1. Fixed Chassis (trax_base)
    n, _, _ = write_binary_stl(ASSETS / "trax_base.stl", build_bucket_meshes(buckets["chassis_black"], per_group_cap=4500), ORIGIN_BASE)
    total_tris += n
    (ASSETS / "trax_chassis_white.stl").unlink(missing_ok=True)
    n, _, _ = write_crease_obj(ASSETS / "trax_chassis_white.obj", build_bucket_meshes(buckets["chassis_white"], per_group_cap=0), ORIGIN_BASE)
    total_tris += n
    n, _, _ = write_binary_stl(ASSETS / "trax_chassis_silver.stl", build_bucket_meshes(buckets["chassis_silver"], per_group_cap=4000), ORIGIN_BASE)
    total_tris += n
    n, _, _ = write_binary_stl(ASSETS / "trax_chassis_gold.stl", build_bucket_meshes(buckets["chassis_gold"], per_group_cap=800), ORIGIN_BASE)
    total_tris += n

    # 2. Rotating Front & Rear Sprockets (centered at exact shaft axes)
    def build_sprocket(items, total_target):
        raw_total = sum(it["faces"] for it in items)
        out = []
        for it in items:
            share = max(400, int(round(total_target * (it["faces"] / raw_total))))
            out.append(make_o3d_mesh(it["v_mj_mm"], it["f_local"], target_tris=share))
        return out

    n, _, _ = write_binary_stl(ASSETS / "trax_sprocket_front_left.stl", build_sprocket(buckets["sprocket_front_left"], 4800), ORIGIN_SP_FL)
    total_tris += n
    n, _, _ = write_binary_stl(ASSETS / "trax_sprocket_front_right.stl", build_sprocket(buckets["sprocket_front_right"], 4800), ORIGIN_SP_FR)
    total_tris += n
    n, _, _ = write_binary_stl(ASSETS / "trax_sprocket_rear_left.stl", build_sprocket(buckets["sprocket_rear_left"], 3000), ORIGIN_SP_RL)
    total_tris += n
    n, _, _ = write_binary_stl(ASSETS / "trax_sprocket_rear_right.stl", build_sprocket(buckets["sprocket_rear_right"], 3000), ORIGIN_SP_RR)
    total_tris += n

    # 3. Pan Link L-Bracket (centered at ORIGIN_PAN)
    n, _, _ = write_binary_stl(ASSETS / "trax_pan_bracket.stl", build_bucket_meshes(buckets["neck_pan_bracket"], per_group_cap=0), ORIGIN_PAN)
    total_tris += n

    # 4. Tilt + Head Link (centered at ORIGIN_TILT)
    n, _, _ = write_binary_stl(ASSETS / "trax_head.stl", build_bucket_meshes(buckets["head_black"], per_group_cap=4500), ORIGIN_TILT)
    total_tris += n
    (ASSETS / "trax_head_white.stl").unlink(missing_ok=True)
    n, _, _ = write_crease_obj(ASSETS / "trax_head_white.obj", build_bucket_meshes(buckets["head_white"], per_group_cap=0), ORIGIN_TILT)
    total_tris += n
    n, _, _ = write_binary_stl(ASSETS / "trax_head_silver.stl", build_bucket_meshes(buckets["head_silver"], per_group_cap=0), ORIGIN_TILT)
    total_tris += n
    n, _, _ = write_binary_stl(ASSETS / "trax_head_lens.stl", build_bucket_meshes(buckets["head_lens"], per_group_cap=0), ORIGIN_TILT)
    total_tris += n

    # 5. Canonical Right & Left 3D CAD Track Shoes (from bottom shoe [73], decimated to ~480 tris)
    canon_info = group_meta[73]
    canon_v_full = canon_info["v_mj_mm"]
    canon_yz_center = canon_v_full[:, 1:3].mean(axis=0)
    canon_origin = np.array([TRACK_RIGHT_X_MM, canon_yz_center[0], canon_yz_center[1]], dtype=np.float64)

    shoe_r_mesh = make_o3d_mesh(canon_v_full, canon_info["f_local"], target_tris=480)
    n_shoe, V_sr, F_sr = write_binary_stl(ASSETS / "trax_track_shoe_right.stl", [shoe_r_mesh], canon_origin)

    # Exact X-mirror of decimated right shoe for left track (reversing face winding to preserve outward normals!)
    shoe_l_mesh = o3d.geometry.TriangleMesh()
    V_sl = V_sr.copy()
    V_sl[:, 0] *= -1.0
    F_sl = F_sr[:, [0, 2, 1]].copy()
    shoe_l_mesh.vertices = o3d.utility.Vector3dVector(V_sl)
    shoe_l_mesh.triangles = o3d.utility.Vector3iVector(F_sl)
    write_binary_stl(ASSETS / "trax_track_shoe_left.stl", [shoe_l_mesh], np.zeros(3))
    total_tris += 40 * n_shoe

    print(f"\nTotal rendered robot triangles (including all 40 instanced 3D track shoes): {total_tris:,d} tris")

    # 6. Register all 20 CAD track shoes via 2D ICP to extract exact 20-station periodic track loop!
    print("\nExtracting exact 20-station CAD track loop via 2D Procrustes ICP ...")
    src_yz = canon_v_full[:, 1:3] - canon_yz_center[None, :]

    def pca_init_angle(yz):
        c = yz - yz.mean(axis=0)
        _, vecs = np.linalg.eigh(c.T @ c / len(c))
        maj = vecs[:, 1]
        mn = np.array([-maj[1], maj[0]])
        if np.mean((c @ mn) ** 3) < 0:
            mn = -mn
        return mn

    min0 = pca_init_angle(src_yz)
    stations = []
    for it in buckets["track_shoes_right"]:
        dst_yz = it["v_mj_mm"][:, 1:3]
        mn = pca_init_angle(dst_yz)
        init_a = float(np.arctan2(min0[0] * mn[1] - min0[1] * mn[0], min0[0] * mn[0] + min0[1] * mn[1]))
        ty, tz, alpha, err = register_2d_icp(src_yz, dst_yz, init_a)
        stations.append({"idx": it["idx"], "name": it["name"], "y_mm": ty, "z_mm": tz, "alpha_rad": alpha, "icp_err_mm": err})

    # Order stations in FORWARD-DRIVING belt circulation direction:
    # Bottom run moves REARWARD (-Y_mj), Rear pulley wraps UPWARD, Top run moves FORWARD (+Y_mj), Front pulley wraps DOWNWARD.
    # Around center (-18.5, 0.0) in (Y_mj, Z_mj), that is CLOCKWISE (decreasing atan2(Z, Y + 18.5))!
    stations.sort(key=lambda s: -np.arctan2(s["z_mm"], s["y_mm"] + 18.5))
    # Rotate list so station #0 is the canonical bottom-center shoe [73]!
    i0 = next(i for i, s in enumerate(stations) if s["idx"] == 73)
    stations = stations[i0:] + stations[:i0]

    # Compute cumulative chord arc-lengths and unwrap alpha_rad continuously (decreasing by -2*pi per full forward loop)
    arc_s = [0.0]
    alphas = [stations[0]["alpha_rad"]]
    for k in range(1, 20):
        prev, cur = stations[k - 1], stations[k]
        ds = float(np.hypot(cur["y_mm"] - prev["y_mm"], cur["z_mm"] - prev["z_mm"]))
        arc_s.append(arc_s[-1] + ds)
        # Unwrap angle close to previous (expecting negative step ~ -2*pi/20)
        da = (cur["alpha_rad"] - alphas[-1] + np.pi) % (2 * np.pi) - np.pi
        alphas.append(alphas[-1] + da)

    close_ds = float(np.hypot(stations[0]["y_mm"] - stations[-1]["y_mm"], stations[0]["z_mm"] - stations[-1]["z_mm"]))
    perimeter_mm = arc_s[-1] + close_ds
    total_dalpha = (stations[0]["alpha_rad"] - alphas[-1] + np.pi) % (2 * np.pi) - np.pi
    end_alpha = alphas[-1] + total_dalpha

    max_icp = max(s["icp_err_mm"] for s in stations)
    mean_icp = float(np.mean([s["icp_err_mm"] for s in stations]))
    print(f"  20-shoe ICP registration accuracy: mean={mean_icp:.4f} mm, max={max_icp:.4f} mm")
    print(f"  Exact loop perimeter = {perimeter_mm:.4f} mm ({perimeter_mm/1000.0:.6f} m), end_alpha = {np.degrees(end_alpha):.2f} deg")

    loop_meta = {
        "perimeter_m": round(perimeter_mm / 1000.0, 6),
        "track_x_m": [round(TRACK_LEFT_X_MM / 1000.0, 6), round(TRACK_RIGHT_X_MM / 1000.0, 6)],
        "sprocket_front_pos_m": {
            "left": [round(x / 1000.0, 6) for x in ORIGIN_SP_FL],
            "right": [round(x / 1000.0, 6) for x in ORIGIN_SP_FR],
            "pitch_radius_m": 0.0192,
        },
        "sprocket_rear_pos_m": {
            "left": [round(x / 1000.0, 6) for x in ORIGIN_SP_RL],
            "right": [round(x / 1000.0, 6) for x in ORIGIN_SP_RR],
            "pitch_radius_m": 0.0124,
        },
        "s_m": [round(s / 1000.0, 6) for s in arc_s] + [round(perimeter_mm / 1000.0, 6)],
        "y_m": [round(st["y_mm"] / 1000.0, 6) for st in stations] + [round(stations[0]["y_mm"] / 1000.0, 6)],
        "z_m": [round(st["z_mm"] / 1000.0, 6) for st in stations] + [round(stations[0]["z_mm"] / 1000.0, 6)],
        "alpha_rad": [round(a, 6) for a in alphas] + [round(end_alpha, 6)],
    }
    meta_out = ASSETS / "trax_cad_kinematics.json"
    meta_out.write_text(json.dumps(loop_meta, indent=2) + "\n")
    print(f"  Saved exact CAD track & joint kinematics metadata to {meta_out}")


if __name__ == "__main__":
    main()
