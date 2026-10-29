"""CAD-derived asymmetric tank track & dual-sprocket visual kinematics for T-Rax.

Extracted via sub-micrometer 2D Procrustes ICP from the 20-shoe Shapr3D assembly
('Trax Robot.obj' / 'Trax Robot.step'). Purely updates massless visual geoms
(group 2, contype=0, conaffinity=0) while composing MuJoCo's internal principal-inertia
mesh frame (mesh_pos, mesh_quat), so physics contact dynamics (geom_surfacevel)
remain strictly decoupled and deterministic whether visual animation is enabled or not.
"""
import mujoco
import numpy as np


class TraxTrackAnimator:
    """Periodic C1 cubic Hermite spline animator for 20 track shoes + 2 sprockets per side."""

    NUM_SHOES = 20
    TRACK_X = (-0.035018, 0.035018)  # Left (-X_mj), Right (+X_mj) lateral centers [m]
    TRACK_WIDTH = 0.070036           # Center-to-center track gauge [m]
    PERIMETER = 0.172561             # Closed 20-shoe belt loop arc length [m]
    FRONT_SPROCKET_RADIUS = 0.0192   # Large front drive sprocket pitch radius [m]
    REAR_SPROCKET_RADIUS = 0.0124    # Smaller rear idler sprocket pitch radius [m]
    FRONT_SPROCKET_POS = (
        np.array([-0.035000, -0.000486, 0.001242], dtype=np.float64),
        np.array([+0.035000, -0.000486, 0.001242], dtype=np.float64),
    )
    REAR_SPROCKET_POS = (
        np.array([-0.035028, -0.036525, -0.001925], dtype=np.float64),
        np.array([+0.035028, -0.036525, -0.001925], dtype=np.float64),
    )

    # Exact 20 CAD shoe stations ordered along forward-driving belt circulation (+seam knot 20)
    LOOP_S = np.array([
        0.000000, 0.008544, 0.017289, 0.027963, 0.035960,
        0.044378, 0.052535, 0.061190, 0.069756, 0.078605,
        0.087436, 0.096237, 0.104828, 0.113419, 0.122004,
        0.130156, 0.138848, 0.147037, 0.155649, 0.164154,
        0.172561,
    ], dtype=np.float64)

    LOOP_Y = np.array([
        -0.014146, -0.022647, -0.031277, -0.041243, -0.046951,
        -0.048289, -0.044073, -0.037044, -0.029259, -0.020744,
        -0.012061, -0.003309,  0.005164,  0.012491,  0.017541,
         0.018919,  0.015907,  0.010551,  0.002604, -0.005749,
        -0.014146,
    ], dtype=np.float64)

    LOOP_Z = np.array([
        -0.019456, -0.018598, -0.017186, -0.013361, -0.007761,
         0.000550,  0.007533,  0.012583,  0.016157,  0.018565,
         0.020176,  0.021102,  0.019684,  0.015197,  0.008255,
         0.000220, -0.007933, -0.014128, -0.017445, -0.019044,
        -0.019456,
    ], dtype=np.float64)

    LOOP_ALPHA = np.array([
         0.000000000, -0.139626340, -0.174532925, -0.331612558, -1.064650844,
        -1.658062789, -2.321287905, -2.617993878, -2.775073511, -2.897246558,
        -2.967059728, -3.089232776, -3.420845334, -3.822271062, -4.223696790,
        -4.782202150, -5.113814708, -5.585053606, -6.003932627, -6.143558967,
        -6.283185307,
    ], dtype=np.float64)

    # Precompute periodic C1 tangents across the 20 segments
    _V = np.column_stack([LOOP_Y, LOOP_Z, LOOP_ALPHA])
    _DS = np.diff(LOOP_S)[:, None]
    _SLOPES = np.diff(_V, axis=0) / _DS
    _TAN = np.empty_like(_V)
    _TAN[0] = 0.5 * (_SLOPES[-1] + _SLOPES[0])
    _TAN[1:-1] = 0.5 * (_SLOPES[:-1] + _SLOPES[1:])
    _TAN[-1] = _TAN[0]

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.tread_geoms = []
        self.shoe_mesh_pos = []
        self.shoe_mesh_quat = []

        for side in ("left", "right"):
            ids = []
            for i in range(self.NUM_SHOES):
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"track_{side}_tread_{i}")
                if gid != -1:
                    ids.append(int(gid))
            self.tread_geoms.append(ids)
            if ids:
                mid = int(model.geom_dataid[ids[0]])
                if mid >= 0:
                    self.shoe_mesh_pos.append(model.mesh_pos[mid].copy())
                    self.shoe_mesh_quat.append(model.mesh_quat[mid].copy())
                else:
                    self.shoe_mesh_pos.append(np.zeros(3))
                    self.shoe_mesh_quat.append(np.array([1.0, 0.0, 0.0, 0.0]))
            else:
                self.shoe_mesh_pos.append(np.zeros(3))
                self.shoe_mesh_quat.append(np.array([1.0, 0.0, 0.0, 0.0]))

        self.front_sprockets = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"sprocket_front_{s}")
            for s in ("left", "right")
        ]
        self.rear_sprockets = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"sprocket_rear_{s}")
            for s in ("left", "right")
        ]
        self.front_mesh_pos = [self._geom_mesh_pos(g) for g in self.front_sprockets]
        self.front_mesh_quat = [self._geom_mesh_quat(g) for g in self.front_sprockets]
        self.rear_mesh_pos = [self._geom_mesh_pos(g) for g in self.rear_sprockets]
        self.rear_mesh_quat = [self._geom_mesh_quat(g) for g in self.rear_sprockets]

        self.has_visual_tracks = all(len(ids) == self.NUM_SHOES for ids in self.tread_geoms)

    def _geom_mesh_pos(self, gid: int) -> np.ndarray:
        if gid != -1 and self.model.geom_dataid[gid] >= 0:
            return self.model.mesh_pos[int(self.model.geom_dataid[gid])].copy()
        return np.zeros(3, dtype=np.float64)

    def _geom_mesh_quat(self, gid: int) -> np.ndarray:
        if gid != -1 and self.model.geom_dataid[gid] >= 0:
            return self.model.mesh_quat[int(self.model.geom_dataid[gid])].copy()
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    @staticmethod
    def _apply_rx_to_mesh(base_x, base_y, base_z, alpha, m_pos: np.ndarray, m_quat: np.ndarray):
        """Compose desired X-axis rotation R_x(alpha) and translation (base_x, base_y, base_z)
        with MuJoCo's internal principal-inertia mesh frame (m_pos, m_quat)."""
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        px = base_x + m_pos[0]
        py = base_y + m_pos[1] * ca - m_pos[2] * sa
        pz = base_z + m_pos[1] * sa + m_pos[2] * ca

        half = 0.5 * alpha
        ch = np.cos(half)
        sh = np.sin(half)
        mw, mx, my, mz = m_quat
        qw = ch * mw - sh * mx
        qx = ch * mx + sh * mw
        qy = ch * my - sh * mz
        qz = ch * mz + sh * my
        return px, py, pz, qw, qx, qy, qz

    @classmethod
    def sample_stations(cls, phase_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate (y, z, alpha) for all 20 shoes at belt displacement phase_m [meters]."""
        s = np.mod(cls.LOOP_S[:-1] + float(phase_m), cls.PERIMETER)
        idx = np.clip(np.searchsorted(cls.LOOP_S, s, side="right") - 1, 0, cls.NUM_SHOES - 1)
        h = cls.LOOP_S[idx + 1] - cls.LOOP_S[idx]
        t = (s - cls.LOOP_S[idx]) / h

        omt = 1.0 - t
        h00 = (1.0 + 2.0 * t) * omt * omt
        h10 = t * omt * omt
        h01 = t * t * (3.0 - 2.0 * t)
        h11 = t * t * (t - 1.0)

        val = (
            h00[:, None] * cls._V[idx]
            + (h10 * h)[:, None] * cls._TAN[idx]
            + h01[:, None] * cls._V[idx + 1]
            + (h11 * h)[:, None] * cls._TAN[idx + 1]
        )
        return val[:, 0], val[:, 1], val[:, 2]

    def update(self, track_phase: np.ndarray) -> None:
        """Update MuJoCo visual geom_pos and geom_quat for left/right tracks and sprockets."""
        if not self.has_visual_tracks:
            return
        pos = self.model.geom_pos
        quat = self.model.geom_quat

        for side in range(2):
            phase = float(track_phase[side])
            y, z, alpha = self.sample_stations(phase)
            ids = self.tread_geoms[side]
            px, py, pz, qw, qx, qy, qz = self._apply_rx_to_mesh(
                self.TRACK_X[side], y, z, alpha, self.shoe_mesh_pos[side], self.shoe_mesh_quat[side]
            )
            pos[ids, 0] = px
            pos[ids, 1] = py
            pos[ids, 2] = pz
            quat[ids, 0] = qw
            quat[ids, 1] = qx
            quat[ids, 2] = qy
            quat[ids, 3] = qz

            fg = self.front_sprockets[side]
            if fg != -1:
                p0 = self.FRONT_SPROCKET_POS[side]
                theta_f = -phase / self.FRONT_SPROCKET_RADIUS
                fpx, fpy, fpz, fqw, fqx, fqy, fqz = self._apply_rx_to_mesh(
                    p0[0], p0[1], p0[2], theta_f, self.front_mesh_pos[side], self.front_mesh_quat[side]
                )
                pos[fg] = (fpx, fpy, fpz)
                quat[fg] = (fqw, fqx, fqy, fqz)

            rg = self.rear_sprockets[side]
            if rg != -1:
                p0 = self.REAR_SPROCKET_POS[side]
                theta_r = -phase / self.REAR_SPROCKET_RADIUS
                rpx, rpy, rpz, rqw, rqx, rqy, rqz = self._apply_rx_to_mesh(
                    p0[0], p0[1], p0[2], theta_r, self.rear_mesh_pos[side], self.rear_mesh_quat[side]
                )
                pos[rg] = (rpx, rpy, rpz)
                quat[rg] = (rqw, rqx, rqy, rqz)
