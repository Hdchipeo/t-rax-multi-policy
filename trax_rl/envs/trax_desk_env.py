"""Desk-scale MuJoCo environment with a free base and two continuous moving tracks.

This scene has a different observation/action contract from the room policies.
Software geofencing uses simulator pose; cliff probes simulate extra hardware.
"""
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from trax_rl.dynamics.desk_humans import DeskHumans
from trax_rl.dynamics.track_kinematics import TraxTrackAnimator


class TraxDeskEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 20}
    DESK_HALF = np.array([.70, .375])
    TABLE_HEIGHT = .76
    ZONE_CENTER = np.array([.40, .02])
    ZONE_HALF = np.array([.25, .20])
    ROBOT_RADIUS = .075
    TRACK_WIDTH = TraxTrackAnimator.TRACK_WIDTH
    MAX_LINEAR = .08
    MAX_ANGULAR = 1.0

    def __init__(self, xml_path=None, max_episode_steps=1800, frameskip=25,
                 restrict_to_zone=True, safety_enabled=True,
                 domain_randomization=False, render_mode=None, human_scenarios=True, animate_tracks=True):
        super().__init__()
        if frameskip <= 0 or max_episode_steps <= 0:
            raise ValueError('frameskip and max_episode_steps must be positive')
        project_root = Path(__file__).resolve().parents[2]
        path = Path(xml_path) if xml_path else Path('assets/trax/desk_scene.xml')
        if not path.is_absolute():path = project_root / path
        if tuple(int(p) for p in mujoco.__version__.split('.')[:2]) < (3,11):
            raise RuntimeError('Continuous tracks require MuJoCo >= 3.11')
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.DESK_HALF = self.model.geom('desk_top').size[:2].copy()
        self.TABLE_HEIGHT = float(self.model.geom('desk_top').pos[2]+self.model.geom('desk_top').size[2])
        self.ZONE_CENTER = self.model.geom('desk_zone_mat').pos[:2].copy()
        self.ZONE_HALF = self.model.geom('desk_zone_mat').size[:2].copy()
        self.data = mujoco.MjData(self.model)
        self.frameskip = frameskip
        self.dt = self.model.opt.timestep * frameskip
        self.max_episode_steps = max_episode_steps
        self.restrict_to_zone = restrict_to_zone
        self.safety_enabled = safety_enabled
        self.domain_randomization = domain_randomization
        self.render_mode = render_mode
        self.animate_tracks = animate_tracks
        self.robot_id = self.model.body('trax_base').id
        self.ball_id = self.model.body('desk_ball').id
        self.base_adr = self.model.jnt_qposadr[self.model.joint('robot_free').id]
        self.ball_adr = self.model.jnt_qposadr[self.model.joint('desk_ball_free').id]
        self.pan_adr = self.model.jnt_qposadr[self.model.joint('joint_pan').id]
        self.tilt_adr = self.model.jnt_qposadr[self.model.joint('joint_tilt').id]
        if not hasattr(self.model,'geom_surfacevel'):
            raise RuntimeError('Continuous tracks require MuJoCo >= 3.11')
        self.model.flg_surfacevel = True
        self.track_geoms = [self.model.geom(f'track_{s}_contact').id for s in ['left','right']]
        self.track_animator = TraxTrackAnimator(self.model)
        self.tread_geoms = self.track_animator.tread_geoms
        self.sprocket_geoms = [self.track_animator.front_sprockets, self.track_animator.rear_sprockets]
        self.track_speeds = np.zeros(2)
        self.track_phase = np.zeros(2)
        self.surface_geoms = {self.model.geom(n).id for n in ['desk_top', 'desk_zone_mat']}
        self.ray_groups = np.array([1,0,0,0,0,0], dtype=np.uint8)
        self.humans = DeskHumans(self.model,self.data) if human_scenarios else None
        self.action_space = spaces.Box(-1., 1., shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'robot': spaces.Box(-1., 1., shape=(10,), dtype=np.float32),
            'target': spaces.Box(-1., 1., shape=(4,), dtype=np.float32),
            'sensors': spaces.Box(0., 1., shape=(5,), dtype=np.float32),
            'workspace': spaces.Box(-1., 1., shape=(6,), dtype=np.float32),
            'humans': spaces.Box(-1., 1., shape=(3,8), dtype=np.float32),
            'objects': spaces.Box(-1., 1., shape=(2,6), dtype=np.float32),
        })
        self.viewer = self.renderer = None
        self.step_count = 0
        self.command = np.zeros(2)
        self.reset(seed=42)

    def _animate_tracks(self):
        # Visual 3D CAD track shoes and dual sprockets follow belt displacement on the exact 20-shoe CAD loop.
        # Contact uses the continuous sole; these massless visual meshes do not add joints or alter dynamics.
        self.track_animator.update(self.track_phase)

    def _pose(self):
        rotation = self.data.xmat[self.robot_id].reshape(3,3)
        yaw = float(np.arctan2(rotation[1,0], rotation[0,0]))
        return self.data.xpos[self.robot_id].copy(), yaw

    def _bounds(self):
        center = self.ZONE_CENTER if self.restrict_to_zone else np.zeros(2)
        half = self.ZONE_HALF if self.restrict_to_zone else self.DESK_HALF
        return center, half - self.ROBOT_RADIUS - .01

    def _ray(self, origin, direction):
        hit_id = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(self.model, self.data,
            np.asarray(origin,dtype=float), np.asarray(direction,dtype=float),
            self.ray_groups, True, self.robot_id, hit_id)
        return distance, int(hit_id[0])

    def _sensors(self):
        position, yaw = self._pose()
        forward = np.array([-np.sin(yaw), np.cos(yaw), 0.])
        origin = position + forward * .065
        origin[2] = self.TABLE_HEIGHT + .025
        distance, _ = self._ray(origin, forward)
        tof = min(1.0, distance) if distance >= 0 else 1.0
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        supported = []
        for x,y in [(-.065,.06),(.065,.06),(-.065,-.06),(.065,-.06)]:
            point = position.copy()
            point[:2] += rotation @ np.array([x,y])
            point[2] = self.TABLE_HEIGHT + .08
            gap, hit = self._ray(point, [0,0,-1])
            supported.append(float(0 <= gap < .10 and hit in self.surface_geoms))
        return float(tof), np.array(supported)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        if self.domain_randomization:
            self.data.qpos[self.ball_adr:self.ball_adr+2] += self.np_random.uniform(-.01,.01,2)
        if self.humans is not None:self.humans.reset(self.np_random,self.domain_randomization)
        self.track_speeds[:] = 0
        self.track_phase[:] = 0
        self.model.geom_surfacevel[self.track_geoms] = 0
        self._animate_tracks()
        # Let the continuous track soles settle onto the physical mat.
        for _ in range(100):
            mujoco.mj_step(self.model,self.data)
        self.episode_start_time = float(self.data.time)
        self.step_count = 0
        self.command = np.zeros(2)
        self.visited = set()
        self._last_pose = self._pose()[0]
        self.last_safety_reason = 'NONE'
        if self.humans is not None:self.humans.hold_objects()
        mujoco.mj_forward(self.model,self.data)
        return self._observation(), self._info()

    def _observation(self):
        position, yaw = self._pose()
        matrix = self.data.xmat[self.robot_id].reshape(3,3)
        roll = float(np.arctan2(matrix[2,1],matrix[2,2]))
        pitch = float(np.arcsin(np.clip(-matrix[2,0],-1,1)))
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,
                                self.robot_id,velocity,0)
        forward = np.array([-np.sin(yaw),np.cos(yaw)])
        pan, tilt = self.data.qpos[self.pan_adr],self.data.qpos[self.tilt_adr]
        target = self.data.xpos[self.ball_id] - position
        bearing = float(np.arctan2(target[0]*np.cos(yaw)+target[1]*np.sin(yaw),
                                   -target[0]*np.sin(yaw)+target[1]*np.cos(yaw)))
        gaze_error = float((bearing+pan+np.pi)%(2*np.pi)-np.pi)
        distance = float(np.linalg.norm(target[:2]))
        visible = distance < 1 and abs(gaze_error) < np.pi/3
        tof, probes = self._sensors()
        zone_margin = float(np.min(self.ZONE_HALF-np.abs(position[:2]-self.ZONE_CENTER))-self.ROBOT_RADIUS)
        desk_margin = float(np.min(self.DESK_HALF-np.abs(position[:2]))-self.ROBOT_RADIUS)
        def vector(values):return np.clip(values,-1,1).astype(np.float32)
        return {
            'robot':vector([np.dot(velocity[3:5],forward)/self.MAX_LINEAR,velocity[2],
                pan/1.047,tilt/.698,roll/.5,pitch/.5,
                self.track_speeds[0]/.13,self.track_speeds[1]/.13,
                self.command[0]/self.MAX_LINEAR,self.command[1]/self.MAX_ANGULAR]),
            'target':vector([float(visible),gaze_error/np.pi if visible else 0,
                             distance if visible else 1,.015/max(.015,distance) if visible else 0]),
            'sensors':np.array([tof,*probes],dtype=np.float32),
            **(self.humans.observation(position) if self.humans is not None else
               dict(humans=np.zeros((3,8),dtype=np.float32),objects=np.zeros((2,6),dtype=np.float32))),
            'workspace':vector([*(position[:2]-self.ZONE_CENTER)/self.ZONE_HALF,
                                np.sin(yaw),np.cos(yaw),zone_margin/.2,desk_margin/.4]),
        }

    def _contacts(self):
        collisions = []
        for contact in self.data.contact[:self.data.ncon]:
            a,b = int(contact.geom1),int(contact.geom2)
            robot_a = self.model.body_rootid[self.model.geom_bodyid[a]] == self.robot_id
            robot_b = self.model.body_rootid[self.model.geom_bodyid[b]] == self.robot_id
            if robot_a == robot_b:
                continue
            other = b if robot_a else a
            if other not in self.surface_geoms:
                collisions.append(mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,other))
        return sorted(set(collisions))

    def _info(self):
        position, yaw = self._pose()
        center,half = self._bounds()
        return dict(pose=dict(x=float(position[0]),y=float(position[1]),z=float(position[2]),yaw=yaw),
            safety_reason=self.last_safety_reason,command=self.command.tolist(),
            zone_clearance=float(np.min(half-np.abs(position[:2]-center))),
            desk_edge_clearance=float(np.min(self.DESK_HALF-np.abs(position[:2]))-self.ROBOT_RADIUS),
            collisions=self._contacts(),
            simulator_ground_truth=True, simulated_cliff_sensors=True,
            drivetrain='continuous_tracks', observation_schema='desk_social_dict_v3',
            human_scenario=self.humans.info() if self.humans is not None else None,
            social_guard_uses_simulator_state=True,
            track_speeds_m_s=self.track_speeds.tolist(),
            fell=bool(position[2] < self.TABLE_HEIGHT-.05))

    def step(self, action):
        action = np.asarray(action,dtype=float)
        if action.shape != (5,) or not np.all(np.isfinite(action)):
            raise ValueError('Desk action must contain five finite values')
        action = np.clip(action,-1,1)
        desired = np.array([action[0]*self.MAX_LINEAR,action[1]*self.MAX_ANGULAR])
        self.command += np.clip(desired-self.command,
                                -np.array([.15,1.2])*self.dt,np.array([.15,1.2])*self.dt)
        position,yaw = self._pose()
        forward = np.array([-np.sin(yaw),np.cos(yaw)])
        self.last_safety_reason = 'NONE'
        if self.safety_enabled:
            center,half = self._bounds()
            lookahead = position[:2] + forward*self.command[0]*.8
            moving_out = np.any(np.abs(lookahead-center) > half)
            tof,cliffs = self._sensors()
            if moving_out:
                self.command[0] = 0
                self.last_safety_reason = 'ZONE_BOUNDARY' if self.restrict_to_zone else 'DESK_EDGE'
            elif self.command[0] > 0 and tof < .045:
                self.command[0] = 0
                self.last_safety_reason = 'FRONT_OBSTACLE'
            elif np.min(cliffs) < .5:
                self.command[0] = 0
                self.last_safety_reason = 'SIMULATED_CLIFF'
        if self.safety_enabled and self.humans is not None and self.humans.hand_clearance(position)<.14:
            self.command[:]=0
            self.last_safety_reason='HUMAN_HAND_NEAR'
        # Surface material moves backwards at the sole. MuJoCo contact
        # friction moves the free chassis forwards; no pose/velocity teleport.
        linear, angular = self.command
        self.track_speeds[:] = [linear-angular*self.TRACK_WIDTH/2,
                                linear+angular*self.TRACK_WIDTH/2]
        self.model.geom_surfacevel[self.track_geoms] = 0
        self.model.geom_surfacevel[self.track_geoms,1] = -self.track_speeds
        self.data.ctrl[0] = np.clip(self.data.ctrl[0]+action[2]*.8*self.dt,-.90,.90)
        self.data.ctrl[1] = np.clip(self.data.ctrl[1]+action[3]*.6*self.dt,-.60,.60)
        # Fifth channel is an expressive intensity placeholder for task training.
        self.step_count += 1
        if self.humans is not None:
            self.humans.begin_step()
            self.humans.update(self.step_count*self.dt)
        for _ in range(self.frameskip):
            if self.humans is not None:self.humans.hold_objects()
            self.track_phase += self.track_speeds*self.model.opt.timestep
            if self.animate_tracks:self._animate_tracks()
            mujoco.mj_step(self.model,self.data)
        if not self.animate_tracks:self._animate_tracks()
        unstable = (not np.all(np.isfinite(self.data.qpos)) or
                    not np.all(np.isfinite(self.data.qvel)) or
                    any(self.data.warning[i].number for i in
                        [mujoco.mjtWarning.mjWARN_BADQPOS,mujoco.mjtWarning.mjWARN_BADQVEL,
                         mujoco.mjtWarning.mjWARN_BADQACC]))
        info = self._info()
        info['simulation_unstable'] = bool(unstable)
        cell = tuple(np.floor(self.data.xpos[self.robot_id,:2]/.02).astype(int))
        discovery = cell not in self.visited
        self.visited.add(cell)
        reward = .02*discovery-.002*float(np.sum(np.abs(action[:4])))
        reward -= .5*bool(info['collisions'])+.02*(self.last_safety_reason!='NONE')
        if info['fell'] or unstable:reward -= 5
        observation = self._observation()
        if unstable:
            observation = {k:np.nan_to_num(v,nan=0,posinf=1,neginf=-1) for k,v in observation.items()}
        return observation,float(reward),bool(info['fell'] or unstable),self.step_count>=self.max_episode_steps,info

    def render(self):
        if self.render_mode=='rgb_array':
            if self.renderer is None:self.renderer=mujoco.Renderer(self.model,height=900,width=1440)
            self.renderer.update_scene(self.data,camera='desk_overview')
            return self.renderer.render()
        if self.viewer is None:
            import mujoco.viewer as viewer
            self.viewer=viewer.launch_passive(self.model,self.data)
            self.viewer.cam.lookat[:]=[.15,0,.85]
            self.viewer.cam.distance=1.9
            self.viewer.cam.azimuth=125
            self.viewer.cam.elevation=-30
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:self.viewer.close();self.viewer=None
        if self.renderer is not None:self.renderer.close();self.renderer=None
