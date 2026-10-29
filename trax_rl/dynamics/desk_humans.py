"""Scripted kinematic people and physical object releases for desk training.

Time is episode-relative seconds. The timeline is simulator ground truth only;
policy observations are gated by the robot camera's field of view and occlusion.
These avatars are motion proxies, not learned human policies or human dynamics.
"""
import mujoco
import numpy as np


def blend(a, b, fraction):
    u = np.clip(fraction, 0., 1.)
    u = u*u*(3-2*u)
    return np.asarray(a)*(1-u)+np.asarray(b)*u


class DeskHumans:
    PARTS = ('torso','head','upper_l','fore_l','hand_l','upper_r','fore_r','hand_r',
             'thigh_l','shin_l','foot_l','thigh_r','shin_r','foot_r')
    # start, end, actor, action; starts are emitted once per episode.
    TIMELINE = (
        (0,3,0,'arrive'), (3,5,0,'sit'), (5,14,0,'type'),
        (8,12,1,'arrive'), (12,15,1,'wave'), (14,17,0,'wave'),
        (17,23,0,'place_glass'), (23,28,0,'type'), (24,28,1,'point'),
        (26,31,2,'arrive'), (28,33,0,'phone'), (31,34,2,'wave'),
        (33,38,1,'move'), (34,39,0,'type'), (38,44,2,'place_object'),
        (44,48,2,'point'), (46,50,0,'wave'), (53,59,1,'leave'),
        (58,64,2,'leave'), (66,69,0,'stand'), (69,74,0,'move'), (74,80,0,'leave'))
    OBJECTS = ('water_glass','new_object')
    TARGETS = np.array([[.08,-.015,.817],[.59,-.10,.787]])

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.mocap = [{part:int(model.body(f'human_{i}_{part}').mocapid[0])
                       for part in self.PARTS} for i in range(3)]
        self.geoms = [{part:model.geom(f'human_{i}_{part}_geom').id for part in self.PARTS}
                      for i in range(3)]
        self.body_ids = [model.body(f'human_{i}_head').id for i in range(3)]
        self.object_ids = [model.body(n).id for n in self.OBJECTS]
        self.qadr = [int(model.jnt_qposadr[model.joint(n+'_free').id]) for n in self.OBJECTS]
        self.vadr = [int(model.jnt_dofadr[model.joint(n+'_free').id]) for n in self.OBJECTS]
        self.object_geoms = [np.flatnonzero(model.geom_bodyid == b) for b in self.object_ids]
        self.rgba = [model.geom_rgba[g].copy() for g in self.object_geoms]
        self.collision_geoms = [model.geom(n+'_collision').id for n in self.OBJECTS]
        self.camera = model.camera('robot_camera').id
        self.ray_groups = np.array([1,1,0,0,0,0],dtype=np.uint8)
        self.reset(np.random.default_rng(42), False)

    def reset(self, rng, randomize):
        self.time = 0.
        self.delay = float(rng.uniform(0,2)) if randomize else 0.
        self.events = []
        self.new_events = []
        self.emitted = set()
        self.present = np.zeros(3,dtype=bool)
        self.actions = ['absent']*3
        self.heads = np.zeros((3,3))
        self.hands = np.zeros((3,2,3))
        self.seated = np.zeros(3)
        self.walking = np.zeros(3)
        self.object_state = ['absent']*2
        self.released = np.zeros(2,dtype=bool)
        self.held_positions = np.zeros((2,3))
        for k in range(2):self._set_object(k, [k,-4,-3], visible=False, collision=False)
        self.begin_step()
        self.update(0.)

    def begin_step(self):
        self.new_events = []

    def _emit(self, key, actor, action, time):
        if key not in self.emitted:
            item=dict(time_s=round(float(time),3),person_id=actor+1,action=action)
            self.events.append(item)
            self.new_events.append(item)
            self.emitted.add(key)

    def _part(self, i, part, position, vector=None):
        mid=self.mocap[i][part]
        self.data.mocap_pos[mid] = position
        self.data.mocap_quat[mid] = [1,0,0,0]
        if vector is not None:
            length=float(np.linalg.norm(vector))
            mujoco.mju_quatZ2Vec(self.data.mocap_quat[mid],np.asarray(vector,dtype=float))
            geom=self.geoms[i][part]
            self.model.geom_size[geom,1] = length/2
            self.model.geom_rbound[geom] = length/2+self.model.geom_size[geom,0]

    def _segment(self, i, part, a, b):
        a,b=np.asarray(a),np.asarray(b)
        self._part(i,part,(a+b)/2,b-a)

    def _set_object(self, k, pos, visible=True, collision=False):
        adr,vel=self.qadr[k],self.vadr[k]
        self.data.qpos[adr:adr+3]=pos
        self.data.qpos[adr+3:adr+7]=[1,0,0,0]
        self.data.qvel[vel:vel+6]=0
        self.model.geom_rgba[self.object_geoms[k]]=self.rgba[k]
        if not visible:self.model.geom_rgba[self.object_geoms[k],3]=0
        g=self.collision_geoms[k]
        self.model.geom_contype[g]=int(collision)
        self.model.geom_conaffinity[g]=int(collision)

    def _location(self, i, t):
        if i==0:
            if t<3:return blend([-1.45,-1.25],[-.23,-.71],t/3)
            if t<69:return np.array([-.23,-.71])
            if t<74:return blend([-.23,-.71],[-.86,-.72],(t-69)/5)
            return blend([-.86,-.72],[-1.65,-1.5],(t-74)/6)
        if i==1:
            if t<12:return blend([-1.65,-1.35],[-.83,-.57],(t-8)/4)
            if t<33:return np.array([-.83,-.57])
            if t<38:return blend([-.83,-.57],[-.97,.12],(t-33)/5)
            if t<53:return np.array([-.97,.12])
            return blend([-.97,.12],[-1.75,-1.30],(t-53)/6)
        if t<31:return blend([1.7,-1.35],[.94,-.58],(t-26)/5)
        if t<58:return np.array([.94,-.58])
        return blend([.94,-.58],[1.75,-1.35],(t-58)/6)

    def update(self, episode_time):
        self.time=float(episode_time)
        t=self.time-self.delay
        for n,(start,end,i,action) in enumerate(self.TIMELINE):
            if t>=start:self._emit(('action',n),i,action,start+self.delay)
        for i,(start,end) in enumerate([(0,80),(8,59),(26,64)]):
            self.present[i]=start<=t<end
            self.actions[i]='idle' if self.present[i] else 'absent'
            for a,b,actor,action in self.TIMELINE:
                if actor==i and a<=t<b:self.actions[i]=action
            if not self.present[i]:
                for part in self.PARTS:self._part(i,part,[i*2,-4,-3])
                self.heads[i]=[i*2,-4,-3]
                self.hands[i]=[i*2,-4,-3]
                self.seated[i]=self.walking[i]=0
                continue
            xy=self._location(i,t)
            sit=0.
            if i==0:
                sit=float(np.clip((t-3)/2,0,1)*(1-np.clip((t-66)/3,0,1)))
            walk=self.actions[i] in ('arrive','move','leave')
            self.seated[i],self.walking[i]=sit,float(walk)
            hip=np.r_[xy, .94-.37*sit]
            shoulder=hip+np.array([0,.025,.40])
            action=self.actions[i]
            if i in (0,2):
                begin,release=(17,22) if i==0 else (38,43)
                amount=float(blend(0.,1.,t-begin)*(1-blend(0.,1.,t-release)))
                lean=np.array([.03,.16,-.04]) if i==0 else np.array([-.20,.20,-.16])
                shoulder+=lean*amount
            head=shoulder+np.array([0,0,.18])
            self.heads[i]=head
            self._part(i,'torso',(hip+shoulder)/2)
            mujoco.mju_quatZ2Vec(self.data.mocap_quat[self.mocap[i]['torso']],shoulder-hip)
            self._part(i,'head',head)
            hands=np.array([shoulder+[-.20,.02,-.43],shoulder+[.20,.02,-.43]])
            action=self.actions[i]
            if action=='type':
                for side,x in enumerate([-.31,-.14]):hands[side]=[x,-.17,.805+.007*np.sin(t*14+side*2)]
            elif action=='wave':hands[1]=shoulder+[.28+.07*np.sin(t*7),.10,.22]
            elif action=='point':hands[1]=[.28,-.18,.96]
            elif action=='phone':hands[1]=head+[.10,0,-.04]
            # Every held object follows the visible right hand before release.
            k=0 if i==0 else 1 if i==2 else None
            if k is not None:
                begin,release=(17,22) if k==0 else (38,43)
                home=shoulder+np.array([.20,.08,-.25])
                target=self.TARGETS[k]
                if begin<=t<release:
                    self.object_state[k]='held'
                    if t<release-1:
                        object_pos=blend(home,target+[0,0,.12],(t-begin)/(release-begin-1))
                    else:object_pos=blend(target+[0,0,.12],target,t-(release-1))
                    hands[1]=object_pos+[.045,0,.015]
                    self.held_positions[k]=object_pos
                    self._set_object(k,object_pos)
                elif t>=release and not self.released[k]:
                    self._set_object(k,target,collision=True)
                    self.released[k]=True
                    self.object_state[k]='on_table'
                    self._emit(('release',k),i,'release_'+self.OBJECTS[k],release+self.delay)
                if release<=t<release+1:hands[1]=blend(target+[.045,0,.015],home,t-release)
            for side,sign in enumerate([-1,1]):
                anchor=shoulder+[sign*.16,0,0]
                delta=hands[side]-anchor
                reach=float(np.linalg.norm(delta))
                if reach>.59:hands[side]=anchor+delta*.59/reach
            self.hands[i]=hands
            for side,label in enumerate(['l','r']):
                sign=-1 if side==0 else 1
                s=shoulder+[sign*.16,0,0]
                # Two-link IK fixes upper/forearm lengths where the target is reachable.
                delta=hands[side]-s
                reach=float(np.linalg.norm(delta))
                axis=delta/max(reach,1e-9)
                bend=np.cross(axis,[0,0,1.])
                if np.linalg.norm(bend)<.001:bend=np.array([1.,0,0])
                bend/=np.linalg.norm(bend)
                elbow=(s+hands[side])/2+sign*bend*np.sqrt(max(0,.30**2-(reach/2)**2))
                self._segment(i,'upper_'+label,s,elbow)
                self._segment(i,'fore_'+label,elbow,hands[side])
                self._part(i,'hand_'+label,hands[side])
                leg=hip+[sign*.105,0,0]
                stride=.12*np.sin(t*7+side*np.pi)*walk
                knee=blend(leg+[0,stride,-.43],leg+[0,.35,-.10],sit)
                foot=blend(leg+[0,-stride,-.88],leg+[0,.36,-.50],sit)
                foot[2]=max(.055,foot[2])+.055*max(0.,np.sin(t*7+side*np.pi))*walk
                self._segment(i,'thigh_'+label,leg,knee)
                self._segment(i,'shin_'+label,knee,foot)
                self._part(i,'foot_'+label,foot+[0,.04,0])
        # Freeze hidden free bodies instead of letting gravity accelerate them forever.
        for k,state in enumerate(self.object_state):
            if state=='absent':self._set_object(k,[k,-4,-3],visible=False)

    def hold_objects(self):
        for k,state in enumerate(self.object_state):
            if state=='held':self._set_object(k,self.held_positions[k])
            elif state=='absent':self._set_object(k,[k,-4,-3],visible=False)

    def hand_clearance(self, position):
        hands=self.hands[self.present]
        if not len(hands):return 10.
        return float(np.min(np.linalg.norm(hands-np.asarray(position),axis=-1)))

    def _visible(self, target, target_body):
        origin=self.data.cam_xpos[self.camera]
        delta=np.asarray(target)-origin
        distance=float(np.linalg.norm(delta))
        if distance<1e-6:return False,0.,0.
        local=self.data.cam_xmat[self.camera].reshape(3,3).T@delta
        depth=-local[2]
        # Square conservative FOV, independent of renderer resolution.
        tangent=np.tan(np.deg2rad(self.model.cam_fovy[self.camera]/2))
        if distance>2.5 or depth<=0 or max(abs(local[0]),abs(local[1]))>depth*tangent:
            return False,0.,0.
        hit=np.array([-1],dtype=np.int32)
        gap=mujoco.mj_ray(self.model,self.data,origin,delta/distance,self.ray_groups,
                          True,self.model.body('trax_base').id,hit)
        visible=gap>=0 and int(self.model.geom_bodyid[hit[0]])==target_body
        return visible,float(np.arctan2(local[0],depth)/np.pi),distance/2.5

    def observation(self, robot_position):
        people=np.zeros((3,8),dtype=np.float32)
        objects=np.zeros((2,6),dtype=np.float32)
        for i in range(3):
            if not self.present[i]:continue
            seen,bearing,distance=self._visible(self.heads[i],self.body_ids[i])
            if seen:
                people[i]=[1,bearing,distance,self.seated[i],self.walking[i],
                           float(self.actions[i]=='wave'),float(self.actions[i] in ('place_glass','place_object','point')),
                           np.clip(np.min(np.linalg.norm(self.hands[i]-robot_position,axis=1)),0,1)]
        for k in range(2):
            if self.object_state[k]=='absent':continue
            seen,bearing,distance=self._visible(self.data.xpos[self.object_ids[k]],self.object_ids[k])
            if seen:
                v=self.data.qvel[self.vadr[k]:self.vadr[k]+2]
                objects[k]=[1,bearing,distance,float(self.object_state[k]=='held'),*np.clip(v,-1,1)]
        return dict(humans=people,objects=objects)

    def info(self):
        return dict(time_s=round(self.time,3),people=[dict(person_id=i+1,present=bool(self.present[i]),
            action=self.actions[i],seated=float(self.seated[i])) for i in range(3)],
            objects=dict(zip(self.OBJECTS,self.object_state)),events=list(self.new_events),
            event_count=len(self.events),kinematic_humans=True,simulated_social_perception=True)
