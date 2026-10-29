"""PPO task layer for the desk: notice, respond, inspect, then rest.

Actor input contains camera-gated perception and observation-derived memory.
Simulator positions may shape rewards, never the actor's target direction.
No scripted motor controller or DeskDemo is used by this environment.
"""
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from trax_rl.envs.trax_desk_env import TraxDeskEnv


class TraxDeskSocialEnv(gym.Wrapper):
    SCHEMA = 'desk_social_rl_v1'

    def __init__(self, training=False, horizon=600, **kwargs):
        if horizon <= 0:raise ValueError('horizon must be positive')
        kwargs.setdefault('domain_randomization',True)
        kwargs.setdefault('animate_tracks',False)
        super().__init__(TraxDeskEnv(**kwargs))
        self.training=training
        self.horizon=horizon
        self.observation_space=spaces.Dict({**self.env.observation_space.spaces,
            'memory':spaces.Box(0.,1.,(5,),dtype=np.float32),
            'elevation':spaces.Box(-1.,1.,(5,),dtype=np.float32)})
        self._clear()

    def _clear(self):
        self.noticed=np.zeros(3,dtype=bool)
        self.inspected=np.zeros(2,dtype=bool)
        self.discovered=np.zeros(5,dtype=bool)
        self.dwell=np.zeros(5)
        self.waves={}
        self.previous_goal=None
        self.previous_cost=0.
        self.last_control=np.zeros(4)
        self.elapsed=0
        self.metrics=dict(steps=0,collisions=0,falls=0,unstable=0,safety_overrides=0,
            wave_opportunities=0,wave_responses=0,people_noticed=0,objects_inspected=0,
            command_variation=0.,idle_steps=0,return_=0.)

    def reset(self, seed=None, options=None):
        observation,info=self.env.reset(seed=seed)
        self._clear()
        rng=self.env.np_random
        start=float((options or {}).get('start_time',0.))
        if self.training and not options:
            if rng.random()<.65:
                start=max(0.,float(rng.choice([8,13,17,27,31,38,44,66,82])+rng.uniform(-2,2)))
            self.env.data.qpos[self.env.base_adr:self.env.base_adr+2]+=rng.uniform(-.025,.025,2)
            yaw=float(rng.uniform(-np.pi,np.pi))
            self.env.data.qpos[self.env.base_adr+3:self.env.base_adr+7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
            self.env.data.qpos[self.env.pan_adr]=rng.uniform(-.3,.3)
            self.env.data.qpos[self.env.tilt_adr]=rng.uniform(-.2,.5)
            self.env.data.ctrl[:]=self.env.data.qpos[[self.env.pan_adr,self.env.tilt_adr]]
        if start<0 or start>89:raise ValueError('start_time must be in [0,89] seconds')
        human=self.env.humans
        if human is None:raise ValueError('Social training requires human_scenarios=True')
        # Replay release boundaries in order so late snapshots retain both props.
        for time in sorted(set([0.,min(start,22+human.delay),min(start,43+human.delay),start])):
            human.update(time)
        self.env.step_count=round(start/self.env.dt)
        self.env.max_episode_steps=self.env.step_count+self.horizon
        human.begin_step()
        human.hold_objects()
        mujoco.mj_forward(self.env.model,self.env.data)
        for _ in range(100):
            human.hold_objects()
            mujoco.mj_step(self.env.model,self.env.data)
        mujoco.mj_forward(self.env.model,self.env.data)
        observation=self.env._observation()
        self.last_control=self._control()
        self._register_waves(initial=True)
        info=self.env._info()
        return self._augment(observation),self._info(info,{})

    def _control(self):
        return np.array([self.env.command[0]/self.env.MAX_LINEAR,
                         self.env.command[1],*self.env.data.ctrl])

    def _camera_error(self, target):
        human=self.env.humans
        delta=np.asarray(target)-self.env.data.cam_xpos[human.camera]
        local=self.env.data.cam_xmat[human.camera].reshape(3,3).T@delta
        return np.array([np.arctan2(local[0],-local[2]),
                         np.arctan2(local[1],np.hypot(local[0],local[2]))])

    def _augment(self, observation):
        elevation=np.zeros(5,dtype=np.float32)
        human=self.env.humans
        for i in range(3):
            if observation['humans'][i,0]:
                elevation[i]=self._camera_error(human.heads[i])[1]/np.pi
        for k in range(2):
            if observation['objects'][k,0]:
                elevation[k+3]=self._camera_error(self.env.data.xpos[human.object_ids[k]])[1]/np.pi
        return {**observation,'memory':np.r_[self.noticed,self.inspected].astype(np.float32),
                'elevation':elevation}

    def _register_waves(self, initial=False):
        human=self.env.humans
        now=self.env.step_count*self.env.dt
        events=list(human.new_events)
        if initial:
            events=[dict(person_id=i+1,action='wave',time_s=now)
                    for i in range(3) if human.actions[i]=='wave']
        for event in events:
            if event['action']!='wave':continue
            key=(event['person_id']-1,event['time_s'])
            if key not in self.waves:
                self.waves[key]=dict(deadline=float(event['time_s'])+3.,dwell=0.,responded=False)
                self.metrics['wave_opportunities']+=1

    def _goal(self):
        human=self.env.humans
        now=self.env.step_count*self.env.dt
        for (actor,start),wave in self.waves.items():
            if now<=wave['deadline'] and not wave['responded'] and human.present[actor]:
                return ('wave',actor,start),human.heads[actor]
        for k in range(2):
            if human.released[k] and not self.inspected[k]:
                return ('object',k),self.env.data.xpos[human.object_ids[k]]
        for i in range(3):
            if human.present[i] and not self.noticed[i]:return ('person',i),human.heads[i]
        return None,None

    def _task_reward(self, observation, info, action):
        human=self.env.humans
        dt=self.env.dt
        reward=dict(discovery=0.,notice=0.,wave=0.,inspection=0.,progress=0.,
                    effort=-.002*dt*float(np.sum(np.abs(action[:4]))),
                    smoothness=0.,safety=0.,collision=0.,failure=0.,quiet=0.)
        self._register_waves()
        control=self._control()
        variation=float(np.abs(control-self.last_control).mean())
        self.last_control=control
        reward['smoothness']=-.04*variation
        still=abs(observation['robot'][0])<.015/self.env.MAX_LINEAR and abs(observation['robot'][1])<.25
        seen=np.r_[observation['humans'][:,0]>0,observation['objects'][:,0]>0]
        fresh=seen & ~self.discovered
        reward['discovery']=float(np.count_nonzero(fresh))*.2
        self.discovered|=seen
        aligned=np.zeros(5,dtype=bool)
        for index in range(5):
            if not seen[index]:continue
            target=human.heads[index] if index<3 else self.env.data.xpos[human.object_ids[index-3]]
            error=self._camera_error(target)
            aligned[index]=np.linalg.norm(error)<.28
        # Dwell must be uninterrupted. A single glimpse is not task completion.
        self.dwell=np.where(aligned & still,self.dwell+dt,0.)
        for i in range(3):
            if not self.noticed[i] and self.dwell[i]>=.4:
                self.noticed[i]=True
                reward['notice']+=1.
                self.metrics['people_noticed']+=1
        for (actor,start),wave in self.waves.items():
            if self.env.step_count*dt>wave['deadline'] or wave['responded']:continue
            wave['dwell']=wave['dwell']+dt if aligned[actor] and still else 0.
            if wave['dwell']>=.4:
                wave['responded']=True
                reward['wave']+=2.
                self.metrics['wave_responses']+=1
        for k in range(2):
            row=observation['objects'][k]
            if (not self.inspected[k] and human.released[k] and self.dwell[k+3]>=.6
                    and row[2]*2.5<.8 and np.linalg.norm(row[4:6])<.02):
                self.inspected[k]=True
                reward['inspection']+=2.
                self.metrics['objects_inspected']+=1
        goal,target=self._goal()
        if target is not None:
            cost=float(np.linalg.norm(self._camera_error(target)))
            if goal==self.previous_goal:
                # Potential difference only: staying aligned cannot farm reward.
                reward['progress']=.6*(self.previous_cost-cost)
            self.previous_cost=cost
        else:
            reward['quiet']=-.03*dt*float(np.sum(np.abs(control[:2])))
        self.previous_goal=goal
        reward['safety']=-.04*float(info['safety_reason']!='NONE')
        reward['collision']=-2.*bool(info['collisions'])
        reward['failure']=-20.*bool(info['fell'] or info.get('simulation_unstable'))
        m=self.metrics
        m['steps']+=1
        m['collisions']+=int(bool(info['collisions']))
        m['falls']+=int(info['fell'])
        m['unstable']+=int(info.get('simulation_unstable',False))
        m['safety_overrides']+=int(info['safety_reason']!='NONE')
        m['command_variation']+=variation
        m['idle_steps']+=int(still)
        total=float(sum(reward.values()))
        m['return_']+=total
        return total,reward

    def _info(self, info, components):
        m=dict(self.metrics)
        steps=max(1,m['steps'])
        m['mean_command_variation']=m['command_variation']/steps
        m['safety_override_rate']=m['safety_overrides']/steps
        m['idle_fraction']=m['idle_steps']/steps
        m['wave_response_rate']=m['wave_responses']/max(1,m['wave_opportunities'])
        return {**info,'observation_schema':self.SCHEMA,'reward_components':components,'social_metrics':m,
                'privileged_reward_shaping':True,'scripted_motor_controller':False}

    def step(self, action):
        observation,_,terminated,truncated,info=self.env.step(action)
        self.elapsed+=1
        reward,components=self._task_reward(observation,info,np.asarray(action))
        return self._augment(observation),reward,terminated,bool(truncated or self.elapsed>=self.horizon),self._info(info,components)
