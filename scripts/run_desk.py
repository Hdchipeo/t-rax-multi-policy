#!/usr/bin/env python3
"""Launch, render, or smoke-test the T-Rax desk environment."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time


def prepare_mujoco_runtime():
    """Work around AppleDouble dylibs introduced when a venv moves to exFAT.

    Keep the installed package untouched; cache a clean copy on the local disk.
    This also lets the native mjpython launcher run from a non-exFAT volume.
    """
    spec = importlib.util.find_spec('mujoco')
    if spec is None:
        raise RuntimeError('MuJoCo is missing from this Python environment')
    package = Path(spec.origin).parent
    if not list((package/'plugin').glob('._*.dylib')):
        return
    key = hashlib.sha256((str(package)+str(Path(spec.origin).stat().st_mtime_ns)).encode()).hexdigest()[:12]
    cache = Path(tempfile.gettempdir())/f'trax-mujoco-{key}'
    if not (cache/'mujoco/__init__.py').is_file():
        shutil.copytree(package,cache/'mujoco',ignore=shutil.ignore_patterns('._*','__pycache__'),dirs_exist_ok=True)
    sys.path.insert(0,str(cache))
    os.environ['PYTHONPATH'] = str(cache)+os.pathsep+os.environ.get('PYTHONPATH','')


prepare_mujoco_runtime()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from trax_rl.envs.trax_desk_env import TraxDeskEnv


class DeskDemo:
    """Known-layout demonstration, not an RL policy trained on the desk."""
    def __init__(self):
        self.points = [np.array(p) for p in [[.46,-.06],[.46,.035],[.30,.035],[.30,-.06]]]
        self.phase = 'REST'
        self.elapsed = 0.
        self.index = 0
        self.completed_laps = 0
        self.social_until = 0.
        self.social_actor = 0

    def act(self, env):
        if env.humans is not None:
            for event in env.humans.new_events:
                if event['action'] in ('arrive','wave','release_water_glass','release_new_object'):
                    self.social_until=env.step_count*env.dt+1.2
                    self.social_actor=event['person_id']-1
            if env.step_count*env.dt<self.social_until and env.humans.present[self.social_actor]:
                position,yaw=env._pose()
                delta=env.humans.heads[self.social_actor]-position
                bearing=np.arctan2(delta[0]*np.cos(yaw)+delta[1]*np.sin(yaw),
                                   -delta[0]*np.sin(yaw)+delta[1]*np.cos(yaw))
                pan=env.data.qpos[env.pan_adr]
                tilt=env.data.qpos[env.tilt_adr]
                target_tilt=np.clip(np.arctan2(delta[2],np.linalg.norm(delta[:2])),-.55,.55)
                return np.array([0,0,np.clip((-bearing-pan)*2,-.6,.6),
                                 np.clip((target_tilt-tilt)*2,-.6,.6),.3],dtype=np.float32)
        self.elapsed += env.dt
        action = np.zeros(5,dtype=np.float32)
        position,yaw = env._pose()
        pan = env.data.qpos[env.pan_adr]
        tilt = env.data.qpos[env.tilt_adr]
        action[3] = np.clip(-tilt*2,-1,1)
        if self.phase=='REST':
            if self.elapsed>=3:
                self.phase='SURVEY';self.elapsed=0
        elif self.phase=='SURVEY':
            target=.45*np.sin(self.elapsed*1.4)
            action[2]=np.clip((target-pan)*2,-.6,.6)
            if self.elapsed>=3:
                self.phase='MOVE';self.elapsed=0
        elif self.phase=='MOVE':
            delta = self.points[self.index]-position[:2]
            distance = float(np.linalg.norm(delta))
            desired_yaw = -np.arctan2(delta[0],delta[1])
            error = float((desired_yaw-yaw+np.pi)%(2*np.pi)-np.pi)
            action[1] = np.clip(error*1.4,-.65,.65)
            if abs(error)<.3:
                action[0]=min(.045,max(.009,distance*.4))/env.MAX_LINEAR
            action[2]=np.clip(-pan*2,-.5,.5)
            if distance<.018:
                action[:2]=0
                self.index=(self.index+1)%len(self.points)
                self.phase='INSPECT';self.elapsed=0
                if self.index==0:self.completed_laps+=1
        elif self.phase=='INSPECT':
            delta=env.data.xpos[env.ball_id,:2]-position[:2]
            bearing=np.arctan2(delta[0]*np.cos(yaw)+delta[1]*np.sin(yaw),
                                -delta[0]*np.sin(yaw)+delta[1]*np.cos(yaw))
            desired=np.clip(-bearing,-.65,.65)
            action[2]=np.clip((desired-pan)*2,-.5,.5)
            if self.elapsed>=2:
                self.phase='REST' if self.index==0 else 'MOVE';self.elapsed=0
        return action


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--policy',type=Path,help='Run a trained desk_social_rl_v1 PPO checkpoint instead of DeskDemo')
    parser.add_argument('--test',action='store_true',help='Run project regression tests with the MuJoCo runtime bootstrap')
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/environments/desk_v1.yaml')
    parser.add_argument('--steps',type=int,default=1800)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--manual',action='store_true',help='Stationary physics inspection; no driving demo')
    parser.add_argument('--render',type=Path,help='Save a PNG and exit; no window')
    parser.add_argument('--render-time',type=float,default=0.,help='Simulate this many seconds before the PNG snapshot')
    parser.add_argument('--no-humans',action='store_true',help='Disable the human timeline for baseline comparisons')
    parser.add_argument('--camera',choices=['social_overview','desk_overview','robot_closeup','desk_topdown','robot_camera'],default='social_overview')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.policy and (args.manual or args.no_humans):
        parser.error('--policy requires humans and cannot be combined with --manual')
    if args.test:
        import unittest
        suite=unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'))
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if sys.platform=='darwin' and not args.headless and not args.render and 'MJPYTHON_BIN' not in os.environ:
        launcher=Path(sys.executable).parent/'mjpython'
        if not launcher.is_file():raise RuntimeError('Use the MuJoCo mjpython interpreter to open the native viewer')
        # Invoke the trampoline through Python: moved venv scripts can have stale shebangs.
        os.execv(sys.executable,[sys.executable,str(launcher),str(Path(__file__).resolve()),*sys.argv[1:]])
    import yaml
    options = dict(yaml.safe_load(args.config.read_text())['environment'])
    options.pop('class',None)
    options['max_episode_steps']=max(args.steps,options.get('max_episode_steps',1200))
    options['human_scenarios']=not args.no_humans
    model=None
    if args.policy:
        import torch
        from stable_baselines3 import PPO
        from trax_rl.envs.trax_desk_social_env import TraxDeskSocialEnv
        torch.set_num_threads(1)
        metadata_path=args.policy.with_suffix('.metadata.json')
        if metadata_path.is_file():
            metadata=json.loads(metadata_path.read_text())
            if metadata.get('observation_schema')!=TraxDeskSocialEnv.SCHEMA:
                raise ValueError('Checkpoint does not use the desk social RL schema')
            options.update(metadata['config']['environment'])
        env=TraxDeskSocialEnv(training=False,horizon=options['max_episode_steps'],**options)
        model=PPO.load(args.policy,device='cpu')
        if model.observation_space!=env.observation_space or model.action_space!=env.action_space:
            env.close()
            raise ValueError('Checkpoint spaces do not match the desk social environment')
    else:
        env=TraxDeskEnv(**options)
    physical=env.unwrapped
    controller=DeskDemo()
    observation,_=env.reset(seed=args.seed)
    def action_for_frame():
        if model is not None:return model.predict(observation,deterministic=True)[0]
        return np.zeros(5,dtype=np.float32) if args.manual else controller.act(physical)
    if args.render:
        if args.render_time<0:parser.error('--render-time must be nonnegative')
        for _ in range(round(args.render_time/physical.dt)):
            observation,_,terminated,_,_=env.step(action_for_frame())
            if terminated:raise RuntimeError('Simulation terminated before snapshot')
        from PIL import Image
        args.render.parent.mkdir(parents=True,exist_ok=True)
        with mujoco.Renderer(physical.model,height=1000,width=1600) as renderer:
            renderer.update_scene(physical.data,camera=args.camera)
            Image.fromarray(renderer.render()).save(args.render)
        print('Rendered:',args.render)
        env.close()
        return
    summary=dict(steps=0,collisions=0,fell=False,unstable=False,min_zone_clearance=1.,completed_laps=0,
                 human_events=list(physical.humans.events) if physical.humans is not None else [],
                 max_people=0,objects={})
    def step():
        nonlocal observation
        action=action_for_frame()
        observation,reward,terminated,truncated,info=env.step(action)
        summary['steps']+=1
        summary['collisions']+=bool(info['collisions'])
        summary['fell']|=info['fell'];summary['unstable']|=info['simulation_unstable']
        summary['min_zone_clearance']=min(summary['min_zone_clearance'],info['zone_clearance'])
        summary['completed_laps']=controller.completed_laps
        if 'social_metrics' in info:summary['social_metrics']=info['social_metrics']
        if info['human_scenario'] is not None:
            social=info['human_scenario']
            summary['human_events'].extend(social['events'])
            summary['max_people']=max(summary['max_people'],sum(p['present'] for p in social['people']))
            summary['objects']=social['objects']
        if summary['steps']%100==0:
            p=info['pose'];print(f"step={summary['steps']} phase={('RL' if model is not None else controller.phase)} x={p['x']:.3f} y={p['y']:.3f} safety={info['safety_reason']}",flush=True)
        return terminated,truncated
    try:
        if args.headless:
            for _ in range(args.steps):
                terminated,truncated=step()
                if terminated or truncated:break
        else:
            from mujoco import viewer as mj_viewer
            with mj_viewer.launch_passive(physical.model,physical.data) as viewer:
                viewer.cam.lookat[:]=[.07,.03,.87]
                viewer.cam.distance=1.9;viewer.cam.azimuth=125;viewer.cam.elevation=-30
                if args.camera=='social_overview':
                    # Start at the overview pose, but allow mouse orbit and zoom.
                    camera_id=physical.model.camera('social_overview').id
                    position=physical.data.cam_xpos[camera_id]
                    forward=-physical.data.cam_xmat[camera_id].reshape(3,3)[:,2]
                    distance=float(np.linalg.norm(position-viewer.cam.lookat))
                    viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
                    viewer.cam.fixedcamid=-1
                    viewer.cam.lookat[:]=position+distance*forward
                    viewer.cam.distance=distance
                    viewer.cam.azimuth=float(np.degrees(np.arctan2(forward[1],forward[0])))
                    viewer.cam.elevation=float(np.degrees(np.arcsin(np.clip(forward[2],-1,1))))
                elif args.camera=='robot_closeup':
                    viewer.cam.lookat[:]=[.40,.02,.82];viewer.cam.distance=.85;viewer.cam.elevation=-35
                elif args.camera=='desk_topdown':
                    viewer.cam.lookat[:]=[0,0,.76];viewer.cam.distance=1.8;viewer.cam.elevation=-89
                elif args.camera=='robot_camera':
                    viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid=physical.model.camera('robot_camera').id
                print(f'T-Rax {"PPO policy" if model is not None else "scripted demo"} running. Mouse: orbit/zoom; close window to exit.',flush=True)
                while viewer.is_running():
                    start=time.monotonic()
                    with viewer.lock():
                        terminated,truncated=step()
                    viewer.sync()
                    if terminated or truncated:
                        with viewer.lock():
                            observation,_=env.reset(seed=args.seed+summary['steps']);controller=DeskDemo()
                    time.sleep(max(0,physical.dt-(time.monotonic()-start)))
    finally:
        env.close()
    print(json.dumps(summary,indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(summary,indent=2))
    return 1 if summary['fell'] or summary['unstable'] else 0


if __name__=='__main__':raise SystemExit(main())
