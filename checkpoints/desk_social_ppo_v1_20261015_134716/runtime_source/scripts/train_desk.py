#!/usr/bin/env python3
"""Train and evaluate a direct PPO policy in the desk human/object context."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# Also bootstraps the clean MuJoCo package on exFAT, in spawned workers.
from scripts.run_desk import prepare_mujoco_runtime
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from trax_rl.envs.trax_desk_social_env import TraxDeskSocialEnv


def write_json(path, payload):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False))
    temporary.replace(path)


def source_hash():
    digest=hashlib.sha256()
    paths=[ROOT/'trax_rl/envs/trax_desk_env.py',ROOT/'trax_rl/envs/trax_desk_social_env.py',
           ROOT/'trax_rl/dynamics/desk_humans.py',ROOT/'assets/trax/desk_scene.xml',
           ROOT/'assets/trax/desk_humans.xml',ROOT/'assets/trax/trax_base.stl',
           ROOT/'assets/trax/trax_head.stl',Path(__file__)]
    for p in paths:
        digest.update(str(p.relative_to(ROOT)).encode())
        digest.update(p.read_bytes())
    return digest.hexdigest()


def make_env(options, horizon, rank, seed):
    def create():
        env=Monitor(TraxDeskSocialEnv(training=True,horizon=horizon,**options))
        env.reset(seed=seed+rank*97)
        return env
    return create


def evaluate(model, config, seeds, yaws=None, baseline=None):
    episodes=[]
    for index,seed in enumerate(seeds):
        env=TraxDeskSocialEnv(training=False,horizon=config['evaluation']['horizon'],**config['environment'])
        try:
            observation,info=env.reset(seed=seed)
            if yaws is not None:
                import mujoco
                yaw=yaws[index%len(yaws)]
                env.env.data.qpos[env.env.base_adr+3:env.env.base_adr+7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
                mujoco.mj_forward(env.env.model,env.env.data)
                observation=env._augment(env.env._observation())
            random=np.random.default_rng(seed+777)
            for _ in range(config['evaluation']['horizon']):
                if baseline=='stationary':action=np.zeros(5,dtype=np.float32)
                elif baseline=='random':action=random.uniform(-.5,.5,5).astype(np.float32)
                else:action,_=model.predict(observation,deterministic=True)
                observation,_,term,trunc,info=env.step(action)
                if term or trunc:break
            episodes.append(dict(seed=seed,initial_yaw=yaws[index%len(yaws)] if yaws else 0.,
                **info['social_metrics']))
        finally:env.close()
    metrics=['return_','wave_response_rate','people_noticed','objects_inspected',
             'safety_override_rate','mean_command_variation','idle_fraction']
    mean={k:float(np.mean([e[k] for e in episodes])) for k in metrics}
    unsafe=sum(bool(e['falls'] or e['unstable'] or e['collisions']) for e in episodes)
    gates=config['evaluation']['gates']
    checks=dict(no_collisions_falls_or_instability=unsafe==0,
        waves=mean['wave_response_rate']>=gates['wave_response_rate'],
        people=mean['people_noticed']>=gates['people_noticed'],
        objects=mean['objects_inspected']>=gates['objects_inspected'],
        safety_overrides=mean['safety_override_rate']<=gates['safety_override_rate'])
    return dict(episodes=episodes,mean=mean,unsafe_episodes=unsafe,gates=checks,passed=all(checks.values()))


def checkpoint_metadata(path, model, config, validation=None):
    import importlib.metadata as metadata
    write_json(Path(path).with_suffix('.metadata.json'),dict(
        checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        runtime_sha256=source_hash(),observation_schema=TraxDeskSocialEnv.SCHEMA,
        observation_shapes={k:list(v.shape) for k,v in model.observation_space.spaces.items()},
        num_timesteps=model.num_timesteps,seed=model.seed,config=config,
        effective_training=dict(n_envs=model.n_envs,n_steps=model.n_steps,n_epochs=model.n_epochs,
            batch_size=model.batch_size,learning_rate=model.learning_rate if isinstance(model.learning_rate,(float,int)) else 'schedule'),
        scripted_motor_controller=False,privileged_reward_shaping=True,
        versions={n:metadata.version(n) for n in ['mujoco','gymnasium','torch','stable-baselines3']},
        validation_passed=bool(validation and validation['passed']),
        validation=validation,created_at=datetime.now(timezone.utc).isoformat()))


class DeskTrainingCallback(BaseCallback):
    def __init__(self, config, output):
        super().__init__()
        self.config=config
        self.output=Path(output)
        self.best_score=None
        self.last_eval=0
        self.started=time.monotonic()
        self.completed_episodes=0

    def validate(self):
        self.model.save(self.output/'latest_model.zip')
        checkpoint_metadata(self.output/'latest_model.zip',self.model,self.config)
        result=evaluate(self.model,self.config,self.config['evaluation']['validation_seeds'])
        result['num_timesteps']=self.num_timesteps
        write_json(self.output/'latest_validation.json',result)
        m=result['mean']
        score=(-result['unsafe_episodes'],m['wave_response_rate']*4+m['people_noticed']+m['objects_inspected'],
               -m['safety_override_rate'],m['return_'])
        if self.best_score is None or score>self.best_score:
            self.best_score=score
            self.model.save(self.output/'best_model.zip')
            checkpoint_metadata(self.output/'best_model.zip',self.model,self.config,result)
            write_json(self.output/'best_validation.json',result)
        print(f"VALIDATION step={self.num_timesteps} waves={m['wave_response_rate']:.2f} people={m['people_noticed']:.1f} objects={m['objects_inspected']:.1f} unsafe={result['unsafe_episodes']} passed={result['passed']}",flush=True)
        self.last_eval=self.num_timesteps
        return result

    def _on_step(self):
        for info in self.locals.get('infos',[]):
            if 'episode' in info:self.completed_episodes+=1
            for key,val in info.get('reward_components',{}).items():
                self.logger.record_mean('desk_reward/'+key,val)
            for key,val in info.get('social_metrics',{}).items():
                self.logger.record_mean('desk_metrics/'+key,val)
        if self.num_timesteps%2048==0:
            elapsed=time.monotonic()-self.started
            payload=dict(state='training',num_timesteps=self.num_timesteps,
                target_timesteps=self.config['training']['total_timesteps'],
                elapsed_seconds=elapsed,steps_per_second=self.num_timesteps/max(elapsed,.001),
                completed_episodes=self.completed_episodes,output=str(self.output))
            write_json(self.output/'status.json',payload)
            print(f"TRAIN step={self.num_timesteps} fps={payload['steps_per_second']:.0f} episodes={self.completed_episodes}",flush=True)
        if self.num_timesteps-self.last_eval>=self.config['evaluation']['frequency']:
            self.validate()
        return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/policies/desk_social_ppo_v1.yaml')
    parser.add_argument('--timesteps',type=int)
    parser.add_argument('--n-envs',type=int)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--evaluate',type=Path)
    parser.add_argument('--compare-baselines',action='store_true')
    args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text())
    train=config['training']
    if args.timesteps is not None:train['total_timesteps']=args.timesteps
    if args.n_envs is not None:train['n_envs']=args.n_envs
    if args.resume and args.compare_baselines:parser.error('Compare baselines on a fresh run, before training')
    if train['total_timesteps']<=0 or train['n_envs']<=0:parser.error('timesteps and n-envs must be positive')
    config['environment']['xml_path']=str(ROOT/config['environment']['xml_path'])
    output=args.output or ROOT/'checkpoints'/('desk_social_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output=output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if not args.evaluate and (output/'config.yaml').exists():parser.error('Use a new output directory to preserve previous runs')
    torch.set_num_threads(train['torch_threads'])
    if args.evaluate:
        model=PPO.load(args.evaluate,device='cpu')
        result=evaluate(model,config,config['evaluation']['test_seeds'],config['evaluation']['test_yaws'])
        write_json(output/'evaluation.json',result)
        print(json.dumps(result,indent=2))
        return
    (output/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    write_json(output/'status.json',dict(state='initializing',target_timesteps=train['total_timesteps']))
    factories=[make_env(config['environment'],train['horizon'],i,train['seed']) for i in range(train['n_envs'])]
    vec=SubprocVecEnv(factories,start_method='spawn') if train['n_envs']>1 else DummyVecEnv(factories)
    callback=DeskTrainingCallback(config,output)
    try:
        if args.resume:
            model=PPO.load(args.resume,env=vec,device='cpu')
        else:
            model=PPO('MultiInputPolicy',vec,learning_rate=train['learning_rate'],n_steps=train['n_steps'],
                batch_size=train['batch_size'],n_epochs=train['n_epochs'],gamma=train['gamma'],
                gae_lambda=train['gae_lambda'],clip_range=train['clip_range'],ent_coef=train['ent_coef'],
                policy_kwargs=dict(net_arch=dict(pi=train['net_arch'],vf=train['net_arch']),log_std_init=-.7),
                tensorboard_log=str(output/'tensorboard'),device='cpu',seed=train['seed'],verbose=0)
        baseline={}
        if args.compare_baselines:
            for kind in ['stationary','random','untrained']:
                result=evaluate(model,config,config['evaluation']['test_seeds'],config['evaluation']['test_yaws'],
                                baseline=kind if kind!='untrained' else None)
                baseline[kind]=result
                write_json(output/'baselines.json',baseline)
                print('BASELINE',kind,result['mean'],flush=True)
        print(f"START PPO {TraxDeskSocialEnv.SCHEMA} steps={train['total_timesteps']} n_envs={train['n_envs']} output={output}",flush=True)
        model.learn(total_timesteps=train['total_timesteps'],callback=callback,
                    reset_num_timesteps=not bool(args.resume),progress_bar=False)
        validation=callback.validate()
        model.save(output/'final_model.zip')
        checkpoint_metadata(output/'final_model.zip',model,config,validation)
        best=PPO.load(output/'best_model.zip',device='cpu')
        test=evaluate(best,config,config['evaluation']['test_seeds'],config['evaluation']['test_yaws'])
        write_json(output/'test_evaluation.json',test)
        write_json(output/'status.json',dict(state='completed',num_timesteps=model.num_timesteps,
            target_timesteps=train['total_timesteps'],test_passed=test['passed'],test_mean=test['mean'],
            output=str(output),elapsed_seconds=time.monotonic()-callback.started))
        print('COMPLETE',json.dumps(test['mean']),f"test_passed={test['passed']}",flush=True)
    except BaseException as exc:
        if 'model' in locals():
            model.save(output/'interrupted_model.zip')
            checkpoint_metadata(output/'interrupted_model.zip',model,config)
        write_json(output/'status.json',dict(state='interrupted',error=repr(exc),
            num_timesteps=getattr(locals().get('model'),'num_timesteps',0)))
        raise
    finally:vec.close()


if __name__=='__main__':main()
