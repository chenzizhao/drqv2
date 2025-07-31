import logging
from collections import OrderedDict

import dm_env
import gymnasium as gym
import numpy as np
from dm_env import TimeStep, specs

import knotgym  # noqa: F401

_logging_level = logging.ERROR
_logging_submodules = ["knotgym"]
logging.basicConfig(level=_logging_level)
for name in logging.root.manager.loggerDict:
  if any(name.startswith(mod) for mod in _logging_submodules):
    logging.getLogger(name).setLevel(_logging_level)


class ReshapePixelsWrapper(dm_env.Environment):
  def __init__(self, env):
    self._env = env
    wrapped_obs_spec = self._env.observation_spec()
    assert len(wrapped_obs_spec) == 1
    pixels_spec = wrapped_obs_spec["pixels"]
    assert pixels_spec.dtype == np.uint8
    assert (pixels_spec.minimum == 0).all()
    assert (pixels_spec.maximum == 255).all()

    # [H, 2W, C] -> [H, W, 2C]
    height, double_width, channels = pixels_spec.shape
    width = double_width // 2
    self._obs_spec = OrderedDict(
      pixels=specs.BoundedArray(
        shape=(height, width, 2 * channels),
        dtype=np.uint8,
        minimum=0,
        maximum=255,
      )
    )

  def action_spec(self):
    return self._env.action_spec()

  def observation_spec(self):
    return self._obs_spec

  def step(self, action):
    time_step = self._env.step(action)
    obs = time_step.observation["pixels"]
    obs = self._proc_pixels(obs)
    return time_step._replace(
      observation=OrderedDict(pixels=obs),
    )

  def reset(self):
    time_step = self._env.reset()
    obs = time_step.observation["pixels"]
    obs = self._proc_pixels(obs)
    return time_step._replace(
      observation=OrderedDict(pixels=obs),
    )

  def _proc_pixels(self, pixels):
    # [H, 2W, C] -> [H, W, 2C]
    double_width = pixels.shape[-2]
    width = double_width // 2
    first_half = pixels[..., :width, :]
    second_half = pixels[..., width : 2 * width, :]
    pixels = np.concatenate([first_half, second_half], axis=-1)
    assert pixels.shape == self._obs_spec["pixels"].shape
    return pixels

  def __getattr__(self, name):
    return getattr(self._env, name)


class Knot(dm_env.Environment):
  # domain == knot
  def __init__(self, task_name, seed=None, **kwargs):
    # TODO logdir = cfg.work_dir / split / f"{rank:04d}" if rank == 1 else None
    env = gym.make(
      "knotgym/Unknot-v0",
      task=task_name,
      logdir=None,
      logfreq=100,  #
      **kwargs,
    )
    self._env = env
    self._reset_next_step = True
    self._last_render = None
    assert kwargs.get("output_pixels", True)

  def step(self, action):
    if self._reset_next_step:
      return self.reset()

    obs, reward, term, trunc, info = self._env.step(action)
    self._step_count += 1

    if trunc:
      discount = 1.0
    else:
      discount = 1.0 if term else None
    del term, trunc

    self._last_render = obs
    obs = OrderedDict(pixels=obs)

    episode_over = discount is not None
    if episode_over:
      self._reset_next_step = True
      return dm_env.TimeStep(
        step_type=dm_env.StepType.LAST,
        reward=reward,
        discount=discount,
        observation=obs,
      )
    return TimeStep(
      step_type=dm_env.StepType.MID,
      reward=reward,
      discount=1.0,
      observation=obs,
    )

  def reset(self):
    self._reset_next_step = False
    self._step_count = 0

    obs, info = self._env.reset()
    self._last_render = obs
    obs = OrderedDict(pixels=obs)
    time_step = TimeStep(
      step_type=dm_env.StepType.FIRST,
      reward=None,
      discount=None,
      observation=obs,
    )
    return time_step

  def observation_spec(self):
    obs_space = self._env.observation_space
    return OrderedDict(
      pixels=specs.BoundedArray(
        shape=obs_space.shape,
        dtype=obs_space.dtype,
        minimum=obs_space.low,
        maximum=obs_space.high,
      ),
    )

  def action_spec(self):
    action_space = self._env.action_space
    return specs.BoundedArray(
      shape=action_space.shape,
      dtype=action_space.dtype,
      minimum=action_space.low,
      maximum=action_space.high,
    )

  def render(self, mode="rgb_array"):
    return self._last_render


def load(name, seed=None, **task_kwargs):
  domain, task = name.split("_", 1)
  env = Knot(task, seed=seed, **task_kwargs)
  env = ReshapePixelsWrapper(env)
  return env
