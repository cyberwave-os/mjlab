"""Useful methods for MDP observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, RayCastSensor
from mjlab.utils.lab_api.math import wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


##
# Root state.
##


def base_lin_vel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b


def base_ang_vel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_ang_vel_b


def projected_gravity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b


##
# Joint state.
##


def joint_pos_rel(
  env: ManagerBasedRlEnv,
  biased: bool = False,
  wrap_continuous: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Joint positions relative to the default (home) pose.

  When ``wrap_continuous`` is set, *continuous* joints (those with non-finite
  ``±inf`` position limits) have the home-relative delta ``(jp - home)`` wrapped
  into ``[-pi, pi]``. Wrapping the *delta* (rather than wrapping the absolute
  position and then subtracting home) puts the 2*pi observation seam on the far
  side of home — out of the joint's working range — instead of at absolute
  ``±pi`` regardless of home. Because ``wrap_to_pi`` is modular
  (``wrap(wrap(jp) - home) == wrap(jp - home)``), this stays consistent with the
  deploy adapter, which wraps the absolute measured qpos in ``positions_to_qpos``
  before mjlab subtracts the home pose — closing the train/deploy gap for
  continuous joints (a wound-up joint no longer feeds an ever-growing raw angle
  to the policy). For a joint whose home is ``0`` (e.g. the Kinova roll joints)
  it is identical to the previous ``wrap(jp) - home``, so trained checkpoints are
  unchanged. Defaults to ``False`` so existing tasks are unchanged; enable it
  only for newly trained policies.
  """
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  jnt_ids = asset_cfg.joint_ids
  joint_pos = asset.data.joint_pos_biased if biased else asset.data.joint_pos
  jp = joint_pos[:, jnt_ids]
  dp = default_joint_pos[:, jnt_ids]
  if not wrap_continuous:
    return jp - dp
  # Continuous joints are exposed by mjlab with non-finite (±inf) position
  # limits (see Entity: an unlimited joint's raw [0, 0] range is overwritten
  # with [-inf, +inf]).
  limits = asset.data.joint_pos_limits[:, jnt_ids]  # (N, J, 2)
  continuous = ~torch.isfinite(limits).all(dim=-1)  # (N, J)
  return torch.where(continuous, wrap_to_pi(jp - dp), jp - dp)


def joint_vel_rel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  jnt_ids = asset_cfg.joint_ids
  return asset.data.joint_vel[:, jnt_ids] - default_joint_vel[:, jnt_ids]


##
# Actions.
##


def last_action(env: ManagerBasedRlEnv, action_name: str | None = None) -> torch.Tensor:
  if action_name is None:
    return env.action_manager.action
  return env.action_manager.get_term(action_name).raw_action


##
# Commands.
##


def generated_commands(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return command


##
# Sensors.
##


def builtin_sensor(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Get observation from a built-in sensor by name."""
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  return sensor.data


def projected_gravity_from_sensor(
  env: ManagerBasedRlEnv, sensor_name: str
) -> torch.Tensor:
  """Projected gravity from a ``framezaxis`` up-vector sensor.

  The sensor is expected to output the world Z-axis expressed in the sensor's frame
  (e.g. ``framezaxis`` with ``objtype=body objname=world`` and ``reftype=site``). That
  is the body-frame "up" vector, so it is negated to point along gravity.

  Unlike :func:`projected_gravity`, which uses the root body orientation, this reads
  the sensor's site frame and therefore reflects IMU site pose randomization.
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  return -sensor.data


def height_scan(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  offset: float = 0.0,
  miss_value: float | None = None,
) -> torch.Tensor:
  """Height scan from a raycast sensor.

  Returns the height of the sensor frame above each hit point.
  Supports multi-frame sensors: each ray uses its own frame's Z.

  Args:
    env: The environment.
    sensor_name: Name of a RayCastSensor in the scene.
    offset: Constant offset subtracted from heights.
    miss_value: Value to use for rays that miss (distance < 0).
      Defaults to the sensor's ``max_distance``.

  Returns:
    Tensor of shape [B, N] where N = num_frames * num_rays_per_frame.
    Rays are ordered frame-major (all rays for frame 0, then frame 1, etc.).
  """
  sensor: RayCastSensor = env.scene[sensor_name]
  if miss_value is None:
    miss_value = sensor.cfg.max_distance

  data = sensor.data
  F, N = sensor.num_frames, sensor.num_rays_per_frame
  B = data.distances.shape[0]

  # Each ray's height = its frame's Z - hit Z. For single-frame sensors (F=1) this
  # reduces to the original pos_w[:, 2] - hit_z broadcast.
  frame_z = data.frame_pos_w[:, :, 2:3]  # [B, F, 1]
  hit_z = data.hit_pos_w[..., 2].view(B, F, N)  # [B, F, N]
  heights = (frame_z - hit_z - offset).view(B, F * N)

  miss_mask = data.distances < 0
  return torch.where(miss_mask, torch.full_like(heights, miss_value), heights)
