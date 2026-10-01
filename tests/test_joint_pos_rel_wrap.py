"""``joint_pos_rel(wrap_continuous=True)`` wraps only continuous joints.

A continuous joint (exposed by mjlab with non-finite ``±inf`` position limits)
has its home-relative delta ``wrap_to_pi(jp - home)`` taken, so the 2*pi seam
sits at ``home ± pi`` (out of the working range) rather than at absolute ``±pi``.
Because ``wrap_to_pi`` is modular this stays consistent with the deploy adapter,
which pre-wraps the absolute qpos before mjlab subtracts home. Bounded joints are
untouched, and the default (``wrap_continuous=False``) preserves the raw delta.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import cast

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.entity import Entity, EntityCfg
from mjlab.envs.mdp.observations import joint_pos_rel
from mjlab.sim.sim import Simulation, SimulationCfg

# One bounded hinge and one honestly-unlimited hinge (limited="false") — the
# shape the robot-format exporter now emits for a continuous joint.
_MIXED_XML = """
<mujoco>
  <worldbody>
    <body name="base">
      <body name="link1" pos="0 0 0.1">
        <joint name="bounded" type="hinge" axis="0 0 1" limited="true" range="-2 2"/>
        <geom type="box" size="0.05 0.05 0.1" mass="1"/>
        <body name="link2" pos="0 0 0.2">
          <joint name="continuous" type="hinge" axis="0 1 0" limited="false"/>
          <geom type="box" size="0.05 0.05 0.1" mass="0.5"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _adapter_wrap_to_pi(v: float) -> float:
  # The deploy adapter's wrap_to_pi (Python modulo → always non-negative).
  return (v + math.pi) % (2.0 * math.pi) - math.pi


def _env(joint_pos: torch.Tensor, default: torch.Tensor, limits: torch.Tensor):
  data = SimpleNamespace(
    joint_pos=joint_pos,
    joint_pos_biased=joint_pos,
    default_joint_pos=default,
    joint_pos_limits=limits,
  )
  entity = SimpleNamespace(data=data)
  return cast("object", SimpleNamespace(scene={"robot": entity}))


def _cfg():
  return cast("object", SimpleNamespace(name="robot", joint_ids=slice(None)))


def test_default_is_raw_delta():
  device = get_test_device()
  jp = torch.tensor([[7.0, 0.3]], device=device)
  default = torch.tensor([[0.0, 0.1]], device=device)
  # joint 0 continuous (±inf), joint 1 bounded — but wrap is OFF by default.
  limits = torch.tensor([[[-math.inf, math.inf], [-2.0, 2.0]]], device=device)
  out = joint_pos_rel(_env(jp, default, limits), asset_cfg=_cfg())  # type: ignore[arg-type]
  assert torch.allclose(out, jp - default)


def test_wraps_only_continuous_joints():
  device = get_test_device()
  # joint 0 bounded (wide finite limits) at 7.0 → must NOT be wrapped.
  # joint 1 continuous (±inf) wound up to 7.0 rad → wrapped into [-pi, pi].
  jp = torch.tensor([[7.0, 7.0]], device=device)
  default = torch.tensor([[0.0, 0.0]], device=device)
  limits = torch.tensor([[[-100.0, 100.0], [-math.inf, math.inf]]], device=device)
  out = joint_pos_rel(_env(jp, default, limits), wrap_continuous=True, asset_cfg=_cfg())  # type: ignore[arg-type]
  assert out[0, 0].item() == pytest.approx(7.0)
  assert out[0, 1].item() == pytest.approx(_adapter_wrap_to_pi(7.0), abs=1e-4)


def test_delta_wrap_matches_adapter_across_revolutions():
  # New semantics: wrap the home-relative DELTA, i.e. wrap_to_pi(jp - home).
  # Sample angles across ±3π (avoiding exact odd multiples of pi where the
  # branch endpoint convention differs by a measure-zero amount).
  device = get_test_device()
  home = -0.5
  angles = [-3.0 * math.pi + 0.2, -4.0, -0.7, 0.0, 2.5, 7.0, 3.0 * math.pi - 0.2]
  jp = torch.tensor([angles], device=device)
  default = torch.full_like(jp, home)
  limits = torch.full((1, len(angles), 2), math.inf, device=device)
  limits[..., 0] = -math.inf
  out = joint_pos_rel(_env(jp, default, limits), wrap_continuous=True, asset_cfg=_cfg())  # type: ignore[arg-type]
  expected = torch.tensor(
    [[_adapter_wrap_to_pi(a - home) for a in angles]], device=device
  )
  assert torch.allclose(out, expected, atol=1e-4)

  # Parity: because wrap_to_pi is modular, feeding the adapter's pre-wrapped
  # absolute qpos yields the IDENTICAL relative observation as the raw angle —
  # wrap(wrap(jp) - home) == wrap(jp - home). This is the train/deploy guarantee.
  jp_prewrapped = torch.tensor([[_adapter_wrap_to_pi(a) for a in angles]], device=device)
  out_prewrapped = joint_pos_rel(
    _env(jp_prewrapped, default, limits), wrap_continuous=True, asset_cfg=_cfg()
  )  # type: ignore[arg-type]
  assert torch.allclose(out, out_prewrapped, atol=1e-4)


def test_seam_is_on_far_side_of_home_not_absolute_pi():
  # With a non-zero home the 2*pi seam sits at home ± pi (far from the working
  # range), NOT at absolute ±pi. A joint sitting NEAR its home yields a small
  # continuous delta even when its absolute angle straddles ±pi — no 2*pi jump.
  device = get_test_device()
  home = math.pi - 0.1  # working point a hair below +pi absolute
  jp = torch.tensor([[home - 0.05, home + 0.05]], device=device)  # straddles +pi
  default = torch.full_like(jp, home)
  limits = torch.full((1, 2, 2), math.inf, device=device)
  limits[..., 0] = -math.inf
  out = joint_pos_rel(_env(jp, default, limits), wrap_continuous=True, asset_cfg=_cfg())  # type: ignore[arg-type]
  assert out[0, 0].item() == pytest.approx(-0.05, abs=1e-4)
  assert out[0, 1].item() == pytest.approx(0.05, abs=1e-4)


def test_wrap_is_idempotent_for_already_wrapped():
  device = get_test_device()
  jp = torch.tensor([[0.5, -1.2]], device=device)  # already inside [-pi, pi]
  default = torch.zeros_like(jp)
  limits = torch.full((1, 2, 2), math.inf, device=device)
  limits[..., 0] = -math.inf
  out = joint_pos_rel(_env(jp, default, limits), wrap_continuous=True, asset_cfg=_cfg())  # type: ignore[arg-type]
  assert torch.allclose(out, jp, atol=1e-5)


def test_real_entity_unlimited_joint_surfaces_inf_and_wraps():
  """Integration smoke on a REAL mjlab Entity (not a fake env).

  Confirms the exact field ``joint_pos_rel`` reads — ``joint_pos_limits`` — is
  ``±inf`` for a ``limited="false"`` joint (the exporter's continuous shape),
  and that a wound-up continuous joint is wrapped into ``[-pi, pi]`` while the
  bounded joint passes through raw. This closes the gap between the fake-env
  unit tests above and the real Entity limit resolution.
  """
  device = get_test_device()
  entity = Entity(EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(_MIXED_XML)))
  model = entity.compile()
  sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)

  # Column order follows the model's joint order: bounded (0), continuous (1).
  limits = entity.data.joint_pos_limits[0]
  assert torch.isfinite(limits[0]).all()  # bounded joint: finite
  assert not torch.isfinite(limits[1]).all()  # continuous joint: ±inf
  assert limits[1, 0] == float("-inf") and limits[1, 1] == float("inf")

  # Wind the continuous joint up past +pi; keep the bounded joint in range.
  entity.write_joint_position_to_sim(torch.tensor([[1.5, 7.0]], device=device))
  sim.forward()

  env = cast("object", SimpleNamespace(scene={"robot": entity}))
  cfg = cast("object", SimpleNamespace(name="robot", joint_ids=slice(None)))
  default = entity.data.default_joint_pos[0]  # zeros for this model

  out = joint_pos_rel(env, wrap_continuous=True, asset_cfg=cfg)[0]  # type: ignore[arg-type]
  # Bounded joint: raw (1.5 - default). Continuous joint: wrap_to_pi(7.0) - default.
  assert out[0].item() == pytest.approx(1.5 - default[0].item(), abs=1e-4)
  expected_cont = _adapter_wrap_to_pi(7.0) - default[1].item()
  assert out[1].item() == pytest.approx(expected_cont, abs=1e-4)

  # Without the flag the continuous joint stays the raw (unwrapped) delta.
  raw = joint_pos_rel(env, asset_cfg=cfg)[0]  # type: ignore[arg-type]
  assert raw[1].item() == pytest.approx(7.0 - default[1].item(), abs=1e-4)
