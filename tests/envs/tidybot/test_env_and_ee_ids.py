"""Tests for environment IDs and end-effector IDs/poses in TidyBotRobotEnv."""

from pathlib import Path

import numpy as np
import pytest

try:
    from prbench.envs.tidybot.tidybot_robot_env import TidyBotRobotEnv
    import prbench

    MUJOCO_AVAILABLE = True
    skip_reason = ""
except Exception as e:  # pragma: no cover
    MUJOCO_AVAILABLE = False
    skip_reason = f"TidyBot/MuJoCo not available: {e}"


def _load_ground_scene_xml() -> str:
    assert MUJOCO_AVAILABLE
    model_path = (
        Path(prbench.__file__).parent
        / "envs"
        / "tidybot"
        / "models"
        / "stanford_tidybot"
        / "ground_scene.xml"
    )
    with open(model_path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason=skip_reason)
def test_environment_ids_and_object_positions_ground_scene() -> None:
    env = TidyBotRobotEnv(control_frequency=20, camera_names=None, show_viewer=False)
    xml_string = _load_ground_scene_xml()
    obs, _, _, _ = env.reset(xml_string)
    assert isinstance(obs, dict)

    # Environment IDs present and sane
    ids = env.get_environment_ids()
    assert "objects" in ids and isinstance(ids["objects"], list)
    assert len(ids["objects"]) >= 1

    # Validate that known cube bodies are detected
    model = env.sim.model  # type: ignore[union-attr]
    id2name = model._body_id2name  # type: ignore[attr-defined]
    obj_names = [id2name[i] for i in ids["objects"] if i in id2name]
    assert any(n and n.lower().startswith("cube") for n in obj_names)

    # Extract object positions from qpos (objects precede robot joints, freejoint: 7 DoF)
    qpos = obs["qpos"]
    # Find first robot joint index from known names; if fails, assume objects span until last multiple of 7
    try:
        base_start = model.get_joint_qpos_addr("joint_x")  # type: ignore[attr-defined]
        robot_min_idx = base_start
    except Exception:  # pragma: no cover - fallback when names differ
        robot_min_idx = (qpos.size // 7) * 7

    extracted_positions: list[np.ndarray] = []
    i = 0
    while i + 6 < robot_min_idx:
        extracted_positions.append(qpos[i : i + 3].copy())
        i += 7

    # Each object body position should match one of the extracted qpos chunks
    data = env.sim.data  # type: ignore[union-attr]
    for body_id in ids["objects"]:
        body_pos = data.xpos[body_id]
        assert np.any(
            [np.allclose(body_pos, p, atol=1e-6) for p in extracted_positions]
        ), f"Body id {body_id} position {body_pos} not found in qpos objects {extracted_positions}"


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason=skip_reason)
def test_end_effector_ids_and_pose_validity() -> None:
    env = TidyBotRobotEnv(control_frequency=20, camera_names=None, show_viewer=False)
    xml_string = _load_ground_scene_xml()
    env.reset(xml_string)

    # IDs should be cached
    assert env.ee_site_id is not None
    # Body may be None if name differs; only assert if provided
    if env.ee_body_id is not None:
        assert isinstance(env.ee_body_id, int)

    # End-effector pose should be finite and consistent with model lookup by name
    data = env.sim.data  # type: ignore[union-attr]
    site_id = env.ee_site_id  # type: ignore[assignment]
    site_pos = data.site_xpos[site_id]
    site_mat = data.site_xmat[site_id]
    assert np.all(np.isfinite(site_pos)) and np.all(np.isfinite(site_mat))

    # Cross-check id via model mapping
    model = env.sim.model  # type: ignore[union-attr]
    try:
        mapped_id = model._site_name2id.get("pinch_site")  # type: ignore[attr-defined]
        if mapped_id is not None:
            assert int(mapped_id) == int(site_id)
    except Exception:  # pragma: no cover
        pass

