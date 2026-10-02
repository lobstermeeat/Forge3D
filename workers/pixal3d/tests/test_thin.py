"""The thin-object rule's measurement: a preview's bounding box to a choice of weights."""

import pytest
import trimesh

from pixal3d_worker import thin


def box(*extents):
    return trimesh.creation.box(extents=extents)


def test_a_thin_box_is_thin_and_a_cube_is_not():
    report = thin.measure(box(1.0, 0.1, 0.8))
    assert report["extents"] == [0.1, 0.8, 1.0] and report["ratio"] == pytest.approx(0.1)
    assert report["thin"] is True and report["threshold"] == thin.THIN_RATIO
    cube = thin.measure(box(1.0, 1.0, 1.0))
    assert cube["ratio"] == pytest.approx(1.0) and cube["thin"] is False


def test_the_threshold_is_honoured_at_and_above():
    plate = box(1.0, 0.15, 1.0)  # the pistol's ratio
    assert thin.measure(plate, 0.20)["thin"] is True
    assert thin.measure(plate, 0.15)["thin"] is True  # at the threshold counts as thin
    assert thin.measure(plate, 0.10)["thin"] is False
    assert thin.is_thin(0.37) is False and thin.is_thin(0.10) is True  # the arcade and the shield


def test_the_box_is_axis_aligned_and_the_pose_does_not_matter_for_a_level_object():
    plate = box(1.0, 0.1, 1.0)
    plate.apply_translation([3.0, -2.0, 5.0])  # where it stands changes nothing
    assert thin.measure(plate)["ratio"] == pytest.approx(0.1)
    diagonal = box(1.4, 0.2, 0.4)
    diagonal.apply_transform(trimesh.transformations.rotation_matrix(0.8, [0, 0, 1]))  # turned in the picture plane
    assert thin.measure(diagonal)["thin"] is False  # the guitar's case: not thin to the rule


def test_degenerate_meshes_are_not_thin():
    flat = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 0], [0, 0, 0]], faces=[[0, 1, 2]], process=False)
    report = thin.measure(flat)
    assert report["ratio"] is None and report["thin"] is False
    with pytest.raises(ValueError):
        thin.extents(trimesh.Trimesh())
