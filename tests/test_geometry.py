"""Geometry checks against the golden-set reference (cvat_labeling pseudo3d.geometry)."""

import sys
from pathlib import Path

import numpy as np
import pytest

from data_engine import geometry
from data_engine.config import load_site

SITE = load_site()
CALIBRATION = Path(SITE["calibration"])
CVAT_LABELING = SITE.repo("cvat_labeling")

pytestmark = pytest.mark.skipif(
    not CALIBRATION.is_file() or not (CVAT_LABELING / "pseudo3d" / "geometry.py").is_file(),
    reason="standard calibration or cvat_labeling reference geometry not available",
)


@pytest.fixture(scope="module")
def reference():
    sys.path.insert(0, str(CVAT_LABELING))
    from pseudo3d import geometry as ref
    return ref, ref.load_cameras(CALIBRATION)


@pytest.fixture(scope="module")
def cams():
    return geometry.load_calibration(CALIBRATION)


def room_points(count: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(-100, 300, count), rng.uniform(-150, 400, count), rng.uniform(5, 70, count)])


def visible(cam, native):
    return np.isfinite(native).all(axis=-1) & (native[..., 0] > 0) & (native[..., 0] < 2048) \
        & (native[..., 1] > 0) & (native[..., 1] < 1536)


def test_projection_matches_reference(reference, cams):
    ref, ref_cams = reference
    points = room_points(200)
    for cid, cam in cams.items():
        mine = cam.project(points, verify=False) * cam.to_cvat
        theirs = ref_cams[int(cid)].project(points)
        ok = ref_cams[int(cid)].depth(points) > 1
        assert np.allclose(mine[ok], theirs[ok], atol=1e-6)


def test_pixel_ray_round_trip(cams):
    """pixel -> ray -> 3D point on the ray -> pixel; un-invertible corners give NaN rays, never wrong ones."""
    rng = np.random.default_rng(1)
    for cam in cams.values():
        pixels = np.column_stack([rng.uniform(20, 2028, 2000), rng.uniform(20, 1516, 2000)])
        rays = cam.rays(pixels)
        valid = np.isfinite(rays).all(axis=1)
        assert valid.mean() > 0.9
        points = cam.center + rays[valid] * rng.uniform(100, 500, (int(valid.sum()), 1))
        back = cam.project(points)
        finite = np.isfinite(back).all(axis=1)
        assert finite.mean() > 0.99
        assert np.linalg.norm(back[finite] - pixels[valid][finite], axis=1).max() < 1.0


def test_projection_never_folds_into_the_image(cams):
    """A point far outside the field of view is NaN or outside the frame, never inside it."""
    for cam in cams.values():
        for sideways in (cam.R[0], -cam.R[0], cam.R[1], -cam.R[1]):
            far_off_axis = cam.center + 200 * cam.R[2] + 2000 * sideways  # ~84 degrees off axis
            pixel = cam.project(far_off_axis[None])[0]
            inside = np.isfinite(pixel).all() and 0 <= pixel[0] < 2048 and 0 <= pixel[1] < 1536
            assert not inside


def test_triangulation_matches_reference(reference, cams):
    ref, ref_cams = reference
    rng = np.random.default_rng(2)
    points = room_points(120, seed=3)
    ids = sorted(cams)
    native = np.stack([cams[c].project(points) for c in ids])
    usable = np.stack([visible(cams[c], native[i]) for i, c in enumerate(ids)])
    noisy = native + rng.normal(0, 1.0, native.shape)
    xyz, used, residual = geometry.triangulate_views([cams[c] for c in ids], np.nan_to_num(noisy), usable)
    solved = np.isfinite(xyz).all(axis=1)
    assert solved.sum() > 60
    assert np.linalg.norm(xyz[solved] - points[solved], axis=1).max() < 6.0  # inches, 1 px native noise
    assert np.nanmax(residual) <= 12.0
    for n in np.flatnonzero(solved)[:25]:
        views = np.flatnonzero(used[:, n])
        point, errors = ref.triangulate([ref_cams[int(ids[v])] for v in views],
                                        noisy[views, n] * 0.625, np.ones(len(views)))
        assert np.linalg.norm(point - xyz[n]) < 1.0  # inches: Gauss-Newton vs robust least squares
        assert abs(errors.max() - np.nanmax(residual[:, n])) < 1.0


def test_gates_refuse_bad_points(cams):
    rng = np.random.default_rng(6)
    ids = sorted(cams)[:4]
    points = room_points(80, seed=4)
    native = np.stack([cams[c].project(points, verify=False) for c in ids])
    usable = np.stack([visible(cams[c], cams[c].project(points)) for c in ids])
    corrupted = np.nan_to_num(native).copy()
    corrupted[0] += rng.normal(0, 150.0, corrupted[0].shape)  # view 0 shows a different person
    xyz, used, _ = geometry.triangulate_views([cams[c] for c in ids], corrupted, usable)
    solved = np.isfinite(xyz).all(axis=1)
    enough = usable[0] & (usable.sum(axis=0) >= 3) & solved
    assert enough.sum() >= 10
    # With two or more good views the wrong view is dropped and the point stays accurate.
    assert used[0, enough].mean() < 0.1
    assert np.median(np.linalg.norm(xyz[enough] - points[enough], axis=1)) < 0.5
    assert np.isnan(xyz[usable.sum(axis=0) < 2]).all()


def test_pair_evidence_separates_people(cams):
    a, b = cams["0"], cams["3"]
    points = room_points(200, seed=5)
    na, nb = a.project(points), b.project(points)
    keep = visible(a, na) & visible(b, nb)
    ra, rb = a.rays(na[keep]), b.rays(nb[keep])
    _, residual, angle, front = geometry.pair_ray_evidence(a.center, ra, a.focal, b.center, rb, b.focal)
    assert residual.max() < 0.05 and front.all()
    shifted = np.roll(rb, 1, axis=0)  # pair each ray with a different point
    _, wrong, _, _ = geometry.pair_ray_evidence(a.center, ra, a.focal, b.center, shifted, b.focal)
    assert np.median(wrong) > 20
