import numpy as np
import pytest

from data_engine import reid


def _unit(rng, n, d=16):
    return reid.l2_normalize(rng.normal(size=(n, d)))


def test_l2_normalize_and_cosine():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 8))
    n = reid.l2_normalize(a)
    assert np.allclose(np.linalg.norm(n, axis=1), 1, atol=1e-6)
    sim = reid.cosine_matrix(a, a)
    assert np.allclose(np.diag(sim), 1, atol=1e-5)
    assert np.allclose(sim, sim.T, atol=1e-6)
    assert reid.l2_normalize(np.zeros((1, 4))).shape == (1, 4)  # no NaN
    assert np.isfinite(reid.l2_normalize(np.zeros((1, 4)))).all()


def test_aggregate_mean_weights_and_empty():
    rng = np.random.default_rng(1)
    e = _unit(rng, 5)
    d = reid.aggregate(e)
    assert np.isclose(np.linalg.norm(d), 1, atol=1e-5)
    assert np.allclose(d, reid.l2_normalize(e.mean(0)), atol=1e-6)
    w = np.array([0, 0, 0, 0, 1.0])
    assert np.allclose(reid.aggregate(e, w), e[4], atol=1e-6)
    assert reid.aggregate(np.empty((0, 16))) is None
    with pytest.raises(ValueError):
        reid.aggregate(e, np.zeros(5))
    with pytest.raises(ValueError):
        reid.aggregate(e, np.ones(4))
    # scale of unnormalised inputs does not matter
    assert np.allclose(reid.aggregate(e * 7), d, atol=1e-5)


def test_quality_weights_monotone():
    boxes = np.array([[0, 0, 40, 60, 0.9], [0, 0, 80, 200, 0.9], [0, 0, 80, 200, 0.3]], dtype=np.float32)
    w = reid.quality_weights(boxes)
    assert w[1] > w[0] and w[1] > w[2] and (w > 0).all() and (w <= 1).all()


def test_decide_best_rule():
    assert reid.decide_best(np.array([0.9, 0.5, 0.4]), s_min=0.8, margin=0.1) == 0
    assert reid.decide_best(np.array([0.9, 0.85]), s_min=0.8, margin=0.1) == -1  # margin
    assert reid.decide_best(np.array([0.7, 0.2]), s_min=0.8, margin=0.1) == -1  # floor
    assert reid.decide_best(np.array([0.7]), s_min=0.6, margin=0.5) == 0  # single candidate: no margin
    assert reid.decide_best(np.array([]), s_min=0.0, margin=0.0) == -1
    assert reid.decide_best(np.array([np.nan, 0.9]), s_min=0.5, margin=0.2) == 1
    assert reid.decide_best(np.array([0.9, 0.1]), s_min=0.5, margin=0.2, mutual=np.array([False, True])) == -1
    assert reid.decide_best(np.array([0.9, 0.1]), s_min=0.5, margin=0.2, mutual=np.array([True, False])) == 0


def test_decide_matrix_mutual():
    sim = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.7]])
    plain = reid.decide_matrix(sim, s_min=0.5, margin=0.1)
    assert list(plain) == [0, 0, 1]
    mutual = reid.decide_matrix(sim, s_min=0.5, margin=0.1, mutual=True)
    assert list(mutual) == [0, -1, 1]  # query 1 is not column 0's best query


def test_threshold_for_false_accepts():
    different = np.array([0.1, 0.2, 0.3, 0.4])
    s0 = reid.threshold_for_false_accepts(np.array([0.9]), different, 0.0)
    assert s0 > 0.4 and (different >= s0).sum() == 0
    s25 = reid.threshold_for_false_accepts(np.array([0.9]), different, 0.25)
    assert (different >= s25).sum() == 1
    assert reid.threshold_for_false_accepts(np.array([]), different, 1.0) == 0.1


def test_roc_auc():
    assert reid.roc_auc(np.array([0.9, 0.8]), np.array([0.1, 0.2])) == 1.0
    assert reid.roc_auc(np.array([0.1]), np.array([0.9])) == 0.0
    assert reid.roc_auc(np.array([0.5]), np.array([0.5])) == 0.5
    assert np.isnan(reid.roc_auc(np.array([]), np.array([1.0])))


def test_prepare_crop_shape_and_degenerate():
    image = np.random.default_rng(0).integers(0, 255, size=(300, 200, 3), dtype=np.uint8)
    x = reid.prepare_crop(image, [10, 20, 90, 250])
    assert x.shape == (3, 256, 128) and x.dtype == np.float32
    assert reid.prepare_crop(image, [10, 20, 12, 250]) is None
    assert reid.prepare_crop(image, [-50, -50, 100, 100]).shape == (3, 256, 128)  # clamped to the image
