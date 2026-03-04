import numpy as np
import torch

from cassava.utils import AverageMeter, cutmix, get_score, rand_bbox, seed_torch


def test_average_meter_single_update():
    meter = AverageMeter()
    meter.update(10.0)
    assert meter.val == 10.0
    assert meter.avg == 10.0
    assert meter.count == 1


def test_average_meter_multiple_updates():
    meter = AverageMeter()
    meter.update(10.0, n=2)
    meter.update(20.0, n=3)
    assert meter.count == 5
    expected_avg = (10.0 * 2 + 20.0 * 3) / 5
    assert abs(meter.avg - expected_avg) < 1e-6


def test_average_meter_reset():
    meter = AverageMeter()
    meter.update(5.0)
    meter.reset()
    assert meter.val == 0
    assert meter.avg == 0
    assert meter.count == 0


def test_get_score_perfect():
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4])
    assert get_score(y_true, y_pred) == 1.0


def test_get_score_all_wrong():
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([1, 2, 3, 4, 0])
    assert get_score(y_true, y_pred) == 0.0


def test_get_score_partial():
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 0, 0, 0])
    assert get_score(y_true, y_pred) == 2 / 5


def test_seed_torch_reproducibility():
    seed_torch(42)
    t1 = torch.randn(5)
    seed_torch(42)
    t2 = torch.randn(5)
    assert torch.equal(t1, t2)


def test_rand_bbox_within_bounds():
    size = (1, 3, 100, 100)
    lam = 0.5
    bbx1, bby1, bbx2, bby2 = rand_bbox(size, lam)
    assert 0 <= bbx1 <= 100
    assert 0 <= bby1 <= 100
    assert 0 <= bbx2 <= 100
    assert 0 <= bby2 <= 100
    assert bbx1 <= bbx2
    assert bby1 <= bby2


def test_cutmix_output_shapes():
    data = torch.randn(4, 3, 32, 32)
    target = torch.tensor([0, 1, 2, 3])
    new_data, targets = cutmix(data, target, alpha=1.0)
    assert new_data.shape == data.shape
    assert len(targets) == 3  # (target, shuffled_target, lam)
    assert targets[0].shape == target.shape
    assert targets[1].shape == target.shape


def test_cutmix_lam_in_valid_range():
    data = torch.randn(4, 3, 32, 32)
    target = torch.tensor([0, 1, 2, 3])
    _, targets = cutmix(data, target, alpha=1.0)
    lam = targets[2]
    assert 0 <= lam <= 1
