import pytest
import torch

from cassava.loss import (
    BiTemperedLogisticLoss,
    FocalCosineLoss,
    bi_tempered_logistic_loss,
    compute_normalization,
    exp_t,
    log_t,
    tempered_softmax,
)


def test_log_t_at_t1_matches_natural_log():
    u = torch.tensor([1.0, 2.0, 3.0])
    result = log_t(u, t=1.0)
    expected = u.log()
    assert torch.allclose(result, expected)


def test_exp_t_at_t1_matches_natural_exp():
    u = torch.tensor([0.0, 1.0, -1.0])
    result = exp_t(u, t=1)
    expected = u.exp()
    assert torch.allclose(result, expected)


def test_log_t_exp_t_inverse():
    u = torch.tensor([0.5, 1.0, 2.0])
    for t in [0.5, 1.0, 1.5]:
        result = exp_t(log_t(u, t), t)
        assert torch.allclose(result, u, atol=1e-5)


def test_tempered_softmax_sums_to_one_at_t1():
    activations = torch.randn(2, 5)
    probs = tempered_softmax(activations, t=1.0)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(2), atol=1e-5)


def test_tempered_softmax_non_negative():
    activations = torch.randn(2, 5)
    for t in [0.5, 1.0, 1.5]:
        probs = tempered_softmax(activations, t)
        assert (probs >= 0).all()


def test_bi_tempered_loss_returns_scalar():
    activations = torch.randn(4, 5)
    labels = torch.randint(0, 5, (4,))
    loss = bi_tempered_logistic_loss(activations, labels, t1=0.3, t2=1.0)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_bi_tempered_loss_with_label_smoothing():
    activations = torch.randn(4, 5)
    labels = torch.randint(0, 5, (4,))
    loss_no_smooth = bi_tempered_logistic_loss(activations, labels, t1=0.3, t2=1.0)
    loss_smooth = bi_tempered_logistic_loss(
        activations, labels, t1=0.3, t2=1.0, label_smoothing=0.1
    )
    # Both should be valid scalars
    assert not torch.isnan(loss_no_smooth)
    assert not torch.isnan(loss_smooth)


def test_bi_tempered_loss_module():
    module = BiTemperedLogisticLoss(t1=0.3, t2=1.0, smoothing=0.05)
    logits = torch.randn(4, 5)
    labels = torch.randint(0, 5, (4,))
    loss = module(logits, labels)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_bi_tempered_loss_gradient_flows():
    activations = torch.randn(4, 5, requires_grad=True)
    labels = torch.randint(0, 5, (4,))
    loss = bi_tempered_logistic_loss(activations, labels, t1=0.3, t2=1.0)
    loss.backward()
    assert activations.grad is not None
    assert not torch.isnan(activations.grad).any()


def test_bi_tempered_loss_reduction_none():
    activations = torch.randn(4, 5)
    labels = torch.randint(0, 5, (4,))
    loss = bi_tempered_logistic_loss(activations, labels, t1=0.3, t2=1.0, reduction="none")
    assert loss.shape == (4,)


def test_bi_tempered_loss_reduction_sum():
    activations = torch.randn(4, 5)
    labels = torch.randint(0, 5, (4,))
    loss = bi_tempered_logistic_loss(activations, labels, t1=0.3, t2=1.0, reduction="sum")
    assert loss.shape == ()


def test_focal_cosine_loss_on_cpu():
    model = FocalCosineLoss(alpha=1, gamma=2, xent=0.1)
    logits = torch.randn(4, 5)
    labels = torch.randint(0, 5, (4,))
    loss = model(logits, labels)
    assert loss.shape == ()
    assert not torch.isnan(loss)


def test_focal_cosine_loss_gradient_flows():
    model = FocalCosineLoss(alpha=1, gamma=2, xent=0.1)
    logits = torch.randn(4, 5, requires_grad=True)
    labels = torch.randint(0, 5, (4,))
    loss = model(logits, labels)
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
