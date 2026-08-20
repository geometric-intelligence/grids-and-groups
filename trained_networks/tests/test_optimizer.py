"""Tests for trained_networks.optimizer."""

import pytest
import torch

import trained_networks.model as model
import trained_networks.optimizer as optimizer


class TestPerNeuronScaledSGD:
    """Tests for optimizer.PerNeuronScaledSGD."""

    @pytest.fixture
    def two_layer_mlp(self):
        return model.TwoLayerMLP(group_size=5, hidden_dim=10, k=3, nonlinearity="power")

    def test_step_updates_parameters(self, two_layer_mlp):
        opt = optimizer.PerNeuronScaledSGD(two_layer_mlp, lr=0.01)

        initial_w_in = two_layer_mlp.W_in.clone()
        initial_w_out = two_layer_mlp.W_out.clone()

        x = torch.randn(4, two_layer_mlp.k, two_layer_mlp.group_size)
        y = two_layer_mlp(x)
        loss = y.sum()
        loss.backward()

        opt.step()

        assert not torch.allclose(two_layer_mlp.W_in, initial_w_in), "W_in not updated"
        assert not torch.allclose(two_layer_mlp.W_out, initial_w_out), "W_out not updated"

    def test_degree_inference(self, two_layer_mlp):
        opt = optimizer.PerNeuronScaledSGD(two_layer_mlp, lr=0.01)
        expected_degree = two_layer_mlp.k
        assert opt.defaults["degree"] == expected_degree

    def test_explicit_degree(self, two_layer_mlp):
        explicit_degree = 5
        opt = optimizer.PerNeuronScaledSGD(two_layer_mlp, lr=0.01, degree=explicit_degree)
        assert opt.defaults["degree"] == explicit_degree

    def test_finite_gradients_after_step(self, two_layer_mlp):
        opt = optimizer.PerNeuronScaledSGD(two_layer_mlp, lr=0.01)

        x = torch.randn(4, two_layer_mlp.k, two_layer_mlp.group_size)
        y = two_layer_mlp(x)
        loss = y.sum()
        loss.backward()

        opt.step()

        for name, param in two_layer_mlp.named_parameters():
            assert torch.isfinite(param).all(), f"Non-finite values in {name}"


class TestHybridRNNOptimizer:
    """Tests for optimizer.HybridRNNOptimizer."""

    @pytest.fixture
    def quadratic_rnn(self):
        return model.QuadraticRNN(group_size=5, hidden_dim=10, k=3)

    def test_step_updates_all_parameters(self, quadratic_rnn):
        opt = optimizer.HybridRNNOptimizer(quadratic_rnn, lr=0.01, adam_lr=0.001)

        initial_params = {name: param.clone() for name, param in quadratic_rnn.named_parameters()}

        x = torch.randn(4, 3, quadratic_rnn.group_size)
        y = quadratic_rnn(x)
        loss = y.sum()
        loss.backward()

        opt.step()

        for name, param in quadratic_rnn.named_parameters():
            assert not torch.allclose(param, initial_params[name]), f"{name} not updated"

    def test_scaled_sgd_for_mlp_params(self, quadratic_rnn):
        opt = optimizer.HybridRNNOptimizer(quadratic_rnn, lr=0.01)

        assert len(opt.param_groups) == 2
        assert opt.param_groups[0]["type"] == "scaled_sgd"
        assert opt.param_groups[1]["type"] == "adam"

    def test_adam_for_w_mix(self, quadratic_rnn):
        opt = optimizer.HybridRNNOptimizer(quadratic_rnn, lr=0.01, adam_lr=0.001)

        adam_params = list(opt.param_groups[1]["params"])
        assert len(adam_params) == 1
        assert adam_params[0] is quadratic_rnn.W_mix

    def test_finite_parameters_after_step(self, quadratic_rnn):
        opt = optimizer.HybridRNNOptimizer(quadratic_rnn, lr=0.01, adam_lr=0.001)

        x = torch.randn(4, 3, quadratic_rnn.group_size)
        y = quadratic_rnn(x)
        loss = y.sum()
        loss.backward()

        opt.step()

        for name, param in quadratic_rnn.named_parameters():
            assert torch.isfinite(param).all(), f"Non-finite values in {name}"

    def test_multiple_steps(self, quadratic_rnn):
        opt = optimizer.HybridRNNOptimizer(quadratic_rnn, lr=0.01, adam_lr=0.001)

        for _ in range(5):
            opt.zero_grad()
            x = torch.randn(4, 3, quadratic_rnn.group_size)
            y = quadratic_rnn(x)
            loss = y.sum()
            loss.backward()
            opt.step()

        for name, param in quadratic_rnn.named_parameters():
            assert torch.isfinite(param).all(), f"Non-finite values in {name}"
