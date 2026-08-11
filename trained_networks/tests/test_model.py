"""Tests for trained_networks.model (QuadraticRNN, TwoLayerMLP)."""

import pytest
import torch

import trained_networks.model as model


class TestQuadraticRNN:
    """Tests for model.QuadraticRNN."""

    @pytest.fixture
    def default_params(self):
        return {"group_size": 7, "hidden_dim": 10, "k": 4}

    def test_output_shape_basic(self, default_params):
        net = model.QuadraticRNN(**default_params)
        batch_size = 8
        k = default_params["k"]
        group_size = default_params["group_size"]

        x = torch.randn(batch_size, k, group_size)
        y = net(x)

        assert y.shape == (batch_size, group_size)

    def test_output_shape_return_all_outputs(self, default_params):
        params = {**default_params, "return_all_outputs": True}
        net = model.QuadraticRNN(**params)
        batch_size = 8
        k = default_params["k"]
        group_size = default_params["group_size"]

        x = torch.randn(batch_size, k, group_size)
        y = net(x)

        expected_shape = (batch_size, k - 1, group_size)
        assert y.shape == expected_shape

    def test_output_shape_k_equals_2(self, default_params):
        params = {**default_params, "k": 2}
        net = model.QuadraticRNN(**params)
        batch_size = 4
        group_size = default_params["group_size"]

        x = torch.randn(batch_size, 2, group_size)
        y = net(x)

        assert y.shape == (batch_size, group_size)

    def test_quadratic_transform(self, default_params):
        net = model.QuadraticRNN(**default_params)
        batch_size = 2
        k = default_params["k"]
        group_size = default_params["group_size"]

        x = torch.randn(batch_size, k, group_size)
        y = net(x)

        assert torch.isfinite(y).all()

    def test_minimum_sequence_length_error(self, default_params):
        net = model.QuadraticRNN(**default_params)
        x = torch.randn(2, 1, default_params["group_size"])

        with pytest.raises(AssertionError, match="Sequence length must be at least 2"):
            net(x)

    def test_gradient_flow(self, default_params):
        net = model.QuadraticRNN(**default_params)
        k = default_params["k"]
        group_size = default_params["group_size"]

        x = torch.randn(4, k, group_size, requires_grad=True)
        y = net(x)
        loss = y.sum()
        loss.backward()

        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_stores_k_and_group_size(self, default_params):
        net = model.QuadraticRNN(**default_params)
        assert net.k == default_params["k"]
        assert net.group_size == default_params["group_size"]
        assert net.hidden_dim == default_params["hidden_dim"]


class TestTwoLayerMLP:
    """Tests for model.TwoLayerMLP."""

    @pytest.fixture
    def default_params(self):
        return {"group_size": 6, "hidden_dim": 20, "k": 2}

    def test_output_shape(self, default_params):
        net = model.TwoLayerMLP(**default_params)
        batch_size = 8
        k = default_params["k"]
        group_size = default_params["group_size"]

        x = torch.randn(batch_size, k, group_size)
        y = net(x)

        assert y.shape == (batch_size, group_size)

    def test_output_shape_k3_power(self):
        net = model.TwoLayerMLP(group_size=5, hidden_dim=8, k=3, nonlinearity="power")
        x = torch.randn(4, 3, 5)
        y = net(x)
        assert y.shape == (4, 5)

    def test_k_mismatch_error(self, default_params):
        net = model.TwoLayerMLP(**default_params)
        wrong_k = default_params["k"] + 1
        x = torch.randn(2, wrong_k, default_params["group_size"])

        with pytest.raises(AssertionError, match="Expected k="):
            net(x)

    def test_different_k_values(self):
        group_size = 5
        hidden_dim = 8

        for k in [2, 3, 4, 5]:
            net = model.TwoLayerMLP(
                group_size=group_size, hidden_dim=hidden_dim, k=k, nonlinearity="power"
            )
            x = torch.randn(4, k, group_size)
            y = net(x)
            assert y.shape == (4, group_size), f"Failed for k={k}"

    def test_square_nonlinearity(self, default_params):
        params = {**default_params, "nonlinearity": "square"}
        net = model.TwoLayerMLP(**params)
        x = torch.randn(4, default_params["k"], default_params["group_size"])
        y = net(x)
        assert torch.isfinite(y).all()

    def test_relu_nonlinearity(self, default_params):
        params = {**default_params, "nonlinearity": "relu"}
        net = model.TwoLayerMLP(**params)
        x = torch.randn(4, default_params["k"], default_params["group_size"])
        y = net(x)
        assert torch.isfinite(y).all()

    def test_tanh_nonlinearity(self, default_params):
        params = {**default_params, "nonlinearity": "tanh"}
        net = model.TwoLayerMLP(**params)
        x = torch.randn(4, default_params["k"], default_params["group_size"])
        y = net(x)
        assert torch.isfinite(y).all()

    def test_gelu_nonlinearity(self, default_params):
        params = {**default_params, "nonlinearity": "gelu"}
        net = model.TwoLayerMLP(**params)
        x = torch.randn(4, default_params["k"], default_params["group_size"])
        y = net(x)
        assert torch.isfinite(y).all()

    def test_linear_nonlinearity(self, default_params):
        params = {**default_params, "nonlinearity": "linear"}
        net = model.TwoLayerMLP(**params)
        x = torch.randn(4, default_params["k"], default_params["group_size"])
        y = net(x)
        assert torch.isfinite(y).all()

    def test_power_nonlinearity(self):
        net = model.TwoLayerMLP(group_size=5, hidden_dim=8, k=3, nonlinearity="power")
        x = torch.randn(4, 3, 5) * 0.1
        y = net(x)
        assert torch.isfinite(y).all()

    def test_invalid_nonlinearity(self, default_params):
        params = {**default_params, "nonlinearity": "invalid"}
        net = model.TwoLayerMLP(**params)
        x = torch.randn(4, default_params["k"], default_params["group_size"])

        with pytest.raises(ValueError, match="Invalid nonlinearity"):
            net(x)

    def test_gradient_flow(self, default_params):
        net = model.TwoLayerMLP(**default_params)
        x = torch.randn(4, default_params["k"], default_params["group_size"], requires_grad=True)
        y = net(x)
        loss = y.sum()
        loss.backward()

        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_default_hidden_dim(self):
        group_size = 8
        net = model.TwoLayerMLP(group_size=group_size, k=2)
        assert net.hidden_dim == 50 * group_size

    def test_output_scale(self, default_params):
        scale_small = 0.1
        scale_large = 10.0

        torch.manual_seed(42)
        net_small = model.TwoLayerMLP(**default_params, output_scale=scale_small)
        torch.manual_seed(42)
        net_large = model.TwoLayerMLP(**default_params, output_scale=scale_large)

        x = torch.randn(4, default_params["k"], default_params["group_size"])
        y_small = net_small(x)
        y_large = net_large(x)

        assert y_large.abs().mean() > y_small.abs().mean()

    def test_stores_k_and_group_size(self, default_params):
        net = model.TwoLayerMLP(**default_params)
        assert net.k == default_params["k"]
        assert net.group_size == default_params["group_size"]
        assert net.hidden_dim == default_params["hidden_dim"]
