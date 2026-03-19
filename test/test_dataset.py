"""Tests for src.dataset module."""

import numpy as np
import pytest
import torch

import src.dataset as dataset


class TestBuildModularAdditionSequenceDataset1D:
    """Tests for OnlineModularAdditionDataset1D.generate_dataset."""

    @pytest.fixture
    def template_1d(self):
        """Create a simple 1D template."""
        p = 7
        template = np.random.randn(p).astype(np.float32)
        return template

    def test_output_shape_sampled(self, template_1d):
        """Test output shapes in sampled mode."""
        p = len(template_1d)
        k = 3
        num_samples = 100

        X, Y, sequence = dataset.OnlineModularAdditionDataset1D.generate_dataset(
            p=p, template=template_1d, k=k, mode="sampled", num_samples=num_samples
        )

        assert X.shape == (num_samples, k, p), f"X shape mismatch: {X.shape}"
        assert Y.shape == (num_samples, p), f"Y shape mismatch: {Y.shape}"
        assert sequence.shape == (num_samples, k), f"sequence shape mismatch: {sequence.shape}"

    def test_output_shape_exhaustive(self, template_1d):
        """Test output shapes in exhaustive mode."""
        p = len(template_1d)
        k = 2

        X, Y, sequence = dataset.OnlineModularAdditionDataset1D.generate_dataset(
            p=p, template=template_1d, k=k, mode="exhaustive"
        )

        expected_n = p**k
        assert X.shape == (expected_n, k, p)
        assert Y.shape == (expected_n, p)
        assert sequence.shape == (expected_n, k)

    def test_output_shape_return_all_outputs(self, template_1d):
        """Test output shapes with return_all_outputs=True."""
        p = len(template_1d)
        k = 4
        num_samples = 50

        X, Y, sequence = dataset.OnlineModularAdditionDataset1D.generate_dataset(
            p=p,
            template=template_1d,
            k=k,
            mode="sampled",
            num_samples=num_samples,
            return_all_outputs=True,
        )

        # Y should have k-1 outputs (one after each pair of tokens)
        assert X.shape == (num_samples, k, p)
        assert Y.shape == (num_samples, k - 1, p)
        assert sequence.shape == (num_samples, k)

    def test_rolling_correctness(self, template_1d):
        """Test that X values are rolled versions of template."""
        p = len(template_1d)
        k = 2

        X, Y, sequence = dataset.OnlineModularAdditionDataset1D.generate_dataset(
            p=p, template=template_1d, k=k, mode="exhaustive"
        )

        # Check first sample
        shift_0 = int(sequence[0, 0])
        expected_x0 = np.roll(template_1d, shift_0)
        np.testing.assert_allclose(X[0, 0, :], expected_x0, rtol=1e-5)


class TestBuildModularAdditionSequenceDataset2D:
    """Tests for OnlineModularAdditionDataset2D.generate_dataset."""

    @pytest.fixture
    def template_2d(self):
        """Create a simple 2D template."""
        p1, p2 = 5, 5
        template = np.random.randn(p1, p2).astype(np.float32)
        return template

    def test_output_shape_sampled(self, template_2d):
        """Test output shapes in sampled mode."""
        p1, p2 = template_2d.shape
        k = 3
        num_samples = 100

        X, Y, sequence_xy = dataset.OnlineModularAdditionDataset2D.generate_dataset(
            p1=p1, p2=p2, template=template_2d, k=k, mode="sampled", num_samples=num_samples
        )

        p_flat = p1 * p2
        assert X.shape == (num_samples, k, p_flat), f"X shape mismatch: {X.shape}"
        assert Y.shape == (num_samples, p_flat), f"Y shape mismatch: {Y.shape}"
        assert sequence_xy.shape == (
            num_samples,
            k,
            2,
        ), f"sequence_xy shape mismatch: {sequence_xy.shape}"

    def test_output_shape_exhaustive(self, template_2d):
        """Test output shapes in exhaustive mode."""
        p1, p2 = 3, 3  # Use small dimensions for exhaustive
        template = np.random.randn(p1, p2).astype(np.float32)
        k = 2

        X, Y, sequence_xy = dataset.OnlineModularAdditionDataset2D.generate_dataset(
            p1=p1, p2=p2, template=template, k=k, mode="exhaustive"
        )

        expected_n = (p1 * p2) ** k
        p_flat = p1 * p2
        assert X.shape == (expected_n, k, p_flat)
        assert Y.shape == (expected_n, p_flat)
        assert sequence_xy.shape == (expected_n, k, 2)


class TestBuildModularAdditionSequenceDatasetGenericD3:
    """Tests for build_modular_addition_sequence_dataset_generic with DihedralGroup."""

    @pytest.fixture
    def d3_group(self):
        """Create a DihedralGroup(N=3) for testing."""
        from escnn.group import DihedralGroup

        return DihedralGroup(N=3)

    @pytest.fixture
    def template_d3(self, d3_group):
        """Create a template for D3 group (order 6)."""
        group_order = d3_group.order()
        template = np.random.randn(group_order).astype(np.float32)
        return template

    def test_output_shape_sampled(self, template_d3, d3_group):
        """Test output shapes in sampled mode."""
        k = 3
        num_samples = 100
        group_order = len(template_d3)

        X, Y, sequence = dataset.build_modular_addition_sequence_dataset_generic(
            template=template_d3, k=k, group=d3_group, mode="sampled", num_samples=num_samples
        )

        assert X.shape == (num_samples, k, group_order), f"X shape mismatch: {X.shape}"
        assert Y.shape == (num_samples, group_order), f"Y shape mismatch: {Y.shape}"
        assert sequence.shape == (num_samples, k), f"sequence shape mismatch: {sequence.shape}"

    def test_output_shape_exhaustive(self, template_d3, d3_group):
        """Test output shapes in exhaustive mode."""
        k = 2
        group_order = len(template_d3)
        n_elements = group_order

        X, Y, sequence = dataset.build_modular_addition_sequence_dataset_generic(
            template=template_d3, k=k, group=d3_group, mode="exhaustive"
        )

        expected_n = n_elements**k
        assert X.shape == (expected_n, k, group_order)
        assert Y.shape == (expected_n, group_order)
        assert sequence.shape == (expected_n, k)

    def test_output_shape_return_all_outputs(self, template_d3, d3_group):
        """Test output shapes with return_all_outputs=True."""
        k = 4
        num_samples = 50
        group_order = len(template_d3)

        X, Y, sequence = dataset.build_modular_addition_sequence_dataset_generic(
            template=template_d3,
            k=k,
            group=d3_group,
            mode="sampled",
            num_samples=num_samples,
            return_all_outputs=True,
        )

        assert X.shape == (num_samples, k, group_order)
        assert Y.shape == (num_samples, k - 1, group_order)
        assert sequence.shape == (num_samples, k)


class TestOnlineModularAdditionDataset1D:
    """Tests for OnlineModularAdditionDataset1D."""

    def test_batch_shape(self):
        """Test that batches have correct shapes."""
        p = 7
        k = 3
        batch_size = 16
        template = np.random.randn(p).astype(np.float32)

        ds = dataset.OnlineModularAdditionDataset1D(
            p=p, template=template, k=k, batch_size=batch_size, device="cpu"
        )

        # Get first batch
        iterator = iter(ds)
        X, Y = next(iterator)

        assert X.shape == (batch_size, k, p), f"X shape mismatch: {X.shape}"
        assert Y.shape == (batch_size, p), f"Y shape mismatch: {Y.shape}"

    def test_batch_shape_return_all_outputs(self):
        """Test batch shapes with return_all_outputs=True."""
        p = 7
        k = 4
        batch_size = 16
        template = np.random.randn(p).astype(np.float32)

        ds = dataset.OnlineModularAdditionDataset1D(
            p=p,
            template=template,
            k=k,
            batch_size=batch_size,
            device="cpu",
            return_all_outputs=True,
        )

        iterator = iter(ds)
        X, Y = next(iterator)

        assert X.shape == (batch_size, k, p)
        assert Y.shape == (batch_size, k - 1, p)


class TestOnlineModularAdditionDataset2D:
    """Tests for OnlineModularAdditionDataset2D."""

    def test_batch_shape(self):
        """Test that batches have correct shapes."""
        p1, p2 = 5, 5
        k = 3
        batch_size = 16
        template = np.random.randn(p1, p2).astype(np.float32)

        ds = dataset.OnlineModularAdditionDataset2D(
            p1=p1, p2=p2, template=template, k=k, batch_size=batch_size, device="cpu"
        )

        iterator = iter(ds)
        X, Y = next(iterator)

        p_flat = p1 * p2
        assert X.shape == (batch_size, k, p_flat), f"X shape mismatch: {X.shape}"
        assert Y.shape == (batch_size, p_flat), f"Y shape mismatch: {Y.shape}"

    def test_batch_shape_return_all_outputs(self):
        """Test batch shapes with return_all_outputs=True."""
        p1, p2 = 5, 5
        k = 4
        batch_size = 16
        template = np.random.randn(p1, p2).astype(np.float32)

        ds = dataset.OnlineModularAdditionDataset2D(
            p1=p1,
            p2=p2,
            template=template,
            k=k,
            batch_size=batch_size,
            device="cpu",
            return_all_outputs=True,
        )

        iterator = iter(ds)
        X, Y = next(iterator)

        p_flat = p1 * p2
        assert X.shape == (batch_size, k, p_flat)
        assert Y.shape == (batch_size, k - 1, p_flat)


class TestCnDataset:
    """Tests for cn_dataset function."""

    def test_output_shape(self):
        """Test that output shapes are correct."""
        group_size = 7
        template = np.random.randn(group_size)

        X, Y = dataset.cn_dataset(template)

        n_samples = group_size**2
        assert X.shape == (n_samples, 2, group_size), f"X shape mismatch: {X.shape}"
        assert Y.shape == (n_samples, group_size), f"Y shape mismatch: {Y.shape}"

    def test_modular_addition_property(self):
        """Test that Y is the rolled template by (a+b) mod p."""
        group_size = 5
        template = np.arange(group_size).astype(float)

        X, Y = dataset.cn_dataset(template)

        # Check a specific case: a=1, b=2 -> q=(1+2)%5=3
        idx = 1 * group_size + 2
        expected_y = np.roll(template, 3)
        np.testing.assert_allclose(Y[idx], expected_y)

    def test_covers_all_pairs(self):
        """Test that all pairs (a, b) are covered."""
        group_size = 4
        template = np.random.randn(group_size)

        X, Y = dataset.cn_dataset(template)

        assert X.shape[0] == group_size**2


class TestCnxcnDataset:
    """Tests for cnxcn_dataset function."""

    def test_output_shape(self):
        """Test that output shapes are correct."""
        image_length = 4
        template = np.random.randn(image_length * image_length)

        X, Y = dataset.cnxcn_dataset(template)

        n_samples = image_length**4
        n_features = image_length * image_length
        assert X.shape == (n_samples, 2, n_features), f"X shape mismatch: {X.shape}"
        assert Y.shape == (n_samples, n_features), f"Y shape mismatch: {Y.shape}"

    def test_covers_all_combinations(self):
        """Test that all combinations are covered."""
        image_length = 3
        template = np.random.randn(image_length * image_length)

        X, Y = dataset.cnxcn_dataset(template)

        expected_n = image_length**4
        assert X.shape[0] == expected_n


class TestGroupDataset:
    """Tests for group_dataset function."""

    @pytest.fixture
    def dihedral_group(self):
        """Create a DihedralGroup for testing."""
        from escnn.group import DihedralGroup

        return DihedralGroup(N=3)

    def test_output_shape(self, dihedral_group):
        """Test that output shapes are correct for D3."""
        group_order = dihedral_group.order()
        template = np.random.randn(group_order)

        X, Y = dataset.group_dataset(dihedral_group, template)

        n_samples = group_order**2
        assert X.shape == (n_samples, 2, group_order), f"X shape mismatch: {X.shape}"
        assert Y.shape == (n_samples, group_order), f"Y shape mismatch: {Y.shape}"

    def test_template_length_mismatch_error(self, dihedral_group):
        """Test that mismatched template length raises error."""
        wrong_size = dihedral_group.order() + 1
        template = np.random.randn(wrong_size)

        with pytest.raises(AssertionError):
            dataset.group_dataset(dihedral_group, template)


class TestMoveDatasetToDeviceAndFlatten:
    """Tests for move_dataset_to_device_and_flatten function."""

    def test_output_shape_and_type(self):
        """Test that output shapes and types are correct."""
        group_size = 5
        n_samples = 10

        X = np.random.randn(n_samples, 2, group_size)
        Y = np.random.randn(n_samples, group_size)

        X_tensor, Y_tensor, device = dataset.move_dataset_to_device_and_flatten(X, Y, device="cpu")

        assert isinstance(X_tensor, torch.Tensor)
        assert isinstance(Y_tensor, torch.Tensor)
        assert X_tensor.shape == (n_samples, 2 * group_size)
        assert Y_tensor.shape == (n_samples, group_size)

    def test_flattening(self):
        """Test that X is correctly flattened."""
        group_size = 4
        n_samples = 5

        X = np.arange(n_samples * 2 * group_size).reshape(n_samples, 2, group_size).astype(float)
        Y = np.random.randn(n_samples, group_size)

        X_tensor, Y_tensor, device = dataset.move_dataset_to_device_and_flatten(X, Y, device="cpu")

        expected_flat = np.concatenate([X[0, 0, :], X[0, 1, :]])
        np.testing.assert_allclose(X_tensor[0].numpy(), expected_flat)

    def test_device_cpu(self):
        """Test explicit CPU device."""
        X = np.random.randn(5, 2, 4)
        Y = np.random.randn(5, 4)

        X_tensor, Y_tensor, device = dataset.move_dataset_to_device_and_flatten(X, Y, device="cpu")

        assert X_tensor.device.type == "cpu"
        assert Y_tensor.device.type == "cpu"
