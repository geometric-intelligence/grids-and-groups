"""Tests for src.dataset module."""

import numpy as np
import pytest
import torch

import src.dataset as dataset


class TestGroupCompositionDatasetOfflineCn:
    """Tests for GroupCompositionDataset with group_name='cn' (offline)."""

    def test_sampled_shapes(self):
        group_size = 7
        template = np.random.randn(group_size).astype(np.float32)
        k = 3
        num_samples = 100

        ds = dataset.GroupCompositionDataset(
            "cn",
            group_size=group_size,
            template=template,
            k=k,
            mode="sampled",
            num_samples=num_samples,
        )

        assert len(ds) == num_samples
        assert ds.X.shape == (num_samples, k, group_size)
        assert ds.Y.shape == (num_samples, group_size)
        assert ds.sequence.shape == (num_samples, k)

    def test_exhaustive_shapes(self):
        group_size = 7
        template = np.random.randn(group_size).astype(np.float32)
        k = 2

        ds = dataset.GroupCompositionDataset(
            "cn",
            group_size=group_size,
            template=template,
            k=k,
            mode="exhaustive",
        )

        expected_n = group_size**k
        assert len(ds) == expected_n
        assert ds.X.shape == (expected_n, k, group_size)
        assert ds.Y.shape == (expected_n, group_size)
        assert ds.sequence.shape == (expected_n, k)

    def test_return_all_outputs(self):
        group_size = 7
        template = np.random.randn(group_size).astype(np.float32)
        k = 4
        num_samples = 50

        ds = dataset.GroupCompositionDataset(
            "cn",
            group_size=group_size,
            template=template,
            k=k,
            mode="sampled",
            num_samples=num_samples,
            return_all_outputs=True,
        )

        assert len(ds) == num_samples
        assert ds.X.shape == (num_samples, k, group_size)
        assert ds.Y.shape == (num_samples, k - 1, group_size)
        assert ds.sequence.shape == (num_samples, k)

    def test_rolling_correctness(self):
        group_size = 7
        template = np.random.randn(group_size).astype(np.float32)
        k = 2

        ds = dataset.GroupCompositionDataset(
            "cn",
            group_size=group_size,
            template=template,
            k=k,
            mode="exhaustive",
        )

        shift_0 = int(ds.sequence[0, 0])
        expected_x0 = np.roll(template, shift_0)
        np.testing.assert_allclose(ds.X[0, 0, :].numpy(), expected_x0, rtol=1e-5)

    def test_getitem(self):
        group_size = 7
        template = np.random.randn(group_size).astype(np.float32)
        k = 2

        ds = dataset.GroupCompositionDataset(
            "cn",
            group_size=group_size,
            template=template,
            k=k,
            mode="exhaustive",
        )

        x_i, y_i = ds[0]
        assert x_i.shape == ds.X.shape[1:]
        assert y_i.shape == ds.Y.shape[1:]
        torch.testing.assert_close(x_i, ds.X[0])
        torch.testing.assert_close(y_i, ds.Y[0])


class TestGroupCompositionDatasetOfflineCnxcn:
    """Tests for GroupCompositionDataset with group_name='cnxcn' (offline)."""

    def test_sampled_shapes(self):
        p1, p2 = 5, 5
        template = np.random.randn(p1, p2).astype(np.float32)
        k = 3
        num_samples = 100

        ds = dataset.GroupCompositionDataset(
            "cnxcn",
            p1=p1,
            p2=p2,
            template=template,
            k=k,
            mode="sampled",
            num_samples=num_samples,
        )

        group_size = p1 * p2
        assert len(ds) == num_samples
        assert ds.X.shape == (num_samples, k, group_size)
        assert ds.Y.shape == (num_samples, group_size)
        assert ds.sequence.shape == (num_samples, k, 2)

    def test_exhaustive_shapes(self):
        p1, p2 = 3, 3
        template = np.random.randn(p1, p2).astype(np.float32)
        k = 2

        ds = dataset.GroupCompositionDataset(
            "cnxcn",
            p1=p1,
            p2=p2,
            template=template,
            k=k,
            mode="exhaustive",
        )

        group_size = p1 * p2
        expected_n = group_size**k
        assert len(ds) == expected_n
        assert ds.X.shape == (expected_n, k, group_size)
        assert ds.Y.shape == (expected_n, group_size)
        assert ds.sequence.shape == (expected_n, k, 2)


class TestGroupCompositionDatasetOfflineGroup:
    """Tests for GroupCompositionDataset with generic groups (offline)."""

    @pytest.fixture
    def d3_group(self):
        from src.groups import DihedralGroup

        return DihedralGroup(N=3)

    @pytest.fixture
    def template_d3(self, d3_group):
        group_size = d3_group.order
        return np.random.randn(group_size).astype(np.float32)

    def test_sampled_shapes(self, template_d3, d3_group):
        k = 3
        num_samples = 100
        group_size = len(template_d3)

        ds = dataset.GroupCompositionDataset(
            "dihedral",
            template=template_d3,
            k=k,
            group=d3_group,
            mode="sampled",
            num_samples=num_samples,
        )

        assert len(ds) == num_samples
        assert ds.X.shape == (num_samples, k, group_size)
        assert ds.Y.shape == (num_samples, group_size)
        assert ds.sequence.shape == (num_samples, k)

    def test_exhaustive_shapes(self, template_d3, d3_group):
        k = 2
        group_size = len(template_d3)

        ds = dataset.GroupCompositionDataset(
            "dihedral",
            template=template_d3,
            k=k,
            group=d3_group,
            mode="exhaustive",
        )

        expected_n = group_size**k
        assert len(ds) == expected_n
        assert ds.X.shape == (expected_n, k, group_size)
        assert ds.Y.shape == (expected_n, group_size)
        assert ds.sequence.shape == (expected_n, k)

    def test_return_all_outputs(self, template_d3, d3_group):
        k = 4
        num_samples = 50
        group_size = len(template_d3)

        ds = dataset.GroupCompositionDataset(
            "dihedral",
            template=template_d3,
            k=k,
            group=d3_group,
            mode="sampled",
            num_samples=num_samples,
            return_all_outputs=True,
        )

        assert len(ds) == num_samples
        assert ds.X.shape == (num_samples, k, group_size)
        assert ds.Y.shape == (num_samples, k - 1, group_size)
        assert ds.sequence.shape == (num_samples, k)

    def test_template_mismatch_error(self, d3_group):
        wrong_size = d3_group.order + 1
        template = np.random.randn(wrong_size).astype(np.float32)

        with pytest.raises(AssertionError):
            dataset.GroupCompositionDataset(
                "dihedral",
                template=template,
                k=2,
                group=d3_group,
                mode="exhaustive",
            )

    def test_getitem(self, template_d3, d3_group):
        ds = dataset.GroupCompositionDataset(
            "dihedral",
            template=template_d3,
            k=2,
            group=d3_group,
            mode="exhaustive",
        )

        x_i, y_i = ds[0]
        assert x_i.shape == ds.X.shape[1:]
        assert y_i.shape == ds.Y.shape[1:]
        torch.testing.assert_close(x_i, ds.X[0])
        torch.testing.assert_close(y_i, ds.Y[0])


class TestGroupCompositionDatasetOnline:
    """Tests for GroupCompositionDataset with online=True."""

    def test_online_cn_shapes(self):
        group_size = 7
        k = 3
        batch_size = 16
        template = np.random.randn(group_size).astype(np.float32)

        ds = dataset.GroupCompositionDataset(
            "cn",
            online=True,
            group_size=group_size,
            template=template,
            k=k,
            batch_size=batch_size,
            device="cpu",
        )

        X, Y = next(iter(ds))
        assert X.shape == (batch_size, k, group_size)
        assert Y.shape == (batch_size, group_size)

    def test_online_cn_return_all_outputs(self):
        group_size = 7
        k = 4
        batch_size = 16
        template = np.random.randn(group_size).astype(np.float32)

        ds = dataset.GroupCompositionDataset(
            "cn",
            online=True,
            group_size=group_size,
            template=template,
            k=k,
            batch_size=batch_size,
            device="cpu",
            return_all_outputs=True,
        )

        X, Y = next(iter(ds))
        assert X.shape == (batch_size, k, group_size)
        assert Y.shape == (batch_size, k - 1, group_size)

    def test_online_cnxcn_shapes(self):
        p1, p2 = 5, 5
        k = 3
        batch_size = 16
        template = np.random.randn(p1, p2).astype(np.float32)

        ds = dataset.GroupCompositionDataset(
            "cnxcn",
            online=True,
            p1=p1,
            p2=p2,
            template=template,
            k=k,
            batch_size=batch_size,
            device="cpu",
        )

        group_size = p1 * p2
        X, Y = next(iter(ds))
        assert X.shape == (batch_size, k, group_size)
        assert Y.shape == (batch_size, group_size)

    def test_online_cnxcn_return_all_outputs(self):
        p1, p2 = 5, 5
        k = 4
        batch_size = 16
        template = np.random.randn(p1, p2).astype(np.float32)

        ds = dataset.GroupCompositionDataset(
            "cnxcn",
            online=True,
            p1=p1,
            p2=p2,
            template=template,
            k=k,
            batch_size=batch_size,
            device="cpu",
            return_all_outputs=True,
        )

        group_size = p1 * p2
        X, Y = next(iter(ds))
        assert X.shape == (batch_size, k, group_size)
        assert Y.shape == (batch_size, k - 1, group_size)

    def test_online_unsupported_group_raises(self):
        with pytest.raises(ValueError, match="Online mode only supported"):
            dataset.GroupCompositionDataset(
                "dihedral",
                online=True,
                template=np.zeros(6),
                k=2,
                batch_size=4,
                device="cpu",
            )
