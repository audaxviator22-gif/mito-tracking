"""Unit tests for mito_tracking.features."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mito_tracking.features import TrackingFeatureExtractor, TrackingFeatures


# Fixtures — reusable test data


@pytest.fixture
def extractor():
    """Default feature extractor with 10 pixels per micrometer."""
    return TrackingFeatureExtractor(pixels_per_micrometer=10.0)


@pytest.fixture
def det_a():
    """A reference detection for tests."""
    return {
        "bbox": [10, 10, 20, 20],
        "center": [20, 20],
        "area_um2": 4.0,
        "mean_intensity": 100.0,
        "aspect_ratio": 1.0,
        "circularity": 0.5,
    }


@pytest.fixture
def det_identical(det_a):
    """A detection identical to det_a (same object, same frame)."""
    return dict(det_a)


@pytest.fixture
def det_nearby(det_a):
    """A detection a few pixels to the right of det_a."""
    return {
        "bbox": [15, 10, 20, 20],
        "center": [25, 20],
        "area_um2": 4.0,
        "mean_intensity": 100.0,
        "aspect_ratio": 1.0,
        "circularity": 0.5,
    }


@pytest.fixture
def det_far(det_a):
    """A detection far from det_a (no overlap, large distance)."""
    return {
        "bbox": [100, 100, 20, 20],
        "center": [110, 110],
        "area_um2": 4.0,
        "mean_intensity": 100.0,
        "aspect_ratio": 1.0,
        "circularity": 0.5,
    }



# Test: TrackingFeatures dataclass


class TestTrackingFeaturesDataclass:
    def test_to_vector_returns_ndarray(self):
        feats = TrackingFeatures(
            position_distance=0.5,
            area_ratio=0.9,
            intensity_similarity=0.8,
            shape_similarity=0.7,
            iou=0.6,
        )
        vec = feats.to_vector()
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (5,)

    def test_to_vector_preserves_order(self):
        feats = TrackingFeatures(
            position_distance=0.1,
            area_ratio=0.2,
            intensity_similarity=0.3,
            shape_similarity=0.4,
            iou=0.5,
        )
        vec = feats.to_vector()
        expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        np.testing.assert_allclose(vec, expected, atol=1e-6)

    def test_to_vector_dtype_is_float32(self):
        feats = TrackingFeatures(0.0, 0.0, 0.0, 0.0, 0.0)
        assert feats.to_vector().dtype == np.float32

    def test_names_constant(self):
        assert TrackingFeatures.NAMES == [
            "position_distance",
            "area_ratio",
            "intensity_similarity",
            "shape_similarity",
            "iou",
        ]



# Test: IoU calculation


class TestIoU:
    def test_identical_boxes_iou_is_one(self, extractor):
        bbox = [0, 0, 10, 10]
        assert extractor._calculate_iou(bbox, bbox) == pytest.approx(1.0)

    def test_non_overlapping_boxes_iou_is_zero(self, extractor):
        a = [0, 0, 10, 10]
        b = [100, 100, 10, 10]
        assert extractor._calculate_iou(a, b) == 0.0

    def test_touching_boxes_iou_is_zero(self, extractor):
        a = [0, 0, 10, 10]
        b = [10, 0, 10, 10]
        assert extractor._calculate_iou(a, b) == 0.0

    def test_half_overlap_iou(self, extractor):
        # Two boxes of 10x10 that overlap on a 5x10 region
        # intersection = 50, union = 100 + 100 - 50 = 150, IoU = 1/3
        a = [0, 0, 10, 10]
        b = [5, 0, 10, 10]
        assert extractor._calculate_iou(a, b) == pytest.approx(1 / 3, abs=1e-6)

    def test_iou_is_symmetric(self, extractor):
        a = [0, 0, 10, 10]
        b = [3, 3, 10, 10]
        assert extractor._calculate_iou(a, b) == extractor._calculate_iou(b, a)

    def test_contained_box_iou(self, extractor):
        # Small box fully inside large box
        # intersection = area of small = 100, union = 400, IoU = 0.25
        big = [0, 0, 20, 20]
        small = [5, 5, 10, 10]
        assert extractor._calculate_iou(big, small) == pytest.approx(0.25)



# Test: center extraction


class TestCenterExtraction:
    def test_uses_explicit_center(self, extractor):
        det = {"bbox": [0, 0, 10, 10], "center": [42, 43]}
        assert extractor._get_center(det) == [42, 43]

    def test_falls_back_to_bbox_center(self, extractor):
        det = {"bbox": [10, 20, 30, 40]}
        # x + w/2 = 10 + 15 = 25; y + h/2 = 20 + 20 = 40
        assert extractor._get_center(det) == [25, 40]

    def test_none_center_falls_back_to_bbox(self, extractor):
        det = {"bbox": [10, 20, 30, 40], "center": None}
        assert extractor._get_center(det) == [25, 40]



# Test: full feature extraction


class TestComputeFeatures:
    def test_identical_detections(self, extractor, det_a):
        """Same detection twice should give perfect scores."""
        feats = extractor.compute_features(det_a, det_a)
        assert feats.iou == pytest.approx(1.0)
        assert feats.area_ratio == pytest.approx(1.0)
        assert feats.intensity_similarity == pytest.approx(1.0)
        assert feats.shape_similarity == pytest.approx(1.0)
        assert feats.position_distance == pytest.approx(0.0)

    def test_far_detections_have_zero_iou(self, extractor, det_a, det_far):
        feats = extractor.compute_features(det_a, det_far)
        assert feats.iou == 0.0

    def test_nearby_detections_have_positive_iou(self, extractor, det_a, det_nearby):
        feats = extractor.compute_features(det_a, det_nearby)
        # Overlap region: 15 pixels wide? Bboxes are [10,10,20,20] and [15,10,20,20]
        # Intersection: x from 15 to 30 = 15 wide, y from 10 to 30 = 20 tall → 300
        # Union: 400 + 400 - 300 = 500 → IoU = 0.6
        assert feats.iou == pytest.approx(0.6, abs=1e-6)

    def test_all_features_are_finite(self, extractor, det_a, det_nearby):
        feats = extractor.compute_features(det_a, det_nearby)
        for name in TrackingFeatures.NAMES:
            value = getattr(feats, name)
            assert math.isfinite(value), f"{name} is not finite: {value}"

    def test_all_features_are_non_negative(self, extractor, det_a, det_nearby):
        feats = extractor.compute_features(det_a, det_nearby)
        for name in TrackingFeatures.NAMES:
            value = getattr(feats, name)
            assert value >= 0, f"{name} is negative: {value}"

    def test_returns_tracking_features_instance(self, extractor, det_a, det_nearby):
        feats = extractor.compute_features(det_a, det_nearby)
        assert isinstance(feats, TrackingFeatures)


# Test: position distance normalization


class TestPositionDistance:
    def test_position_distance_scales_with_ppm(self, det_a, det_nearby):
        """Larger pixels_per_micrometer should give smaller normalized distance."""
        ex_low = TrackingFeatureExtractor(pixels_per_micrometer=1.0)
        ex_high = TrackingFeatureExtractor(pixels_per_micrometer=100.0)

        d_low = ex_low.compute_features(det_a, det_nearby).position_distance
        d_high = ex_high.compute_features(det_a, det_nearby).position_distance

        # Larger ppm → same pixel distance is fewer micrometers → smaller cost
        assert d_high < d_low

    def test_position_distance_capped_at_two(self, det_a, det_far):
        """Distance is capped at 2.0 to keep the feature bounded."""
        ex = TrackingFeatureExtractor(pixels_per_micrometer=1.0)
        feats = ex.compute_features(det_a, det_far)
        assert feats.position_distance == 2.0

    def test_identical_centers_give_zero_distance(self, extractor, det_a):
        feats = extractor.compute_features(det_a, det_a)
        assert feats.position_distance == pytest.approx(0.0)



# Test: area ratio


class TestAreaRatio:
    def test_equal_areas_give_ratio_one(self, extractor, det_a):
        feats = extractor.compute_features(det_a, det_a)
        assert feats.area_ratio == pytest.approx(1.0)

    def test_double_area_gives_ratio_half(self, extractor, det_a):
        det_big = dict(det_a)
        det_big["area_um2"] = 8.0  # double the area
        feats = extractor.compute_features(det_a, det_big)
        assert feats.area_ratio == pytest.approx(0.5)

    def test_area_ratio_is_symmetric(self, extractor, det_a):
        det_big = dict(det_a)
        det_big["area_um2"] = 8.0
        f1 = extractor.compute_features(det_a, det_big)
        f2 = extractor.compute_features(det_big, det_a)
        assert f1.area_ratio == pytest.approx(f2.area_ratio)

    def test_zero_area_does_not_crash(self, extractor):
        a = {"bbox": [0, 0, 10, 10], "area_um2": 0.0, "mean_intensity": 50,
             "aspect_ratio": 1.0, "center": [5, 5]}
        b = {"bbox": [0, 0, 10, 10], "area_um2": 0.0, "mean_intensity": 50,
             "aspect_ratio": 1.0, "center": [5, 5]}
        feats = extractor.compute_features(a, b)
        assert feats.area_ratio == 0.0


# Test: intensity similarity


class TestIntensitySimilarity:
    def test_equal_intensities_give_one(self, extractor, det_a):
        feats = extractor.compute_features(det_a, det_a)
        assert feats.intensity_similarity == pytest.approx(1.0)

    def test_different_intensities_reduce_similarity(self, extractor, det_a):
        det_dim = dict(det_a)
        det_dim["mean_intensity"] = 50.0
        feats = extractor.compute_features(det_a, det_dim)
        # 1 - |100-50|/100 = 0.5
        assert feats.intensity_similarity == pytest.approx(0.5)

    def test_missing_intensity_defaults_to_50(self, extractor):
        a = {"bbox": [0, 0, 10, 10], "center": [5, 5], "area_um2": 1.0,
             "aspect_ratio": 1.0}
        b = {"bbox": [0, 0, 10, 10], "center": [5, 5], "area_um2": 1.0,
             "aspect_ratio": 1.0}
        feats = extractor.compute_features(a, b)
        # Both default to 50, so similarity = 1.0
        assert feats.intensity_similarity == pytest.approx(1.0)



# Test: shape similarity


class TestShapeSimilarity:
    def test_equal_aspect_ratios_give_one(self, extractor, det_a):
        feats = extractor.compute_features(det_a, det_a)
        assert feats.shape_similarity == pytest.approx(1.0)

    def test_elongated_vs_round_reduces_similarity(self, extractor, det_a):
        det_elongated = dict(det_a)
        det_elongated["aspect_ratio"] = 4.0
        feats = extractor.compute_features(det_a, det_elongated)
        # 1 - |1 - 4|/4 = 1 - 0.75 = 0.25
        assert feats.shape_similarity == pytest.approx(0.25)

    def test_missing_aspect_ratio_uses_bbox(self, extractor):
        a = {"bbox": [0, 0, 20, 10], "center": [10, 5], "area_um2": 2.0,
             "mean_intensity": 50}
        b = {"bbox": [0, 0, 20, 10], "center": [10, 5], "area_um2": 2.0,
             "mean_intensity": 50}
        feats = extractor.compute_features(a, b)
        # Both have ar = 20/10 = 2, similarity = 1.0
        assert feats.shape_similarity == pytest.approx(1.0)



# Test: feature vector


class TestFeatureVector:
    def test_feature_vector_shape_and_dtype(self, extractor, det_a, det_nearby):
        feats = extractor.compute_features(det_a, det_nearby)
        vec = extractor.feature_vector(feats)
        assert vec.shape == (5,)
        assert vec.dtype == np.float32

    def test_feature_vector_matches_dataclass_to_vector(self, extractor, det_a, det_nearby):
        feats = extractor.compute_features(det_a, det_nearby)
        v1 = extractor.feature_vector(feats)
        v2 = feats.to_vector()
        np.testing.assert_array_equal(v1, v2)



# Integration tests


class TestIntegration:
    def test_symmetry_of_feature_vector(self, extractor, det_a, det_nearby):
        """Most features should be symmetric (order of arguments doesn't matter)."""
        f_ab = extractor.compute_features(det_a, det_nearby)
        f_ba = extractor.compute_features(det_nearby, det_a)

        assert f_ab.iou == pytest.approx(f_ba.iou)
        assert f_ab.area_ratio == pytest.approx(f_ba.area_ratio)
        assert f_ab.intensity_similarity == pytest.approx(f_ba.intensity_similarity)
        assert f_ab.shape_similarity == pytest.approx(f_ba.shape_similarity)
        assert f_ab.position_distance == pytest.approx(f_ba.position_distance)

    def test_more_similar_detections_have_higher_iou(self, extractor, det_a, det_nearby, det_far):
        """A closer detection should have higher IoU than a distant one."""
        near_iou = extractor.compute_features(det_a, det_nearby).iou
        far_iou = extractor.compute_features(det_a, det_far).iou
        assert near_iou > far_iou
