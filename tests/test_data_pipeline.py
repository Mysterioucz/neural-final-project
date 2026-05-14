"""
Integration tests for the data preprocessing pipeline.

Verifies:
- Feature matrix X has shape (569, 30).
- Labels are correctly mapped to {1.0, -1.0}.
- Features are standardized (mean ~0, std ~1).
"""

import numpy as np
import pytest

from src.data_loader import load_and_clean_data
from src.label_encoder import encode_labels
from src.preprocessing import ManualStandardScaler
from src.data_utils import train_test_split


DATA_PATH = "data/data.csv"


@pytest.fixture(scope="module")
def raw_data():
    """Load and return the raw X, y from the dataset."""
    return load_and_clean_data(DATA_PATH)


@pytest.fixture(scope="module")
def pipeline_data(raw_data):
    """Run the full preprocessing pipeline and return (X_scaled, y_encoded)."""
    X, y = raw_data
    y_encoded = encode_labels(y)
    scaler = ManualStandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y_encoded


class TestDataLoader:
    """Tests for load_and_clean_data."""

    def test_feature_matrix_shape(self, raw_data):
        """X must have 569 samples and 30 features."""
        X, _ = raw_data
        assert X.shape == (569, 30), f"Expected (569, 30), got {X.shape}"

    def test_id_column_removed(self, raw_data):
        """The 'id' column must be dropped."""
        X, _ = raw_data
        assert "id" not in X.columns

    def test_unnamed_32_column_removed(self, raw_data):
        """The 'Unnamed: 32' artifact column must be dropped."""
        X, _ = raw_data
        assert "Unnamed: 32" not in X.columns

    def test_diagnosis_not_in_features(self, raw_data):
        """The 'diagnosis' target column must not be in X."""
        X, _ = raw_data
        assert "diagnosis" not in X.columns

    def test_y_has_correct_length(self, raw_data):
        """Target y must have 569 entries."""
        _, y = raw_data
        assert len(y) == 569


class TestLabelEncoder:
    """Tests for encode_labels."""

    def test_labels_are_binary(self, pipeline_data):
        """Encoded labels must contain only 1.0 and -1.0."""
        _, y_encoded = pipeline_data
        unique = set(np.unique(y_encoded))
        assert unique == {1.0, -1.0}, f"Expected {{1.0, -1.0}}, got {unique}"

    def test_output_dtype_is_float64(self, pipeline_data):
        """Encoded label array must be float64."""
        _, y_encoded = pipeline_data
        assert y_encoded.dtype == np.float64

    def test_label_length_matches_dataset(self, pipeline_data):
        """Encoded label array must have 569 entries."""
        _, y_encoded = pipeline_data
        assert len(y_encoded) == 569

    def test_malignant_encoded_as_positive(self):
        """'M' must map to 1.0."""
        result = encode_labels(np.array(["M"]))
        assert result[0] == 1.0

    def test_benign_encoded_as_negative(self):
        """'B' must map to -1.0."""
        result = encode_labels(np.array(["B"]))
        assert result[0] == -1.0

    def test_encode_labels_basic(self):
        """encode_labels(['M', 'B']) must return [1., -1.]."""
        result = encode_labels(np.array(["M", "B"]))
        np.testing.assert_array_equal(result, np.array([1.0, -1.0]))


class TestManualStandardScaler:
    """Tests for ManualStandardScaler."""

    def test_scaled_features_mean_near_zero(self, pipeline_data):
        """After standardization, each feature must have mean approximately 0."""
        X_scaled, _ = pipeline_data
        col_means = np.mean(X_scaled, axis=0)
        np.testing.assert_allclose(
            col_means, np.zeros(X_scaled.shape[1]), atol=1e-6,
            err_msg="Scaled feature means are not approximately zero."
        )

    def test_scaled_features_std_near_one(self, pipeline_data):
        """After standardization, each feature must have std approximately 1."""
        X_scaled, _ = pipeline_data
        col_stds = np.std(X_scaled, axis=0)
        np.testing.assert_allclose(
            col_stds, np.ones(X_scaled.shape[1]), atol=1e-4,
            err_msg="Scaled feature std values are not approximately one."
        )

    def test_scaled_output_shape(self, pipeline_data):
        """Scaled feature matrix must retain the (569, 30) shape."""
        X_scaled, _ = pipeline_data
        assert X_scaled.shape == (569, 30)

    def test_fit_stores_mean_and_std(self):
        """fit() must populate mean_ and std_ attributes."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = ManualStandardScaler()
        scaler.fit(X)
        assert scaler.mean_ is not None
        assert scaler.std_ is not None
        np.testing.assert_allclose(scaler.mean_, [3.0, 4.0])

    def test_transform_before_fit_raises(self):
        """transform() called before fit() must raise RuntimeError."""
        scaler = ManualStandardScaler()
        with pytest.raises(RuntimeError):
            scaler.transform(np.array([[1.0, 2.0]]))

    def test_fit_transform_matches_fit_then_transform(self):
        """fit_transform(X) must equal fit(X).transform(X)."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler1 = ManualStandardScaler()
        result1 = scaler1.fit_transform(X)
        scaler2 = ManualStandardScaler()
        scaler2.fit(X)
        result2 = scaler2.transform(X)
        np.testing.assert_array_equal(result1, result2)


class TestTrainTestSplit:
    """Tests for train_test_split."""

    def test_split_sizes_default(self):
        """Default 20% test split on 100 samples must give (80, 20)."""
        X = np.zeros((100, 5))
        y = np.zeros(100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20

    def test_split_shapes_on_actual_data(self, raw_data):
        """Split on (569, 30) must give correct train/test sizes."""
        X, y = raw_data
        y_enc = encode_labels(y)
        scaler = ManualStandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42)
        assert X_train.shape[0] + X_test.shape[0] == 569
        assert X_train.shape[1] == 30
        assert X_test.shape[1] == 30

    def test_reproducible_with_random_state(self):
        """Same random_state must produce identical splits."""
        X = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        X_train1, X_test1, _, _ = train_test_split(X, y, random_state=0)
        X_train2, X_test2, _, _ = train_test_split(X, y, random_state=0)
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)

    def test_no_overlap_between_train_and_test(self):
        """Train and test indices must not overlap."""
        X = np.arange(50).reshape(50, 1)
        y = np.arange(50)
        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=7)
        train_vals = set(X_train.flatten())
        test_vals = set(X_test.flatten())
        assert train_vals.isdisjoint(test_vals)

    def test_invalid_test_size_raises(self):
        """test_size outside (0, 1) must raise ValueError."""
        X, y = np.zeros((10, 2)), np.zeros(10)
        with pytest.raises(ValueError):
            train_test_split(X, y, test_size=1.5)
        with pytest.raises(ValueError):
            train_test_split(X, y, test_size=0.0)
