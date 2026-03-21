import numpy as np
import pytest
import nnfs
nnfs.init()

# --- Classes under test (extracted from test_check2.ipynb) ---

class layer_dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class activation_relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class activation_softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)


class loss_categorical(loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        return -np.log(correct_confidences)


# --- Tests ---

class TestLayerDense:
    def test_output_shape(self):
        layer = layer_dense(4, 5)
        X = np.random.randn(10, 4)
        layer.forward(X)
        assert layer.output.shape == (10, 5)

    def test_weights_shape(self):
        layer = layer_dense(3, 7)
        assert layer.weights.shape == (3, 7)

    def test_biases_initialized_to_zero(self):
        layer = layer_dense(3, 5)
        assert np.all(layer.biases == 0)

    def test_forward_uses_biases(self):
        layer = layer_dense(2, 2)
        layer.weights = np.zeros((2, 2))
        layer.biases = np.array([[1.0, 2.0]])
        X = np.ones((3, 2))
        layer.forward(X)
        expected = np.tile([1.0, 2.0], (3, 1))
        np.testing.assert_array_almost_equal(layer.output, expected)


class TestActivationReLU:
    def test_zeros_negative(self):
        relu = activation_relu()
        X = np.array([[-1.0, -2.0], [3.0, 0.0]])
        relu.forward(X)
        expected = np.array([[0.0, 0.0], [3.0, 0.0]])
        np.testing.assert_array_equal(relu.output, expected)

    def test_positive_unchanged(self):
        relu = activation_relu()
        X = np.array([[1.0, 2.0, 3.0]])
        relu.forward(X)
        np.testing.assert_array_equal(relu.output, X)

    def test_output_shape(self):
        relu = activation_relu()
        X = np.random.randn(5, 4)
        relu.forward(X)
        assert relu.output.shape == (5, 4)


class TestActivationSoftmax:
    def test_output_sums_to_one(self):
        softmax = activation_softmax()
        X = np.random.randn(10, 3)
        softmax.forward(X)
        row_sums = np.sum(softmax.output, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(10))

    def test_output_all_positive(self):
        softmax = activation_softmax()
        X = np.random.randn(5, 4)
        softmax.forward(X)
        assert np.all(softmax.output > 0)

    def test_output_shape(self):
        softmax = activation_softmax()
        X = np.random.randn(7, 3)
        softmax.forward(X)
        assert softmax.output.shape == (7, 3)

    def test_uniform_input_gives_uniform_output(self):
        softmax = activation_softmax()
        X = np.zeros((1, 3))
        softmax.forward(X)
        np.testing.assert_array_almost_equal(softmax.output, [[1/3, 1/3, 1/3]])


class TestLossCategorical:
    def setup_method(self):
        self.loss_fn = loss_categorical()

    def test_perfect_prediction_sparse(self):
        # Predicted probability 1.0 for the correct class → loss ≈ 0
        y_pred = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        y_true = np.array([0, 1])
        result = self.loss_fn.calculate(y_pred, y_true)
        assert result < 1e-5

    def test_uniform_prediction_loss(self):
        # Uniform softmax output for 3 classes → loss ≈ ln(3) ≈ 1.0986
        y_pred = np.full((100, 3), 1/3)
        y_true = np.zeros(100, dtype=int)
        result = self.loss_fn.calculate(y_pred, y_true)
        assert abs(result - np.log(3)) < 1e-4

    def test_one_hot_encoded_labels(self):
        y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        y_true = np.array([[1, 0, 0], [0, 1, 0]])  # one-hot
        result = self.loss_fn.calculate(y_pred, y_true)
        expected = np.mean([-np.log(0.7), -np.log(0.8)])
        assert abs(result - expected) < 1e-6

    def test_output_is_scalar(self):
        y_pred = np.array([[0.5, 0.3, 0.2]])
        y_true = np.array([0])
        result = self.loss_fn.calculate(y_pred, y_true)
        assert np.isscalar(result) or result.ndim == 0

    def test_clipping_prevents_log_zero(self):
        y_pred = np.array([[0.0, 1.0, 0.0]])
        y_true = np.array([0])  # predicted 0.0 for correct class
        result = self.loss_fn.calculate(y_pred, y_true)
        assert np.isfinite(result)


class TestForwardPass:
    """Integration test: full forward pass matches expected loss range."""

    def test_loss_near_ln3_on_init(self):
        from nnfs.datasets import spiral_data
        X, y = spiral_data(samples=100, classes=3)
        dense1 = layer_dense(2, 3)
        act1 = activation_relu()
        dense2 = layer_dense(3, 3)
        act2 = activation_softmax()
        loss_fn = loss_categorical()

        dense1.forward(X)
        act1.forward(dense1.output)
        dense2.forward(act1.output)
        act2.forward(dense2.output)
        result = loss_fn.calculate(act2.output, y)

        # Untrained network should produce loss close to ln(3) ≈ 1.0986
        assert abs(result - np.log(3)) < 0.1
