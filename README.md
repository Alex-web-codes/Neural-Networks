# Neural Networks

A from-scratch implementation of neural network components in Python, built while following the *Neural Networks from Scratch* (NNFS) series.

## Contents

| File | Description |
|---|---|
| `test_check2.ipynb` | Jupyter notebook with layer, activation, and loss implementations + spiral dataset demo |
| `test_check2.py` | Pytest test suite for all components |

## Components Implemented

- **`layer_dense`** — fully connected layer with weight/bias initialization and forward pass
- **`activation_relu`** — ReLU activation function
- **`activation_softmax`** — Softmax activation with numerical stability
- **`loss_categorical`** — Categorical cross-entropy loss supporting both sparse and one-hot labels

## Getting Started

### Install dependencies

```bash
pip install numpy nnfs pytest
```

### Run the notebook

Open `test_check2.ipynb` in Jupyter to see the forward pass and loss calculation on a spiral dataset.

### Run the tests

```bash
python -m pytest test_check2.py -v
```

## Dependencies

- [NumPy](https://numpy.org/)
- [nnfs](https://github.com/Sentdex/nnfs) — dataset helpers and reproducibility utilities
