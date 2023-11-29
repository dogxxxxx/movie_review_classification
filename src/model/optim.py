import numpy as np

from model.activation import sigmoid
from model.loss_function import binary_loss


def predict_proba(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    z = np.dot(x, weight.T)
    probability = sigmoid(z)
    return probability


def predict_class(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    predict_probability = predict_proba(x, weight)
    return np.round(predict_probability)


def SGD(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weight: np.ndarray,
    x_dev: np.ndarray = np.array([]),
    y_dev: np.ndarray = np.array([]),
    lr: float = 0.01,
    batch_size: int = 32,
    alpha: float = 0.00001,
    epochs: int = 5,
    tolerance: float = 0.0001,
    print_progress: bool = True,
):
    training_loss_history = []
    validation_loss_history = []

    for i in range(epochs):
        p = np.random.permutation(x_train.shape[0])
        x_train = x_train[p]
        y_train = y_train[p]
        for j in range(0, x_train.shape[0], batch_size):
            x_batch = x_train[j : j + batch_size]
            y_batch = y_train[j : j + batch_size]
            gradient = (
                np.mean(x_batch * (sigmoid(np.dot(x_batch, weight.T)) - y_batch), axis=0) + 2 * alpha * weight
                )
            weight = weight - lr * gradient

        y_prob = predict_proba(x_train, weight)
        train_loss = binary_loss(y_prob, y_train, weight)
        training_loss_history.append(train_loss)

        if print_progress and (i + 1) % 1 == 0:
            print(f"Epoch {i + 1}/{epochs} - Training Loss: {train_loss:.4f}    ", end="")

        if x_dev.size > 0 and y_dev.size > 0:
            dev_prob = predict_proba(x_dev, weight)
            validation_loss = binary_loss(dev_prob, y_dev, weight)
            validation_loss_history.append(validation_loss)
            if print_progress and (i + 1) % 1 == 0:
                print(f"Validation Loss: {validation_loss:.4f}", end="")
        if (i + 1) % 1 == 0:
            print()
        if i == 0:
                pass
        elif abs(validation_loss - validation_loss_history[-2]) < tolerance:
            print(f"Early stopping at epoch: {i + 1}")
            break

    return weight, training_loss_history, validation_loss_history


if __name__ == "__main__":
    pass
