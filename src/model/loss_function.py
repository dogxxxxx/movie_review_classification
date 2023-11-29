import numpy as np


def binary_loss(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    weight: np.ndarray,
    regularisation: str = "l2",
    alpha: float = 0.00001,
    epsilon: float = 1e-15
):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = np.mean(-1 * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
    if regularisation == "l2":
        reg = np.dot(weight, weight.T)
    loss += alpha * reg
    return loss.item()


if __name__ == "__main__":
    pass
