import numpy as np

from model.optim import predict_class
from model.optim import SGD


class LogisticRegression:
    def __init__(self):
        pass

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_dev: np.ndarray = np.array([]),
        y_dev: np.ndarray = np.array([]),
        lr: float = 0.01,
        batch_size: int = 32,
        epochs: int = 100,
        alpha: float = 0.00001,
        tolerance: float = 0.0001,
    ):
        self.weight = np.random.uniform(low=-1, high=1, size=[1, x_train.shape[1]])
        self.weight, training_loss_history, validation_loss_history = SGD(
            x_train,
            y_train,
            self.weight,
            x_dev,
            y_dev,
            lr,
            batch_size,
            alpha,
            epochs,
            tolerance,
        )


    def predict(self, x_test: np.ndarray):
        y_pred = predict_class(x_test, self.weight)
        return y_pred

if __name__ == "__main__":
    x = np.array([[0.2, 0.02, 0.01], [0.7, 0.2, 0.1]])
    w = np.random.uniform(low=-1, high=1, size=[1, x.shape[1]])
    y_true = np.array([1, 0])
    print("w: ", w)
    logistic_regression = LogisticRegression()
    pred_class = logistic_regression.predict_class(x, w)
    print("pred_class: ", pred_class)
    loss = logistic_regression.l2_loss(y_true, w)
    print("loss: ", loss)
