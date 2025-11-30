import numpy as np


class LogisticRegressionNumpy:
    def __init__(self, lr=0.1, n_iters=1000, verbose=False):
        self.lr = lr
        self.n_iters = n_iters
        self.verbose = verbose
        self.W = None
        self.b = None

    @staticmethod
    def sigmoid(z):
        # ổn định số học tránh overflow
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X = X.astype(float)
        y = y.astype(float).reshape(-1, 1)

        self.W = np.zeros((n_features, 1))
        self.b = 0.0

        for i in range(self.n_iters):
            # dự đoán
            z = X @ self.W + self.b  # (n,1)
            y_hat = self.sigmoid(z)

            # hàm mất mát: binary cross-entropy
            eps = 1e-10
            loss = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

            # gradient
            dz = y_hat - y  # (n,1)
            dW = (X.T @ dz) / n_samples
            db = np.mean(dz)

            # cập nhật
            self.W -= self.lr * dW
            self.b -= self.lr * db

            if self.verbose and i % 100 == 0:
                print(f"Iter {i}: loss = {loss:.4f}")

    def predict_proba(self, X):
        z = X @ self.W + self.b
        return self.sigmoid(z).ravel()

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


def confusion_matrix(y_true, y_pred):
    """
    Trả về ma trận 2x2: [[TN, FP],[FN, TP]]
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    return np.array([[TN, FP],
                     [FN, TP]])


def accuracy_score(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    return np.mean(y_true == y_pred)


def precision_recall_f1(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return precision, recall, f1


def k_fold_indices(n_samples: int, k: int, shuffle=True, random_state=42):
    """
    Sinh ra index cho K-Fold cross validation.
    """
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[: n_samples % k] += 1

    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_idx, test_idx))
        current = stop

    return folds


def cross_validate_logreg_numpy(X, y, k=5, lr=0.1, n_iters=1000):
    """
    K-fold cross validation cho LogisticRegressionNumpy.
    """
    folds = k_fold_indices(len(y), k)
    accs, pres, recs, f1s = [], [], [], []

    for train_idx, val_idx in folds:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = LogisticRegressionNumpy(lr=lr, n_iters=n_iters)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        p, r, f1 = precision_recall_f1(y_val, y_pred)

        accs.append(acc)
        pres.append(p)
        recs.append(r)
        f1s.append(f1)

    return {
        "accuracy": np.mean(accs),
        "precision": np.mean(pres),
        "recall": np.mean(recs),
        "f1": np.mean(f1s),
    }


# ============ model dùng scikit-learn để so sánh ============

def sklearn_logreg_train_eval(X_train, y_train, X_test, y_test, C=1.0, max_iter=1000):
    """
    Huấn luyện LogisticRegression của scikit-learn và trả về kết quả + dự đoán.
    """
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=C, max_iter=max_iter)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    p, r, f1 = precision_recall_f1(y_test, y_pred)

    return {
        "model": clf,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
    }
