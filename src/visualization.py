import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_target_distribution(y):
    vals, counts = np.unique(y, return_counts=True)
    fig, ax = plt.subplots()
    ax.bar(vals, counts)
    ax.set_xticks(vals)
    ax.set_xlabel("target (0: không đổi việc, 1: đổi việc)")
    ax.set_ylabel("Số lượng")
    ax.set_title("Phân bố nhãn target")
    return fig, ax


def plot_categorical_vs_target(cat_col, y, title=""):
    cat = np.array(cat_col)
    y = np.array(y)

    uniq = np.unique(cat)
    rates = []
    for u in uniq:
        mask = cat == u
        if mask.sum() == 0:
            rates.append(0.0)
        else:
            rates.append(y[mask].mean())

    x = np.arange(len(uniq))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, rates)
    ax.set_xticks(x)
    ax.set_xticklabels(uniq, rotation=45, ha="right")
    ax.set_ylabel("Tỷ lệ muốn đổi việc (target=1)")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_numeric_hist(x, bins=30, title="", xlabel=""):
    fig, ax = plt.subplots()
    ax.hist(x, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Tần số")
    return fig, ax


def plot_correlation_heatmap(X, feature_names=None, title="Ma trận tương quan"):
    corr = np.corrcoef(X.T)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, ax=ax, xticklabels=feature_names, yticklabels=feature_names, annot=False)
    ax.set_title(title)
    return fig, ax


def plot_confusion_matrix(cm, labels=("0", "1"), title="Confusion Matrix"):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(title)
    return fig, ax


def plot_roc_curve(y_true, y_score):
    y_true = y_true.astype(int)
    y_score = np.array(y_score)

    # sort theo score giảm dần
    desc_idx = np.argsort(-y_score)
    y_true = y_true[desc_idx]
    y_score = y_score[desc_idx]

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    tprs = [0.0]
    fprs = [0.0]
    thresholds = [1.0]

    TP = 0
    FP = 0
    prev_score = 1.0

    for i in range(len(y_score)):
        score = y_score[i]
        if score != prev_score:
            tprs.append(TP / P if P > 0 else 0.0)
            fprs.append(FP / N if N > 0 else 0.0)
            thresholds.append(score)
            prev_score = score

        if y_true[i] == 1:
            TP += 1
        else:
            FP += 1

    tprs.append(TP / P if P > 0 else 0.0)
    fprs.append(FP / N if N > 0 else 0.0)
    thresholds.append(0.0)

    fig, ax = plt.subplots()
    ax.plot(fprs, tprs)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    return fig, ax
