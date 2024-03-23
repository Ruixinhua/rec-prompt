import numpy as np


# Source: http://ethen8181.github.io/machine-learning/model_selection/auc/auc.html#Implementation
def binary_clf_curve(y_true, y_score):
    """
    Calculate true and false positives per binary classification
    threshold (can be used for roc curve or precision/recall curve);
    the calcuation makes the assumption that the positive case
    will always be labeled as 1
    Parameters
    ----------
    y_true : 1d ndarray, shape = [n_samples]
        True targets/labels of binary classification
    y_score : 1d ndarray, shape = [n_samples]
        Estimated probabilities or scores
    Returns
    -------
    tps : 1d ndarray
        True positives counts, index i records the number
        of positive samples that got assigned a
        score >= thresholds[i].
        The total number of positive samples is equal to
        tps[-1] (thus false negatives are given by tps[-1] - tps)
    fps : 1d ndarray
        False positives counts, index i records the number
        of negative samples that got assigned a
        score >= thresholds[i].
        The total number of negative samples is equal to
        fps[-1] (thus true negatives are given by fps[-1] - fps)
    thresholds : 1d ndarray
        Predicted score sorted in decreasing order
    References
    ----------
    Github: scikit-learn _binary_clf_curve
    - https://github.com/scikit-learn/scikit-learn/blob/ab93d65/sklearn/metrics/ranking.py#L263
    """

    # sort predicted scores in descending order
    # and also reorder corresponding truth values
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically consists of tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve
    distinct_indices = np.where(np.diff(y_score))[0]
    end = np.array([y_true.size - 1])
    threshold_indices = np.hstack((distinct_indices, end))

    thresholds = y_score[threshold_indices]
    tps = np.cumsum(y_true)[threshold_indices]

    # (1 + threshold_indices) = the number of positives
    # at each index, thus number of data points minus true
    # positives = false positives
    fps = (1 + threshold_indices) - tps
    return tps, fps, thresholds


def roc_auc_score(y_true, y_score):
    """
    Compute Area Under the Curve (AUC) from prediction scores
    Parameters
    ----------
    y_true : 1d ndarray, shape = [n_samples]
        True targets/labels of binary classification
    y_score : 1d ndarray, shape = [n_samples]
        Estimated probabilities or scores
    Returns
    -------
    auc : float
    """

    # ensure the target is binary
    if np.unique(y_true).size != 2:
        raise ValueError('Only two class should be present in y_true. ROC AUC score '
                         'is not defined in that case.')

    tps, fps, _ = binary_clf_curve(y_true, y_score)

    # convert count to rate
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]

    # compute AUC using the trapezoidal rule;
    # appending an extra 0 is just to ensure the length matches
    zero = np.array([0])
    tpr_diff = np.hstack((np.diff(tpr), zero))
    fpr_diff = np.hstack((np.diff(fpr), zero))
    auc = np.dot(tpr, fpr_diff) + np.dot(tpr_diff, fpr_diff) / 2
    return auc


def accuracy(pred, target):
    return np.sum(np.array(pred) == np.array(target)).item() / len(target)


def mrr_score(y_true, y_score):
    """Computing mrr score metric.

    Args:
        y_true (np.ndarray): ground-truth label.
        y_score (np.ndarray): predicted label.

    Returns:
        np.ndarray: mrr scores.
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.

    Args:
        y_true (np.ndarray): ground-truth label.
        y_score (np.ndarray): predicted label.
        k

    Returns:
        np.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.

    Args:
        y_true (np.ndarray): ground-truth label.
        y_score (np.ndarray): predicted label.
        k

    Returns:
        np.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def group_auc(label, pred):
    """
    Compute the area under the ROC curve
    :param label: List[np.ndarray] or np.ndarray
    :param pred: List[np.ndarray] or np.ndarray
    :return: roc auc score
    """
    if isinstance(label, list) or len(label.shape) > 1:
        return np.round(np.mean([roc_auc_score(l, p) for l, p in zip(label, pred)]).item(), 4)
    else:
        return roc_auc_score(label, pred)


def mean_mrr(label, pred):
    """
    Compute the mean reciprocal rank
    :param label: List[np.ndarray] or np.ndarray
    :param pred: List[np.ndarray] or np.ndarray
    :return: MRR score
    """
    if isinstance(label, list) or len(label.shape) > 1:
        return np.round(np.mean([mrr_score(l, p) for l, p in zip(label, pred)]).item(), 4)
    else:
        return mrr_score(label, pred)


def ndcg(label, pred, k):
    """
    Compute the normalized discounted cumulative gain
    :param label: List[np.ndarray] or np.ndarray
    :param pred: List[np.ndarray] or np.ndarray
    :param k: the number of evaluated items
    :return: NDCG score
    """
    if isinstance(label, list) or len(label.shape) > 1:
        return np.round(np.mean([ndcg_score(l, p, k) for l, p in zip(label, pred)]).item(), 4)
    else:
        return ndcg_score(label, pred, k)


def ndcg_5(label, pred):
    return ndcg(label, pred, 5)


def ndcg_10(label, pred):
    return ndcg(label, pred, 10)
