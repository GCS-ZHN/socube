# MIT License
#
# Copyright (c) 2022 Zhang.H.N
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional, Tuple
import numpy as np
import pandas as pd
import sklearn.metrics as metrics

from socube.data import (onehot, plotAUC)

__all__ = ["getCurve", "evaluateReport", "binaryRate"]
"""
This module provides metric tools for training.
"""


def binaryRate(label: np.ndarray,
               predict: np.ndarray,
               positive_first: bool = False) -> Tuple[float]:
    """
    For binary classification task, calculate its confusion matrix
    and true positive rate (TPR), false negative rate (FNR), false
    positive rate (FPR) and true negative rate (TNR).

    Parameters
    ------------
    label: np.ndarray
        the true label 1D ndarray vector
    predict: np.ndarray
        the predict label 1D ndarray vector
    positive_first :If True, it will regard 0 as positive class. Otherwise
                it will regard 1 as positive class. By default, it is
                False.

    Returns
    ------------
    a tuple of TPR, FNR, FPR, TNR
    """
    confusion_matrix = metrics.confusion_matrix(label, predict)
    tnr, fpr, fnr, tpr = (confusion_matrix /
                          confusion_matrix.sum(axis=1).reshape(
                              (-1, 1))).flatten()
    if positive_first:
        tnr, fpr = tpr, fnr
    return tpr, fnr, fpr, tnr


def getCurve(label: np.ndarray,
             score: np.ndarray,
             average: str = "macro",
             curve: str = "roc"):
    """
    Evaluate receiver operating characteristic (ROC) for multiclass task
    based on false positive rate(FPR) and true positive rate(TPR). The area
    under curve(AUC) is used.

    Parameters
    ------------
    label: np.ndarray
        the true label 1D ndarray vector
    score: np.ndarray
        the predict score 1D ndarray vector
    average: str
        the average method, "macro" , "micro" or "binary"
        how to average multiclass. Because conventional
        ROC is designed for binary classification task. For
        detail about this, please search it on the Internet.

    Returns
    ------------
        A triple tuple of (x, y, AUC). When curve is "roc",
        x is FPR and y is TPR. When curve is "prc", x is FNR and y is TNR.
    """
    assert len(label) == len(score), "Unmatched label and score"
    assert average in ["macro", "micro",
                       "binary"], f"Invalid average mode {average}"

    def calc(y_true: np.ndarray, y_score: np.ndarray):
        if curve == "roc":
            x, y, _ = metrics.roc_curve(y_true, y_score)
        elif curve == "prc":
            y, x, _ = metrics.precision_recall_curve(y_true, y_score)
        else:
            raise NotImplementedError(f"Unsupport curve type '{curve}'")
        return x, y

    if average == "binary":
        assert len(label.shape) == 1, "Required 1D label for binary"
        assert len(score.shape) == 1, "Required 1D score for binary"
        x, y = calc(label, score)
    elif average == "macro":
        x_list = list()
        y_list = list()
        class_nums = score.shape[1]
        label = onehot(label, class_nums)
        for subtype in range(class_nums):
            sub_label = label[:, subtype]
            if not sub_label.max() > 0:
                continue
            sub_score = score[:, subtype]
            x_array, y_array = calc(sub_label, sub_score)
            x_list.append(x_array)
            y_list.append(y_array)
        x = np.unique(np.concatenate(x_list))
        y = np.zeros_like(x)
        for i in range(len(x_list)):
            y += np.interp(x, x_list[i], y_list[i])
        y /= i + 1
    elif average == "micro":
        label = onehot(label, score.shape[1])
        x, y = calc(label.ravel(), score.ravel())
    auc = metrics.auc(x, y)
    return x, y, auc


def evaluateReport(label: np.ndarray,
                   score: np.ndarray,
                   roc_plot_file: Optional[str] = None,
                   prc_plot_file: Optional[str] = None,
                   average: str = "macro",
                   threshold: float = 0.5) -> pd.Series:
    """
    Evaluate model performance with multiple indicator and generate file report.

    Parameters
    ------------
    label: np.ndarray
        the true label 1D ndarray vector
    score: np.ndarray
        the predict score 1D ndarray vector
    roc_plot_file: Optional[str]
        the file name of ROC curve plot. If None, it will not save plot.
    prc_plot_file: Optional[str]
        the file name of PR curve plot. If None, it will not save plot.
    average: str
        the average method, "macro" , "micro" or "binary"
        how to average multiclass. Because conventional
        ROC is designed for binary classification task. For
        detail about this, please search it on the Internet.
    threshold: float
        the threshold of predict score.

    Returns
    ------------
    a pandas.Series object as the report.
    """
    result = pd.Series(dtype="object")
    if len(score.shape) == 1:
        predict = (score >= threshold).astype(np.int64)
        average = "binary"
    elif len(score.shape) == 2:
        predict = score.argmax(1)
    else:
        raise NotImplementedError("Unsupport yscore dimension")

    report = metrics.classification_report(label, predict, output_dict=True)
    result["ACC"] = report["accuracy"]
    result["MCC"] = metrics.matthews_corrcoef(label, predict)
    result["AP"] = metrics.average_precision_score(label, score)

    roc_data_dict = dict()
    prc_data_dict = dict()

    fpr, tpr, auroc = getCurve(label, score, average, curve="roc")
    recall, precision, auprc = getCurve(label, score, average, curve="prc")
    roc_data_dict['%s ROC curve (area = %0.4f)' % (average, auroc)] = (fpr,
                                                                       tpr)
    prc_data_dict['%s PRC curve (area = %0.4f)' %
                  (average, auprc)] = (recall, precision)

    result[f"{average}_AUROC"] = auroc
    result[f"{average}_AUPRC"] = auprc

    result[f"{average}_F1"] = metrics.f1_score(label, predict, average=average)
    result[f"{average}_precision"] = metrics.precision_score(label,
                                                             predict,
                                                             average=average)
    result[f"{average}_recall"] = metrics.recall_score(label,
                                                       predict,
                                                       average=average)

    plotAUC(roc_data_dict, 'AUROC Curve', 'False Positive Rate',
            'True Positive Rate', roc_plot_file, 1)
    plotAUC(prc_data_dict, 'AUPRC Curve', 'Recall', 'Precision', prc_plot_file,
            -1)
    return result
