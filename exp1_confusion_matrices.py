
"""Experiment 1: Construct and verify bi-level and multi-level confusion matrices.
Input: small example labels (or replace with your dataset).
Output: printed TP,FN,FP,TN and a plotted multi-class confusion matrix.
"""
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_true = np.array([1,0,1,1,0])
y_pred = np.array([1,0,0,1,1])

cm = confusion_matrix(y_true, y_pred)
TN, FP, FN, TP = cm.ravel()
print("Bi-level confusion matrix counts -> TP={}, TN={}, FP={}, FN={}".format(TP, TN, FP, FN))

y_true_multi = np.array([0,0,1,2,1,2,0,1,2,2])
y_pred_multi = np.array([0,1,1,2,1,0,0,2,2,2])
cm_multi = confusion_matrix(y_true_multi, y_pred_multi)
print("Multi-class confusion matrix:\n", cm_multi)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_multi)
disp.plot()
plt.title("Multi-class Confusion Matrix")
plt.savefig("exp1_confusion_multi.png", dpi=150)
print("Saved multi-class confusion matrix to exp1_confusion_multi.png")
