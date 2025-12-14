from utils import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from scipy import interpolate
from scipy import integrate
import matplotlib.pyplot as plt
from sklearn import preprocessing


def compute_abroca(
    df,
    pred_col,
    label_col,
    protected_attr_col,
    majority_protected_attr_val,
    n_grid=10000,
    plot_slices=False,
    lb=0,
    ub=1,
    limit=1000,
    file_name="slice_image.png",
    majority_group_name='Male',
    minority_group_name='Female'
):

    # ---- FIX 1: pandas >= 2.0, không dùng inclusive=True nữa ----
    # Thay bằng inclusive='both' hoặc kiểm tra thủ công
    # old: df[pred_col].between(0, 1, inclusive=True)
    if ((df[pred_col] >= 0) & (df[pred_col] <= 1)).all():
        pass
    else:
        print("predictions must be in range [0,1]")
        exit(1)

    # Check label column binary
    if len(df[label_col].value_counts()) == 2:
        pass
    else:
        print("The label column should be binary")
        exit(1)

    # Check protected attr binary
    if len(df[protected_attr_col].value_counts()) == 2:
        pass
    else:
        print("The protected attribute column should be binary")
        exit(1)

    # Init
    prot_attr_values = df[protected_attr_col].value_counts().index.values
    fpr_tpr_dict = {}

    # Compute ROC per group
    for pa_value in prot_attr_values:
        if pa_value != majority_protected_attr_val:
            minority_protected_attr_val = pa_value

        pa_df = df[df[protected_attr_col] == pa_value]
        fpr_tpr_dict[pa_value] = compute_roc(pa_df[pred_col], pa_df[label_col])

    # Interpolation
    majority_roc_x, majority_roc_y = interpolate_roc_fun(
        fpr_tpr_dict[majority_protected_attr_val][0],
        fpr_tpr_dict[majority_protected_attr_val][1],
        n_grid
    )

    minority_roc_x, minority_roc_y = interpolate_roc_fun(
        fpr_tpr_dict[minority_protected_attr_val][0],
        fpr_tpr_dict[minority_protected_attr_val][1],
        n_grid
    )

    # Integrate |majority - minority|
    if list(majority_roc_x) == list(minority_roc_x):
        f1 = interpolate.interp1d(x=majority_roc_x, y=(majority_roc_y - minority_roc_y))
        f2 = lambda x, acc: abs(f1(x))
        slice_val, _ = integrate.quad(f2, lb, ub, limit)
    else:
        print("Majority and minority FPR are different")
        exit(1)

    # Plotting
    if plot_slices:
        slice_plot(
            majority_roc_x,
            minority_roc_x,
            majority_roc_y,
            minority_roc_y,
            majority_group_name=majority_group_name,
            minority_group_name=minority_group_name,
            fout=file_name,
            value=round(slice_val, 4),
        )

    return slice_val
