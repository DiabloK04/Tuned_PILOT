import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pilot.Pilot import tree_summary

"""
File: paperplots.py
Author: Ivo Klazema
Description: This file contains the methods to make the plots for the extension part of the paper: "Extending PILOT by tuning the BIC penalty".
The calls are in "SimulationStudy.py".
"""

LABEL_FS = 14  # axis label font size
TICK_FS = 12  # tick label font size
LEGEND_FS = 12  # legend font size


def _make_X_label(idx):
    return f"X{idx + 1}"


def plot_pilot_root(model_tree, X_train, y_train, X_val=None, y_val=None):
    """
    Makes a scatter plot of the target variable and the variable used in the first split by PILOT.
    The first fitted model is extracted from the pilot object and is printed over the scatterplot.
    """

    root = model_tree
    node_type = root.node
    pivot_val = root.pivot[1]
    interval = root.interval
    feat_id = root.pivot[0]
    xlabel = _make_X_label(feat_id)

    x_tr = X_train.iloc[:, feat_id]
    y_tr = y_train
    x_va = X_val.iloc[:, feat_id] if X_val is not None else None
    y_va = y_val if X_val is not None else None

    # determine plot bounds
    lo, hi = interval if interval is not None else (x_tr.min(), x_tr.max())

    plt.figure(figsize=(9, 6))

    plt.scatter(x_tr, y_tr, color="blue", alpha=0.3, s=20)
    if x_va is not None:
        plt.scatter(x_va, y_va, color="blue", alpha=0.3, s=20)

    # red PILOT curve
    xx = np.linspace(lo, hi, 300)
    if node_type in ("plin", "pcon", "blin"):
        # two segments
        xx_left = xx[xx <= pivot_val]
        xx_right = xx[xx >= pivot_val]
        y_left = root.lm_l[0] * xx_left + root.lm_l[1]
        y_right = root.lm_r[0] * xx_right + root.lm_r[1]
        plt.plot(xx_left, y_left, color="red", linewidth=2, label="PILOT")
        plt.plot(xx_right, y_right, color="red", linewidth=2)
    elif node_type == "lin":
        y_pred = root.lm_l[0] * xx + root.lm_l[1]
        plt.plot(xx, y_pred, color="red", linewidth=2, label="PILOT")
    else:  # constant
        avg = (pd.concat([y_tr, y_va]).mean() if x_va is not None else y_tr.mean())
        plt.hlines(avg, lo, hi, colors="red", linewidth=2, label="PILOT")

    plt.xlabel(xlabel, fontsize=LABEL_FS)
    plt.ylabel("Target", fontsize=LABEL_FS)
    plt.xticks(fontsize=TICK_FS)
    plt.yticks(fontsize=TICK_FS)
    plt.legend(loc="best", fontsize=LEGEND_FS)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_compare_cart_pilot(cart_model, pilot_tree, X_train, y_train, X_val=None, y_val=None):
    """
    Makes a scatter plot of the target variable and the variable used in the first split by PILOT and CART.
    The first fitted models are extracted from the pilot object and the cart object and are printed over the scatterplot.
    """

    # extract feature and interval from pilot_tree
    feat_id, pivot_val = pilot_tree.pivot
    interval = pilot_tree.interval
    xlabel = _make_X_label(feat_id)

    x_tr = X_train.iloc[:, feat_id]
    y_tr = y_train
    x_va = X_val.iloc[:, feat_id] if X_val is not None else None
    y_va = y_val if (X_val is not None and y_val is not None) else None

    lo, hi = interval if interval is not None else (x_tr.min(), x_tr.max())

    plt.figure(figsize=(9, 6))
    plt.scatter(x_tr, y_tr, color="blue", alpha=0.3, s=20)
    if x_va is not None:
        plt.scatter(x_va, y_va, color="blue", alpha=0.3, s=20)

    # CART root-split on same feature
    tree = cart_model.tree_
    if tree.feature[0] != feat_id:
        raise ValueError(f"CART root split on X{tree.feature[0] + 1}, "
                         f"but PILOT uses X{feat_id + 1}")
    thresh = tree.threshold[0]
    left_val = y_tr[x_tr <= thresh].mean()
    right_val = y_tr[x_tr > thresh].mean()
    plt.hlines(left_val, lo, thresh, colors="black", linewidth=2, label="CART")
    plt.hlines(right_val, thresh, hi, colors="black", linewidth=2)

    # PILOT curve
    xx = np.linspace(lo, hi, 300)
    node_type = pilot_tree.node
    if node_type in ("plin", "pcon", "blin"):
        xx_l = xx[xx <= pivot_val]
        xx_r = xx[xx >= pivot_val]
        y_l = pilot_tree.lm_l[0] * xx_l + pilot_tree.lm_l[1]
        y_r = pilot_tree.lm_r[0] * xx_r + pilot_tree.lm_r[1]
        plt.plot(xx_l, y_l, color="red", linewidth=2, label="PILOT")
        plt.plot(xx_r, y_r, color="red", linewidth=2)
    elif node_type == "lin":
        y_pred = pilot_tree.lm_l[0] * xx + pilot_tree.lm_l[1]
        plt.plot(xx, y_pred, color="red", linewidth=2, label="PILOT")
    else:
        vals = np.concatenate([y_tr.values, y_va.values]) if y_va is not None else y_tr.values
        plt.hlines(vals.mean(), lo, hi, colors="red", linewidth=2, label="PILOT")

    plt.xlabel(xlabel, fontsize=LABEL_FS)
    plt.ylabel("Target", fontsize=LABEL_FS)
    plt.xticks(fontsize=TICK_FS)
    plt.yticks(fontsize=TICK_FS)
    plt.legend(loc="best", fontsize=LEGEND_FS)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_cart_feature_importance(cart_model, threshold=0.01):
    """
    Plots the feature importances for the CART model (DecisionTreeRegressor) of Scikit-Learn.
    """

    imp = cart_model.feature_importances_
    idxs = np.where(imp >= threshold)[0]
    vals = imp[idxs]
    labels = [_make_X_label(i) for i in idxs]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, vals, color="blue")
    plt.ylabel("Importance", fontsize=LABEL_FS)
    plt.xticks(fontsize=TICK_FS)
    plt.yticks(fontsize=TICK_FS)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_pilot_feature_importance(pilot_model, threshold=0.01):
    """
    Plots the feature importances for the PILOT model by aggregating the RSS drops of all the splits and normalizing these.
    """

    # build summary of splits
    summary = []
    tree_summary(pilot_model.model_tree, level=0, tree_id='root',
                 parent_id=None, summary=summary)
    df = pd.DataFrame(summary)

    # compute gain per feature
    gains = {}
    intern = df[df['node'] != 'END']
    for _, row in intern.iterrows():
        nid = row['tree_id']
        children = df[df['parent_id'] == nid]
        if len(children) == 2:
            child_sum = children['Rt'].sum()
        elif len(children) == 1:
            child_sum = children['Rt'].iloc[0]
        else:
            continue
        gain = row['Rt'] - child_sum
        fidx = int(row['pivot_idx'])
        gains[fidx] = gains.get(fidx, 0.0) + gain

    total = sum(gains.values())

    # select features above threshold and sort by index
    idxs = sorted(i for i, g in gains.items() if (g / total) >= threshold)
    vals = [(gains[i] / total) for i in idxs]
    labels = [f"X{i + 1}" for i in idxs]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, vals, color="blue")
    plt.ylabel("Importance", fontsize=LABEL_FS)
    plt.xticks(fontsize=TICK_FS)
    plt.yticks(fontsize=TICK_FS)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_cart_root_red(cart_model, X_train, y_train, X_val=None, y_val=None):
    """
    Makes a scatter plot of the target variable and the variable used in the first split by CART.
    The first fitted model is extracted from the DecisionTreeRegressor object and is printed over the scatterplot.
    """
    tree = cart_model.tree_
    feat = tree.feature[0]
    thresh = tree.threshold[0]
    xlabel = _make_X_label(feat)

    x_tr = X_train.iloc[:, feat]
    y_tr = y_train
    x_va = X_val.iloc[:, feat] if X_val is not None else None
    y_va = y_val if (X_val is not None and y_val is not None) else None

    lo, hi = x_tr.min(), x_tr.max()

    plt.figure(figsize=(9, 6))
    # blue scatter
    plt.scatter(x_tr, y_tr, color="blue", alpha=0.3, s=20)
    if x_va is not None:
        plt.scatter(x_va, y_va, color="blue", alpha=0.3, s=20)

    # red CART step at root
    left_val = y_tr[x_tr <= thresh].mean()
    right_val = y_tr[x_tr > thresh].mean()
    plt.hlines(left_val, lo, thresh, colors="red", linewidth=2, label="CART")
    plt.hlines(right_val, thresh, hi, colors="red", linewidth=2)

    plt.xlabel(xlabel, fontsize=LABEL_FS)
    plt.ylabel("Target", fontsize=LABEL_FS)
    plt.xticks(fontsize=TICK_FS)
    plt.yticks(fontsize=TICK_FS)
    plt.legend(loc="best", fontsize=LEGEND_FS)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_pdp_3d(model, X, features, nr, grid_resolution=30):
    """
    Makes a three-dimensional partial dependence plot for the inputted model and the data based on the two features that are inputted as an argument.
    """
    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    feat1, feat2 = features

    # Extract feature columns
    col1 = X_arr[:, feat1]
    col2 = X_arr[:, feat2]

    # Build a grid
    x1_vals = np.linspace(col1.min(), col1.max(), grid_resolution)
    x2_vals = np.linspace(col2.min(), col2.max(), grid_resolution)

    # Placeholder for PD values
    Z = np.zeros((grid_resolution, grid_resolution))

    # Compute partial dependence by averaging predictions
    for i, v1 in enumerate(x1_vals):
        for j, v2 in enumerate(x2_vals):
            X_temp = X_arr.copy()
            X_temp[:, feat1] = v1
            X_temp[:, feat2] = v2
            Z[i, j] = model.predict(X_temp).mean()

    # Plot the surface
    xx, yy = np.meshgrid(x1_vals, x2_vals)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, Z.T, rstride=1, cstride=1, alpha=0.7)

    xlabel1 = _make_X_label(feat1)
    xlabel2 = _make_X_label(feat2)

    ax.set_title(f"PDP for Function {nr}, {xlabel1} & {xlabel2}.", fontsize=LABEL_FS)
    ax.set_xlabel(xlabel1, fontsize=LABEL_FS, labelpad=10)
    ax.set_ylabel(xlabel2, fontsize=LABEL_FS, labelpad=10)
    ax.set_zlabel("Partial dependence", fontsize=LABEL_FS, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
    ax.tick_params(axis='z', which='major', labelsize=TICK_FS)
    plt.tight_layout()

    return fig


def plot_prediction(model, nr, a=6, n_useless=0):
    """
        Makes a three-dimensional prediction plot for the inputted model and the domain that are inputted as an argument.
        """
    grid_size = 50
    x1_grid = np.linspace(-a, a, grid_size)
    x2_grid = np.linspace(-a, a, grid_size)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)

    z_fixed = np.zeros((X1.ravel().shape[0], n_useless))  # shape (2500, 5)

    X_pred = np.column_stack([X1.ravel(), X2.ravel(), z_fixed])  # shape (2500, 7)
    y_pred = model.predict(X_pred)
    Y_pred_grid = y_pred.reshape(X1.shape)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X1, X2, Y_pred_grid, cmap='plasma', edgecolor='none')

    ax.set_title(f"Prediction Surface for Function {nr}", fontsize=LABEL_FS)
    ax.set_xlabel("X1", fontsize=LABEL_FS, labelpad=10)
    ax.set_ylabel("X2", fontsize=LABEL_FS, labelpad=10)
    ax.set_zlabel("y", fontsize=LABEL_FS, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
    ax.tick_params(axis='z', which='major', labelsize=TICK_FS)
    plt.tight_layout()
    plt.show()

    return fig
