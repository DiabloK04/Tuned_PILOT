import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import random

np.random.seed(42)
random.seed(42)
def true_dgp(X, index, domain=6):
    X = np.asarray(X)
    if index == 1:
        x1, x2 = X[:,0], X[:,1]
        return np.where(x2 < 0, 6*x1, 6*x1 + 3*x2)
    elif index == 2:
        x1, x2 = X[:,0], X[:,1]
        return 10*np.sin(x1) + 3*(x2**2)
    elif index == 3:
        x1, x2 = X[:,0], X[:,1]
        return 6*x1*x2 + 3*x2
    elif index == 4:
        return (10*np.sin(np.pi*X[:,0]*X[:,1])
                + 20*(X[:,2]-0.5)**2
                + 10*X[:,3]
                + 5*X[:,4])
    elif index == 5:
        return np.sqrt(X[:,0]**2
                       + (X[:,1]*X[:,2] - 1.0/(X[:,1]*X[:,3]))**2)
    elif index == 6:
        return np.arctan((X[:,1]*X[:,2] - 1.0/(X[:,1]*X[:,3]))/X[:,0])
    else:
        raise ValueError("index must be in 1...6")

class TrueDGPRegressor(BaseEstimator, RegressorMixin):
    """
    A “model” whose predict(X) returns the noise-free true DGP f(X).
    """
    def __init__(self, index, domain=6):
        self.index = index
        self.domain = domain

    def fit(self, X, y=None):
        # no training needed
        return self

    def predict(self, X):
        return true_dgp(X, self.index, self.domain)

def plot_dgp(a=3):

    grid_size = 50
    x1_grid = np.linspace(-a, a, grid_size)
    x2_grid = np.linspace(-a, a, grid_size)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)

    # Define the true functions (no noise)
    alpha, beta, gamma = 6.0, 3.0, 10.0

    # Dataset 1: Broken linear function
    F1 = np.where(X2 < 0, alpha * X1, alpha * X1 + beta * X2)
    # Dataset 2: Nonlinear, no interaction
    F2 = gamma * np.sin(X1) + beta * (X2 ** 2)
    # Dataset 3: Interaction term
    F3 = alpha * X1 * X2 + beta * X2

    # Plot true DGPs
    fig = plt.figure(figsize=(18, 5))

    titles = ['True DGP: Broken Linear', 'True DGP: Nonlinear (No Interaction)', 'True DGP: Interaction']
    functions = [F1, F2, F3]

    for i, (Z, title) in enumerate(zip(functions, titles)):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.plot_surface(X1, X2, Z, cmap='plasma', edgecolor='none')
        ax.set_title(title)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')

    plt.tight_layout()
    plt.show()

def plot_prediction(model, name, a=3, n_useless=5):
    grid_size = 50
    x1_grid = np.linspace(-a, a, grid_size)
    x2_grid = np.linspace(-a, a, grid_size)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)

    z_fixed = np.zeros((X1.ravel().shape[0], n_useless))  # shape (2500, 5)

    X_pred = np.column_stack([X1.ravel(), X2.ravel(), z_fixed])  # shape (2500, 7)
    y_pred = model.predict(X_pred)
    Y_pred_grid = y_pred.reshape(X1.shape)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X1, X2, Y_pred_grid, cmap='plasma', edgecolor='none')
    ax.set_title(name)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    plt.tight_layout()

    return fig

def generate_selected_datasets(selected_indices, n_samples=1000, n_useless=0, noise=0.1, random=None, domain=6):
    """
    Generate a selection of synthetic datasets based on index choices.

    Index mapping:
      1: Custom broken linear function
      2: Custom nonlinear no-interaction function
      3: Custom non-additive interaction function
      4: Friedman 1
      5: Friedman 2
      6: Friedman 3

    Parameters:
        selected_indices (list of int): Dataset indices to generate (from 1 to 6).
        n_samples (int): Number of observations per dataset.
        n_useless (int): Number of additional useless (noise) features to append.
        noise (float): Standard deviation of Gaussian noise in the target.
        random: (int, optional): Seed for reproducibility.
        domain: (int): domain of x1, x2 of first three DGP's.

    Returns:
        dict[str, pd.DataFrame]: Mapping "dataset_<index>"  DataFrame with features and 'y'.
    """

    rng = np.random.RandomState(random)

    result = {}

    # Prepare common random inputs for custom functions
    x1 = rng.uniform(-domain, domain, n_samples)
    x2 = rng.uniform(-domain, domain, n_samples)
    eps = rng.normal(0, noise, n_samples)

    # Define custom dataset formulas
    custom_formulas = {
        1: lambda: np.where(x2 < 0, 6 * x1 + eps, 6 * x1 + 3 * x2 + eps),
        2: lambda: 10 * np.sin(x1) + 3 * (x2 ** 2) + eps,
        3: lambda: 6 * x1 * x2 + 3 * x2 + eps
    }

    # Define Friedman dataset generators
    friedman_generators = {
        4: lambda: make_friedman1(n_samples=n_samples, n_features=5, noise=noise, random_state=random),
        5: lambda: make_friedman2(n_samples=n_samples, noise=noise, random_state=random),
        6: lambda: make_friedman3(n_samples=n_samples, noise=noise, random_state=random)
    }

    for idx in selected_indices:
        if idx in custom_formulas:
            # Generate custom dataset
            y = custom_formulas[idx]()
            df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
        elif idx in friedman_generators:
            # Generate Friedman dataset
            X, y = friedman_generators[idx]()
            df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
            df['y'] = y
        else:
            raise ValueError(f"Index {idx} is not in [1..6].")

        # Append useless features if requested
        if n_useless > 0:
            noise_feats = rng.normal(0, 1, size=(n_samples, n_useless))
            df = pd.concat([
                df,
                pd.DataFrame(noise_feats, columns=[f"z{i + 1}" for i in range(n_useless)])
            ], axis=1)

        result[idx] = df

    return result

