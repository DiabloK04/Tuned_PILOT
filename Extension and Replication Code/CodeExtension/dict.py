import lightgbm as lgb
from scipy.stats import loguniform
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from pilot.Pilot import PILOT
from pyearth import Earth

model_configs = {
    "MARS": {
        "model": Earth(),
        "param_grid": {
            'max_terms': [10, 50, 100, None],
            'max_degree': [1, 2, 3],
            'penalty': [1, 2, 3, None],
            'endspan': [10, 20, None],
            'minspan_alpha': [0.0, 0.1, 0.05, None],
            'thresh': [0.001, 0.01, 0.1, None],
        }
    },
    "CART": {
        "model": DecisionTreeRegressor(),
        "param_grid": {
            'max_depth': [3, 6, 9, 12, 15, 18],
            'min_samples_split': [2, 10, 20, 40],
            'min_samples_leaf': [1, 5, 10, 20],
            'ccp_alpha': [0, 0.001, 0.005, 0.01, 0.015, 0.03] #added 0.03
        }
    },
    "PILOT": {
        "model": PILOT(),
        "param_grid": {
            'max_depth': [3, 6, 9, 12, 15, 18],
            'min_sample_split': [2, 10, 20, 40],
            'min_sample_leaf': [1, 5, 10, 20],
            'bic_factor': loguniform(1e-6, 1.1),
        }
    },
    "PILOT_Base": {
        "model": PILOT(),
        "param_grid": {
            'max_depth': [3, 6, 9, 12, 15, 18],
            'min_sample_split': [2, 10, 20, 40],
            'min_sample_leaf': [1, 5, 10, 20],
            'bic_factor': [1]
        }
    },
    "LightGBMLinear": {
        "model": lgb.LGBMRegressor(
            objective="regression",
            device="cpu",
            linear_tree=True,
            tree_learner="serial",
            verbosity=-1,
            force_col_wise=True,
        ),
        "param_grid": {
            "max_depth": [3, 6, 9, 12, 15, 18],
            "min_data": [1, 5, 10, 20, 30]
        }
    },
    "Ridge": {
        "model": Ridge(),
        "param_grid": {
            'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]
        }
    },
    "Lasso": {
        "model": Lasso(),
        "param_grid": {
            'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]
        }
    }
}
