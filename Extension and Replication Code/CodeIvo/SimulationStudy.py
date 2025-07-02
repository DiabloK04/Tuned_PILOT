import os
import pandas as pd

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from CodeIvo.Simulation import generate_selected_datasets, TrueDGPRegressor
from dict import model_configs
import time

from paperplots import plot_prediction, plot_pdp_3d
import numpy as np
import random

random.seed(42)
np.random.seed(42)
import warnings
warnings.filterwarnings("ignore")
a = 6
n = [500]
random = 42
t = 0.2
n_iter = 75
replications = 1
domain = 6
experiment_name = "PILOT_PLOTSS"
n_test = 5000
indices = [1,2,3,4,5,6]
plot = True

# TODO: - ISE Computations
#       - PDP Plots True DGP
#       - Feature importance's True DGP

results = []
plot_dir = os.path.join("Experiments", experiment_name, "Plots")
os.makedirs(plot_dir, exist_ok=True)

valid_pairs = [
    (0, 0.0),
]

wholetime = time.perf_counter()
for useless, noise in valid_pairs:

        for n_obs in n:

            start = time.perf_counter()
            selected_keys = [
                             'PILOT_Base',
                             'CART',
                             'PILOT',
                             'TrueDGP',
                             'LightGBMLinear',
                             'MARS',
            ]
            selected_configs = {k: model_configs[k] for k in selected_keys if k != 'TrueDGP'}

            for run_id in range(replications):

                test_sets = generate_selected_datasets(
                    selected_indices=indices,
                    n_samples=n_test,
                    n_useless=useless,
                    noise=0.0,
                    random=random+run_id,
                    domain=domain
                )
                datasets = generate_selected_datasets(indices, n_obs, useless, noise, random+run_id, domain)

                for idx, dataset in datasets.items():
                    dataset_name = f"dataset_{idx}_useless_{useless}_noise_{noise}_n_{n_obs}"

                    for model_name in selected_keys:
                        modeltime = time.perf_counter()
                        if model_name == 'TrueDGP' and run_id != 0:
                            continue

                        if model_name == "LightGBMLinear" or model_name == "LightGBMLinear_Poly":
                            X = dataset.drop(columns=["y"]).to_numpy() #or values
                            y = dataset["y"]
                        else:
                            X = dataset.drop(columns=["y"]).to_numpy()
                            y = dataset["y"].to_numpy()


                        if model_name == 'TrueDGP':
                            best_model = TrueDGPRegressor(index=idx, domain=domain)
                            best_params = {}

                        else:
                            config = selected_configs[model_name]
                            model = config["model"]
                            param_grid = config["param_grid"]

                            grid = RandomizedSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error",
                                                      verbose=1,
                                                      n_iter=n_iter, random_state=random+run_id, n_jobs=-1)

                            grid.fit(X, y)

                            best_model = grid.best_estimator_
                            best_params = grid.best_params_

                        y_pred = best_model.predict(X)
                        r2 = r2_score(y, y_pred)

                        test_df = test_sets[idx]
                        X_nf = test_df.drop(columns="y").values
                        y_nf = test_df["y"].values  # f(x), noise=0

                        # predictions and MSE on noise free
                        y_pred_nf = best_model.predict(X_nf)
                        mse_test = mean_squared_error(y_nf, y_pred_nf)
                        r2_test = r2_score(y_nf, y_pred_nf)
                        # domain volume
                        if idx in (1, 2, 3):
                            volume = (2 * domain) ** 2
                        elif idx == 4:
                            volume = 1.0
                        else:
                            volume = 100 * (560 * np.pi - 40 * np.pi) * 1 * (11 - 1)

                        #ise, nise
                        ise = mse_test * volume
                        var_f = np.var(y_nf, ddof=0)
                        nise = ise / var_f

                        flat_params = {f"param_{k}": v for k, v in best_params.items()}

                        modeltimeend = time.perf_counter()
                        modelevaltime = modeltimeend - modeltime

                        results.append({
                            "run_id": run_id,
                            "model": model_name,
                            "dataset": dataset_name,
                            "n_useless": useless,
                            "n_obs": n_obs,
                            "idx": idx,
                            "test_frac": t,
                            "r_squared": r2,
                            "nise": nise,
                            "ise": ise,
                            "noise": noise,
                            "mse_test": mse_test,
                            "r2_test": r2_test,
                            "var_f": var_f,
                            "time": modelevaltime,
                            **flat_params
                        })

                        print(f"Added results for run {run_id}, dataset {dataset_name}.")
                        if run_id == 0 and plot == True:

                            print(f"Generating plots for {model_name} on {dataset_name}...")

                            if idx <= 3:  # eerste drie datasets
                                fig = plot_prediction(best_model, name=f"{model_name} on {dataset_name}", a=domain,
                                                      n_useless=useless, nr=idx)
                                fig.savefig(
                                    os.path.join(plot_dir, f"PredictedFunction_{dataset_name}_{model_name}.png"))
                                plt.close(fig)

                            else:  # laatste drie datasets: PDP + permutation importance
                                result = permutation_importance(
                                    best_model, X, y, n_repeats=10, random_state=42, scoring="r2"
                                )
                                importances = result.importances_mean

                                #Top 2 features
                                top_features = sorted(
                                    enumerate(importances), key=lambda x: x[1], reverse=True
                                )[:2]
                                feature_indices = [idx for idx, _ in top_features]

                                labels = [str(i + 1) for i in range(len(importances))]

                                LABEL_FS = 14  # axis label font size
                                TICK_FS = 12  # tick label font size
                                LEGEND_FS = 12  # legend font size

                                # Barplot
                                plt.figure(figsize=(8, 4))
                                plt.bar(labels, importances, color="blue")
                                plt.title(f"Permutation Importance for Function {idx}.")

                                plt.ylabel("Importance", fontsize=LABEL_FS)
                                plt.xlabel("Feature", fontsize=LABEL_FS)
                                plt.xticks(fontsize=TICK_FS)
                                plt.yticks(fontsize=TICK_FS)
                                plt.grid(axis='y', linestyle='--', alpha=0.7)
                                plt.tight_layout()

                                fname = f"Importance_{dataset_name}_{model_name}.png"
                                plt.savefig(os.path.join(plot_dir, fname))
                                plt.close()

                                fig = plot_pdp_3d(best_model, X, feature_indices, nr=idx, grid_resolution=50)
                                fig.savefig(
                                    os.path.join(plot_dir, f"PDP_{dataset_name}_{model_name}_f{feature_indices}.png"))
                                plt.close(fig)
            end = time.perf_counter()
            print(f"simulate_dataset(useless={useless}, noise={noise}) took {end - start:.4f} seconds")
results_df = pd.DataFrame(results)
output_dir = os.path.join("Experiments", experiment_name)
os.makedirs(output_dir, exist_ok=True)

#Save results
results_df.to_csv(
    os.path.join(output_dir, f"summary_{experiment_name}_output.csv"),
    index=False
)

wholetime_end = time.perf_counter()

print(f"Results saved, took {wholetime_end - wholetime:.4f} seconds")
