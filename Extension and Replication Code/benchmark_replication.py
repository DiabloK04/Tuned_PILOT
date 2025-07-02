from sklearn.metrics import mean_squared_error
from sklearn.model_selection import  train_test_split, RandomizedSearchCV
from benchmark_config import IGNORE_COLUMNS
from benchmark_util import *
from CodeIvo.dict import model_configs

OUTPUTFOLDER = pathlib.Path(__file__).parent / "Output"

def run_benchmark(experiment_name):

    experiment_folder = OUTPUTFOLDER / experiment_name
    experiment_folder.mkdir(exist_ok=True)
    experiment_file = experiment_folder / "results.csv"
    print(f"Results will be stored in {experiment_file}")


    repo_ids_to_process = [1, 183, 186, 275, 291, 999]
    results = []

    for repo_id in repo_ids_to_process:

        selected_keys = ['PILOT', 'CART']
        selected_configs = {k: model_configs[k] for k in selected_keys}

        for model_name, config in selected_configs.items():
            model = config["model"]
            param_grid = config["param_grid"]

            print(f"Starting gridsearch for {model_name}, dataset_{repo_id}")
            grid = RandomizedSearchCV(model, param_grid, n_iter=75, cv=5, scoring="neg_mean_squared_error", verbose=3, n_jobs=-1, random_state=42)
            fit_params = {}

            if repo_id == 999:
                dataset = pd.read_csv("DataCsv/Admission_Predict.csv")
                X = dataset.drop(columns=["Serial No.", "Chance of Admit "]).to_numpy()
                y = dataset["Chance of Admit "].to_numpy()
            else:

                dataset = load_data(repo_id, ignore_feat=IGNORE_COLUMNS.get(repo_id))
                if model_name == "PILOT" or model_name == "PILOT_Base":
                    X = dataset.X_label_encoded.values
                    y = dataset.y.values
                    fit_params = {'categorical': dataset.categorical}

                elif model_name == "LightGBMLinear":

                    X = dataset.X_oh_encoded
                    y = dataset.y

                else:
                    X = dataset.X_oh_encoded
                    y = dataset.y

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            grid.fit(X_train, y_train, **fit_params)
            best_model = grid.best_estimator_
            best_params = grid.best_params_
            flat_params = {f"param_{k}": v for k, v in best_params.items()}

            y_pred = best_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            results.append({
                "repo_id": repo_id,
                "model": model_name,
                "mse": mse,
                **flat_params
            })


            # best_model.fit(X,y, **fit_params)
            # model_path = f"retrained_models/{repo_id}_{model_name.lower()}_best_retrained_model.pkl"
            # joblib.dump(best_model, model_path)
            # print(f"Saved {model_name} model to: {model_path}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(experiment_file, index=False)
    print(f"Results saved to {experiment_file}")


run_benchmark("pilot_iii")
