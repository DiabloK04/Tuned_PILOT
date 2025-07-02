import joblib
from paperplots import *


model = joblib.load("../retrained_models/186_pilot_base_best_retrained_model.pkl")
cart_model = joblib.load("../retrained_models/186_cart_best_retrained_model.pkl")
repo_id = 186
dataset = joblib.load(f"../Data/{repo_id}.pkl")

X = dataset.X_label_encoded
y = dataset.y

plot_pilot_feature_importance(model, X, 0)
plot_cart_feature_importance(cart_model, 0)

plot_compare_cart_pilot(
    cart_model,
    model.model_tree,
    X,
    y,
)

model = joblib.load("../retrained_models/183_pilot_base_best_retrained_model.pkl")
cart_model = joblib.load("../retrained_models/183_cart_best_retrained_model.pkl")
repo_id = 183
dataset = joblib.load(f"../Data/{repo_id}.pkl")

X = dataset.X_label_encoded
y = dataset.y

plot_pilot_feature_importance(model, X, 0.01)
plot_cart_feature_importance(cart_model, 0.01)

plot_pilot_root(
    model_tree=model.model_tree,
    X_train=X,
    y_train= y,
)
plot_cart_root_red(cart_model, X, y)



model = joblib.load("../retrained_models/999_pilot_base_best_retrained_model.pkl")
cart_model = joblib.load("../retrained_models/999_cart_best_retrained_model.pkl")

dataset = pd.read_csv("../DataCsv/Admission_Predict.csv")

X = dataset.drop(columns=["Serial No.", "Chance of Admit "])
y = dataset["Chance of Admit "]

plot_compare_cart_pilot(
    cart_model,
    model.model_tree,
    X,
    y,
)