# %% Loading packages
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from bayes_opt import BayesianOptimization
from catboost import CatBoostClassifier
from IPython.core.interactiveshell import InteractiveShell
from scikitplot.metrics import plot_roc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier

# Setting styles
InteractiveShell.ast_node_interactivity = "all"
sns.set(style="whitegrid", color_codes=True, rc={"figure.figsize": (12.7, 9.27)})

random_state = 123
init_points = 10
n_iter = 30

# %% Load data

df = pd.read_csv(os.path.join("data", "processed", "training.csv"), index_col="ID")
df_validation = pd.read_csv(
    os.path.join("data", "processed", "validation.csv"), index_col="ID"
)

# Keeping only columns that credit rating agency would surely have
cols = ["credit_given", "sex", "education", "marriage", "age", "defaulted"]
df = df[cols]
df_validation = df_validation[cols]
# There are a few missing values and dropping them makes other stuff easier
# without affecting the amount of data much
df.dropna(inplace=True)
df_validation.dropna(inplace=True)
# many of our models can only handle numerical values so dummifying
df = pd.get_dummies(df, prefix=["sex", "education", "marriage"])
df_validation = pd.get_dummies(df_validation, prefix=["sex", "education", "marriage"])

X = df.drop(columns="defaulted")
y = df["defaulted"]

X_validation = df_validation.drop(columns="defaulted")
y_validation = df_validation["defaulted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)

# %%


def xgb_optimization(gamma, max_depth, cv=5):
    score = cross_val_score(
        XGBClassifier(
            objective="binary:logistic",
            gamma=max(gamma, 0),
            max_depth=int(max_depth),
            seed=random_state,
            nthread=-1,
        ),
        X=X_train,
        y=y_train,
        cv=cv,
        scoring="roc_auc",
        fit_params={
            "early_stopping_rounds": 10,
            "eval_metric": "auc",
            "eval_set": [(X_train, y_train), (X_test, y_test)],
        },
        n_jobs=-1,
    ).mean()

    return score


def rf_optimization(max_depth, min_samples_split, cv=5):
    score = cross_val_score(
        RandomForestClassifier(
            max_depth=int(max(max_depth, 1)),
            min_samples_split=int(max(min_samples_split, 2)),
            n_jobs=-1,
            random_state=random_state,
            class_weight="balanced",
        ),
        X=X_train,
        y=y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    ).mean()

    return score


def cat_optimization(depth, l2_leaf_reg, cv=5):
    score = cross_val_score(
        CatBoostClassifier(
            l2_leaf_reg=l2_leaf_reg,
            depth=int(depth),
            loss_function="Logloss",
            verbose=True,
            random_state=random_state,
        ),
        X=X_train,
        y=y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    ).mean()

    return score


def bayesian_optimization(function, parameters, n_iter, init_points):
    BO = BayesianOptimization(f=function, pbounds=parameters, random_state=random_state)
    BO.maximize(init_points=init_points, n_iter=n_iter)

    return BO.max


# %%

# can't add more parameters without changing the xgb_optimization function
xgb_parameters = {"gamma": (0, 20), "max_depth": (1, 2000)}

xgb_best_solution = bayesian_optimization(
    function=xgb_optimization,
    parameters=xgb_parameters,
    n_iter=n_iter,
    init_points=init_points,
)

# %%

# can't add more parameters without changing the optimization function
rf_parameters = {"max_depth": (1, 150), "min_samples_split": (2, 10)}

rf_best_solution = bayesian_optimization(
    function=rf_optimization,
    parameters=rf_parameters,
    n_iter=n_iter,
    init_points=init_points,
)

# %%

# can't add more parameters without changing the optimization function
cat_parameters = {"depth": (4, 10), "l2_leaf_reg": (2, 4)}

cat_best_solution = bayesian_optimization(
    function=cat_optimization,
    parameters=cat_parameters,
    n_iter=n_iter,
    init_points=init_points,
)


# %%

params = xgb_best_solution["params"]
xgb_model = XGBClassifier(
    objective="binary:logistic",
    gamma=int(max(params["gamma"], 0)),
    max_depth=int(max(params["max_depth"], 1)),
    seed=random_state,
    nthread=-1,
)


xgb_model.fit(X, y)

xgb_pred_train = xgb_model.predict_proba(X)
xgb_pred_validation = xgb_model.predict_proba(X_validation)

# %%

params = rf_best_solution["params"]
rf_model = RandomForestClassifier(
    max_depth=int(max(params["max_depth"], 1)),
    min_samples_split=int(max(params["min_samples_split"], 2)),
    n_jobs=-1,
    random_state=random_state,
    class_weight="balanced",
)

rf_model.fit(X, y)

rf_pred_train = rf_model.predict_proba(X)
rf_pred_validation = rf_model.predict_proba(X_validation)

# %%

params = cat_best_solution["params"]
cat_model = CatBoostClassifier(
    l2_leaf_reg=params["l2_leaf_reg"],
    depth=int(params["depth"]),
    loss_function="Logloss",
    verbose=False,
    random_state=random_state,
)

cat_model.fit(X, y)

cat_pred_train = cat_model.predict_proba(X)
cat_pred_validation = cat_model.predict_proba(X_validation)

# %%


fig, ax = plt.subplots()
plot_roc(
    y,
    xgb_pred_train,
    plot_micro=False,
    plot_macro=False,
    title="ROC Curves",
    classes_to_plot=[1],
    ax=ax,
    figsize=None,
    cmap="Reds",
)
plot_roc(
    y_validation,
    xgb_pred_validation,
    plot_micro=False,
    plot_macro=False,
    classes_to_plot=[1],
    ax=ax,
    figsize=None,
    cmap="Reds",
)
plot_roc(
    y,
    rf_pred_train,
    plot_micro=False,
    plot_macro=False,
    classes_to_plot=[1],
    ax=ax,
    figsize=None,
    cmap="Blues",
)
plot_roc(
    y_validation,
    rf_pred_validation,
    plot_micro=False,
    plot_macro=False,
    classes_to_plot=[1],
    ax=ax,
    figsize=None,
    cmap="Blues",
)
plot_roc(
    y,
    cat_pred_train,
    plot_micro=False,
    plot_macro=False,
    classes_to_plot=[1],
    ax=ax,
    figsize=None,
    cmap="Greens",
)
plot_roc(
    y_validation,
    cat_pred_validation,
    plot_micro=False,
    plot_macro=False,
    classes_to_plot=[1],
    ax=ax,
    figsize=None,
    cmap="Greens",
)
plt.show()

# %%

with open(os.path.join("models", "xgboost_best_params.txt"), "w") as output_file:
    print(xgb_best_solution, file=output_file)

with open(os.path.join("models", "rf_best_params.txt"), "w") as output_file:
    print(rf_best_solution, file=output_file)

with open(os.path.join("models", "cat_best_params.txt"), "w") as output_file:
    print(cat_best_solution, file=output_file)
