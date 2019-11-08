# %% [markdown]

# # Finding Customers Where We Should Limit Credit Limit
#
# Lowering the credit limit for customers that have too high credit limits
# should mean that the number of defaults goes down


# %% Loading packages
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikitplot as skplt
import seaborn as sns
import shap
from catboost import CatBoostClassifier
from IPython.core.interactiveshell import InteractiveShell
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.optimization.credit_limit import (
    calculate_effect_of_credit_drop,
    form_results_to_ordered_df,
    order_effects_within_customers,
)

# Setting styles
InteractiveShell.ast_node_interactivity = "all"
sns.set(style="whitegrid", color_codes=True, rc={"figure.figsize": (12.7, 9.27)})

random_state = 123

# %% Load data

df = pd.read_csv(os.path.join("data", "processed", "training.csv"), index_col="ID")
df_validation = pd.read_csv(
    os.path.join("data", "processed", "validation.csv"), index_col="ID"
)

# %%

# Keeping only columns that don't contain absolute pay figures except for
# credit given (trying to give it some meaning)

cols = [
    "credit_given",
    "sex",
    "education",
    "marriage",
    "age",
    "defaulted",
    "perc_credit_used1",
    "perc_credit_used2",
    "perc_credit_used3",
    "perc_credit_used4",
    "perc_credit_used5",
    "perc_credit_used6",
    "perc_credit_change1",
    "perc_credit_change2",
    "perc_credit_change3",
    "perc_credit_change4",
    "perc_credit_change5",
]


df = df[cols]
df_validation = df_validation[cols]

# There are a few missing values and dropping them makes other stuff easier
# without affecting the amount of data much
df.dropna(inplace=True)
df_validation.dropna(inplace=True)
# many of our models can only handle numerical values so dummifying
df = pd.get_dummies(df, prefix=["sex", "education", "marriage"])
df_validation = pd.get_dummies(df_validation, prefix=["sex", "education", "marriage"])

X_train = df.drop(columns="defaulted")
y_train = df["defaulted"]

X_validation = df_validation.drop(columns="defaulted")
y_validation = df_validation["defaulted"]

# %% [markdown]

# ## Credit Limit Affects the Default Probability
#
# * If we drop credit limit from the features the power of the model suffers a lot
# * Shapley values also show that the highest influences (wide spread) in the
# predictions from low and high credit limits

# %%

xgb_credit_limit = XGBClassifier(random_state=random_state, objective="binary:logistic")
xgb_credit_limit.fit(X_train, y_train)
xgb_credit_limit_pred = xgb_credit_limit.predict_proba(X_train)

xgb_no_credit_limit = XGBClassifier(
    random_state=random_state, objective="binary:logistic"
)
xgb_no_credit_limit.fit(X_train.drop(columns="credit_given"), y_train)
xgb_no_credit_limit_pred = xgb_no_credit_limit.predict_proba(
    X_train.drop(columns="credit_given")
)

fig, ax = plt.subplots()
skplt.metrics.plot_roc(
    y_train,
    xgb_credit_limit_pred,
    plot_micro=False,
    plot_macro=False,
    classes_to_plot=[1],
    ax=ax,
    cmap="Blues",
)
skplt.metrics.plot_roc(
    y_train,
    xgb_no_credit_limit_pred,
    plot_micro=False,
    plot_macro=False,
    classes_to_plot=[1],
    ax=ax,
    cmap="Reds",
)
plt.legend(["Model With Credit Limit", "Random Guess", "Model Without Credit Limit"])
plt.show()

# load JS visualization code to notebook
shap.initjs()

explainer = shap.TreeExplainer(xgb_credit_limit)
shap_values = explainer.shap_values(X_train)

# visualize the training set predictions
shap.summary_plot(shap_values, X_train)

# %% [markdown]

# # Training Multiple Models to Predict Default
#
# * We try training extreme gradient boosting, random forest and catboost
# * Model need to have good lift for its most extreme prediction, but not the whole way
# (we only need to find a small fraction of the defaulters to make things better)

# Visualize how the models create their predictions with shap
# Create out of sample predictions with different levels of credit

# %%

# hyperparameters come from a separate tuning phase
xgb_model = XGBClassifier(
    random_state=random_state,
    objective="binary:logistic",
    gamma=20,
    max_depth=2,
    seed=random_state,
    nthread=-1,
)

# we care that the predicted probabilites are well tuned and we need to
# adjust this separately for random forests
rf_model = CalibratedClassifierCV(
    RandomForestClassifier(
        random_state=random_state,
        max_depth=1,
        min_samples_split=9,
        n_estimators=1000,
        n_jobs=-1,
        class_weight="balanced",
    ),
    cv=5,
)

cat_model = CatBoostClassifier(
    l2_leaf_reg=3,
    depth=4,
    loss_function="Logloss",
    verbose=False,
    random_state=random_state,
)


# %%

xgb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
cat_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict_proba(X_validation)
rf_pred = rf_model.predict_proba(X_validation)
cat_pred = cat_model.predict_proba(X_validation)

# %% ROC

fig, ax = plt.subplots()
skplt.metrics.plot_roc(
    y_validation,
    xgb_pred,
    plot_micro=False,
    plot_macro=False,
    classes_to_plot=[1],
    ax=ax,
    cmap="Blues",
)
skplt.metrics.plot_roc(
    y_validation,
    rf_pred,
    plot_micro=False,
    plot_macro=False,
    classes_to_plot=[1],
    ax=ax,
    cmap="Reds",
)
skplt.metrics.plot_roc(
    y_validation,
    cat_pred,
    plot_micro=False,
    plot_macro=False,
    classes_to_plot=[1],
    ax=ax,
    cmap="Greens",
)
plt.show()

# %% Lift

skplt.metrics.plot_lift_curve(y_validation, xgb_pred, title="XGBoost Lift Curve")
skplt.metrics.plot_lift_curve(y_validation, rf_pred, title="Random Forest Lift Curve")
skplt.metrics.plot_lift_curve(y_validation, cat_pred, title="CatBoost Lift Curve")
plt.show()


# %% Checking that probabilities are tuned

xgb_fop, xgb_mpv = calibration_curve(
    y_validation, xgb_pred[:, 1], normalize=False, n_bins=30
)
random_forest_fop, random_forest_mpv = calibration_curve(
    y_validation, rf_pred[:, 1], normalize=False, n_bins=30
)
cat_fop, cat_mpv = calibration_curve(
    y_validation, cat_pred[:, 1], normalize=False, n_bins=30
)

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle="--")
ax.plot(xgb_mpv, xgb_fop, linewidth=3, marker=".", markersize=18, label="XGBoost")
ax.plot(
    random_forest_mpv,
    random_forest_fop,
    linewidth=3,
    marker=".",
    markersize=18,
    label="Random Forest",
)
ax.plot(cat_mpv, cat_fop, linewidth=3, marker=".", markersize=18, label="CatBoost")
plt.legend()
plt.title("Calibration plot")
plt.show()

fig, ax = plt.subplots()
sns.distplot(xgb_pred[:, 1], ax=ax, label="XGBoost")
sns.distplot(rf_pred[:, 1], ax=ax, label="Random Forest")
sns.distplot(cat_pred[:, 1], ax=ax, label="CatBoost")
plt.title("Distribution of Out of Sample Predicted Probabilities")
plt.legend()
plt.show()


# %% Calculating effects of limit credit

# %%

models = [xgb_model, rf_model, cat_model]
model_names = ["XGBoost", "Random Forest", "CatBoost"]
credit_limit_factors = np.around(np.arange(0.7, 0.90, 0.1), 2)


for model_name, model in zip(model_names, models):

    (
        all_probs_changes,
        all_credit_amount_changes,
        all_expected_costs_in_credit,
    ) = calculate_effect_of_credit_drop(
        model=model, X=X_validation, credit_factors=credit_limit_factors
    )

    (
        all_processed_costs,
        all_processed_factors,
        all_processed_credit_changes,
        all_processed_probs_changes,
    ) = order_effects_within_customers(
        X=X_validation,
        credit_factors=credit_limit_factors,
        all_probs_changes=all_probs_changes,
        all_credit_amount_changes=all_credit_amount_changes,
        all_expected_costs_in_credit=all_expected_costs_in_credit,
    )

    costs_df = form_results_to_ordered_df(
        y=y_validation,
        X=X_validation,
        probs=model.predict_proba(X_validation)[:, 1],
        all_processed_costs=all_processed_costs,
        all_processed_factors=all_processed_factors,
        all_processed_credit_changes=all_processed_credit_changes,
        all_processed_probs_changes=all_processed_probs_changes,
    )

    plt.plot(costs_df.defaults_prevented_perc, label="Defaults avoided")
    plt.plot(costs_df.credit_cost_perc, label="Cost in credit")
    # plt.xlim([0, 1000])
    # plt.ylim([0, 0.25])
    plt.title(model_name + ": Defaults Avoided vs Defaults Cost in Credit")
    plt.xlabel("Number of Steps of Credit Limiting")
    plt.ylabel("%")
    plt.legend()
    plt.show()

    plt.plot(costs_df.customers_affected_perc, label="Customers affected")
    plt.plot(costs_df.defaulters_affected_perc, label="Defaulters affected")
    plt.plot(costs_df.non_defaulters_affected_perc, label="Non-defaulters affected")
    # plt.xlim([0, 1000])
    # plt.ylim([0, 25])
    plt.title(model_name + ": Proportion of Different Types of Customers affected")
    plt.xlabel("Number of Steps of Credit Limiting")
    plt.ylabel("%")
    plt.legend()
    plt.show()


# %%
