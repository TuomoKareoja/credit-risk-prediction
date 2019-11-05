# TODO: create dataset with all combinations of demographic values with age in 5 year
# brackets

# TODO: for each of this groups give them max_credit from min to min in 1000 intervals

# TODO: calculate proportion of customers that fit into each group and combine this
# to the dataset

# %%

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from IPython.core.interactiveshell import InteractiveShell
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Setting styles
InteractiveShell.ast_node_interactivity = "all"
sns.set(style="whitegrid", color_codes=True, rc={"figure.figsize": (12.7, 9.27)})

random_state = 123

# %% [markdown]

# # Using TPOT to automatically generate data pipeline and optimize a model
# %%

df = pd.read_csv(os.path.join("data", "processed", "training.csv"), index_col="ID")

# Keeping only columns that credit rating agency would surely have
cols = ["credit_given", "sex", "education", "marriage", "age", "defaulted"]
df = df[cols]
# tpot can't handle missing values or non numeric columns
df.dropna(inplace=True)
df = pd.get_dummies(df, prefix=["sex", "education", "marriage"])

# %%

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns="defaulted"),
    df["defaulted"],
    test_size=0.2,
    random_state=random_state,
)

X = pd.concat([X_train, X_test])
y = pd.concat([y_train, y_test])


# %%

# Average CV AUC-score on the training set was:0.6192352078198122
gb_model = GradientBoostingClassifier(
    learning_rate=0.1,
    max_depth=1,
    max_features=0.9000000000000001,
    min_samples_leaf=7,
    min_samples_split=16,
    n_estimators=100,
    subsample=0.25,
    random_state=random_state,
)

xgb_model = XGBClassifier(
    max_depth=1,
    learning_rate=0.1,
    n_estimators=100,
    verbosity=1,
    n_jobs=3,
    min_child_weight=1,
    max_delta_step=0,
    subsample=1,
    colsample_bytree=1,
    colsample_bylevel=1,
    colsample_bynode=1,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    base_score=0.5,
    random_state=random_state,
)


# %%

gb_model.fit(X, y)
gb_pred = gb_model.predict_proba(X)
gb_pred_08 = gb_model.predict_proba(X.assign(credit_given=X["credit_given"] * 0.8))
gb_pred_09 = gb_model.predict_proba(X.assign(credit_given=X["credit_given"] * 0.9))
gb_pred_11 = gb_model.predict_proba(X.assign(credit_given=X["credit_given"] * 1.1))
gb_pred_12 = gb_model.predict_proba(X.assign(credit_given=X["credit_given"] * 1.2))

# %%

xgb_model.fit(X, y)
xgb_pred = xgb_model.predict_proba(X)
xgb_pred_08 = xgb_model.predict_proba(X.assign(credit_given=X["credit_given"] * 0.8))
xgb_pred_09 = xgb_model.predict_proba(X.assign(credit_given=X["credit_given"] * 0.9))
xgb_pred_11 = xgb_model.predict_proba(X.assign(credit_given=X["credit_given"] * 1.1))
xgb_pred_12 = xgb_model.predict_proba(X.assign(credit_given=X["credit_given"] * 1.2))


# %%

probs_list = [gb_pred, xgb_pred]
model_names = ["gradient boosting model", "extreme gradient boosting model"]

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle="--")
for probs, model in zip(probs_list, model_names):
    fop, mpv = calibration_curve(y, probs[:, 1], n_bins=20, normalize=False)
    ax.plot(mpv, fop, linewidth=3, marker=".", markersize=18, label=model)
plt.legend()
plt.title("Calibration plot")
plt.show()


# %%

sns.distplot(gb_pred[:, 1] - gb_pred_08[:, 1], hist=False, label="credit given 80 %")
sns.distplot(gb_pred[:, 1] - gb_pred_09[:, 1], hist=False, label="credit given 90 %")
sns.distplot(gb_pred[:, 1] - gb_pred_11[:, 1], hist=False, label="credit given 110 %")
sns.distplot(gb_pred[:, 1] - gb_pred_12[:, 1], hist=False, label="credit given 120 %")
plt.show()

# %%

sns.distplot(xgb_pred[:, 1] - xgb_pred_08[:, 1], hist=False, label="credit given 80 %")
sns.distplot(xgb_pred[:, 1] - xgb_pred_09[:, 1], hist=False, label="credit given 90 %")
sns.distplot(xgb_pred[:, 1] - xgb_pred_11[:, 1], hist=False, label="credit given 110 %")
sns.distplot(xgb_pred[:, 1] - xgb_pred_12[:, 1], hist=False, label="credit given 120 %")
plt.legend()
plt.show()


# %%

high_risk_credit_df = X[gb_pred[:, 1] - gb_pred_09[:, 1] < -0.05]
high_risk_credit_df.head()
high_risk_credit_df.describe()

# %% Trying out Shapley values

# load JS visualization code to notebook
shap.initjs()

# Use Kernel SHAP to explain test set predictions

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X)
# visualize the training set predictions
shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])
shap.force_plot(explainer.expected_value, shap_values[:200, :], X.iloc[:200, :])

# %%
