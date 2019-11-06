# %% [markdown]

# # Finding Customers Where We Should Limit Credit Limit

# %% Loading packages
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scikitplot as skplt
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_predict
from xgboost import XGBClassifier

# Setting styles
InteractiveShell.ast_node_interactivity = "all"
sns.set(style="whitegrid", color_codes=True, rc={"figure.figsize": (12.7, 9.27)})

random_state = 123

# %% Load data

df = pd.read_csv(os.path.join("data", "processed", "training.csv"), index_col="ID")

# Keeping only columns that credit rating agency would surely have
cols = ["credit_given", "sex", "education", "marriage", "age", "defaulted"]
df = df[cols]
# There are a few missing values and dropping them makes other stuff easier
# without affecting the amount of data much
df.dropna(inplace=True)
# many of our models can only handle numerical values so dummifying
df = pd.get_dummies(df, prefix=["sex", "education", "marriage"])


X = df.drop(columns="defaulted")
y = df["defaulted"]

# %% [markdown]

# ## Credit Limit Affects the Default Probability

# %%

xgb_credit_limit = XGBClassifier(random_state=random_state)
xgb_credit_limit.fit(X, y)
xgb_credit_limit_pred = xgb_credit_limit.predict_proba(X)

xgb_no_credit_limit = XGBClassifier(random_state=random_state)
xgb_no_credit_limit.fit(X.drop(columns="credit_given"), y)
xgb_no_credit_limit_pred = xgb_no_credit_limit.predict_proba(
    X.drop(columns="credit_given")
)

fig, ax = plt.subplots()
skplt.metrics.plot_roc(
    y,
    xgb_credit_limit_pred,
    plot_micro=False,
    plot_macro=False,
    classes_to_plot=[1],
    ax=ax,
    cmap="Blues",
)
skplt.metrics.plot_roc(
    y,
    xgb_no_credit_limit_pred,
    plot_micro=False,
    plot_macro=False,
    classes_to_plot=[1],
    ax=ax,
    cmap="Reds",
)
plt.legend(["Model With Credit Limit", "Random Guess", "Model Without Credit Limit"])
plt.show()

# %% [markdown]

# # Training Multiple Models to Predict Default
#
# * We try training multiple models and keep the best
# * Model need to have good lift for its most extreme prediction, but not the whole way
# (we only need to find a small fraction of the defaulters to make things better)

# Visualize how the models create their predictions with shap
# Create out of sample predictions with different levels of credit

# %%

cv = 5

xgb_model = XGBClassifier(random_state=random_state)
# we care that the predicted probabilites are well tuned and we need to
# adjust this separately for random forests
random_forest_model = CalibratedClassifierCV(
    RandomForestClassifier(random_state=random_state), cv=5
)

xgb_pred = cross_val_predict(xgb_model, X, y, cv=cv, method="predict_proba")
random_forest_pred = cross_val_predict(
    random_forest_model, X, y, cv=cv, method="predict_proba"
)

# %% Combination model

comb_pred_class1 = np.mean([xgb_pred[:, 1], random_forest_pred[:, 1]], axis=0)
comb_pred_class0 = 1 - comb_pred_class1
comb_pred = np.array(
    [[class0, class1] for class0, class1 in zip(comb_pred_class0, comb_pred_class1)]
)

# %% ROC

fig, ax = plt.subplots()
skplt.metrics.plot_roc(
    y,
    xgb_pred,
    plot_micro=False,
    plot_macro=False,
    classes_to_plot=[1],
    ax=ax,
    cmap="Blues",
)
skplt.metrics.plot_roc(
    y,
    random_forest_pred,
    plot_micro=False,
    plot_macro=False,
    classes_to_plot=[1],
    ax=ax,
    cmap="Reds",
)
skplt.metrics.plot_roc(
    y,
    comb_pred,
    plot_micro=False,
    plot_macro=False,
    classes_to_plot=[1],
    ax=ax,
    cmap="Greens",
)
plt.show()

# %% Lift

skplt.metrics.plot_lift_curve(y, xgb_pred, title="XGBoost Lift Curve")
skplt.metrics.plot_lift_curve(y, random_forest_pred, title="Random Forest Lift Curve")
skplt.metrics.plot_lift_curve(y, comb_pred, title="Combination Model Lift Curve")
plt.show()


# %% Tuning of probabilities

xgb_fop, xgb_mpv = calibration_curve(y, xgb_pred[:, 1], normalize=False, n_bins=30)
random_forest_fop, random_forest_mpv = calibration_curve(
    y, random_forest_pred[:, 1], normalize=False, n_bins=30
)
comb_fop, comb_mpv = calibration_curve(y, comb_pred[:, 1], normalize=False, n_bins=30)

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
ax.plot(
    comb_mpv,
    comb_fop,
    linewidth=3,
    marker=".",
    markersize=18,
    label="Combination Model",
)
plt.legend()
plt.title("Calibration plot")
plt.show()

fig, ax = plt.subplots()
sns.distplot(xgb_pred[:, 1], ax=ax, label="XGBoost")
sns.distplot(random_forest_pred[:, 1], ax=ax, label="Random Forest")
sns.distplot(comb_pred[:, 1], ax=ax, label="Combination Model")
plt.title("Distribution of Out of Sample Predicted Probabilities")
plt.legend()
plt.show()


# %% Calculating effects of limit credit

xgb_model.fit(X, y)
xgb_pred_08 = xgb_model.predict_proba(X.assign(credit_given=X["credit_given"] * 0.8))
xgb_pred_085 = xgb_model.predict_proba(X.assign(credit_given=X["credit_given"] * 0.85))
xgb_pred_09 = xgb_model.predict_proba(X.assign(credit_given=X["credit_given"] * 0.9))
xgb_pred_095 = xgb_model.predict_proba(X.assign(credit_given=X["credit_given"] * 0.95))
xgb_pred_11 = xgb_model.predict_proba(X.assign(credit_given=X["credit_given"] * 1.1))
sns.distplot(xgb_pred[:, 1] - xgb_pred_08[:, 1], hist=False, label="credit given 80 %")
sns.distplot(xgb_pred[:, 1] - xgb_pred_085[:, 1], hist=False, label="credit given 85 %")
sns.distplot(xgb_pred[:, 1] - xgb_pred_09[:, 1], hist=False, label="credit given 90 %")
sns.distplot(xgb_pred[:, 1] - xgb_pred_095[:, 1], hist=False, label="credit given 95 %")
sns.distplot(xgb_pred[:, 1] - xgb_pred_11[:, 1], hist=False, label="credit given 110 %")
plt.legend()
plt.show()

# %%

results = pd.DataFrame(
    {
        "defaulted": y,
        "credit_given": X["credit_given"],
        "prob": xgb_pred[:, 1],
        "prob_loss95": np.where(
            xgb_pred[:, 1] - xgb_pred_095[:, 1] <= 0,
            np.nan,
            (xgb_pred[:, 1] - xgb_pred_095[:, 1]) * 100,
        ),
        "prob_loss90": np.where(
            xgb_pred[:, 1] - xgb_pred_09[:, 1] <= 0,
            np.nan,
            (xgb_pred[:, 1] - xgb_pred_09[:, 1]) * 100,
        ),
        "prob_loss85": np.where(
            xgb_pred[:, 1] - xgb_pred_085[:, 1] <= 0,
            np.nan,
            (xgb_pred[:, 1] - xgb_pred_085[:, 1]) * 100,
        ),
        "prob_loss80": np.where(
            xgb_pred[:, 1] - xgb_pred_08[:, 1] <= 0,
            np.nan,
            (xgb_pred[:, 1] - xgb_pred_08[:, 1]) * 100,
        ),
        "credit_loss95": X["credit_given"] - X["credit_given"] * 0.95,
        "credit_loss90": X["credit_given"] - X["credit_given"] * 0.9,
        "credit_loss85": X["credit_given"] - X["credit_given"] * 0.85,
        "credit_loss80": X["credit_given"] - X["credit_given"] * 0.8,
    }
)

# if giving less debt actually

results = results.assign(
    prob_change95=results.prob_loss95,
    prob_change90=np.where(
        results.prob_loss90 - results.prob_loss95 <= 0,
        np.nan,
        results.prob_loss90 - results.prob_loss95,
    ),
    prob_change85=np.where(
        results.prob_loss85 - results.prob_loss90 <= 0,
        np.nan,
        results.prob_loss85 - results.prob_loss90,
    ),
    prob_change80=np.where(
        results.prob_loss80 - results.prob_loss85 <= 0,
        np.nan,
        results.prob_loss80 - results.prob_loss85,
    ),
)

results = results.assign(
    credit_change95=results.credit_loss95,
    credit_change90=results.credit_loss90 - results.credit_loss95,
    credit_change85=results.credit_loss85 - results.credit_loss90,
    credit_change80=results.credit_loss80 - results.credit_loss85,
)

results = results.assign(
    cost_95_prob=results.credit_change95 / results.prob_change95,
    cost_90_prob=results.credit_change90 / results.prob_change90,
    cost_85_prob=results.credit_change85 / results.prob_change85,
    cost_80_prob=results.credit_change80 / results.prob_change80,
)


# %%

# mean of y and prob should be closely aligned
results.describe()
results.head(50)

# %% trying what we get with just 5 % credit lowering option

# results.sort_values(by="cost_95_prob", inplace=True)
# results.reset_index(inplace=True)
defaults_prevented_perc = results.prob_loss95.cumsum() / results.prob.sum()
customers_affected_perc = (results.index + 1) * 100 / len(results)
credit_cost_perc = results.credit_loss95.cumsum() * 100 / results.credit_given.sum()


# %%

plt.plot(defaults_prevented_perc, label="defaults")
plt.plot(customers_affected_perc, label="customers affected")
plt.plot(credit_cost_perc, label="credit cost")
plt.axis([0, 12000, 0, 30])
plt.legend()
plt.show()

# %%


# CALCULATE EFFECTS
# Calculate the average drop in probs between models for each drop
# Calculate the absolute drop in credit for each drop
# Add the probs of default from original models as an average
# Make sure that the probs are nicely align with actual number of defaults

# customer1, prob of default, original_credit, drop1_prob change, drop1_credit_change...

# Create function that optimizes range of prob drops with minimal credit drop

# for each solution plot loss credit percent and drop in default rate
# also include the variation of predicted defaulting for people
# who would lose credit
# AND the variation of percent credit lost per model
# and the percent of customer who would lose credit

# visualize for one solution who would lose credit
# Visualize the correlation between risk and losing of credit

# what is the cost of lost credit and what is the cost of default


# %%
