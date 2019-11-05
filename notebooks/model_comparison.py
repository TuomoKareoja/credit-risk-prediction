# %%

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikitplot as skplt
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator

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


# %%

# Average CV AUC-score on the training set was:0.6192352078198122
auc_pipeline = make_pipeline(
    StackingEstimator(
        estimator=GradientBoostingClassifier(
            learning_rate=0.1,
            max_depth=1,
            max_features=0.9000000000000001,
            min_samples_leaf=7,
            min_samples_split=16,
            n_estimators=100,
            subsample=0.25,
        )
    ),
    LogisticRegression(C=1.0, dual=False, penalty="l1"),
)

# %%

# Average CV logloss score on the training set was:-0.5174687874907155
logloss_pipeline = make_pipeline(
    StackingEstimator(estimator=LogisticRegression(C=1.0, dual=True, penalty="l2")),
    StackingEstimator(estimator=BernoulliNB(alpha=100.0, fit_prior=True)),
    StackingEstimator(estimator=MultinomialNB(alpha=1.0, fit_prior=True)),
    LogisticRegression(C=5.0, dual=False, penalty="l1"),
)

# %%

X = pd.concat([X_train, X_test])
y = pd.concat([y_train, y_test])

auc_pipeline.fit(X, y)
logloss_pipeline.fit(X, y)
no_def_pred_proba = np.array([[1, 0] for i in range(len(X))])
auc_pred_proba = auc_pipeline.predict_proba(X)
logloss_pred_proba = logloss_pipeline.predict_proba(X)
no_def_pred = np.zeros(len(X))
auc_pred = auc_pipeline.predict(X)
logloss_pred = logloss_pipeline.predict(X)

# %%

fig, ax = plt.subplots()
skplt.metrics.plot_roc(
    y,
    auc_pred_proba,
    ax=ax,
    plot_micro=False,
    plot_macro=False,
    classes_to_plot=[1],
    cmap="twilight",
)
skplt.metrics.plot_roc(
    y,
    logloss_pred_proba,
    ax=ax,
    plot_micro=False,
    plot_macro=False,
    classes_to_plot=[1],
    cmap="hot",
)
ax.legend(["auc_model", "guessing", "logloss_model"])
plt.show()

# %%

fig, ax = plt.subplots()
skplt.metrics.plot_precision_recall(
    y, auc_pred_proba, ax=ax, classes_to_plot=[1], cmap="twilight", plot_micro=False
)
skplt.metrics.plot_precision_recall(
    y, logloss_pred_proba, ax=ax, classes_to_plot=[1], cmap="hot", plot_micro=False
)
skplt.metrics.plot_precision_recall(
    y, no_def_pred_proba, ax=ax, classes_to_plot=[1], cmap="Greens", plot_micro=False
)
ax.legend(["auc_model", "logloss_model", "no_defaults"])
plt.show()

# %%

skplt.metrics.plot_ks_statistic(y, auc_pred_proba, title="auc_model")
skplt.metrics.plot_ks_statistic(y, logloss_pred_proba, title="logloss_model")
plt.show()


# %%

skplt.metrics.plot_confusion_matrix(y, auc_pred, title="auc_model")
skplt.metrics.plot_confusion_matrix(y, logloss_pred, title="logloss_model")
skplt.metrics.plot_confusion_matrix(y, no_def_pred, title="no_defaults")
plt.show()

# %%

probs_list = [auc_pred_proba, logloss_pred_proba]
model_names = ["auc_model", "logloss_model"]

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle="--")
for probs, model in zip(probs_list, model_names):
    fop, mpv = calibration_curve(y, probs[:, 1], n_bins=10, normalize=False)
    ax.plot(mpv, fop, linewidth=3, marker=".", markersize=18, label=model)
plt.legend()
plt.title("Calibration plot")
plt.show()

# %%

skplt.metrics.plot_lift_curve(y, auc_pred_proba, title="auc_model")
skplt.metrics.plot_lift_curve(y, logloss_pred_proba, title="logloss_model")
plt.show()

# %%
