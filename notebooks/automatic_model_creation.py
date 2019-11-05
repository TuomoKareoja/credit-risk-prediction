# %%

import os

import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

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

pipeline_optimizer_logloss = TPOTClassifier(
    generations=20,
    population_size=10,
    cv=3,
    random_state=random_state,
    verbosity=2,
    scoring="neg_log_loss",
    n_jobs=4,
    memory="auto",
)

pipeline_optimizer_auc = TPOTClassifier(
    generations=20,
    population_size=10,
    cv=3,
    random_state=random_state,
    verbosity=2,
    scoring="roc_auc",
    n_jobs=4,
    memory="auto",
)

# %%

pipeline_optimizer_logloss.fit(X_train, y_train)
pipeline_optimizer_logloss.export(os.path.join("models", "tpot_exported_logloss.py"))

# %%

pipeline_optimizer_auc.fit(X_train, y_train)
pipeline_optimizer_auc.export(os.path.join("models", "tpot_exported_auc.py"))
