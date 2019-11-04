#%%

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from dotenv import find_dotenv, load_dotenv
from IPython.core.interactiveshell import InteractiveShell
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot import TPOTClassifier
from tpot.builtins import StackingEstimator

# Setting styles
InteractiveShell.ast_node_interactivity = "all"
sns.set(style="whitegrid", color_codes=True, rc={"figure.figsize": (12.7, 9.27)})

random_state = 123

#%% [markdown]

# # Using TPOT to automatically generate data pipeline and optimize a model
#%%

df = pd.read_csv(os.path.join("data", "processed", "training.csv"), index_col="ID")

# Keeping only columns that credit rating agency would surely have
cols = ["credit_given", "sex", "education", "marriage", "age", "defaulted"]
df = df[cols]
# tpot can't handle missing values or non numeric columns
df.dropna(inplace=True)
df = pd.get_dummies(df, prefix=["sex", "education", "marriage"])

#%%

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns="defaulted"),
    df["defaulted"],
    test_size=0.2,
    random_state=random_state,
)

#%%

pipeline_optimizer = TPOTClassifier()

pipeline_optimizer = TPOTClassifier(
    generations=20,
    population_size=15,
    cv=4,
    random_state=random_state,
    verbosity=2,
    scoring="roc_auc",
    n_jobs=4,
    memory="auto",
)

#%%

pipeline_optimizer.fit(X_train, y_train)

#%%

print(pipeline_optimizer.score(X_test, y_test))

#%%

pipeline_optimizer.export(os.path.join("models", "tpot_exported_pipeline.py"))


# %%

# Average CV score on the training set was:0.6192352078198122
exported_pipeline = make_pipeline(
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

X = pd.concat([X_train, X_test])
pd.concat([y_train, y_test])
exported_pipeline.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))
print(pipeline_optimizer.score(X_test, y_test))
results = exported_pipeline.predict(pd.concat[X_train, X_test])


# %%
