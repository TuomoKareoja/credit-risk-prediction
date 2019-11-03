# %% Importing packages

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell

# Setting styles
InteractiveShell.ast_node_interactivity = "all"
sns.set(style="whitegrid", color_codes=True, rc={"figure.figsize": (12.7, 9.27)})

# %% [markdown]

# # Loading the data

# * Skipping first row as it contains less informative heading
# * Using provided ID column as index

# %% Loading the data

df = pd.read_csv(
    os.path.join("data", "raw", "defaults.csv"), index_col="ID", skiprows=1
)

# %% [markdown]

# # Basic info
#
# * Only 30000 rows
# * Many columns need to converted to strings with non numerical values (e.g. education)
# * No obvious missing values

# %%

print("Rows in dataset", df.shape)

# %%

df.head()

# %%

df.dtypes
# %%

df.isnull().sum()

# %% lowercasing column names for better handling
df.columns = [
    "credit_given",
    "sex",
    "education",
    "marriage",
    "age",
    "pay_status0",
    "pay_status2",
    "pay_status3",
    "pay_status4",
    "pay_status5",
    "pay_status6",
    "bill_amt1",
    "bill_amt2",
    "bill_amt3",
    "bill_amt4",
    "bill_amt5",
    "bill_amt6",
    "pay_amt1",
    "pay_amt2",
    "pay_amt3",
    "pay_amt4",
    "pay_amt5",
    "pay_amt6",
    "defaulted",
]

# %% [markdown]

# # Distributions
#
# * Education and marriage contain values that are not described in dataset info
# * Codes for payment status seem to have no bearing to what is described in
# dataset info
# * There are lots of values of billable amount that is negative. Does this mean
# that the credit card company was paying the customer or is this an error?
# * The proportion of customers that defaulted is around 20 %. This seems very
# high and raises the question if the dataset is biased

# %% credit given

df.credit_given.hist(bins=100)
plt.title("credit_given")
plt.show()

# %% sex

df.sex.value_counts().plot.bar()
plt.title("sex")
plt.show()
df.sex.replace({1: "Male", 2: "Female"}, inplace=True)
df.sex.value_counts().plot.bar()
plt.title("sex (recoded)")
plt.show()

# %% education

df.education.value_counts().sort_index().plot.bar()
plt.title("education")
plt.show()
# Categories 0, 5 and 6 are not mentioned in dataset info.
# Coding 5 and 6 as Other and guessing that 0 is missing
df.education.replace(
    {
        0: np.nan,
        1: "Graduate School",
        2: "University",
        3: "High School",
        4: "Others",
        5: "Others",
        6: "Others",
    },
    inplace=True,
)
df.education.value_counts().sort_index().plot.bar()
plt.title("education (recoded)")
plt.show()

# %% marriage

df.marriage.value_counts().sort_index().plot.bar()
plt.title("marriage")
plt.show()
# Zero not in dataset info. Guessing this to be missing
df.marriage.replace({0: np.nan, 1: "Married", 2: "Single", 3: "Others"}, inplace=True)
df.marriage.value_counts().sort_index().plot.bar()
plt.title("marriage (recoded)")
plt.show()


# %% age

# no weird values
df.age.value_counts().sort_index().plot.bar()
df.age.value_counts().sort_index()

# %% Pay statuses

# The values of these columns does not match the dataset description
# Leaving as is until more info found.
# Also weird that there is no pay_status1
pay_status_cols = [
    "pay_status0",
    "pay_status2",
    "pay_status3",
    "pay_status4",
    "pay_status5",
    "pay_status6",
]
for col in pay_status_cols:
    df[col].value_counts().sort_index().plot.bar()
    plt.title(col)
    plt.show()

# %% Bill amount

bill_cols = [
    "bill_amt1",
    "bill_amt2",
    "bill_amt3",
    "bill_amt4",
    "bill_amt5",
    "bill_amt6",
]

# How are negative numbers possible
for col in bill_cols:
    df[col].plot.hist(bins=100)
    plt.title(col)
    plt.show()

# %% amount paid

pay_cols = ["pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6"]

# How are negative numbers possible
for col in pay_cols:
    df[col].plot.hist(bins=100)
    plt.title(col)
    plt.show()

# %% Defaulted

# defaulting next payment is rare, but not exceedingly so
df["defaulted"].value_counts().plot.bar()
plt.show()

# %%
