#%%

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from IPython.core.interactiveshell import InteractiveShell
from statsmodels.graphics.mosaicplot import mosaic

# Setting styles
InteractiveShell.ast_node_interactivity = "all"
sns.set(style="whitegrid", color_codes=True, rc={"figure.figsize": (12.7, 9.27)})

# %% load data

df = pd.read_csv(os.path.join("data", "processed", "clean.csv"), index_col="ID")

# %% view of the data

print("Shape of the data", df.shape)
df.head()

# %% data types

df.dtypes

# %% Number of missing values

print("almost not missing data")
df.isnull().sum()

#%% [markdown]

# # Default Rate
#
# * Default rate is quite high, but most people don't default

#%%

sns.countplot(x="defaulted", data=df)
plt.show()

#%% [markdown]

# # Distribution of Credit Given
#
# * Most people have relatively small credit
# * Very few people have credit over 500000, but the right tail is long

# %% distribution of credit given

sns.distplot(df.credit_given, rug=True)
plt.show()

# %% [markdown]

# # Demographic Information
#
# * More females than men
# * Marital status highly related to education, with more educated being more likely singe.
# This effect is less strong with men
# * There is a around 10 years age difference between married and single customers, with
# married people being unsurprisingly older
# * The distribution of age between genders in different demographic groups is very close
# * Males with university degrees default more often than women with university degrees
# * Women with graduate degrees default more often than men with graduate degrees

# %%

mosaic(
    df,
    ["sex", "education", "marriage"],
    title="Distribution of Customers by Demographic Factors",
)
plt.show()

# %% Age

ax = sns.FacetGrid(df, col="education", row="marriage")
ax = ax.map(sns.violinplot, "sex", "age")
plt.show()

# %% Default rate in different demographic groups

ax = sns.FacetGrid(df, col="education", row="marriage", hue="defaulted")
ax = ax.map(sns.countplot, "sex")
plt.show()


# %% [Markdown]

# # Relationship between the amount paid and the total bill in in different months
#
# * Total bill in different months highly correlated. This is because we are often talking about
# long term debts that not easily paid within few months
# * Amount paid is much less correlated than the amount of the bill. This means that people
# are not terribly consistant in the amount their debt payments
# The amount paid is quite correlated with the to the amount of the bill in the previous month.
# Peoples contributions are related to the immediate amount of debt

# %%

bill_amount_and_paid_cols = [
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
]

cor = df[bill_amount_and_paid_cols].corr()
sns.heatmap(cor, annot=True, cmap="Reds")
plt.show()


# %% [markdown]

# # Big heatmap
#
# * Interestingly the bill amount is negative correlated with defaulting. Taking this naively
# would mean that giving people more credit will make them pay more likely!
# * Of course people who can pay get more credit and a bigger bill
# * We need to create some new feature that will capture the risk of giving big
# loans

# %%

cor = df.select_dtypes(include=np.number).corr()
sns.heatmap(cor, annot=True, cmap="Reds")
plt.show()

# %% [markdown]

# # Amount of Maximum Credit Used
#
# * If we divide the bill amount with the credit given we get how much of the credit is used. Logic here
# is that the more close people go to their maximum credit the more likely is that they default (that
# limit is there for a reason)
# * The bigger the proportion of credit used, the more likely the customer is to default
# * Bizarrely the farther we go in time the higher the correlation to defaulting is. This makes intuitively
# little sense as you would guess that the time when people are most likely to default is
# when they are closest to their credit limit
# * It is quite common for people to be at their credit limit and it is possible to even go above it!

#%%


df = df.assign(
    perc_credit_used1=df["bill_amt1"].divide(df["credit_given"]),
    perc_credit_used2=df["bill_amt2"].divide(df["credit_given"]),
    perc_credit_used3=df["bill_amt3"].divide(df["credit_given"]),
    perc_credit_used4=df["bill_amt4"].divide(df["credit_given"]),
    perc_credit_used5=df["bill_amt5"].divide(df["credit_given"]),
    perc_credit_used6=df["bill_amt6"].divide(df["credit_given"]),
)

cols = [
    "credit_given",
    "bill_amt1",
    "bill_amt2",
    "bill_amt3",
    "bill_amt4",
    "bill_amt5",
    "bill_amt6",
    "perc_credit_used1",
    "perc_credit_used2",
    "perc_credit_used3",
    "perc_credit_used4",
    "perc_credit_used5",
    "perc_credit_used6",
    "defaulted",
]

cor = df[cols].corr()
sns.heatmap(cor, annot=True, cmap="Reds")
plt.show()

# %%

sns.scatterplot("bill_amt1", "perc_credit_used1", hue="defaulted", alpha=0.02, data=df)
plt.show()

#%% [markdown]

# # Amount of Maximum Credit Used Change Over Time
#
# * Maybe changes in the credit used are more important than overall level
# * Calculating change in credit use percentage within month
# * Correlations are quite small, but negative as we would suspect (paying of larger chunks of the
# the dept makes default in the next month less likely)

# %%

df = df.assign(
    perc_credit_change1=df["perc_credit_used1"] - df["perc_credit_used2"],
    perc_credit_change2=df["perc_credit_used2"] - df["perc_credit_used3"],
    perc_credit_change3=df["perc_credit_used3"] - df["perc_credit_used5"],
    perc_credit_change4=df["perc_credit_used4"] - df["perc_credit_used5"],
    perc_credit_change5=df["perc_credit_used5"] - df["perc_credit_used6"],
)

cols = [
    "credit_given",
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
    "defaulted",
]

cor = df[cols].corr()
sns.heatmap(cor, annot=True, cmap="Reds")
plt.show()


# %%

