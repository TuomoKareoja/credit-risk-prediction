# -*- coding: utf-8 -*-
import os

import pandas as pd
import numpy as np


def add_perc_credit_used_and_change(df):
    df = df.assign(
        perc_credit_used1=df["bill_amt1"].divide(df["credit_given"]),
        perc_credit_used2=df["bill_amt2"].divide(df["credit_given"]),
        perc_credit_used3=df["bill_amt3"].divide(df["credit_given"]),
        perc_credit_used4=df["bill_amt4"].divide(df["credit_given"]),
        perc_credit_used5=df["bill_amt5"].divide(df["credit_given"]),
        perc_credit_used6=df["bill_amt6"].divide(df["credit_given"]),
    )

    df = df.assign(
        perc_credit_change1=df["perc_credit_used1"] - df["perc_credit_used2"],
        perc_credit_change2=df["perc_credit_used2"] - df["perc_credit_used3"],
        perc_credit_change3=df["perc_credit_used3"] - df["perc_credit_used5"],
        perc_credit_change4=df["perc_credit_used4"] - df["perc_credit_used5"],
        perc_credit_change5=df["perc_credit_used5"] - df["perc_credit_used6"],
    )

    return df
