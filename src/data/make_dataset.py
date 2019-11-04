# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import find_dotenv, load_dotenv
from src.features.make_features import add_perc_credit_used_and_change

random_state = 123


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    logger.info("loading the data")
    df = pd.read_csv(
        os.path.join("data", "raw", "defaults.csv"), index_col="ID", skiprows=1
    )

    logger.info("lowercasing and cleaning column names")
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

    logger.info("recoding categorical variables with text")
    df.sex.replace({1: "Male", 2: "Female"}, inplace=True)
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
    # Zero not in dataset info. Guessing this to be missing
    df.marriage.replace(
        {0: np.nan, 1: "Married", 2: "Single", 3: "Others"}, inplace=True
    )

    # TODO:
    # The values of paystatus columns does not match the dataset description
    # Leaving as is until more info found.

    logger.info("Saving as clean.csv to data/processed")
    df.to_csv(os.path.join("data", "processed", "clean.csv"))

    print()
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("+++++++++++++++ Feature Engineering ++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print()
    logger.info("Creating columns for percentage of credit used and change in this")
    df = add_perc_credit_used_and_change(df)

    logger.info("Separating the validation set")
    validation = df.sample(frac=0.2, random_state=random_state)
    training = df[~df.index.isin(validation.index)]

    logger.info("Saving as training and validation dataset to data/processed")
    validation.to_csv(os.path.join("data", "processed", "validation.csv"))
    training.to_csv(os.path.join("data", "processed", "training.csv"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
