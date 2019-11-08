# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def calculate_effect_of_credit_drop(model, X, credit_factors):

    probs = model.predict_proba(X)[:, 1]

    all_probs_changes = {}
    all_credit_amount_changes = {}
    all_expected_costs_in_credit = {}

    for factor in credit_factors:

        probs_modified = model.predict_proba(
            X.assign(credit_given=X["credit_given"] * factor)
        )[:, 1]

        probs_change = (probs - probs_modified) * 100
        probs_change = np.where(probs_change <= 0, np.nan, probs_change)

        credit_change = np.array(np.round(X.credit_given * (1 - factor), 0))
        cost_in_credit = np.divide(credit_change, probs_change)

        all_probs_changes[factor] = probs_change
        all_credit_amount_changes[factor] = credit_change
        all_expected_costs_in_credit[factor] = cost_in_credit

    return all_probs_changes, all_credit_amount_changes, all_expected_costs_in_credit


def order_effects_within_customers(
    X,
    credit_factors,
    all_probs_changes,
    all_credit_amount_changes,
    all_expected_costs_in_credit,
):

    all_processed_costs = []
    all_processed_factors = []
    all_processed_credit_changes = []
    all_processed_probs_changes = []

    for customer in range(len(X)):

        costs = []
        factors = []
        credit_change = []
        probs_change = []

        for factor in credit_factors:
            costs.append(all_expected_costs_in_credit[factor][customer])
            factors.append(factor)
            credit_change.append(all_credit_amount_changes[factor][customer])
            probs_change.append(all_probs_changes[factor][customer])

        sorted_costs = sorted(costs)
        sorted_factors = [x for _, x in sorted(zip(costs, factors))]
        sorted_credit_change = [x for _, x in sorted(zip(costs, credit_change))]
        sorted_probs_change = [x for _, x in sorted(zip(costs, probs_change))]

        # assign na to costs and credit change if factor of the next
        # best change is not bigger than in the previous drop
        smallest_factor = None
        for i, factor in enumerate(sorted_factors):
            if (not smallest_factor or factor < smallest_factor) and (
                not np.isnan(sorted_costs[i])
            ):
                smallest_factor = factor
            else:
                sorted_costs[i] = np.nan

        # removing indices from list where costs is nan
        sorted_factors = [
            factor
            for factor, cost in zip(sorted_factors, sorted_costs)
            if not np.isnan(cost)
        ]
        sorted_credit_change = [
            factor
            for factor, cost in zip(sorted_credit_change, sorted_costs)
            if not np.isnan(cost)
        ]
        sorted_probs_change = [
            probs
            for probs, cost in zip(sorted_probs_change, sorted_costs)
            if not np.isnan(cost)
        ]
        sorted_costs = [cost for cost in sorted_costs if not np.isnan(cost)]

        if len(sorted_costs) > 1:
            # change in probs is the current value minus previous except for
            # the first one that stays the same
            sorted_probs_change = [
                current_change - previous_change
                for current_change, previous_change in zip(
                    sorted_probs_change, [0] + sorted_probs_change[:-1]
                )
            ]

            # keeping only values where the default risk is actually lessened
            sorted_credit_change = [
                change
                for change, probs in zip(sorted_credit_change, sorted_probs_change)
                if probs > 0
            ]
            sorted_factors = [
                factor
                for factor, probs in zip(sorted_factors, sorted_probs_change)
                if probs > 0
            ]
            sorted_probs_change = [probs for probs in sorted_probs_change if probs > 0]

            # calculating the change in credit for each viable option
            sorted_credit_change = [
                current_change - previous_change
                for current_change, previous_change in zip(
                    sorted_credit_change, [0] + sorted_credit_change[:-1]
                )
            ]

            # calculating the cost (percent default drop per dollar) for
            # each viable option for credit limit drop
            sorted_costs = [
                credit_change / probs_change
                for credit_change, probs_change in zip(
                    sorted_credit_change, sorted_probs_change
                )
            ]

        all_processed_costs.append(sorted_costs)
        all_processed_factors.append(sorted_factors)
        all_processed_credit_changes.append(sorted_credit_change)
        all_processed_probs_changes.append(sorted_probs_change)

    return (
        all_processed_costs,
        all_processed_factors,
        all_processed_credit_changes,
        all_processed_probs_changes,
    )


def form_results_to_ordered_df(
    y,
    X,
    probs,
    all_processed_costs,
    all_processed_factors,
    all_processed_credit_changes,
    all_processed_probs_changes,
):

    costs_df = pd.DataFrame(
        {
            "defaulted": y,
            "credit_given": X["credit_given"],
            "prob": probs,
            "factors": all_processed_factors,
            "costs": all_processed_costs,
            "credit_losses": all_processed_credit_changes,
            "probs_changes": all_processed_probs_changes,
        }
    )

    # unpacking the list of options and then sorting by it
    # the within customer drops are already sorted (we start from the best one)
    # the order within customers is in right order (smaller credit drops first)

    factors = np.array(costs_df[["factors"]].explode("factors"))
    costs = np.array(costs_df[["costs"]].explode("costs"))
    credit_losses = np.array(costs_df[["credit_losses"]].explode("credit_losses"))
    probs_changes = np.array(costs_df[["probs_changes"]].explode("probs_changes"))

    # df to same size as exploded columns
    costs_df = costs_df.explode("factors")

    # overwriting old columns
    costs_df = costs_df.assign(
        factors=factors,
        costs=costs,
        credit_losses=credit_losses,
        probs_changes=probs_changes,
    )

    costs_df.sort_values(by="costs", inplace=True)

    first_instance_of_customer = ~costs_df.index.duplicated()
    costs_df = costs_df.assign(first_instance_of_customer=first_instance_of_customer)

    costs_df = costs_df.assign(
        defaults_prevented_perc=(
            costs_df["probs_changes"].cumsum()
            / costs_df["first_instance_of_customer"].sum()
        ),
        credit_cost_perc=(
            costs_df["credit_losses"].cumsum()
            * 100
            / costs_df["credit_given"]
            .multiply(costs_df["first_instance_of_customer"])
            .sum()
        ),
        customers_affected_perc=costs_df["first_instance_of_customer"].cumsum()
        * 100
        / costs_df["first_instance_of_customer"].sum(),
        defaulters_affected_perc=costs_df["first_instance_of_customer"]
        .multiply(costs_df["defaulted"])
        .cumsum()
        * 100
        / costs_df["first_instance_of_customer"].multiply(costs_df["defaulted"]).sum(),
        non_defaulters_affected_perc=costs_df["first_instance_of_customer"]
        .multiply(costs_df["defaulted"].subtract(1).abs())
        .cumsum()
        * 100
        / costs_df["first_instance_of_customer"]
        .multiply(costs_df["defaulted"].subtract(1).abs())
        .sum(),
    )

    costs_df.reset_index(inplace=True)

    return costs_df
