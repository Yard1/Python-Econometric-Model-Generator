# By Antoni Baum, Bartosz Dąbrowski, Bartłomiej Gąsior, Michał Kędra, 2020

#!/usr/bin/python3
import itertools
import math
import argparse
import numpy as np
import pandas as pd
import collections
from tqdm import tqdm
from joblib import Parallel, delayed
from operator import itemgetter
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.stats.api as sms
from statsmodels.stats.diagnostic import *
from statsmodels.stats.outliers_influence import *
import warnings

ALPHA = 0.95


def main(df, dependent_variable):
    df.sort_values(by=[dependent_variable], inplace=True)
    df.reset_index(inplace=True, drop=True)
    independent_variables = list(df.columns)
    print(independent_variables)
    independent_variables.remove(dependent_variable)
    print(dependent_variable)

    if len(independent_variables) < 10:
        combinations = Parallel(n_jobs=4)(delayed(get_combinations)(
            independent_variables, l) for l in tqdm(range(1, len(independent_variables)+1)))
        combinations = [item for sublist in combinations for item in sublist]
        results_tuple = Parallel(n_jobs=4)(delayed(get_model_result)(
            dependent_variable, c, df) for c in tqdm(combinations))
        results, results_df = map(list, zip(*results_tuple))
    else:
        yx_corrs = collections.OrderedDict()
        for var in independent_variables:
            yx_corrs[var] = df[dependent_variable].corr(df[var])
        combinations = Parallel(n_jobs=4)(delayed(get_combinations_generator)(
            independent_variables, i) for i in range(1, len(independent_variables)+1))
        results_tuple_backwards = stepwise_backwards(
            dependent_variable, independent_variables, df, ALPHA)
        results = [results_tuple_backwards[0]]
        results_df = [results_tuple_backwards[1]]
        results_tuple_forwards = stepwise_forwards(
            dependent_variable, independent_variables, df, ALPHA)
        if {x[0] for x in results_tuple_forwards[0][0].pvalues.items()} != {x[0] for x in results[0][0].pvalues.items()}:
            results.append(results_tuple_forwards[0])
            results_df.append(results_tuple_forwards[1])
    results.sort(key=lambda x: x[0].aic)
    return (results, pd.concat(results_df))


def create_dataframe(item):
    col = pd.DataFrame({'Formula': [item[1]]})
    results_as_html = item[0].summary().tables[0].as_html()
    r_df = pd.read_html(results_as_html)[0]
    r_df.drop([0, 1], axis=1, inplace=True)
    r_df.dropna(axis=0, inplace=True)
    r_df = r_df.T
    colnames = list(r_df.iloc[0])
    colnames = [x[:-1] for x in colnames]
    r_df.columns = colnames
    r_df.drop([2], axis=0, inplace=True)
    r_df.reset_index(inplace=True, drop=True)
    col = col.join(r_df)
    col = col.join(pd.DataFrame.from_dict([item[2]]))
    return col


def get_combinations(col, l):
    s = []
    for subset in itertools.combinations(col, l):
        s.append(get_formula(subset))
    return s


def get_formula(iterable):
    return "1+" + "+".join(iterable)


def get_combinations_generator(col, l):
    return itertools.combinations(col, l)


def hellwig(correlation_matrix, dependent_var_correlation_matrix, var_combinations):
    best_info = []
    for combination in tqdm(var_combinations):
        h = Parallel(n_jobs=-1)(delayed(hellwig_singular)(correlation_matrix,
                                                          dependent_var_correlation_matrix, c) for c in combination)
        h = max(h, key=itemgetter(1))
        best_info.append(h)
    best_info = max(best_info, key=itemgetter(1))
    return best_info


def hellwig_singular(correlation_matrix, dependent_var_correlation_matrix, var_combination):
    h = 0
    var_combination = list(var_combination)
    denominator = 0
    for var in var_combination:
        denominator += abs(correlation_matrix[var_combination[0]][var])
    for var in var_combination:
        h += (dependent_var_correlation_matrix[var]**2)/denominator
    return (var_combination, h)


def get_model_result(dependent_variable, rhs, df):
    mod = sm.OLS.from_formula(dependent_variable + "~" + rhs, data=df).fit()
    tests = test_result(mod, df, ALPHA)
    item = (mod, dependent_variable + "~" + rhs, tests)
    r_df = create_dataframe(item)
    return (item, r_df)


def stepwise_backwards(dependent_variable, independent_variables, df, alpha=0.95):
    while(True):
        model = get_model_result(
            dependent_variable, get_formula(independent_variables), df)
        pvalues = list(model[0][0].pvalues.items())
        min_pvalues = min(pvalues, key=lambda x: x[1])
        if min_pvalues[1] <= 1-alpha:
            break
        independent_variables.remove(min_pvalues[0])
    return model


def stepwise_forwards(dependent_variable, independent_variables, df, alpha=0.95):
    checked_variables = []
    while(True):
        best_pvalue = -1
        best_var = None
        for var in independent_variables:
            vars = checked_variables.copy()
            vars.append(var)
            model = get_model_result(dependent_variable, get_formula(vars), df)
            pvalues = list(model[0][0].pvalues.items())
            for pval in pvalues:
                if pval[0] != "Intercept" and pval[0] not in checked_variables and pval[1] > best_pvalue:
                    best_pvalue = pval[1]
                    best_var = pval[0]
        if not best_var or best_pvalue > 1-alpha:
            break
        independent_variables.remove(best_var)
        checked_variables.append(best_var)
    model = get_model_result(
        dependent_variable, get_formula(checked_variables), df)
    return model


def test_result(res, df, alpha=0.95):
    try:
        each_variable_important_result = each_variable_important(res, alpha)
    except:
        each_variable_important_result = False
    try:
        het_white_result = het_white(res.resid, res.model.exog)[1] < 1-alpha
    except:
        het_white_result = False
    try:
        het_breuschpagan_result = het_breuschpagan(
            res.resid, res.model.exog)[1] < 1-alpha
    except:
        het_breuschpagan_result = False
    try:
        reset_ramsey_result = reset_ramsey(res).pvalue < 1-alpha
    except:
        reset_ramsey_result = False
    try:
        linear_harvey_collier_result = linear_harvey_collier(
            res).pvalue < 1-alpha
    except:
        linear_harvey_collier_result = False
    try:
        vif_result = vif(res, df)
    except:
        vif_result = False
    return {"each_variable_important_result": each_variable_important_result, "het_white_result": het_white_result, "het_breuschpagan_result": het_breuschpagan_result, "reset_ramsey_result": reset_ramsey_result, "linear_harvey_collier_result": linear_harvey_collier_result, "vif_result": vif_result}


def vif(res, df):
    for i in range(len(df.columns)):
        if df.columns[i] in res.model.exog_names:
            v = variance_inflation_factor(np.matrix(df), i)
            if v > 10:
                return True
    return False


def each_variable_important(res, alpha):
    for param, pvalue in res.pvalues.items():
        if pvalue > 1-alpha:
            return True
    return False


def build_plot(res, df, dependent_variable):
    dependent_variable_str = dependent_variable
    dependent_variable = df[dependent_variable]
    prstd, iv_l, iv_u = wls_prediction_std(res)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(dependent_variable, '-', label="data")
    ax.plot(res.fittedvalues, 'ro-', label="OLS")
    ax.plot(iv_u, 'r--')
    ax.plot(iv_l, 'r--')
    ax.legend(loc='best')
    ax.set_title('Obs. sorted by ' + dependent_variable_str)
    plt.show()


def read_csv(fname, delimeter, decimal):
    df = pd.read_csv(fname, delimiter=str(delimeter), decimal=str(decimal))
    df.apply(pd.to_numeric, errors='coerce')
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Give the best combination of variables for modelling using Hellwig\'s method.')
    parser.add_argument('file', help='Path to the data in .csv format')
    parser.add_argument('dependent_variable', help='Dependent variable')
    parser.add_argument('-d', '--delimeter', required=False, default=";",
                        help='csv delimeter (Default: ;)')
    parser.add_argument('-s', '--decimal_separator', required=False, default=",",
                        help='csv decimal separator (Default: ,)')
    args = parser.parse_args()
    main(read_csv(args.file, args.delimeter,
                  args.decimal_separator), args.dependent_variable)
