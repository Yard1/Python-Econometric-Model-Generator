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

#############################
###
### Hellwig's method/method of the capacity of information bearers for Python, by Antoni Baum
### Written in Python 3.6.7
### Requires numpy, pandas, tqdm, joblib packages
###
### Copyright (c) 2019 Antoni Baum (Yard1)
### Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
### The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
### THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
###
### usage: hellwig.py [-h] [-d DELIMETER] [-s DECIMAL_SEPARATOR] [-min MIN]
###                   [-max MAX]
###                   [-id INDEPENDENT_VARIABLES [INDEPENDENT_VARIABLES ...]]
###                   file dependent_variable
### 
### Give the best combination of variables for modelling using Hellwig's method.
### 
### positional arguments:
###   file                  Path to the data in .csv format
###   dependent_variable    Dependent variable
###
### optional arguments:
###   -h, --help            show this help message and exit
###   -d DELIMETER, --delimeter DELIMETER
###                         csv delimeter (Default: ;)
###   -s DECIMAL_SEPARATOR, --decimal_separator DECIMAL_SEPARATOR
###                         csv decimal separator (Default: ,)
###   -min MIN, --min MIN   The smallest number of items in a combination
###                         (Default: 1)
###   -max MAX, --max MAX   The largest number of items in a combination (Default:
###                         number of variables)
###   -id INDEPENDENT_VARIABLES [INDEPENDENT_VARIABLES ...], --independent_variables INDEPENDENT_VARIABLES [INDEPENDENT_VARIABLES ...]
###                         Independent variables to check. If not given, will
###                         check all independent variables in the dataset
###
#############################

def main(fname, dependent_variable, delimeter, decimal):
    df = read_csv(fname, delimeter, decimal)
    df.sort_values(by=[dependent_variable], inplace=True)
    df.reset_index(inplace=True, drop=True)
    independent_variables = list(df.columns)
    print(independent_variables)
    independent_variables.remove(dependent_variable)
    combinations = Parallel(n_jobs=4)(delayed(get_combinations)(independent_variables, l) for l in tqdm(range(1, len(independent_variables)+1)))
    combinations = [item for sublist in combinations for item in sublist]
    results_tuple = Parallel(n_jobs=4)(delayed(get_model_result)(dependent_variable, c, df) for c in tqdm(combinations))
    results, results_df = map(list, zip(*results_tuple))
    results.sort(key=lambda x: x[0].aic)
    best_result = results[0][0]
    best_result_passing_tests = next((x for x in results if sum(x[2].values()) == 0), None)
    #best_result = sm.OLS.from_formula(dependent_variable + "~ 1+hsGPA+PC+skipped+alcohol+gradMI", data=df).fit()
    #results = [best_result]
    print(best_result.summary())
    build_plot(best_result, df, dependent_variable)
    if best_result_passing_tests:
        best_result_passing_tests = best_result_passing_tests[0]
        print(best_result_passing_tests.summary())
        build_plot(best_result_passing_tests, df, dependent_variable)
    results_df = pd.concat(results_df)
    results_df.to_csv (r'results_df.csv', index = None, header=True)

def create_dataframe(item):
    col = pd.DataFrame({'Formula': [item[1]]})
    results_as_html = item[0].summary().tables[0].as_html()
    r_df = pd.read_html(results_as_html)[0]
    r_df.drop([0, 1], axis=1, inplace=True)
    r_df.dropna(axis=0, inplace=True)
    r_df=r_df.T
    colnames = list(r_df.iloc[0])
    colnames = [x[:-1] for x in colnames]
    r_df.columns=colnames
    r_df.drop([2], axis=0, inplace=True)
    r_df.reset_index(inplace=True, drop=True)
    col = col.join(r_df)
    col = col.join(pd.DataFrame.from_dict([item[2]]))
    return col

def get_combinations(col, l):
    s = []
    for subset in itertools.combinations(col, l):
        s.append("1+" + "+".join(subset))
    return s

def get_model_result(dependent_variable, rhs, df):
    mod = sm.OLS.from_formula(dependent_variable + "~" + rhs, data=df).fit()
    tests = test_result(mod, df)
    item = (mod, dependent_variable + "~" + rhs, tests)
    r_df = create_dataframe(item)
    return (item, r_df)

def test_result(res, df, alpha = 0.95):
    try:
        each_variable_important_result = each_variable_important(res, alpha)
    except:
        each_variable_important_result = False
    try:
        het_white_result = het_white(res.resid, res.model.exog)[1] < 1-alpha
    except:
        het_white_result = False
    try:
        het_breuschpagan_result = het_breuschpagan(res.resid, res.model.exog)[1] < 1-alpha
    except:
        het_breuschpagan_result = False
    try:
        reset_ramsey_result = reset_ramsey(res).pvalue < 1-alpha
    except:
        reset_ramsey_result = False
    try:
        linear_harvey_collier_result = linear_harvey_collier(res).pvalue < 1-alpha
    except:
        linear_harvey_collier_result = False
    try:
        vif_result = vif(res, df)
    except:
        vif_result = False
    return {"each_variable_important_result": each_variable_important_result, "het_white_result": het_white_result, "het_breuschpagan_result": het_breuschpagan_result, "reset_ramsey_result": reset_ramsey_result, "linear_harvey_collier_result": linear_harvey_collier_result, "vif_result": vif_result }

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
    fig, ax = plt.subplots(figsize=(8,6))
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
    parser = argparse.ArgumentParser(description='Give the best combination of variables for modelling using Hellwig\'s method.')
    parser.add_argument( 'file', help='Path to the data in .csv format')
    parser.add_argument( 'dependent_variable', help='Dependent variable')
    parser.add_argument('-d', '--delimeter', required=False, default=";",
                    help='csv delimeter (Default: ;)')
    parser.add_argument('-s', '--decimal_separator', required=False, default=",",
                    help='csv decimal separator (Default: ,)')
    args = parser.parse_args()
    main(args.file, args.dependent_variable, args.delimeter, args.decimal_separator)