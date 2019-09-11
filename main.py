import csv
import os
import re
from collections import namedtuple
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

# %% Load all the data

path = os.path.join('data', "trail1.tsv")

# Regexs for parsing the actions
participant = re.compile(r"""Set\(([\w\,\s–\-\.\(\)%&']+)\)""")
action = re.compile(r"""^(\w+)\(""")

# This will be our Action data structure
Action = namedtuple("Action", "action pa pb")


def parse_action(action_string):
    """Parses an action string into an Action tuple"""

    if action_string:
        a = action.findall(action_string)
        assert len(a) == 1, "There should only be one action on this string"
        p = participant.findall(action_string)
        if len(p) == 2:
            return Action(a[0], p[0], p[1])
        else:
            return Action(a[0], p[0], None)
    else:
        return None


def process_frame(fr):
    """Post process a stats dump data frame to marshall the columns into the correct type"""

    fr['iterations'] = fr.iterations.astype(int)
    fr['papers'] = fr.papers.astype(int)
    fr['success'] = fr.success.map(lambda x: True if x == 'true' else False)

    for col in fr.columns:
        if col.startswith('action_'):
            fr[col] = fr[col].map(parse_action)

    return fr


# Load the tsv file
with open(path) as f:
    r = csv.reader(f, delimiter='\t')
    rows = list(r)

# Compute the number of columns. This number is dynamic as the maximum number of actions may change depending on the run
num_cols = max(len(r) for r in rows)

# Generate the headers for the data frame
columns = ['id', 'iterations', 'papers', 'success']
columns = columns + ['action_%i' % i for i in list(range(1, num_cols - len(columns) + 1))]

# Build the pandas data frame
frame = process_frame(pd.DataFrame(rows, columns=columns))


# %% Analysis
def plot_with_regression(series):
    """Plots a series with its linear regression"""
    domain = np.arange(series.size).reshape(-1, 1)
    reg = linear_model.LinearRegression()
    reg.fit(domain, series)
    plt.figure()
    plt.scatter(domain, series)
    plt.plot(domain, reg.predict(domain), color='r')
    plt.show()


groups = frame.groupby(lambda ix: ix % 100)
X = np.arange(100).reshape(-1, 1)
# Compute the average iteration number each 100 epochs
avg_iterations = groups['iterations'].aggregate(np.mean)
plot_with_regression(avg_iterations)
# Compute the average number papers read each 100 epochs
avg_papers = groups['papers'].aggregate(np.mean)
# Do a linear regression to compute the number of iterations
plot_with_regression(avg_papers)
