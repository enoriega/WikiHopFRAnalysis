import csv
import os
import re
from matplotlib.ticker import FuncFormatter
from collections import namedtuple
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

# %% Load all the data

path = os.path.join('data', "trail3.tsv")

# Regexs for parsing the actions
participant = re.compile(r"""Set\(([\w\,\s–\-\.%&'`\$]+)\)""")
action = re.compile(r"""^([\d\.]+)/(\w+)\(""")

# This will be our Action data structure
Action = namedtuple("Action", "epsilon action pa pb")


def parse_action(action_string):
    """Parses an action string into an Action tuple"""

    if action_string:
        # Preprocess the action string
        action_string = action_string.replace("()", "").replace("(,", ",")
        a = action.findall(action_string)
        epsilon, act = float(a[0][0]), a[0][1]
        p = participant.findall(action_string)
        if len(p) == 2:
            return Action(epsilon, act, p[0], p[1])
        else:
            return Action(epsilon, act, p[0], None)
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
action_columns = ['action_%i' % i for i in list(range(1, num_cols - len(columns) + 1))]
columns = columns + action_columns

# Build the pandas data frame
frame = process_frame(pd.DataFrame(rows, columns=columns))


# %% Analysis
def plot_with_regression(ax, series, epsilons=None):
    """Plots a series with its linear regression"""
    domain = np.arange(series.size).reshape(-1, 1)
    reg = linear_model.LinearRegression()
    reg.fit(domain, series)
    ax.scatter(domain, series, s=7)
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.plot(domain, reg.predict(domain), color='y', lw=3)
    ax.set_xlabel("Training Epoch")
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: "0" if x == 0 else r"%i$\times10^{100}$" % x))
    for label in ax.get_xaxis().get_ticklabels():
        label.set_rotation(45)
        label.set_fontsize(8)

    if epsilons is not None:
        ax2 = ax.twinx()
        ax2.plot(domain, epsilons, 'r--')
        ax2.set_ylabel("$\epsilon$-greedy schedule", color='tab:red')
        for label in ax2.yaxis.get_ticklabels():
            label.set_color('tab:red')


def avg_epsilon(grp):
    "Returns the average epsilon for each batch of epochs"
    # Create a series with all the actions
    all_actions = pd.concat(grp[col] for col in action_columns)
    return all_actions.dropna().map(lambda a: a.epsilon).mean()


groups = frame.groupby(lambda ix: ix // 100)
X = np.arange(1000).reshape(-1, 1)
epsilons = groups.apply(avg_epsilon)
# Compute the average iteration number each 100 epochs
fig, ax = plt.subplots()
avg_iterations = groups['iterations'].mean()
plot_with_regression(ax, avg_iterations, epsilons.values)
ax.set_title("Average Iterations per epoch")
ax.set_ylabel("Avg iterations", color='tab:blue')
fig.show()
# Compute the average number papers read each 100 epochs
fig, ax = plt.subplots()
avg_papers = groups['papers'].mean()
# Do a linear regression to compute the number of iterations
plot_with_regression(ax, avg_papers, epsilons.values)
ax.set_title("Average Papers")
ax.set_ylabel("Avg papers read", color='tab:blue')
fig.show()
