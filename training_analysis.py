import os
from matplotlib.ticker import PercentFormatter
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from utils import load_frame, outcome_distributions, group_by_action_distribution


# %% Helper functions
def plot_with_regression(ax, series, ylabel, title, epsilons=None, regression=True):
    """Plots a series with its linear regression"""
    domain = np.arange(series.size).reshape(-1, 1)
    if regression:
        reg = linear_model.LinearRegression()
        reg.fit(domain, series)
        ax.plot(domain, reg.predict(domain), color='y', lw=3)
    ax.scatter(domain, series, s=7)
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel("Batch repetition")
    ax.set_ylabel(ylabel, color='tab:blue')
    # ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: "0" if x == 0 else r"%i$\times10^{100}$" % x))
    # for label in ax.get_xaxis().get_ticklabels():
    #     label.set_rotation(45)
    #     label.set_fontsize(8)

    if epsilons is not None:
        ax2 = ax.twinx()
        ax2.plot(domain, epsilons, 'r--')
        ax2.set_ylabel("$\epsilon$-greedy schedule", color='tab:red')
        ax2.autoscale(False)
        for label in ax2.yaxis.get_ticklabels():
            label.set_color('tab:red')


def avg_epsilon(grp):
    """Returns the average epsilon for each batch of epochs"""
    # Create a series with all the actions
    all_actions = pd.concat(grp[col] for col in action_columns)
    return all_actions.dropna().map(lambda a: a.epsilon).mean()


def plot_action_distributions(ax, frm, ylabel, title, labels=None, normalized=False, epsilons=None):
    """Creates a stack plot with the distribution of exploration/exploitation actions"""
    domain = np.arange(frm.shape[0])
    ax.stackplot(domain, frm.values[:, 0], frm.values[:, 1], labels=labels)
    ax.legend(loc='lower left')
    ax.grid(True, linestyle='--')
    ax.set_xlabel("Batch repetition")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    # ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: "0" if x == 0 else r"%i$\times10^{100}$" % x))
    if normalized:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.))
    ax.set_ymargin(0.)
    ax.set_xmargin(0.)
    ax.autoscale(True)
    # for label in ax.get_xaxis().get_ticklabels():
    #     label.set_rotation(45)
    #     label.set_fontsize(8)

    if epsilons is not None:
        ax2 = ax.twinx()
        ax2.set_ymargin(0.)
        ax2.set_xmargin(0.)
        ax2.autoscale(False)
        ax2.plot(domain, epsilons, 'r--', alpha=0.5)
        ax2.set_ylabel("$\epsilon$-greedy schedule", color='tab:red')
        for label in ax2.yaxis.get_ticklabels():
            label.set_color('tab:red')


# %% Load all the data
path = os.path.join('data', "trial30.tsv")
frame, action_columns = load_frame(path)

# %% Analysis
ONLY_SUCCESSES = False
SHOW_REGRESSIONS = False
NORMALIZE_DISTS = True

if ONLY_SUCCESSES:
    frame = frame.loc[frame.success]

groups = frame.groupby(lambda ix: ix // 251)

X = np.arange(1000).reshape(-1, 1)
epsilons = groups.apply(avg_epsilon)
# Compute the average iteration number each 100 epochs
fig, ax = plt.subplots()
avg_iterations = groups['iterations'].mean()
plot_with_regression(ax, avg_iterations, epsilons=epsilons.values, title="Average Iterations per epoch",
                     ylabel="Avg iterations", regression=SHOW_REGRESSIONS)
fig.show()
# Compute the average number papers read each 100 epochs
fig, ax = plt.subplots()
avg_papers = groups['papers'].mean() #/ groups['success'].apply(lambda s: s.astype(int).sum())
plot_with_regression(ax, avg_papers, "Average Papers", "Avg papers read", epsilons.values, regression=SHOW_REGRESSIONS)
fig.show()

# Compute the distribution of actions
dists = group_by_action_distribution(groups, action_columns, NORMALIZE_DISTS)
if NORMALIZE_DISTS:
    ylabel = "Percentage of actions"
    normalized = True
else:
    normalized = False
    ylabel = "Number of actions"

fig, ax = plt.subplots()
plot_action_distributions(ax, dists, ylabel, "Explore/Exploit tradeoff", labels=["Exploration", "Exploitation"],
                          normalized=normalized, epsilons=epsilons)
fig.show()

if not ONLY_SUCCESSES:
    # Compute the distribution of successes/failures
    outcomes = groups.apply(outcome_distributions)
    outcomes.columns = ["Success", "Failure"]
    # outcomes = pd.concat([outcomes[:, "Success"], outcomes[:, "Failure"]], axis=1)
    outcomes = outcomes.div(outcomes.sum(axis=1), axis=0)  # Normalize the rows
    outcomes.columns = ["Success", 'Failure']
    fig, ax = plt.subplots()
    plot_action_distributions(ax, outcomes, "Percentage of outcome", "Outcome distribution over epochs",
                              labels=["Success", "Failure"], epsilons=epsilons)
    fig.show()
