import os

import matplotlib.pyplot as plt
from matplotlib.ticker import *
import numpy as np

from utils import load_frame, bootsrapping_test, frame_action_distribution, plot_dist

# %% Load the data
path_policy = os.path.join('data', "trial11_eval.tsv")
path_random = os.path.join('data', "baseline_random.tsv")
path_cascade = os.path.join('data', "baseline_cascade.tsv")

policy_orig, policy_action_cols = load_frame(path_policy)
policy_orig.set_index("id")
random_orig, random_action_cols = load_frame(path_random)
random_orig.set_index("id")
cascade_orig, cascade_action_cols = load_frame(path_cascade)
cascade_orig.set_index("id")


# %% Analysis tools
def average_papers_read(f):
    return f.papers.mean()


def success_rate(f):
    return f.success.astype(int).sum() / len(f)


def papers_histogram(f, name, ylim=None):
    plt.figure()
    plt.title("%s papers read" % name)
    plt.ylabel("Frequency")
    plt.xlabel("# of papers read")
    # frequencies = f.papers.value_counts()
    if ylim:
        plt.ylim(top=(ylim + ylim * .05))
    plt.bar(f.index, f.values)
    plt.show()


def overlapping_index(frame_a, frame_b):
    return frame_a.index & frame_b.index


def max_val(*series):
    vals = (f.max() for f in series)
    return max(vals)


# %% Analysis
SUCCESS_ONLY = True

if SUCCESS_ONLY:
    policy = policy_orig[policy_orig.success]
    random = random_orig[random_orig.success]
    cascade = cascade_orig[cascade_orig.success]
else:
    policy = policy_orig
    random = random_orig
    cascade = cascade_orig

# Success rate
policy_success_rate = success_rate(policy)
random_success_rate = success_rate(random)
cascade_success_rate = success_rate(cascade)

# Average papers read
policy_papers_read = average_papers_read(policy)
random_papers_read = average_papers_read(random)
cascade_papers_read = average_papers_read(cascade)

# Papers distribution
NORMALIZE = True

if NORMALIZE:
    p_papers = policy.papers.value_counts() / policy.papers.value_counts().sum()
    r_papers = random.papers.value_counts() / random.papers.value_counts().sum()
    c_papers = cascade.papers.value_counts() / cascade.papers.value_counts().sum()
else:
    p_papers = policy.papers.value_counts()
    r_papers = random.papers.value_counts()
    c_papers = cascade.papers.value_counts()

ylim = max_val(p_papers,
               r_papers,
               c_papers)

papers_histogram(p_papers, "Deep learning", ylim)
papers_histogram(r_papers, "Random baseline", ylim)
papers_histogram(c_papers, "Cascade baseline", ylim)


# %% Action distributions
policy_dist = frame_action_distribution(policy, policy_action_cols, True)
cascade_dist = frame_action_distribution(cascade, policy_action_cols, True)
random_dist = frame_action_distribution(random, policy_action_cols, True)
# fig = plt.figure(figsize=[9.6, 4.8])
fig = plt.figure(figsize=[12.8, 9.6])
ax = fig.subplots(nrows=1, ncols=3)
plot_dist(ax[0], policy_dist, title="Policy action distribution")
plot_dist(ax[1], cascade_dist, title="Cascade baseline action distribution")
plot_dist(ax[2], random_dist, title="Cascade baseline action distribution")
fig.show()


#%% Significance tests
def average_comparison(a, b):
    a_papers = average_papers_read(a)
    b_papers = average_papers_read(b)
    return a_papers > b_papers


iters = 100000
print("Doing significance tests with bootstraping with %i iterations" % iters)
shared_ix = overlapping_index(policy, cascade)
bootstrapped_average_significance = bootsrapping_test(policy.loc[shared_ix],
                                                      cascade.loc[shared_ix],
                                                      average_comparison, iters=iters)

print("The average of papers read by the policy is less than that of cascade with p = %.3f significance" % bootstrapped_average_significance)
