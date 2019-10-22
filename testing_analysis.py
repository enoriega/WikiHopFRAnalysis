import os

import matplotlib.pyplot as plt
from matplotlib.ticker import *
import numpy as np

from utils import load_frame, bootsrapping_test, frame_action_distribution, plot_dist

# %% Load the data
path_policy = os.path.join('data', "trial19_eval_2.tsv")
path_random = os.path.join('data', "baseline_balancedrandom.tsv")
path_cascade = os.path.join('data', "baseline_cascade.tsv")
path_exploit = os.path.join('data', "baseline_exploit.tsv")
path_explore = os.path.join('data', "baseline_explore.tsv")

policy_orig, policy_action_cols = load_frame(path_policy)
policy_orig.set_index("id", inplace=True)
random_orig, random_action_cols = load_frame(path_random)
random_orig.set_index("id", inplace=True)
cascade_orig, cascade_action_cols = load_frame(path_cascade)
cascade_orig.set_index("id", inplace=True)
exploit_orig, exploit_action_cols = load_frame(path_exploit)
exploit_orig.set_index("id", inplace=True)
explore_orig, explore_action_cols = load_frame(path_explore)
explore_orig.set_index("id", inplace=True)


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
SUCCESS_ONLY = False

if SUCCESS_ONLY:
    policy = policy_orig[policy_orig.success]
    random = random_orig[random_orig.success]
    cascade = cascade_orig[cascade_orig.success]
    exploit = exploit_orig[exploit_orig.success]
    explore = explore_orig[explore_orig.success]
else:
    policy = policy_orig
    random = random_orig
    cascade = cascade_orig
    exploit = exploit_orig
    explore = explore_orig

# Success rate
policy_success_rate = success_rate(policy)
random_success_rate = success_rate(random)
cascade_success_rate = success_rate(cascade)
exploit_success_rate = success_rate(exploit)
explore_success_rate = success_rate(explore)

print("Success rate for the policy: %.3f" % policy_success_rate)
print("Success rate for Random: %.3f" % random_success_rate)
print("Success rate for Cascade: %.3f" % cascade_success_rate)
print("Success rate for Exploit: %.3f" % exploit_success_rate)
print("Success rate for Explore: %.3f" % explore_success_rate)

# Average papers read
policy_papers_read = average_papers_read(policy)
random_papers_read = average_papers_read(random)
cascade_papers_read = average_papers_read(cascade)
exploit_papers_read = average_papers_read(exploit)
explore_papers_read = average_papers_read(explore)

print("Avg papers ready by DL: %.3f" % policy_papers_read)
print("Avg papers ready by Random: %.3f" % random_papers_read)
print("Avg papers ready by Cascade: %.3f" % cascade_papers_read)
print("Avg papers ready by Exploit: %.3f" % exploit_papers_read)
print("Avg papers ready by Explore: %.3f" % explore_papers_read)

# Papers distribution
NORMALIZE = True

if NORMALIZE:
    p_papers = policy.papers.value_counts() / policy.papers.value_counts().sum()
    r_papers = random.papers.value_counts() / random.papers.value_counts().sum()
    c_papers = cascade.papers.value_counts() / cascade.papers.value_counts().sum()
    e_papers = exploit.papers.value_counts() / exploit.papers.value_counts().sum()
    ep_papers = explore.papers.value_counts() / explore.papers.value_counts().sum()
else:
    p_papers = policy.papers.value_counts()
    r_papers = random.papers.value_counts()
    c_papers = cascade.papers.value_counts()
    e_papers = exploit.papers.value_counts()
    ep_papers = explore.papers.value_counts()

ylim = max_val(p_papers,
               r_papers,
               c_papers,
               e_papers,
               ep_papers)

papers_histogram(p_papers, "Deep learning", ylim)
papers_histogram(r_papers, "Random baseline", ylim)
papers_histogram(c_papers, "Cascade baseline", ylim)
papers_histogram(e_papers, "Exploit baseline", ylim)
papers_histogram(ep_papers, "Explore baseline", ylim)


# %% Action distributions
policy_dist = frame_action_distribution(policy, policy_action_cols, True)
cascade_dist = frame_action_distribution(cascade, policy_action_cols, True)
random_dist = frame_action_distribution(random, policy_action_cols, True)
exploit_dist = frame_action_distribution(exploit, policy_action_cols, True)
explore_dist = frame_action_distribution(explore, policy_action_cols, True)
# fig = plt.figure(figsize=[9.6, 4.8])
fig = plt.figure(figsize=[12.8, 9.6])
ax = fig.subplots(nrows=1, ncols=3)
plot_dist(ax[0], policy_dist, title="Policy action distribution")
plot_dist(ax[1], cascade_dist, title="Cascade baseline action distribution")
plot_dist(ax[2], random_dist, title="Random baseline action distribution")
# plot_dist(ax[3], exploit_dist, title="Exploit baseline action distribution")
# plot_dist(ax[4], explore_dist, title="Explore baseline action distribution")
fig.show()


#%% Significance tests
def average_comparison(a, b):
    a_papers = average_papers_read(a)
    b_papers = average_papers_read(b)
    return a_papers > b_papers


def efficiency_ratio(f):
    return f.success.astype(int).sum() / f.papers.sum()


def efficiency_comparison(a, b):
    return efficiency_ratio(a) < efficiency_ratio(b)


def statistical_comparison(policy, baseline, iters = 1000):
    shared_ix = overlapping_index(policy, baseline)
    print("Policy: %.2f\tBaseline: %.2f" % (
    average_papers_read(policy.loc[shared_ix]), average_papers_read(baseline.loc[shared_ix])))
    print("Doing significance tests with bootstraping with %i iterations" % iters)
    bootstrapped_average_significance = bootsrapping_test(policy.loc[shared_ix],
                                                          baseline.loc[shared_ix],
                                                          average_comparison, iters=iters)
    print(
        "The average of papers read by the policy is less than that of the baseline with p = %.3f significance" % bootstrapped_average_significance)


statistical_comparison(policy, cascade)
