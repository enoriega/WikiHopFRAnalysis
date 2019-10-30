from os import path
import glob
import csv
import pandas as pd

from matplotlib.ticker import FixedFormatter

from utils import load_frame
from collections import namedtuple
import itertools as it
from operator import attrgetter
import matplotlib.pyplot as plt

DIR_RB = path.join('data', 'baselines_lucene')
PATTERN_RB = path.join(DIR_RB, '*')

DIR_CS = path.join('data', 'cascades_lucene')
PATTERN_CS = path.join(DIR_CS, '*')

files_random_baselines = glob.glob(PATTERN_RB)
files_cascade = glob.glob(PATTERN_CS)

Instance = namedtuple("Instance", 'name p seed frame')
Stats = namedtuple("Stats", 'p success_rates avg_papers avg_iterations efficiency size')

instances_rb = list()

for f in files_random_baselines:
    name = f.split(path.sep)[-1][:-4]
    _, __, p, seed = name.split('_')
    frame, _ = load_frame(f)
    instance = Instance(name, float(p), int(seed), frame)
    instances_rb.append(instance)

instances_cs = list()

for f in files_cascade:
    name = f.split(path.sep)[-1][:-4]
    _, seed = name.split('_')
    frame, _ = load_frame(f)
    instance = Instance(name, 0, int(seed), frame)
    instances_cs.append(instance)

with open(path.join('data', 'testing_instances.txt')) as f:
    testing_ids = {l[:-1] for l in f}

get_p = attrgetter('p')
groups = it.groupby(sorted(instances_rb, key=get_p), key=get_p)

frame_policy, _ = load_frame(path.join('data', 'trial29_eval.tsv'))


def success_rate(f):
    return f.success.value_counts().loc[True] / f.shape[0]


def avg_iterations(f):
    return f.iterations.mean()


def avg_papers(f):
    return f.papers.mean()


def efficiency(f):
    return f.success.astype(int).sum() / f.papers.sum()


def prep_4_plot(stats, field):
    getter = attrgetter(field)
    x = it.chain.from_iterable(it.repeat(st.p, st.size) for st in stats)
    y = it.chain.from_iterable(getter(st) for st in stats)

    return list(x), list(y)


stats = list()
plt.figure()
plt.title("Baseline Performance Comparison")
plt.xlabel("Average papers read")
plt.ylabel("Success rate")
plt.ylim(0, 1)
plt.grid(True)
for name, group in groups:
    frames = [i.frame for i in group]
    success_rates = [success_rate(f) for f in frames]
    papers = [avg_papers(f) for f in frames]
    iterations = [avg_iterations(f) for f in frames]
    eff = [efficiency(f) for f in frames]
    st = Stats(name, success_rates, papers, iterations, eff, len(frames))
    stats.append(st)

    plt.scatter(papers, success_rates, label="$p = %0.2f$" % name)

plt.scatter([avg_papers(f.frame) for f in instances_cs], [success_rate(f.frame) for f in instances_cs], label="Cascade")

# The optimal performer
with open(path.join('data', 'min_docs.tsv')) as f:
    reader = csv.reader(f, delimiter='\t')
    min_necessary = {row[0]: len(row[1:]) for row in reader if row[0] in testing_ids}

optimal_success_rate = len(min_necessary) / frames[0].shape[0]
optimal_avg_papers = pd.Series(list(min_necessary.values())).mean()

plt.scatter([optimal_avg_papers], [optimal_success_rate], label="Optimal Agent")
plt.scatter([avg_papers(frame_policy)], [success_rate(frame_policy)], label="Policy")

plt.legend(loc='lower right')
plt.show()
plt.close()


def box_plot(data, title, ylabel):
    plt.figure()
    plt.grid(True)
    plt.title(title)
    plt.xlabel("$p$ of exploration")
    plt.ylabel(ylabel)
    plt.boxplot(data)
    ax = plt.gca()
    ax.get_xaxis().set_major_formatter(FixedFormatter(["0\n\nFull Exploit", .25, .5, .75, "1\n\nFull Explore"]))
    plt.show()
    plt.close()

# box_plot([s.avg_papers for s in stats], "Baselines' Papers Consumption", "Average papers")
# box_plot([s.avg_iterations for s in stats], "Baselines' Iterations", "Average iterations")
# box_plot([s.success_rates for s in stats], "Baselines' Success Rate", "Success rate")
# box_plot([s.efficiency for s in stats], "Baselines' Efficiency", "Efficiency")
