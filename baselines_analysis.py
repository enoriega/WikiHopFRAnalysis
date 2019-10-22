from os import path
import glob

from matplotlib.ticker import FixedFormatter

from utils import load_frame
from collections import namedtuple
import itertools as it
from operator import attrgetter
import matplotlib.pyplot as plt

DIR = path.join('data', 'baselines')
PATTERN = path.join(DIR, '*')

files = glob.glob(PATTERN)
Instance = namedtuple("Instance", 'name p seed frame')
Stats = namedtuple("Stats", 'p success_rates avg_papers avg_iterations size')

instances = list()

for f in files:
    name = f.split(path.sep)[-1][:-4]
    _, __, p, seed = name.split('_')
    frame, _ = load_frame(f)
    instance = Instance(name, float(p), int(seed), frame)
    instances.append(instance)


get_p = attrgetter('p')
groups = it.groupby(sorted(instances, key=get_p), key=get_p)


def success_rate(f):
    return f.success.value_counts().loc[True] / f.shape[0]


def avg_iterations(f):
    return f.iterations.mean()


def avg_papers(f):
    return f.papers.mean()


def prep_4_plot(stats, field):
    getter = attrgetter(field)
    x = it.chain.from_iterable(it.repeat(st.p, st.size) for st in stats)
    y = it.chain.from_iterable(getter(st) for st in stats)

    return list(x), list(y)


stats = list()
for name, group in groups:
    frames = [i.frame for i in group]
    success_rates = [success_rate(f) for f in frames]
    papers = [avg_papers(f) for f in frames]
    iterations = [avg_iterations(f) for f in frames]
    st = Stats(name, success_rates, papers, iterations, len(frames))
    stats.append(st)


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


box_plot([s.avg_papers for s in stats], "Baselines' Papers Consumption", "Average papers")
box_plot([s.avg_iterations for s in stats], "Baselines' Iterations", "Average iterations")
box_plot([s.success_rates for s in stats], "Baselines' Success Rate", "Success rate")



