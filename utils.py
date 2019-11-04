# Regexs for parsing the actions
import csv
import re
from collections import namedtuple
import random
import numpy as np
from os import path

from matplotlib.ticker import PercentFormatter
from tqdm import tqdm
import pandas as pd

participant = re.compile(r"""Set\(([\w\,\s–\-\.%&'`\$]+)\)""")
action = re.compile(r"""^(([\d\.E\-]+)/)+(\w+)\(""")

# This will be our Action data structure
Action = namedtuple("Action", "epsilon reward action pa pb")


def parse_action(action_string):
    """Parses an action string into an Action tuple"""

    if action_string:
        # Preprocess the action string
        action_string = action_string.replace("()", "").replace("(,", ",")
        # a = action.findall(action_string)
        action_ix = action_string.find("Explo")
        a = [t for t in action_string[:action_ix].split("/") if t]
        if len(a) == 2:
            epsilon, reward = float(a[0]), float(a[1])
        elif len(a) == 1:
            epsilon, reward = float(a[0]), 0.

        act = action_string[action_ix:].split("(")[0]

        p = participant.findall(action_string)
        if len(p) == 2:
            return Action(epsilon, reward, act, p[0], p[1])
        else:
            # return Action(epsilon, act, p[0], None)
            return Action(epsilon, reward, act, None, None)
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


def load_frame(path):
    # Load the tsv file
    with open(path) as f:
        r = csv.reader(f, delimiter='\t')
        rows = list(r)

    # Compute the number of columns. This number is dynamic as the maximum number of actions may change depending on
    # the run
    num_cols = max(len(r) for r in rows)

    # Generate the headers for the data frame
    columns = ['id', 'iterations', 'papers', 'success']
    action_columns = ['action_%i' % i for i in list(range(1, num_cols - len(columns) + 1))]
    columns = columns + action_columns

    # Build the pandas data frame
    frame = process_frame(pd.DataFrame(rows, columns=columns))

    return frame, action_columns


def bootsrapping_test(seq1, seq2, predicate, iters=1000):
    size = len(seq1)

    values = list()
    for _ in tqdm(range(iters)):
        indices = random.choices(range(size), k=size)
        result = predicate(seq1.ix[indices], seq2.ix[indices])
        values.append(result)

    s = pd.Series(values)
    normalized_estimates = s.value_counts() / len(s)

    return normalized_estimates[True] if True in normalized_estimates else 0.


def frame_action_distribution(frm, action_columns, normalized=False):
    """Computes the distribution of actions from a data frame"""
    all_actions = pd.concat(frm[col] for col in action_columns).dropna()
    simplified_actions = all_actions.map(
        lambda a: "Exploration" if a.action.startswith('Exploration') else 'Exploitation')
    if normalized:
        return simplified_actions.value_counts() / simplified_actions.value_counts().sum()
    else:
        return simplified_actions.value_counts()


def outcome_distributions(frm):
    """Computes the distribution of actions from a data frame"""
    outcomes = frm.success
    dist = outcomes.value_counts()

    return dist


def group_by_action_distribution(gb, action_columns, normalized):
    gb = gb.apply(frame_action_distribution, action_columns).unstack()
    gb = pd.concat([gb['Exploration'], gb['Exploitation']], axis=1).fillna(0.0)

    if normalized:
        gb = gb.div(gb.sum(axis=1), axis=0)  # Normalize the rows

    gb.columns = ["Exploration", 'Exploitation']

    return gb


def plot_dist(ax, dist, title):
    bottom = 0.
    ordered = [(dist.loc[label], label) for label in sorted(dist.index, reverse=True)]
    for y_val, label in ordered:
        ax.bar(0, dist.loc[label], bottom=bottom, width=.3, label=label)
        bottom += dist.loc[label]

    ax.set_xlim(-0.5, 0.5)
    ax.set_title(title)
    ax.set_xticks([])
    percentages = np.asarray([v for v, _ in ordered])
    y_tick_positions = percentages.cumsum()
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels("%i" % (v * 100) + "%" for v in percentages)
    ax.legend()


def min_papers():
    with open(path.join('data', 'min_docs.tsv')) as f:
        reader = csv.reader(f, delimiter='\t')
        ret = {r[0]: len(r[1:]) for r in reader}

    return ret


def missing_min_papers():
    with open(path.join('data', 'missing.txt')) as f:
        missing = {l.strip() for l in f}

    return missing


def observed_reward(frame: pd.DataFrame):
    max_num_actions = max(int(c.split('_')[1]) for c in frame.columns if c.startswith("action_"))

    def get_last_action(row):
        for i in range(max_num_actions, 0, -1):
            col = 'action_%i' % i
            if row[col]:
                return row[col].reward

    return frame.apply(get_last_action, axis=1)
