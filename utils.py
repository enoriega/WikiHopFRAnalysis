# Regexs for parsing the actions
import csv
import re
from collections import namedtuple

import pandas as pd

participant = re.compile(r"""Set\(([\w\,\s–\-\.%&'`\$]+)\)""")
action = re.compile(r"""^(([\d\.E\-]+)/)+(\w+)\(""")

# This will be our Action data structure
Action = namedtuple("Action", "epsilon action pa pb")


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
            return Action(epsilon, act, p[0], p[1])
        else:
            # return Action(epsilon, act, p[0], None)
            return Action(epsilon, act, None, None)
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
