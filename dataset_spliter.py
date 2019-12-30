import csv
import itertools as it
import numpy as np
from random import sample, shuffle
from sklearn.feature_extraction.text import CountVectorizer
from simanneal import Annealer

BINARY = False


def bump(source, dest):
    size = int(len(source) * .5)
    elems = sample(source, size)
    dest.extend(elems)
    for elem in elems:
        source.remove(elem)


class DataSetAnnealer(Annealer):

    def __init__(self, state, matrix):
        self.matrix = matrix
        super(DataSetAnnealer, self).__init__(state)  # important!

    def move(self):
        left, right = self.state

        if len(left) < 100:
            bump(right, left)
        elif len(right) > 100:
            bump(left, right)
        else:
            coin = np.random.randint(2)
            if coin == 0:
                bump(right, left)
            else:
                bump(left, right)

        self.state = (left, right)

    def energy(self):
        left, right = self.state
        matrix = self.matrix

        e = error(matrix[left, :], matrix[right, :])

        print(e)

        return e


def error(left, right):
    agg_l = left.sum(axis=0)
    agg_r = right.sum(axis=0)

    if BINARY:
        agg_l[agg_l > 0] = 1
        agg_r[agg_r > 0] = 1

    # DO MSE
    return np.square(agg_l - agg_r).mean()


def read_data():
    with open('entities_in_instances.tsv') as f:
        r = csv.reader(f, delimiter='\t')
        rows = list(r)
    docs = {row[0]: ' '.join(row[1:]) for row in rows}
    voc = set(it.chain.from_iterable(row[1:] for row in rows))
    cv = CountVectorizer(binary=BINARY, vocabulary=voc)
    cv.fit(docs.values())

    index = dict(enumerate(sorted(docs.keys())))
    matrix = cv.transform([docs[k] for k in sorted(docs.keys())]).toarray()

    return matrix, index


def anneal():
    matrix, index = read_data()

    pool = list(index.keys())
    shuffle(pool)

    annealer = DataSetAnnealer(([], pool), matrix)
    # annealer.updates = 10
    # annealer.steps = 200

    final, e = annealer.anneal()

    print(final, e)


def parse_result(path):
    _, index = read_data()

    # Invert the index
    with open(path) as f:
        reader = csv.reader(f)
        rows = list(reader)

    left = [index[int(i)] for i in rows[0]]
    right = [index[int(i)] for i in rows[1]]

    return left, right


if __name__ == '__main__':
    # anneal()

    left, right = parse_result('data_partition_multinomial_2.csv')
    print(len(left), len(right))