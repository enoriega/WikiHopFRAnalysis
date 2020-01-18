import csv
import itertools as it
from heapq import heappush, heappop

import numpy as np
from random import sample, shuffle, choice
from sklearn.feature_extraction.text import CountVectorizer
from simanneal import Annealer
from tqdm import tqdm

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
        matrix = self.matrix
        heap = []
        k = 10
        current_energy = self.energy()

        candidates_left, candidates_right = 0, 0

        for e in left:
            new_energy = error(matrix[list(left - {e}), :], matrix[list(right | {e}), :])
            if new_energy < current_energy:
                heappush(heap, (new_energy, (e, "left")))
                candidates_left += 1
            if candidates_left == k:
                break

        for e in right:
            new_energy = error(matrix[list(left | {e}), :], matrix[list(right - {e}), :])
            if new_energy < current_energy:
                candidates_right += 1
                heappush(heap, (new_energy, (e, "right")))
            if candidates_right == k:
                break

        top_k = [heappop(heap) for i in range(len(heap))]

        _, (ix, source) = choice(top_k)

        if source == 'right':
            right -= {ix}
            left |= {ix}
        else:
            right |= {ix}
            left -= {ix}

        self.state = (left, right)

    def energy(self):
        left, right = self.state
        matrix = self.matrix

        e = error(matrix[list(left), :], matrix[list(right), :])

        # print(e)
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

    left, right = set(pool[:len(pool)//2]), set(pool[len(pool)//2:])

    annealer = DataSetAnnealer((left, right), matrix)
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

    left = {index[int(i)] for i in rows[0]}
    right = {index[int(i)] for i in rows[1]}

    return left, right


if __name__ == '__main__':
    # anneal()

    matrix, _ = read_data()

    # Invert the index
    # with open('data_partition_multinomial_2.csv') as f:
    #     reader = csv.reader(f)
    #     rows = list(reader)
    #
    # l = [int(i) for i in rows[0]]
    # r = [int(i) for i in rows[1]]
    # print(error(matrix[l, :], matrix[r, :]))

    left, right = parse_result('data_partition_multinomial_2.csv')

    with open('training_multinomial.txt', 'w') as f, open('testing_multinomial.txt', 'w') as g:
        for i in left:
            f.write('{}\n'.format(i))

        for j in right:
            g.write('{}\n'.format(j))

    print(len(left), len(right))
