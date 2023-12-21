import itertools
import numpy as np
import pandas as pd

from collections import Counter


def calculate_marginal_probability(data, X):
    probs = dict(Counter(data[X]))
    probs = {x: probs[x]/len(data[X]) for x in probs}
    return probs


def calculate_joint_probability(data, X, Y):
    probs = {}
    X_values = list(set(data[X]))
    Y_values = list(set(data[Y]))
    cartesian_product_values = list(itertools.product(X_values, Y_values))
    dim = len(data[X])
    for x, y in zip(data[X], data[Y]):
        if ((x, y) in probs.keys()):
            probs[(x, y)] += 1
        else:
            probs[(x, y)] = 1
    for value in cartesian_product_values:
        if value not in probs:
            probs[value] = 0
    for x in probs:
        probs[x] = probs[x]/dim
    return probs


def calculate_mi(data, X, Y):
    X_Y_probs = calculate_joint_probability(data, X, Y)
    X_probs = calculate_marginal_probability(data, X)
    Y_probs = calculate_marginal_probability(data, Y)
    MI = 0
    for (x, y) in X_Y_probs:
        x_y_prob = X_Y_probs[(x, y)]
        x_prob = X_probs[x]
        y_prob = Y_probs[y]
        if (x_y_prob/(x_prob*y_prob) != 0):
            term = x_y_prob*np.log(x_y_prob/(x_prob*y_prob))
            MI += term
    return MI


def calculate_empiric_enthropy_of_vertex(data, vertex, parent_vertices):
    H = 0
    if (len(parent_vertices) == 0):
        vertex_values = Counter(data[vertex])
        size = len(data[vertex])
        for item in vertex_values:
            if (vertex_values[item] != 0):
                term = 1
                term = vertex_values[item] * np.log(vertex_values[item] / size) * (-1)
                H += term
    elif (len(parent_vertices) >= 1):
        for parent_vertex in parent_vertices:
            parent_vertex_values = list(set(data[parent_vertex]))
            vertex_values = list(set(data[vertex]))
            all_possible_values = list(itertools.product(parent_vertex_values, vertex_values))
            occurences = {}
            for (item_1,item_2) in zip(data[parent_vertex], data[vertex]):
                if (item_1,item_2) in occurences:
                    occurences[(item_1, item_2)] += 1
                else:
                    occurences[(item_1, item_2)] = 1
            for (item_1, item_2) in all_possible_values:
                if (item_1, item_2) not in occurences:
                    occurences[(item_1, item_2)] = 0
            table_data = []
            for (item_1, item_2) in occurences:
                table_data.append([item_1, item_2, occurences[(item_1, item_2)]])
            table = pd.DataFrame(data=table_data, columns=[parent_vertex,vertex, "Occurence"])
            table.set_index(parent_vertex, inplace=True)
            for i in parent_vertex_values:
                values = table["Occurence"][i].values
                values_sum = np.sum(values)
                for val in values:
                    if val != 0:
                        term = val * np.log(val / values_sum) * (-1)
                        H += term
    return H


def calculate_number_of_independent_conditional_probabilities(data, vertex, parent_vertices):
    vertex_values = list(set(data[vertex]))
    k = len(vertex_values) - 1
    for parent_vertex in parent_vertices:
        parent_vertex_values = list(set(data[parent_vertex]))
        k *= len(parent_vertex_values)
    return k


def calculate_MDL_of_vertex(data, vertex, parent_vertices):
    n = len(data)
    H = calculate_empiric_enthropy_of_vertex(data, vertex, parent_vertices)
    k = calculate_number_of_independent_conditional_probabilities(data, vertex, parent_vertices)
    L = H + 0.5 * k * np.log(n)
    return L


def calculate_MDL_of_graph(data, graph):
    L_vertices = []
    for vertex in graph:
        L_vertices.append(calculate_MDL_of_vertex(data, vertex, graph[vertex]))
    return np.sum(L_vertices)


def cyclic(g):
    path = set()
    def visit(vertex):
        path.add(vertex)
        for neighbour in g.get(vertex):
            if neighbour in path or visit(neighbour):
                return True
        path.remove(vertex)
        return False

    return any(visit(v) for v in g)
