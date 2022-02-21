"""kruskal's and prim's algorythms"""
from itertools import combinations, groupby
import random
import math
import timeit
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

def gnp_random_connected_graph(num_of_nodes,completeness,draw=False):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi
    graph, but enforcing that the resulting graph is conneted
    """
    edges = combinations(range(num_of_nodes), 2)
    Graph = nx.Graph()
    Graph.add_nodes_from(range(num_of_nodes))

    for _, node_edges in groupby(edges, key = lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        Graph.add_edge(*random_edge)
        for elem in node_edges:
            if random.random() < completeness:
                Graph.add_edge(*elem)
    info = []
    for (u,v,w) in Graph.edges(data=True):
        w['weight'] = random.randint(0,10)
        info.append([u,v,w['weight']])
    if draw:
        plt.figure(figsize=(10,6))
        nx.draw(Graph, node_color='lightblue', with_labels=True, node_size=500)
    return info, num_of_nodes


def create_graph(nodes, completeness):
    graph, n = gnp_random_connected_graph(nodes, completeness)
    graph = sorted(graph, key=lambda graph:graph[2])
    return graph, n



def kruskal(graph):
    """
    function creates a list of edges
    that are included in the base of the graph
    return: base - list of edges in MST
    """
    nodes_set = set()
    edge_dict = {}
    base = []

    #adding the edge from the graph to base if it
    #is minimum weighted and not in the cycle
    for edge in graph:
        #check if possible edge(nodes not in one set)
        if edge[0] not in nodes_set or edge[1] not in nodes_set:
            if edge[0] not in nodes_set and edge[1] not in nodes_set:
                edge_dict[edge[0]] = [edge[0], edge[1]]
                edge_dict[edge[1]] = edge_dict[edge[0]]
            else:
                if not edge_dict.get(edge[0]):
                    edge_dict[edge[1]].append(edge[0])
                    edge_dict[edge[0]] = edge_dict[edge[1]]
                else:
                    edge_dict[edge[0]].append(edge[1])
                    edge_dict[edge[1]] = edge_dict[edge[0]]

            base.append(edge)
            nodes_set.add(edge[0])
            nodes_set.add(edge[1])

    #second round check to join sets
    for edge in graph:
        if edge[1] not in edge_dict[edge[0]]:
            base.append(edge)
            gr1 = edge_dict[edge[0]]
            edge_dict[edge[0]] += edge_dict[edge[1]]
            edge_dict[edge[1]] += gr1
    return base


def get_min(graph, nodes_set):
    """
    the function gets the minimum
    possible weighted edge
    """
    for node in nodes_set:
        current_example = min(graph, \
            key=lambda x: x[2] if \
            (x[0] not in nodes_set or x[1] not in nodes_set) and\
            (x[0] == node or x[1] == node) else
            math.inf)
        if current_example[2] != math.inf :
            return current_example
    return (0,0, math.inf)

def prim(graph, nodes_number):
    """
    creates the base of the graph by prim's algorythm
    current Nodes_set is the set that must be full of nodes
    then base is the list of edges in MST

    for future computing function sets inf for the weight
    of the first point to avoid weight compting problem and
    edge choosing problem

    return: base
    """
    Nodes_set = {0} #starting node: 0
    base = []
    graph[0][2] = math.inf
    while len(Nodes_set) < nodes_number:
        edge = get_min(graph, Nodes_set)
        if edge[2] == math.inf: #no minimum weight possible
            break
        base.append(edge)       #adding edge to the base graph
        Nodes_set.add(edge[0])  #addind nodes to the set of used nodes
        Nodes_set.add(edge[1])
    return base


def visual(completeness):
    """
    vunction creates dictionaries for
    timing of algorythms for nodes in the list
    """
    list2 = [10, 20, 50, 100,150, 200]
    visual_kruskal={} #dicts {nodes_number:time_calculating}
    visual_prim={}

    for nodes in list2:
        graph, nodes = create_graph(nodes,completeness)
        start = timeit.default_timer()
        kruskal(graph)
        stop = timeit.default_timer()
        visual_kruskal[nodes] = stop - start

    for nodes in list2:
        graph, nodes = create_graph(nodes,completeness)
        graph[0][2] = math.inf
        start = timeit.default_timer()
        prim(graph, nodes)
        stop = timeit.default_timer()
        visual_prim[nodes] = stop - start

    print(visual_prim, '\n', visual_kruskal)

    x=list(visual_kruskal.keys())
    x1=list(visual_prim.keys())
    fig, ax = plt.subplots()
    l1, = ax.plot(x, [visual_kruskal[i] for i in x], label='kraskala')
    l2, = ax.plot(x1, [visual_prim[i] for i in x1], label = 'prima')
    ax.legend(handles=[l1, l2], loc='upper right')
    plt.ylabel(f'time(s), completeness={completeness}')
    plt.xlabel('number of nodes')
    plt.show()

def decision_tree():
    """
    creates visualisation of decision tree of graphs with
    different number of nodes and probability of connection
    """
    list2 = [10, 20, 50, 100]
    probability = [0.1, 0.25, 0.5, 0.75, 0.9]
    X = []
    Y = []
    for prb in probability:
        for nodes in list2:
            graph, nodes = create_graph(nodes,p)
            start = timeit.default_timer()
            kruskal(graph)
            stop = timeit.default_timer()
            time = stop-start
            X.append([nodes, prb, time])
            Y.append(0)

        for nodes in list2:
            graph, nodes = create_graph(nodes,p)
            graph[0][2] = math.inf
            start = timeit.default_timer()
            prim(graph, nodes)
            stop = timeit.default_timer()
            X.append([nodes, prb, time])
            Y.append(1)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
    clf = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0)
    clf.fit(X_train, y_train)
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]
    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    tree.plot_tree(clf, feature_names=['number of nodes', 'probability', 'time'],\
        class_names=['kruskal', 'prima'])
    plt.show()
