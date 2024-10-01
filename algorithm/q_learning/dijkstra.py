import numpy as np
import heapq


def edge_costs(feature_function, vertex):
    # print(feature_function)
    return feature_function[vertex][0] + 2 * feature_function[vertex][1]
    # return feature_function[vertex]


class Vertex:
    def __init__(self):
        self.d = float('Inf')
        self.parent = None
        self.finished = False


def dijkstra(feature_function, vertices, source, node_f, neighbors):
    nodes = {}
    for node in vertices:
        nodes[node] = Vertex()
    nodes[source].d = 0
    queue = [(0, source)]  # priority queue
    counter = 0
    while queue:
        d, node = heapq.heappop(queue)
        if nodes[node].finished:
            continue
        nodes[node].finished = True
        if node == node_f:
            break
        neighbor_nodes = neighbors.loc[node].to_numpy()[1:5]
        for neighbor in neighbor_nodes:
            neighbor = int(neighbor)
            if neighbor == 625:
                continue
            if nodes[neighbor].finished:
                continue
            new_d = d + edge_costs(feature_function, neighbor)
            if new_d < nodes[neighbor].d:
                nodes[neighbor].d = new_d
                nodes[neighbor].parent = node
                heapq.heappush(queue, (new_d, neighbor))
        counter = counter + 1
    return nodes, counter
