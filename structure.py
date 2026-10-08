import math

import networkx as nx
import numpy as np
import torch


def build_structure(num_nodes, train_edge_index, eta, temperature, rank, seed):
    """FluidC anchors, shortest-path distances and anchor-kernel coordinates."""
    if num_nodes < 2 or eta <= 0 or temperature <= 0 or rank < 1:
        raise ValueError("Invalid structural parameters.")
    edges = train_edge_index.detach().cpu().long()
    if edges.ndim != 2 or edges.size(0) != 2:
        raise ValueError("train_edge_index must have shape [2, E].")
    if edges.numel() and (edges.min() < 0 or edges.max() >= num_nodes):
        raise ValueError("Node indices must be in [0, num_nodes).")
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(sorted({tuple(sorted(e)) for e in edges.t().tolist()
                                 if e[0] != e[1]}))
    count = math.ceil(eta * math.log(num_nodes))
    if not 2 <= count <= num_nodes or rank >= count:
        raise ValueError("Require 2 <= ceil(eta * log(N)) <= N and r < P.")

    components = sorted((sorted(c) for c in nx.connected_components(graph)),
                        key=lambda c: (-len(c), c[0]))
    if count < len(components):
        raise ValueError("P is smaller than the number of connected components. "
                         "Component-wise FluidC cannot form P covering communities.")
    sizes = np.array([len(c) for c in components])
    target = count * sizes / num_nodes
    quotas = np.ones(len(components), dtype=int)
    for _ in range(count - len(components)):
        remaining = np.where(quotas < sizes, target - quotas, -np.inf)
        quotas[int(remaining.argmax())] += 1

    communities = []
    sentinel = np.empty(num_nodes)
    for index, (nodes, quota) in enumerate(zip(components, quotas)):
        subgraph = graph.subgraph(nodes).copy()
        sentinel[nodes] = nx.diameter(subgraph, usebounds=True) + 1
        groups = [set(nodes)] if quota == 1 else list(nx.community.asyn_fluidc(
            subgraph, int(quota), seed=seed + index))
        communities.extend(sorted(groups, key=min))
    anchors = [max(c, key=lambda u: (graph.degree(u), -u)) for c in communities]
    membership = np.empty(num_nodes, dtype=np.int64)
    for index, nodes in enumerate(communities):
        membership[list(nodes)] = index

    # Eqs. (9)--(14).
    distance = np.full((num_nodes, count), np.inf)
    for column, anchor in enumerate(anchors):
        for node, length in nx.single_source_shortest_path_length(graph, anchor).items():
            distance[node, column] = length
    anchor_distance = distance[anchors].copy()
    kernel = np.exp(-np.square(anchor_distance) / temperature)
    inv_degree = kernel.sum(axis=1) ** -0.5
    laplacian = np.eye(count) - inv_degree[:, None] * kernel * inv_degree[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    nontrivial = np.flatnonzero(eigenvalues > 1e-10)
    if len(nontrivial) < rank:
        raise ValueError(f"Only {len(nontrivial)} nontrivial eigenvectors are available.")
    coordinates = eigenvectors[:, nontrivial[:rank]]
    for column in range(rank):
        pivot = np.abs(coordinates[:, column]).argmax()
        if coordinates[pivot, column] < 0:
            coordinates[:, column] *= -1
    distance = np.where(np.isfinite(distance), distance, sentinel[:, None])
    return {
        "distance": torch.tensor(distance, dtype=torch.float32),
        "community": torch.tensor(coordinates[membership], dtype=torch.float32),
        "anchors": torch.tensor(anchors),
        "membership": torch.tensor(membership),
    }
