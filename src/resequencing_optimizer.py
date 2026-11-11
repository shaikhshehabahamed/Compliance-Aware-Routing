"""
Part B (secondary, supporting analysis): compliance-characterised
re-sequencing using an EMPIRICALLY MINED inter-address distance graph.

Why not synthetic coordinates. The raw data gives Haversine leg distances
along two specific sequences per route (planned, actual) -- not full
pairwise coordinates. Rather than reconstructing approximate geometry (which
would inject synthetic, unverifiable distances into the analysis), we mine
every consecutive (address_i, address_j) pair that was EVER observed --
planned or actual -- ANYWHERE in the training+validation period, and use
ONLY those empirically observed distances. This is exact, real data; it is
simply incomplete (most address pairs are never observed adjacent to each
other), so route-level re-optimization is only attempted where the route's
own stops are sufficiently interconnected in this empirical graph, and the
coverage achieved is reported in full rather than assumed.

Scope (deliberately limited, stated up front). We solve a standard
minimum-distance Hamiltonian-path re-planning (TSP) over the empirical graph
-- we do NOT embed the compliance model inside this optimization's objective
(that would require a position-dependent cost term the standard TSP/VRP
formulation does not natively support without added, harder-to-verify
machinery). Instead we solve for the distance-optimal replan and then
SEPARATELY score its predicted compliance-risk profile with the T1 model,
alongside the same score for the historical planned and actual sequences.
This keeps every step individually verifiable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

BIG_M = 1_000_000.0  # effectively "no edge" penalty distance


def mine_adjacency_graph(df: pd.DataFrame) -> dict[tuple[int, int], float]:
    """df: raw (or cleaned) stop-level rows, already restricted to the
    weeks the graph is allowed to learn from (train+val only). Returns a
    dict {(addr_a, addr_b): distance} with addr_a < addr_b (undirected;
    Haversine distance is symmetric), averaged over repeated observations
    of the same pair."""
    edges: dict[tuple[int, int], list[float]] = {}

    def add(a, b, dist):
        if a == b or not np.isfinite(dist):
            return
        key = (a, b) if a < b else (b, a)
        edges.setdefault(key, []).append(dist)

    for (rid, wid), g in df.groupby(["Route ID", "Week ID"]):
        gp = g.sort_values("IndexP")
        addr_p = gp["Address ID"].values
        dist_p = gp["DistanceP"].values
        for k in range(1, len(addr_p)):
            add(addr_p[k - 1], addr_p[k], dist_p[k])

        ga = g.sort_values("IndexA")
        addr_a = ga["Address ID"].values
        dist_a = ga["DistanceA"].values
        for k in range(1, len(addr_a)):
            add(addr_a[k - 1], addr_a[k], dist_a[k])

    return {k: float(np.mean(v)) for k, v in edges.items()}


def route_coverage(addresses: list[int], graph: dict[tuple[int, int], float]) -> float:
    n = len(addresses)
    if n < 2:
        return 1.0
    total_pairs = n * (n - 1) / 2
    known = 0
    for i in range(n):
        for j in range(i + 1, n):
            key = (addresses[i], addresses[j]) if addresses[i] < addresses[j] else (addresses[j], addresses[i])
            if key in graph:
                known += 1
    return known / total_pairs


def build_distance_matrix(addresses: list[int], graph: dict[tuple[int, int], float]) -> np.ndarray:
    n = len(addresses)
    D = np.full((n, n), BIG_M)
    np.fill_diagonal(D, 0.0)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = addresses[i], addresses[j]
            key = (a, b) if a < b else (b, a)
            if key in graph:
                D[i, j] = graph[key]
    return D


def solve_tsp_path(dist_matrix: np.ndarray, start_idx: int = 0, time_limit_s: float = 3.0) -> dict:
    """Open-path TSP: start fixed at `start_idx` (the depot), end free
    (wherever is cheapest), every other node visited exactly once. OR-Tools'
    routing library only solves closed tours natively, so we use the
    standard phantom-end-node construction: an (n+1)-th node with distance 0
    to and from every real node, and tell the manager the vehicle must END
    there. That final "arrival at the phantom" leg is excluded from the
    reported distance, so the objective is exactly the true open-path
    length. Returns the visiting order (real node indices only) and whether
    any BIG_M ("no empirically observed distance") edge had to be used,
    which flags an instance the empirical graph could not fully cover."""
    n = dist_matrix.shape[0]
    phantom = n
    D = np.zeros((n + 1, n + 1))
    D[:n, :n] = dist_matrix
    D[phantom, :] = 0.0
    D[:, phantom] = 0.0

    manager = pywrapcp.RoutingIndexManager(n + 1, 1, [start_idx], [phantom])
    routing = pywrapcp.RoutingModel(manager)

    scale = 100  # integer distance costs, 2 decimal places of precision

    def dist_cb(i, j):
        return int(round(D[manager.IndexToNode(i), manager.IndexToNode(j)] * scale))

    transit_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    # FromSeconds takes an int, which would silently floor any sub-second
    # budget to 0 (i.e. "stop immediately"); use milliseconds so small time
    # limits (these are ~10-15 node instances, sub-second is plenty) work as
    # intended instead of aborting the search before it starts.
    search_params.time_limit.FromMilliseconds(max(1, int(round(time_limit_s * 1000))))

    solution = routing.SolveWithParameters(search_params)
    if solution is None:
        return {"order": None, "total_distance": None, "used_unknown_edge": True, "status": "no_solution"}

    order = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        if node != phantom:
            order.append(node)
        index = solution.Value(routing.NextVar(index))

    total = 0.0
    used_unknown = False
    for a, b in zip(order[:-1], order[1:]):
        d = dist_matrix[a, b]
        total += d
        if d >= BIG_M / 2:
            used_unknown = True
    return {"order": order, "total_distance": total, "used_unknown_edge": used_unknown, "status": "solved"}
