"""
Part A: Compliance-aware time-window budget allocation.

Setting. For a delivery route, the dispatcher currently promises each stop a
time window [Earliest, Latest]. We hold the SEQUENCE fixed (as historically
planned) and ask a narrower, fully data-grounded question: given the SAME
total promised-window "budget" (sum of historical window widths on that
route -- no extra generosity handed out), how should that budget be spread
across stops to maximize the expected number of on-time arrivals?

Why this is a sound counterfactual. The arrival-time model (T2) is trained
WITHOUT the historical window as an input (see features.py), so it predicts
when a stop will actually be reached as a function of planning-time-only
information. That prediction, plus an empirical residual (uncertainty)
distribution estimated on a held-out calibration split, lets us score ANY
candidate window width for ANY stop without ever needing to have observed
that exact window historically.

Method. Discretize width into a grid; for each (stop, width) pair compute
predicted P(on time) from the calibrated residual model; solve the resulting
multiple-choice knapsack (one width level per stop, total width <= budget,
maximize expected on-time count) EXACTLY with OR-Tools CP-SAT.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

MIN_WIDTH = 15.0     # minutes; realistic operational floor
MAX_WIDTH = 1200.0   # minutes (20h); realistic operational ceiling, well above P99
WIDTH_STEP = 15.0     # discretization granularity (minutes)
SCALE = 10_000        # CP-SAT needs integers; scale probabilities to this resolution


def width_grid(budget: float, n_stops: int) -> np.ndarray:
    hi = min(MAX_WIDTH, max(MIN_WIDTH, budget))  # no single stop needs more than the whole budget
    levels = np.arange(MIN_WIDTH, hi + WIDTH_STEP, WIDTH_STEP)
    return levels


class ResidualCalibrator:
    """Stratified empirical distribution of |actual - predicted| arrival
    time, fit ONLY on a calibration split (the validation weeks), stratified
    by decile of an independent risk score (T1-predicted reorder
    probability). Never touches the test split during fitting."""

    def __init__(self, n_strata: int = 5):
        self.n_strata = n_strata
        self.bin_edges_: np.ndarray | None = None
        self.abs_resid_by_stratum_: dict[int, np.ndarray] = {}
        self.signed_resid_by_stratum_: dict[int, np.ndarray] = {}

    def fit(self, risk_score: np.ndarray, residual: np.ndarray):
        self.bin_edges_ = np.quantile(risk_score, np.linspace(0, 1, self.n_strata + 1))
        self.bin_edges_[0] -= 1e-9
        self.bin_edges_[-1] += 1e-9
        strata = np.digitize(risk_score, self.bin_edges_[1:-1])
        for s in range(self.n_strata):
            vals = residual[strata == s]
            fallback = len(vals) <= 5
            self.abs_resid_by_stratum_[s] = np.sort(np.abs(vals if not fallback else residual))
            self.signed_resid_by_stratum_[s] = np.sort(vals if not fallback else residual)
        return self

    def stratum_of(self, risk_score: np.ndarray) -> np.ndarray:
        return np.digitize(risk_score, self.bin_edges_[1:-1])

    def p_on_time(self, half_width: np.ndarray, stratum: np.ndarray) -> np.ndarray:
        """Convenience wrapper for a window SYMMETRIC around the model's
        prediction (offsets [-half_width, +half_width]). This is what every
        counterfactual method in this project actually uses, since we always
        recentre candidate windows at the predicted arrival time. It is
        NOT valid for the historical status-quo window, whose centre is the
        dispatcher's own midpoint, not the model's prediction -- use
        p_within_offsets for that (see run_timewindow_evaluation.py)."""
        return self.p_within_offsets(-half_width, half_width, stratum)

    def p_within_offsets(self, lower_offset: np.ndarray, upper_offset: np.ndarray,
                          stratum: np.ndarray) -> np.ndarray:
        """General form: P(lower_offset <= residual <= upper_offset | stratum),
        where residual = true_arrival - predicted_arrival. Handles windows
        that are NOT centred on the prediction (e.g. the real historical
        window, centred on whatever the dispatcher originally set)."""
        out = np.empty(len(lower_offset))
        for s in np.unique(stratum):
            mask = stratum == s
            signed = self.signed_resid_by_stratum_[s]
            hi = np.searchsorted(signed, upper_offset[mask], side="right")
            lo = np.searchsorted(signed, lower_offset[mask], side="left")
            out[mask] = (hi - lo) / len(signed)
        return out


def build_probability_matrix(pred_arrival: np.ndarray, budget: float,
                              calibrator: ResidualCalibrator, risk_score: np.ndarray):
    """Shared by the exact (CP-SAT) and greedy allocators below: an n_stops x
    n_levels matrix of P(on time) for each candidate width level on the same
    discretization grid, both centred at the model's prediction. Factored out
    so the exact/greedy comparison (see greedy_allocation) differs ONLY in
    the allocation algorithm, never in the inputs it optimizes over."""
    n = len(pred_arrival)
    levels = width_grid(budget, n)
    strata = calibrator.stratum_of(risk_score)
    P = np.zeros((n, len(levels)))
    for j, w in enumerate(levels):
        half = np.full(n, w / 2.0)
        P[:, j] = calibrator.p_on_time(half, strata)
    return levels, P


def solve_route_allocation(pred_arrival: np.ndarray, budget: float,
                            calibrator: ResidualCalibrator, risk_score: np.ndarray,
                            time_limit_s: float = 2.0) -> dict:
    """One (Route,Week) instance. Returns chosen widths and expected on-time count."""
    n = len(pred_arrival)
    levels, P = build_probability_matrix(pred_arrival, budget, calibrator, risk_score)
    strata = calibrator.stratum_of(risk_score)  # only needed by the infeasible-budget fallback below
    P_scaled = np.round(P * SCALE).astype(int)

    model = cp_model.CpModel()
    x = {}
    for i in range(n):
        for j in range(len(levels)):
            x[i, j] = model.NewBoolVar(f"x_{i}_{j}")
        model.Add(sum(x[i, j] for j in range(len(levels))) == 1)

    budget_units = int(round(budget))
    model.Add(sum(int(levels[j]) * x[i, j] for i in range(n) for j in range(len(levels))) <= budget_units)
    model.Maximize(sum(int(P_scaled[i, j]) * x[i, j] for i in range(n) for j in range(len(levels))))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = 4
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # infeasible (budget too small for even MIN_WIDTH per stop) -> fall back to
        # equal split of the budget, floored at 0
        chosen_widths = np.full(n, max(0.0, budget / n))
        expected_on_time = calibrator.p_on_time(chosen_widths / 2.0, strata).sum()
        return {"widths": chosen_widths, "expected_on_time": float(expected_on_time),
                "status": "infeasible_fallback"}

    chosen_widths = np.zeros(n)
    for i in range(n):
        for j in range(len(levels)):
            if solver.Value(x[i, j]) == 1:
                chosen_widths[i] = levels[j]
    expected_on_time = solver.ObjectiveValue() / SCALE
    return {"widths": chosen_widths, "expected_on_time": float(expected_on_time),
            "status": "optimal" if status == cp_model.OPTIMAL else "feasible"}


def oracle_allocation(pred_arrival: np.ndarray, true_arrival: np.ndarray, budget: float,
                       time_limit_s: float = 2.0) -> dict:
    """Hindsight upper bound for solve_route_allocation: given the SAME width
    grid and SAME total budget as every other method, and (unlike every
    other method) the TRUE arrival times, what is the best achievable count
    of on-time stops? This is a deterministic 0/1 multi-choice knapsack (a
    width level either covers the true arrival or it doesn't -- no
    probability involved, since hindsight has no uncertainty left), solved
    exactly with the same CP-SAT approach as solve_route_allocation so the
    comparison uses an identical feasible set (same MIN_WIDTH floor, same
    discretization). A "fund cheapest-to-guarantee stops first, leave
    whatever's left at zero" greedy is WRONG here: it can leave stops below
    MIN_WIDTH -- a width no other method is ever allowed to choose -- which
    lets a few over-funded stops starve the rest and can push the aggregate
    BELOW non-oracle strategies. Solving the identical knapsack instead
    guarantees oracle >= every other strategy on this budget, as an upper
    bound must be."""
    n = len(pred_arrival)
    levels = width_grid(budget, n)
    n_levels = len(levels)
    need = np.abs(true_arrival - pred_arrival)  # half-width required to cover the true arrival
    hit = (levels[None, :] / 2.0) >= need[:, None]  # n x n_levels: would this level be on time?

    model = cp_model.CpModel()
    x = {}
    for i in range(n):
        for j in range(n_levels):
            x[i, j] = model.NewBoolVar(f"x_{i}_{j}")
        model.Add(sum(x[i, j] for j in range(n_levels)) == 1)
    budget_units = int(round(budget))
    model.Add(sum(int(levels[j]) * x[i, j] for i in range(n) for j in range(n_levels)) <= budget_units)
    model.Maximize(sum(int(hit[i, j]) * x[i, j] for i in range(n) for j in range(n_levels)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = 4
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        chosen_widths = np.full(n, max(0.0, budget / n))
        true_on_time = float(((chosen_widths / 2.0) >= need).sum())
        return {"widths": chosen_widths, "true_on_time_count": true_on_time,
                "status": "infeasible_fallback"}

    chosen_widths = np.zeros(n)
    for i in range(n):
        for j in range(n_levels):
            if solver.Value(x[i, j]) == 1:
                chosen_widths[i] = levels[j]
    true_on_time = float(solver.ObjectiveValue())
    return {"widths": chosen_widths, "true_on_time_count": true_on_time,
            "status": "optimal" if status == cp_model.OPTIMAL else "feasible"}


def greedy_allocation(pred_arrival: np.ndarray, budget: float,
                       calibrator: ResidualCalibrator, risk_score: np.ndarray) -> dict:
    """Ablation baseline for solve_route_allocation: a standard discrete
    greedy / water-filling heuristic on the EXACT SAME (stop, width-level)
    probability grid built by build_probability_matrix. Every stop starts at
    the cheapest level (MIN_WIDTH); repeatedly spend one more WIDTH_STEP of
    budget on whichever stop currently gets the largest probability gain
    from its next available level, until the budget runs out, no stop can
    be upgraded further, or the best available gain is <= 0. No external
    solver, no optimality guarantee -- this answers the reviewer question
    "why not just do this instead of an ILP solve" directly, on identical
    inputs, so any gap measured against solve_route_allocation is
    attributable purely to exact vs. greedy allocation."""
    n = len(pred_arrival)
    levels, P = build_probability_matrix(pred_arrival, budget, calibrator, risk_score)
    n_levels = len(levels)

    cur_level = np.zeros(n, dtype=int)
    remaining = budget - float(levels[0]) * n
    if remaining < 0:
        # even MIN_WIDTH per stop exceeds budget -- same fallback as the
        # exact solver's infeasible branch, for a like-for-like comparison
        strata = calibrator.stratum_of(risk_score)
        chosen_widths = np.full(n, max(0.0, budget / n))
        expected_on_time = calibrator.p_on_time(chosen_widths / 2.0, strata).sum()
        return {"widths": chosen_widths, "expected_on_time": float(expected_on_time),
                "status": "infeasible_fallback"}

    while remaining >= WIDTH_STEP - 1e-9:
        can_upgrade = cur_level < (n_levels - 1)
        if not np.any(can_upgrade):
            break
        idx = np.where(can_upgrade)[0]
        gains = P[idx, cur_level[idx] + 1] - P[idx, cur_level[idx]]
        best_local = int(np.argmax(gains))
        if gains[best_local] <= 0:
            break  # no remaining upgrade improves expected on-time -- stop spending
        cur_level[idx[best_local]] += 1
        remaining -= WIDTH_STEP

    chosen_widths = levels[cur_level]
    expected_on_time = float(P[np.arange(n), cur_level].sum())
    return {"widths": chosen_widths, "expected_on_time": expected_on_time, "status": "greedy"}


def equal_split_allocation(n: int, budget: float) -> np.ndarray:
    """Equal split, discretized to the SAME width grid solve_route_allocation
    and greedy_allocation must use. Earlier this returned a continuous
    budget/n width, which gave it an unfair advantage over the grid-limited
    allocators (needed for exact optimization to be tractable) -- occasionally
    letting naive equal-split score above the exact optimizer purely from
    finer-grained rounding, not from a smarter allocation. Discretizing here
    too makes "optimized >= equal_split" a guaranteed property (equal_split's
    output becomes a valid candidate the exact solver's search always covers)
    instead of something that has to be explained away.

    Every stop gets the same base grid level (the largest level `n` of them
    fit inside budget); leftover budget from flooring to the grid is spent
    by bumping one grid step onto as many stops as it covers, in index order
    -- this stays "equal, non-risk-informed" (all but a few stops share the
    exact same width) while respecting the discretization."""
    levels = width_grid(budget, n)
    affordable = levels[levels * n <= budget]
    if len(affordable) == 0:
        # even the cheapest grid level doesn't fit every stop -- same
        # degenerate case the exact/greedy allocators fall back on: plain
        # continuous equal split, ungridded, for consistency across methods
        return np.full(n, max(0.0, budget / n))
    base = affordable[-1]
    widths = np.full(n, base)
    leftover = budget - base * n
    n_bumps = int(leftover // WIDTH_STEP)
    for i in range(min(n_bumps, n)):
        if widths[i] + WIDTH_STEP <= levels[-1]:
            widths[i] += WIDTH_STEP
    return widths


def ga_allocation(pred_arrival: np.ndarray, budget: float,
                   calibrator: ResidualCalibrator, risk_score: np.ndarray,
                   population_size: int = 30, generations: int = 40,
                   crossover_rate: float = 0.8, mutation_rate: float = 0.1,
                   tournament_size: int = 3, elitism: int = 2,
                   seed: int = 0) -> dict:
    """Supplementary robustness check (not a proposed method): does a
    population-based metaheuristic reach solutions competitive with exact
    CP-SAT optimization on the IDENTICAL discretized (stop, width-level)
    probability grid build_probability_matrix produces? Genetic algorithm,
    not particle swarm, because this is a discrete multiple-choice knapsack
    -- PSO is fundamentally a continuous-optimization method and would need
    a non-standard discrete adaptation, whereas integer-encoded GAs are the
    long-established, natural fit for knapsack-family combinatorial problems.

    Encoding: one gene per stop, indexing into `levels` (same grid the exact
    and greedy allocators use). Tournament selection, uniform crossover,
    per-gene mutation, elitism. Infeasible offspring are repaired (not
    penalised) by repeatedly downgrading whichever gene has the LOWEST
    value-lost-per-width-freed ratio until the budget constraint holds --
    the standard repair strategy for knapsack GAs (e.g. Chu & Beasley 1997),
    which keeps every evaluated individual feasible instead of wasting
    search on infeasible directions.

    Fixed, standard-ish hyperparameters throughout -- this deliberately does
    NOT get its own hyperparameter search. That would be a materially larger
    undertaking (and an odd asymmetry against the zero-hyperparameter exact
    solver) without changing the question this baseline exists to answer:
    given a REASONABLY configured metaheuristic, does it beat, match, or
    trail exact optimization on this problem size?"""
    n = len(pred_arrival)
    levels, P = build_probability_matrix(pred_arrival, budget, calibrator, risk_score)
    n_levels = len(levels)

    if levels[0] * n > budget:
        # even the cheapest level for every stop exceeds budget -- same
        # fallback the exact/greedy allocators use, for a fair comparison
        strata = calibrator.stratum_of(risk_score)
        chosen_widths = np.full(n, max(0.0, budget / n))
        expected_on_time = calibrator.p_on_time(chosen_widths / 2.0, strata).sum()
        return {"widths": chosen_widths, "expected_on_time": float(expected_on_time),
                "status": "infeasible_fallback"}

    rng = np.random.default_rng(seed)
    idx_n = np.arange(n)

    def repair(ind: np.ndarray) -> np.ndarray:
        ind = ind.copy()
        total = levels[ind].sum()
        while total > budget:
            can_downgrade = ind > 0
            if not np.any(can_downgrade):
                break  # degenerate case already handled above; defensive only
            j = np.where(can_downgrade)[0]
            loss = P[j, ind[j]] - P[j, ind[j] - 1]                 # value given up
            width_freed = levels[ind[j]] - levels[ind[j] - 1]      # budget recovered
            density = loss / np.maximum(width_freed, 1e-9)
            worst = j[np.argmin(density)]                          # cheapest to downgrade
            ind[worst] -= 1
            total -= width_freed[np.argmin(density)]
        return ind

    def random_individual() -> np.ndarray:
        return repair(rng.integers(0, n_levels, size=n))

    def fitness(pop: np.ndarray) -> np.ndarray:
        return P[idx_n[None, :], pop].sum(axis=1)

    population = np.stack([random_individual() for _ in range(population_size)])
    fits = fitness(population)
    best_i = int(np.argmax(fits))
    best_ind, best_fit = population[best_i].copy(), float(fits[best_i])

    for _ in range(generations):
        order = np.argsort(-fits)
        new_pop = [population[order[i]].copy() for i in range(min(elitism, population_size))]

        while len(new_pop) < population_size:
            i1 = rng.integers(0, population_size, size=tournament_size)
            i2 = rng.integers(0, population_size, size=tournament_size)
            p1 = population[i1[np.argmax(fits[i1])]]
            p2 = population[i2[np.argmax(fits[i2])]]
            if rng.random() < crossover_rate:
                mask = rng.random(n) < 0.5
                child = np.where(mask, p1, p2)
            else:
                child = p1.copy()
            mut_mask = rng.random(n) < mutation_rate
            if np.any(mut_mask):
                child = child.copy()
                child[mut_mask] = rng.integers(0, n_levels, size=int(mut_mask.sum()))
            new_pop.append(repair(child))

        population = np.stack(new_pop[:population_size])
        fits = fitness(population)
        gen_best_i = int(np.argmax(fits))
        if fits[gen_best_i] > best_fit:
            best_fit, best_ind = float(fits[gen_best_i]), population[gen_best_i].copy()

    chosen_widths = levels[best_ind]
    return {"widths": chosen_widths, "expected_on_time": best_fit, "status": "ga"}
