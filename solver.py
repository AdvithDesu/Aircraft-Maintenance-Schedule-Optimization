"""
solver.py
=========
Simulated Annealing (SA) solver for AMCS QUBO problems, plus Section 6
post-processing repair pipeline.

Sections implemented
--------------------
Section 5.2  Simulated Annealing
Section 6    Post-processing repair (one-hot -> capacity -> spacing ->
             blackout audit -> airworthiness verification)

SA flip rule (from QUBOProblem module docstring)
------------------------------------------------
    Delta-E when flipping bit k  =  Q[k,k]  +  2*(1 - 2*x[k]) * h[k]
    After accepted flip:          h  +=  (1 - 2*x_k_old) * Q[k, :]
    where h = Q @ x is maintained incrementally (O(n) per flip).

Public API
----------
simulated_annealing(qubo, data, ...)          -> SAResult
repair_solution(committed, qubo, data, ...)   -> List[CommittedCheck]
solve_stage(qubo, data, ...)                  -> Tuple[List[CommittedCheck], SAResult]
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from data_utils import AMCSData, day_to_date
from preprocessing import CheckWindowInfo, CommittedCheck
from qubo_builder import QUBOProblem, decode_solution, evaluate_energy


# ---------------------------------------------------------------------------
# SA result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SAResult:
    """
    Output of one simulated annealing run.

    best_bits    : binary solution vector that achieved best_energy
    best_energy  : E(x*) = x*^T Q x* at the best solution found
    init_energy  : E(x_init) at the starting point
    n_iter       : total iterations executed
    n_accepted   : total accepted flips (downhill + accepted uphill)
    n_improved   : number of times best_energy was updated
    T_init       : initial temperature used
    T_final      : temperature at termination
    elapsed_s    : wall-clock time in seconds
    """
    best_bits:   np.ndarray
    best_energy: float
    init_energy: float
    n_iter:      int
    n_accepted:  int
    n_improved:  int
    T_init:      float
    T_final:     float
    elapsed_s:   float


# ---------------------------------------------------------------------------
# Greedy initialiser
# ---------------------------------------------------------------------------

def _greedy_init(qubo: QUBOProblem, data: AMCSData) -> np.ndarray:
    """
    Build a warm-start binary vector:

    Scheduling bits
        For each check window, set the valid_day nearest to due_day.
        This gives zero one-hot violations and minimal capacity violations.

    Slack bits
        Set the first max(0, R_t - H_t) slack bits for each active day to 1
        so that the sum H_t + Y_t = R_t wherever H_t <= R_t.
        This keeps the capacity-constraint term near its minimum.
    """
    M = (data.params.max_c_check if qubo.check_type == "C"
         else data.params.max_a_check)

    bits = np.zeros(qubo.n_vars, dtype=np.int8)

    # --- Scheduling bits + track per-day hangar occupancy ------------------
    load: Dict[int, int] = {}   # day -> count of new checks active on that day

    for wi in qubo.windows:
        if not wi.valid_days:
            continue
        t_best = min(wi.valid_days, key=lambda t: abs(t - wi.due_day))
        bits[qubo.var_index[(wi.tail, wi.k, t_best)]] = 1

        dur = wi.duration if qubo.check_type == "C" else 1
        for occ in range(t_best, t_best + dur):
            load[occ] = load.get(occ, 0) + 1

    # --- Slack bits: fill unused residual capacity -------------------------
    for (t, j), idx in qubo.slack_index.items():
        R_t    = max(0, M - int(qubo.phi[t]))
        H_t    = load.get(t, 0)
        unused = max(0, R_t - H_t)
        if j < unused:
            bits[idx] = 1

    return bits


# ---------------------------------------------------------------------------
# Temperature auto-estimation
# ---------------------------------------------------------------------------

def _estimate_T_init(
    Q:             np.ndarray,
    x:             np.ndarray,
    h:             np.ndarray,
    n_warmup:      int,
    target_accept: float,
    rng:           np.random.Generator,
) -> float:
    """
    Estimate a starting temperature by sampling flip-energy magnitudes.

    T_init is chosen so that a flip whose energy cost equals the mean uphill
    delta-E would be accepted with probability `target_accept`:

        P_accept = exp(-mean_uphill_dE / T_init) = target_accept
        => T_init = -mean_uphill_dE / ln(target_accept)

    Falls back to max|Q_diag| / 10 if no uphill flips are sampled.
    """
    n = len(x)
    uphill_dE: List[float] = []

    for _ in range(n_warmup):
        k = int(rng.integers(n))
        dE = float(Q[k, k]) + 2.0 * float(1 - 2 * int(x[k])) * float(h[k])
        if dE > 0.0:
            uphill_dE.append(dE)

    if not uphill_dE:
        fallback = float(np.max(np.abs(np.diag(Q)))) / 10.0
        return max(fallback, 1.0)

    mean_uphill = float(np.mean(uphill_dE))
    return max(-mean_uphill / math.log(target_accept), 1.0)


# ---------------------------------------------------------------------------
# Simulated Annealing core
# ---------------------------------------------------------------------------

def simulated_annealing(
    qubo:          QUBOProblem,
    data:          AMCSData,
    initial_bits:  Optional[np.ndarray] = None,
    max_iter:      int   = 200_000,
    T_init:        Optional[float] = None,
    alpha:         float = 0.9995,
    T_min:         float = 1.0,
    frozen_limit:  int   = 20_000,
    n_warmup:      int   = 2_000,
    target_accept: float = 0.8,
    rng:           Optional[np.random.Generator] = None,
) -> SAResult:
    """
    Simulated annealing for a QUBO problem  E(x) = x^T Q x.

    Uses the O(n) incremental flip rule:
        delta-E when flipping bit k  =  Q[k,k]  +  2*(1 - 2*x[k]) * h[k]
        After accepted flip k:          h  +=  (1 - 2*x_k_old) * Q[k, :]

    h = Q @ x is initialised once and maintained incrementally so that
    each SA step costs O(n) instead of O(n^2).

    Cooling schedule: geometric  T_{i+1} = alpha * T_i

    Early stopping: terminate when T < T_min OR when no improvement to
    best_energy has occurred for `frozen_limit` consecutive iterations.

    Parameters
    ----------
    qubo          : QUBOProblem from build_c_qubo() or build_a_qubo()
    data          : AMCSData (for greedy initialiser)
    initial_bits  : warm-start vector of length n_vars (dtype int8).
                    If None, the greedy initialiser is used.
    max_iter      : maximum number of flip attempts
    T_init        : starting temperature; auto-estimated if None
    alpha         : geometric cooling factor per iteration
    T_min         : cooling stops (run ends) below this temperature
    frozen_limit  : early-stop after this many consecutive non-improving iters
    n_warmup      : random-flip samples used for T_init auto-estimation
    target_accept : desired acceptance probability for an average uphill flip
                    (used only when T_init is None)
    rng           : numpy Generator for reproducibility

    Returns
    -------
    SAResult with best_bits and run statistics.
    """
    t_wall = time.perf_counter()

    if rng is None:
        rng = np.random.default_rng()

    Q = qubo.Q
    n = qubo.n_vars

    # --- Initial solution ---------------------------------------------------
    x = (_greedy_init(qubo, data) if initial_bits is None
         else initial_bits.copy().astype(np.int8))

    # --- h = Q @ x maintained incrementally --------------------------------
    h = Q @ x.astype(np.float64)

    # --- Energy (computed from h to avoid a separate matrix multiply) -------
    current_E = float(x.astype(np.float64) @ h)
    init_E    = current_E
    best_E    = current_E
    best_x    = x.copy()

    # --- Temperature --------------------------------------------------------
    if T_init is None:
        T_init = _estimate_T_init(Q, x, h, n_warmup, target_accept, rng)

    T            = T_init
    n_accepted   = 0
    n_improved   = 0
    frozen_count = 0
    last_iter    = 0

    # --- Main loop ----------------------------------------------------------
    for i in range(max_iter):
        if T < T_min or frozen_count >= frozen_limit:
            last_iter = i
            break

        k    = int(rng.integers(n))
        x_k  = int(x[k])

        # Metropolis acceptance
        dE = Q[k, k] + 2.0 * float(1 - 2 * x_k) * h[k]

        if dE < 0.0 or rng.random() < math.exp(-dE / T):
            # Accept: flip bit and update h incrementally
            x[k]      = 1 - x_k
            h        += float(1 - 2 * x_k) * Q[k, :]
            current_E += dE
            n_accepted += 1

            if current_E < best_E - 1e-9:
                best_E       = current_E
                best_x       = x.copy()
                n_improved  += 1
                frozen_count = 0
            else:
                frozen_count += 1
        else:
            frozen_count += 1

        T *= alpha
    else:
        last_iter = max_iter

    return SAResult(
        best_bits   = best_x,
        best_energy = best_E,
        init_energy = init_E,
        n_iter      = last_iter,
        n_accepted  = n_accepted,
        n_improved  = n_improved,
        T_init      = T_init,
        T_final     = T,
        elapsed_s   = time.perf_counter() - t_wall,
    )


# ---------------------------------------------------------------------------
# Section 6 repair helpers
# ---------------------------------------------------------------------------

def _build_load(
    assignment: Dict[Tuple[str, int], int],
    win_map:    Dict[Tuple[str, int], CheckWindowInfo],
    check_type: str,
    T:          int,
) -> np.ndarray:
    """Daily hangar load from the current (tail, k) -> start_day assignment."""
    load = np.zeros(T, dtype=int)
    for (tail, k), start in assignment.items():
        dur = win_map[(tail, k)].duration if check_type == "C" else 1
        load[start : min(T, start + dur)] += 1
    return load


def _capacity_repair(
    assignment: Dict[Tuple[str, int], int],
    win_map:    Dict[Tuple[str, int], CheckWindowInfo],
    qubo:       QUBOProblem,
    data:       AMCSData,
    max_passes: int  = 20,
    verbose:    bool = False,
) -> None:
    """
    Resolve capacity violations in-place (Section 6, Step 2).

    Per pass: find the first overloaded day, pick the most-flexible check
    occupying it (widest valid_days), and move it to the nearest alternative
    valid day that does not introduce a new violation elsewhere.
    Repeats until no violations remain or max_passes is reached.
    """
    T = data.params.horizon_days
    M = (data.params.max_c_check if qubo.check_type == "C"
         else data.params.max_a_check)

    for _ in range(max_passes):
        load = _build_load(assignment, win_map, qubo.check_type, T)

        # Find first overloaded day
        t_over = None
        for t in range(T):
            R_t = max(0, M - int(qubo.phi[t]))
            if load[t] > R_t:
                t_over = t
                break

        if t_over is None:
            break   # no violations remain

        R_t = max(0, M - int(qubo.phi[t_over]))

        # Checks active (occupying hangar) on t_over
        active: List[Tuple[str, int]] = []
        for key, start in assignment.items():
            dur = win_map[key].duration if qubo.check_type == "C" else 1
            if start <= t_over < start + dur:
                active.append(key)

        if not active:
            break   # should not happen

        # Most flexible check = longest valid_days list
        key_mv = max(active, key=lambda k: len(win_map[k].valid_days))
        wi      = win_map[key_mv]
        old_s   = assignment[key_mv]
        dur     = wi.duration if qubo.check_type == "C" else 1

        moved = False
        for t_new in sorted(wi.valid_days, key=lambda t: abs(t - wi.due_day)):
            if t_new == old_s:
                continue

            # Check that adding occupancy at t_new (after removing from old_s)
            # does not violate capacity on any newly-occupied day.
            ok = True
            for occ in range(t_new, min(T, t_new + dur)):
                # Net load delta at occ after the move
                delta = 1
                if old_s <= occ < old_s + dur:
                    delta = 0   # occ was and will be occupied by this check
                if load[occ] + delta > max(0, M - int(qubo.phi[occ])):
                    ok = False
                    break

            if ok:
                assignment[key_mv] = t_new
                moved = True
                if verbose:
                    print(f"  [cap-repair] {key_mv}: moved day {old_s} -> {t_new}")
                break

        if not moved and verbose:
            print(f"  [cap-repair] Cannot move {key_mv} off overloaded day {t_over}")


def _spacing_repair(
    assignment: Dict[Tuple[str, int], int],
    win_map:    Dict[Tuple[str, int], CheckWindowInfo],
    delta_min:  int,
    verbose:    bool = False,
) -> None:
    """
    Resolve C-check minimum spacing violations in-place (Section 6, Step 3).

    For any pair of C-check start days {s1, s2} with |s1 - s2| < delta_min:
      1. Try to push the later check to s1 + delta_min (earliest valid day).
      2. If that fails, try to pull the earlier check back to s2 - delta_min
         (latest valid day at or before the target).
    Repeats until no violations remain or max_rounds reached.
    """
    if not assignment:
        return

    changed   = True
    max_rounds = 20

    for _ in range(max_rounds):
        if not changed:
            break
        changed = False

        sorted_keys = sorted(assignment, key=lambda k: assignment[k])

        for i in range(len(sorted_keys) - 1):
            k1 = sorted_keys[i]
            k2 = sorted_keys[i + 1]
            s1, s2 = assignment[k1], assignment[k2]

            if s2 - s1 < delta_min:
                # Try to push k2 forward
                target_fwd = s1 + delta_min
                fwd = [t for t in win_map[k2].valid_days if t >= target_fwd]
                if fwd:
                    assignment[k2] = fwd[0]
                    changed = True
                    if verbose:
                        print(f"  [space-repair] {k2}: day {s2} -> {fwd[0]}")
                    break

                # Try to pull k1 backward
                target_bwd = s2 - delta_min
                bwd = [t for t in win_map[k1].valid_days if t <= target_bwd]
                if bwd:
                    assignment[k1] = bwd[-1]
                    changed = True
                    if verbose:
                        print(f"  [space-repair] {k1}: day {s1} -> {bwd[-1]}")
                    break

                if verbose:
                    print(f"  [space-repair] Cannot resolve: {k1}(d={s1}) "
                          f"vs {k2}(d={s2}) gap={s2-s1} < {delta_min}")


def _blackout_audit(
    assignment: Dict[Tuple[str, int], int],
    qubo:       QUBOProblem,
    data:       AMCSData,
    verbose:    bool = False,
) -> int:
    """
    Count checks assigned to soft-blackout days (Section 6, Step 4).

    Blackout is a soft preference only; no hard repair is applied.
    Returns the count of assignments on soft-blackout days.
    """
    cap_arr = (data.capacity.c_capacity if qubo.check_type == "C"
               else data.capacity.a_capacity)
    M = (data.params.max_c_check if qubo.check_type == "C"
         else data.params.max_a_check)

    count = 0
    for key, start in assignment.items():
        c = int(cap_arr[start])
        if 0 < c < M:   # soft-blackout heuristic (mirrors qubo_builder)
            count += 1
            if verbose:
                print(f"  [blackout] {key} on day {start} "
                      f"(cap={c}, nominal={M})")
    return count


def _airworthiness_check(
    assignment: Dict[Tuple[str, int], int],
    win_map:    Dict[Tuple[str, int], CheckWindowInfo],
    verbose:    bool = False,
) -> int:
    """
    Verify each check starts on or before its due_day (Section 6, Step 5).

    Returns the number of airworthiness violations (start > due_day).
    """
    violations = 0
    for key, start in assignment.items():
        wi = win_map[key]
        if start > wi.due_day:
            violations += 1
            if verbose:
                overdue = start - wi.due_day
                print(f"  [airworthiness] {key} start={start} "
                      f"due={wi.due_day} (overdue by {overdue}d)")
    return violations


# ---------------------------------------------------------------------------
# Repair pipeline entry point
# ---------------------------------------------------------------------------

def repair_solution(
    committed:  List[CommittedCheck],
    qubo:       QUBOProblem,
    data:       AMCSData,
    verbose:    bool = False,
) -> List[CommittedCheck]:
    """
    Apply Section 6 post-processing repair to a decoded solution.

    Pipeline
    --------
    Step 1  One-hot repair     — already applied by decode_solution(); skipped.
    Step 2  Capacity repair    — move overloaded checks to feasible days.
    Step 3  Spacing repair     — C-checks only; enforce |s_i - s_j| >= delta_min.
    Step 4  Blackout audit     — log soft-blackout assignments (no hard change).
    Step 5  Airworthiness check — warn if any check starts after its due_day.

    Parameters
    ----------
    committed : decoded CommittedCheck list (from decode_solution)
    qubo      : QUBOProblem used for this stage
    data      : AMCSData
    verbose   : print repair diagnostics to stdout

    Returns
    -------
    New List[CommittedCheck] with repaired start days.
    """
    if not committed:
        return []

    win_map: Dict[Tuple[str, int], CheckWindowInfo] = {
        (wi.tail, wi.k): wi for wi in qubo.windows
    }

    # Mutable assignment dict: (tail, k) -> start_day
    assignment: Dict[Tuple[str, int], int] = {
        (cc.tail, cc.k): cc.start_day for cc in committed
    }

    # Step 2
    _capacity_repair(assignment, win_map, qubo, data, verbose=verbose)

    # Step 3 (C only)
    if qubo.check_type == "C":
        _spacing_repair(
            assignment, win_map, data.params.start_day_interval, verbose=verbose
        )

    # Step 4
    n_bo = _blackout_audit(assignment, qubo, data, verbose=verbose)
    if n_bo and verbose:
        print(f"  [blackout] {n_bo} assignment(s) on soft-blackout days (soft only)")

    # Step 5
    n_aw = _airworthiness_check(assignment, win_map, verbose=verbose)
    if n_aw and verbose:
        print(f"  [airworthiness] {n_aw} check(s) scheduled past due date")

    # Rebuild CommittedCheck list from updated assignment
    result: List[CommittedCheck] = []
    for cc in committed:
        key       = (cc.tail, cc.k)
        new_start = assignment.get(key, cc.start_day)
        dur       = win_map[key].duration if qubo.check_type == "C" else 1
        result.append(CommittedCheck(
            tail       = cc.tail,
            k          = cc.k,
            check_type = cc.check_type,
            start_day  = new_start,
            duration   = dur,
        ))

    return result


# ---------------------------------------------------------------------------
# Convenience wrapper: SA + decode + repair
# ---------------------------------------------------------------------------

def solve_stage(
    qubo:      QUBOProblem,
    data:      AMCSData,
    sa_kwargs: Optional[Dict] = None,
    rng:       Optional[np.random.Generator] = None,
    verbose:   bool = False,
) -> Tuple[List[CommittedCheck], SAResult]:
    """
    End-to-end: SA solve -> decode -> repair -> CommittedCheck list.

    Parameters
    ----------
    qubo      : QUBO problem (build_c_qubo or build_a_qubo output)
    data      : AMCSData
    sa_kwargs : extra keyword arguments forwarded to simulated_annealing()
    rng       : numpy Generator for reproducibility
    verbose   : print repair diagnostics

    Returns
    -------
    (committed_checks, sa_result)
    """
    result    = simulated_annealing(qubo, data, rng=rng, **(sa_kwargs or {}))
    committed = decode_solution(qubo, result.best_bits, data)
    committed = repair_solution(committed, qubo, data, verbose=verbose)
    return committed, result


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

def print_solver_summary(
    sa:        SAResult,
    committed: List[CommittedCheck],
    qubo:      QUBOProblem,
    data:      AMCSData,
) -> None:
    """Print a concise summary of one SA run and its decoded solution."""
    begin = data.params.begin_day
    print("=" * 65)
    print(f"SA  type={qubo.check_type}  day={qubo.current_day}  "
          f"vars={qubo.n_vars} ({qubo.n_sched} sched + {qubo.n_slack} slack)")
    print(f"  T_init={sa.T_init:.3e}  T_final={sa.T_final:.3e}  "
          f"alpha per-step implicit")
    print(f"  Iterations : {sa.n_iter:>8,}")
    print(f"  Accepted   : {sa.n_accepted:>8,}  "
          f"({100*sa.n_accepted/max(sa.n_iter,1):.1f}%)")
    print(f"  Improved   : {sa.n_improved:>8,}")
    print(f"  E init     : {sa.init_energy:>15.4f}")
    print(f"  E best     : {sa.best_energy:>15.4f}")
    print(f"  Elapsed    : {sa.elapsed_s:.3f}s")
    print(f"\n  Committed checks ({len(committed)}):")
    for cc in committed:
        d = day_to_date(cc.start_day, begin)
        print(f"    {cc.tail:<14} k={cc.k}  "
              f"day={cc.start_day} ({d})  "
              f"dur={cc.duration}d  ends={cc.end_day}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from data_utils import load_amcs_data
    from preprocessing import (
        init_runtime_states, init_committed_from_c_initial,
        run_preprocessing, RollingConfig,
    )
    from qubo_builder import (
        build_c_qubo, build_a_qubo,
        constraint_violations, print_qubo_summary,
    )

    path = sys.argv[1] if len(sys.argv) > 1 else "Scheduling_Input_2017.xlsx"
    data = load_amcs_data(path)

    states      = init_runtime_states(data)
    committed_c = init_committed_from_c_initial(data)
    committed_a: List[CommittedCheck] = []
    config      = RollingConfig()
    rng         = np.random.default_rng(42)

    prep = run_preprocessing(
        data, states, current_day=0,
        committed_c=committed_c, committed_a=committed_a, config=config,
    )

    # ---- Stage 1: C-check QUBO -----------------------------------------------
    c_qubo = build_c_qubo(prep.c_windows, committed_c, data, current_day=0)
    print_qubo_summary(c_qubo, data)

    c_committed, c_sa = solve_stage(
        c_qubo, data,
        sa_kwargs=dict(max_iter=100_000, alpha=0.9995, frozen_limit=15_000),
        rng=rng, verbose=True,
    )
    print_solver_summary(c_sa, c_committed, c_qubo, data)

    viol_c = constraint_violations(c_qubo, c_sa.best_bits, data)
    print(f"  Post-SA violations (raw bits) : {viol_c}")

    # ---- Stage 2: A-check QUBO -----------------------------------------------
    # Merge committed_c from Stage 1 with the pre-existing committed_c
    all_committed_c = committed_c + c_committed

    a_qubo = build_a_qubo(
        prep.a_windows, all_committed_c, committed_a, data, current_day=0
    )
    print()
    print_qubo_summary(a_qubo, data)

    a_committed, a_sa = solve_stage(
        a_qubo, data,
        sa_kwargs=dict(max_iter=200_000, alpha=0.9995, frozen_limit=20_000),
        rng=rng, verbose=True,
    )
    print_solver_summary(a_sa, a_committed, a_qubo, data)

    viol_a = constraint_violations(a_qubo, a_sa.best_bits, data)
    print(f"  Post-SA violations (raw bits) : {viol_a}")

    print("\nDone.")
