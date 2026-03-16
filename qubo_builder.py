"""
qubo_builder.py
===============
Sections 3 and 4 of the AMCS Two-Stage QUBO Formulation.

Stage 1 – C-check QUBO  (Section 3)
Stage 2 – A-check QUBO  (Section 4)

QUBO matrix convention
----------------------
Q is a dense NxN symmetric numpy array.  Energy:

    E(x) = x^T Q x
           = Σ_i Q[i,i]·x_i  +  2·Σ_{i<j} Q[i,j]·x_i·x_j

Because Q is symmetric, Σ_{i,j} Q[i,j]·x_i·x_j already doubles off-diagonal
terms, so for a desired pairwise coefficient c·x_a·x_b we store Q[a,b]=Q[b,a]=c/2.
However, in all three constraint types the formulation already produces even
coefficients (2λ for one-hot, 2λ for capacity, λ for each ordered spacing pair),
so every Q[a,b] entry below ends up being a whole multiple of the relevant λ.

Efficient SA flip rule (stored in QUBOProblem.flip_note):
    ΔE when flipping bit k  =  Q[k,k]  +  2·(1−2·x[k])·(Q @ x)[k]
    After accepted flip:    h  +=  (1−2·x_k_old)·Q[k,:]
    where h = Q @ x is maintained incrementally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from data_utils import AMCSData, day_to_date
from preprocessing import CheckWindowInfo, CommittedCheck, RollingConfig


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PenaltyWeights:
    """
    Penalty weight hierarchy (Sections 3.6 and 4.6).

    Hierarchy:  λ1 >> λ2 >> λ3 >> β >> max(U)
                1e5    1e4   1e3  1e2   ×max_U
    """
    max_U:    float   # largest |U_{i,k,t}| across all active variables
    lambda_1: float   # one-hot (hard: exactly one day per check)
    lambda_2: float   # hangar capacity (hard: respect MC/MA)
    lambda_3: float   # minimum C-check spacing (hard: Δmin = 3 days)
    beta:     float   # blackout soft preference


@dataclass
class QUBOProblem:
    """
    Fully-built QUBO problem for one stage of one rolling solve.

    Variables 0 … n_sched-1  are scheduling bits  x_{i,k,t} / a_{i,k,t}.
    Variables n_sched … n_vars-1  are slack bits  y_{t,j} / z_{t,j}.

    Flip rule for simulated annealing (see module docstring):
        ΔE = Q[k,k]  +  2·(1−2·x[k])·h[k]   where h = Q @ x
        After flip k:  h  +=  (1−2·x_k_old) · Q[k,:]
    """
    Q:          np.ndarray   # shape (n_vars, n_vars), symmetric, dtype float64
    n_vars:     int
    n_sched:    int          # scheduling variable count
    n_slack:    int          # slack variable count

    # index maps
    var_index:   Dict[Tuple[str, int, int], int]  # (tail, k, day) -> idx
    slack_index: Dict[Tuple[int, int], int]       # (day, j)       -> idx

    windows:    List[CheckWindowInfo]
    check_type: str          # 'C' or 'A'
    current_day: int
    penalties:  PenaltyWeights
    phi:        np.ndarray   # background hangar load Φ_t, shape (T,)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _month_idx(t: int, begin_day) -> int:
    return day_to_date(t, begin_day).month - 1


def _background_load(
    committed: List[CommittedCheck],
    check_type: str,
    T: int,
) -> np.ndarray:
    """Φ_t: count of committed checks of `check_type` active on each day."""
    phi = np.zeros(T, dtype=int)
    for cc in committed:
        if cc.check_type != check_type:
            continue
        t0 = max(0, cc.start_day)
        t1 = min(T, cc.end_day)
        if t0 < t1:
            phi[t0:t1] += 1
    return phi


def _penalty_weights(
    windows: List[CheckWindowInfo],
    data:    AMCSData,
) -> PenaltyWeights:
    """
    Compute penalty weight hierarchy from Sections 3.6 / 4.6.

    max_U = max |U_{i,k,t}| over all (i,k,t) in the active variable set.
    A floor of 1.0 prevents division-by-zero for degenerate cases.
    """
    begin_day = data.params.begin_day
    max_U = 0.0
    for wi in windows:
        for t in wi.valid_days:
            mu = float(data.dfh[wi.tail][_month_idx(t, begin_day)])
            U  = mu * (wi.due_day - t)
            if abs(U) > max_U:
                max_U = abs(U)
    max_U = max(max_U, 1.0)
    return PenaltyWeights(
        max_U=max_U,
        lambda_1=1e5 * max_U,
        lambda_2=1e4 * max_U,
        lambda_3=1e3 * max_U,
        beta=1e2 * max_U,
    )


def _soft_blackout(t: int, nominal_cap: int, cap_arr: np.ndarray) -> bool:
    """
    Proxy for B_C / B_A membership (Section 3.5 / 4.4).

    Days that were originally in C_PEAK + C_NOT_ALLOWED (or A_NOT_ALLOWED)
    but were restored by MORE_*_SLOTS will typically have 0 < cap < nominal.
    This heuristic covers that case without requiring a separate raw-blackout set.
    """
    c = int(cap_arr[t])
    return 0 < c < nominal_cap


# ---------------------------------------------------------------------------
# Stage 1: C-check QUBO  (Section 3)
# ---------------------------------------------------------------------------

def build_c_qubo(
    c_windows:   List[CheckWindowInfo],
    committed_c: List[CommittedCheck],
    data:        AMCSData,
    current_day: int,
    config:      Optional[RollingConfig] = None,
) -> QUBOProblem:
    """
    Build the Stage-1 C-check QUBO Q_C(x, y).

    Sections 3.1–3.5 are implemented in order.  Returns a QUBOProblem
    whose Q matrix is ready for simulated annealing.
    """
    if config is None:
        config = RollingConfig()

    T         = data.params.horizon_days
    M_C       = data.params.max_c_check
    delta_min = data.params.start_day_interval   # Δmin = 3
    begin_day = data.params.begin_day

    # ---- Background load and residual capacity ----------------------------
    phi = _background_load(committed_c, "C", T)

    # ---- Variable indexing (scheduling vars) ------------------------------
    # x_{i,k,t}: one bit per (tail, k, valid_day).
    var_index: Dict[Tuple[str, int, int], int] = {}
    n_sched = 0

    # Also track which days have at least one active occupancy
    # (a check starting on day s with duration L occupies days s…s+L-1).
    active_days: Set[int] = set()

    for wi in c_windows:
        for t in wi.valid_days:
            var_index[(wi.tail, wi.k, t)] = n_sched
            n_sched += 1
            for occ in range(t, min(T, t + wi.duration)):
                active_days.add(occ)

    # ---- Variable indexing (slack vars) -----------------------------------
    # y_{t,j} for j = 0 … R_t-1, for each "active" day t.
    slack_index: Dict[Tuple[int, int], int] = {}
    n_slack = 0

    for t in sorted(active_days):
        R_t = max(0, M_C - int(phi[t]))
        for j in range(R_t):
            slack_index[(t, j)] = n_sched + n_slack
            n_slack += 1

    n_vars = n_sched + n_slack

    # ---- Allocate symmetric Q matrix --------------------------------------
    Q = np.zeros((n_vars, n_vars), dtype=np.float64)

    # ---- Penalty weights --------------------------------------------------
    pw = _penalty_weights(c_windows, data)
    lam1, lam2, lam3, beta = pw.lambda_1, pw.lambda_2, pw.lambda_3, pw.beta

    # ====================================================================
    # Section 3.1  Objective: minimise wasted utilisation
    #   U_{i,k,t} = µ^FH_i(m) · (d_{i,k,C} - t)      (diagonal only)
    # ====================================================================
    for wi in c_windows:
        for t in wi.valid_days:
            idx = var_index[(wi.tail, wi.k, t)]
            mu  = float(data.dfh[wi.tail][_month_idx(t, begin_day)])
            U   = mu * (wi.due_day - t)
            Q[idx, idx] += U

    # ====================================================================
    # Section 3.2  One-hot constraint
    #   P_C1 = λ1 · Σ_{(i,k)} (Σ_t x_{i,k,t} - 1)²
    #
    #   Expansion (ignoring constant):
    #     diagonal:    -λ1 per variable
    #     off-diagonal: +λ1 in each direction per intra-check pair
    #       → x^T Q x contribution: 2·λ1·x_a·x_b  (from Q[a,b]+Q[b,a])
    # ====================================================================
    for wi in c_windows:
        idxs = [var_index[(wi.tail, wi.k, t)] for t in wi.valid_days]
        # Diagonal
        for idx in idxs:
            Q[idx, idx] -= lam1
        # Off-diagonal (all pairs within this check)
        for p in range(len(idxs)):
            for q in range(p + 1, len(idxs)):
                a, b = idxs[p], idxs[q]
                Q[a, b] += lam1
                Q[b, a] += lam1

    # ====================================================================
    # Section 3.3  Hangar capacity (one-sided via slack)
    #   P_C2 = λ2 · Σ_t (H_t + Σ_j y_{t,j} - R_t)²
    #
    #   With Z_t = H_t + Y_t, expansion of (Z_t - R_t)² gives:
    #     diagonal:    +λ2·(1 - 2·R_t) per variable in Z_t
    #     off-diagonal: +λ2 in each direction per pair in Z_t
    #       → x^T Q x contribution: 2·λ2·z_a·z_b
    #
    #   A_{t,s,i,k} = 1  iff  s ≤ t < s + L^C_{i,k}
    # ====================================================================
    for t in sorted(active_days):
        R_t = max(0, M_C - int(phi[t]))

        # Scheduling vars active (occupying hangar) on day t
        active_x: List[int] = []
        for wi in c_windows:
            for s in wi.valid_days:
                if s <= t < s + wi.duration:
                    active_x.append(var_index[(wi.tail, wi.k, s)])

        # Slack vars for day t
        active_y: List[int] = [
            slack_index[(t, j)]
            for j in range(R_t)
            if (t, j) in slack_index
        ]

        all_z = active_x + active_y

        # Diagonal contributions
        for v in all_z:
            Q[v, v] += lam2 * (1 - 2 * R_t)

        # Off-diagonal contributions (all pairs in Z_t)
        for p in range(len(all_z)):
            for q in range(p + 1, len(all_z)):
                a, b = all_z[p], all_z[q]
                Q[a, b] += lam2
                Q[b, a] += lam2

    # ====================================================================
    # Section 3.4  Minimum spacing between C-check start dates
    #   P_C3 = λ3 · Σ_{(i,k)≠(i',k')} Σ_{|s-s'|<Δmin} x_{i,k,s}·x_{i',k',s'}
    #
    #   Ordered-pair sum → each unordered pair {v1,v2} contributes 2·λ3·x1·x2.
    #   Stored as Q[a,b] += λ3, Q[b,a] += λ3 (total 2·λ3 in x^T Q x). ✓
    # ====================================================================
    for p in range(len(c_windows)):
        for q in range(p + 1, len(c_windows)):
            wi1, wi2 = c_windows[p], c_windows[q]
            for s1 in wi1.valid_days:
                idx1 = var_index[(wi1.tail, wi1.k, s1)]
                for s2 in wi2.valid_days:
                    if abs(s1 - s2) < delta_min:
                        idx2 = var_index[(wi2.tail, wi2.k, s2)]
                        Q[idx1, idx2] += lam3
                        Q[idx2, idx1] += lam3

    # ====================================================================
    # Section 3.5  Soft penalty: C-check blackout days
    #   P_CB = Σ_{(i,k)} Σ_{t∈T^C_{i,k}} B^C_t · x_{i,k,t}
    #   B^C_t = β_C if t ∈ B_C, else 0     (diagonal only)
    # ====================================================================
    for wi in c_windows:
        for t in wi.valid_days:
            if _soft_blackout(t, M_C, data.capacity.c_capacity):
                Q[var_index[(wi.tail, wi.k, t)], var_index[(wi.tail, wi.k, t)]] += beta

    return QUBOProblem(
        Q=Q,
        n_vars=n_vars,
        n_sched=n_sched,
        n_slack=n_slack,
        var_index=var_index,
        slack_index=slack_index,
        windows=c_windows,
        check_type="C",
        current_day=current_day,
        penalties=pw,
        phi=phi,
    )


# ---------------------------------------------------------------------------
# Stage 2: A-check QUBO  (Section 4)
# ---------------------------------------------------------------------------

def build_a_qubo(
    a_windows:   List[CheckWindowInfo],
    committed_c: List[CommittedCheck],
    committed_a: List[CommittedCheck],
    data:        AMCSData,
    current_day: int,
    config:      Optional[RollingConfig] = None,
) -> QUBOProblem:
    """
    Build the Stage-2 A-check QUBO Q_A(a, z).

    Differences from C-check QUBO (Section 4 vs Section 3):
      - No spacing constraint (Section 4.5).
      - A-check duration L^A = 1 day → occupancy indicator trivially = 1 on
        start day only; no multi-day overlap tracking needed.
      - Background load Φ^A_t comes from committed_a only.
        Committed C-checks ground the aircraft but do NOT consume an A-check
        hangar slot; their effect is already captured by valid_days exclusion
        in preprocessing (Section 2.6).
      - Use a_capacity and M_A.
    """
    if config is None:
        config = RollingConfig()

    T         = data.params.horizon_days
    M_A       = data.params.max_a_check
    begin_day = data.params.begin_day

    LA = 1  # fixed A-check duration

    # ---- Background load from committed A-checks only ---------------------
    phi = _background_load(committed_a, "A", T)

    # ---- Variable indexing (scheduling vars) ------------------------------
    var_index: Dict[Tuple[str, int, int], int] = {}
    n_sched = 0

    # Active days: for LA=1, each valid_day t is only active on day t itself.
    active_days: Set[int] = set()

    for wi in a_windows:
        for t in wi.valid_days:
            var_index[(wi.tail, wi.k, t)] = n_sched
            n_sched += 1
            active_days.add(t)

    # ---- Variable indexing (slack vars) -----------------------------------
    slack_index: Dict[Tuple[int, int], int] = {}
    n_slack = 0

    for t in sorted(active_days):
        R_t = max(0, M_A - int(phi[t]))
        for j in range(R_t):
            slack_index[(t, j)] = n_sched + n_slack
            n_slack += 1

    n_vars = n_sched + n_slack

    # ---- Allocate symmetric Q matrix --------------------------------------
    Q = np.zeros((n_vars, n_vars), dtype=np.float64)

    # ---- Penalty weights --------------------------------------------------
    pw = _penalty_weights(a_windows, data)
    lam1, lam2, beta = pw.lambda_1, pw.lambda_2, pw.beta
    # lambda_3 = 0 for A-checks (no spacing constraint, Section 4.5)

    # ====================================================================
    # Section 4.1  Objective: minimise wasted A-check utilisation
    # ====================================================================
    for wi in a_windows:
        for t in wi.valid_days:
            idx = var_index[(wi.tail, wi.k, t)]
            mu  = float(data.dfh[wi.tail][_month_idx(t, begin_day)])
            U   = mu * (wi.due_day - t)
            Q[idx, idx] += U

    # ====================================================================
    # Section 4.2  One-hot constraint
    #   P_A1 = λ4 · Σ_{(i,k)} (Σ_t a_{i,k,t} - 1)²
    # ====================================================================
    for wi in a_windows:
        idxs = [var_index[(wi.tail, wi.k, t)] for t in wi.valid_days]
        for idx in idxs:
            Q[idx, idx] -= lam1
        for p in range(len(idxs)):
            for q in range(p + 1, len(idxs)):
                a, b = idxs[p], idxs[q]
                Q[a, b] += lam1
                Q[b, a] += lam1

    # ====================================================================
    # Section 4.3  A-check hangar capacity (one-sided via slack)
    #   P_A2 = λ5 · Σ_t (Σ_{(i,k)} a_{i,k,t} + Σ_j z_{t,j} - R^A_t)²
    #
    #   Since L^A = 1, each a_{i,k,t} only contributes to day t (no
    #   multi-day occupancy).  H^A_t = Σ_{(i,k)} a_{i,k,t} (count of
    #   A-checks starting on day t).
    # ====================================================================
    for t in sorted(active_days):
        R_t = max(0, M_A - int(phi[t]))

        # A-check vars that start on day t (each occupies only day t)
        active_x: List[int] = []
        for wi in a_windows:
            if t in wi.valid_days:
                active_x.append(var_index[(wi.tail, wi.k, t)])

        active_y: List[int] = [
            slack_index[(t, j)]
            for j in range(R_t)
            if (t, j) in slack_index
        ]

        all_z = active_x + active_y

        for v in all_z:
            Q[v, v] += lam2 * (1 - 2 * R_t)

        for p in range(len(all_z)):
            for q in range(p + 1, len(all_z)):
                a, b = all_z[p], all_z[q]
                Q[a, b] += lam2
                Q[b, a] += lam2

    # ====================================================================
    # Section 4.4  Soft penalty: A-check blackout days
    # ====================================================================
    for wi in a_windows:
        for t in wi.valid_days:
            if _soft_blackout(t, M_A, data.capacity.a_capacity):
                Q[var_index[(wi.tail, wi.k, t)], var_index[(wi.tail, wi.k, t)]] += beta

    return QUBOProblem(
        Q=Q,
        n_vars=n_vars,
        n_sched=n_sched,
        n_slack=n_slack,
        var_index=var_index,
        slack_index=slack_index,
        windows=a_windows,
        check_type="A",
        current_day=current_day,
        penalties=pw,
        phi=phi,
    )


# ---------------------------------------------------------------------------
# Solution decoding
# ---------------------------------------------------------------------------

def decode_solution(
    qubo: QUBOProblem,
    bits: np.ndarray,
    data: AMCSData,
) -> List[CommittedCheck]:
    """
    Translate a binary solution vector into CommittedCheck objects.

    One-hot repair (Section 6, step 1) is applied here:
      - If no day selected for (i,k): pick the valid_day closest to due_day.
      - If multiple days selected for (i,k): keep only the LATEST
        (minimises wasted FH per formulation Section 6).

    Returns only check instances where a start day could be determined.
    """
    # Invert var_index: idx -> (tail, k, day)
    idx_to_var: Dict[int, Tuple[str, int, int]] = {
        v: k for k, v in qubo.var_index.items()
    }

    # Group selected days by (tail, k)
    selected: Dict[Tuple[str, int], List[int]] = {}
    for idx in range(qubo.n_sched):
        if bits[idx] == 1:
            tail, k, t = idx_to_var[idx]
            selected.setdefault((tail, k), []).append(t)

    committed: List[CommittedCheck] = []

    for wi in qubo.windows:
        key = (wi.tail, wi.k)
        days = selected.get(key, [])

        if len(days) == 0:
            # No day selected — repair: pick valid_day nearest to due_day
            if not wi.valid_days:
                continue   # no feasible day; post-processing will handle
            t_best = min(wi.valid_days, key=lambda t: abs(t - wi.due_day))
        elif len(days) == 1:
            t_best = days[0]
        else:
            # Multiple selected — keep the latest feasible day (Section 6)
            t_best = max(d for d in days if d in wi.valid_days)

        duration = wi.duration if qubo.check_type == "C" else 1
        committed.append(CommittedCheck(
            tail=wi.tail,
            k=wi.k,
            check_type=qubo.check_type,
            start_day=t_best,
            duration=duration,
        ))

    return committed


# ---------------------------------------------------------------------------
# QUBO energy evaluation (for testing / validation)
# ---------------------------------------------------------------------------

def evaluate_energy(qubo: QUBOProblem, bits: np.ndarray) -> float:
    """Compute E(x) = x^T Q x for a given binary vector."""
    x = bits.astype(np.float64)
    return float(x @ qubo.Q @ x)


def constraint_violations(
    qubo:    QUBOProblem,
    bits:    np.ndarray,
    data:    AMCSData,
) -> Dict[str, int]:
    """
    Count hard-constraint violations in a decoded solution.
    Used for post-solve validation before committing.

    Returns dict with keys: 'one_hot', 'capacity', 'spacing'.
    """
    idx_to_var: Dict[int, Tuple[str, int, int]] = {
        v: k for k, v in qubo.var_index.items()
    }

    # ---- One-hot: each (tail, k) must select exactly 1 day ---------------
    selected: Dict[Tuple[str, int], List[int]] = {}
    for idx in range(qubo.n_sched):
        if bits[idx] == 1:
            tail, k, t = idx_to_var[idx]
            selected.setdefault((tail, k), []).append(t)

    one_hot_violations = 0
    for wi in qubo.windows:
        cnt = len(selected.get((wi.tail, wi.k), []))
        if cnt != 1:
            one_hot_violations += 1

    # ---- Capacity: hangar load ≤ residual capacity on each day -----------
    T   = data.params.horizon_days
    cap = (data.capacity.c_capacity if qubo.check_type == "C"
           else data.capacity.a_capacity)
    M   = (data.params.max_c_check if qubo.check_type == "C"
           else data.params.max_a_check)

    # Build load array from selected bits
    load = np.zeros(T, dtype=int)
    for idx in range(qubo.n_sched):
        if bits[idx] == 1:
            tail, k, t = idx_to_var[idx]
            wi = next(w for w in qubo.windows if w.tail == tail and w.k == k)
            dur = wi.duration if qubo.check_type == "C" else 1
            load[t:min(T, t + dur)] += 1

    capacity_violations = 0
    for t in range(T):
        R_t = max(0, M - int(qubo.phi[t]))
        if load[t] > R_t:
            capacity_violations += 1

    # ---- Spacing (C-checks only): no two starts within Δmin days ----------
    spacing_violations = 0
    if qubo.check_type == "C":
        delta_min = data.params.start_day_interval
        starts = []
        for (tail, k), days in selected.items():
            if days:
                starts.append(days[0])
        starts.sort()
        for i in range(len(starts)):
            for j in range(i + 1, len(starts)):
                if abs(starts[i] - starts[j]) < delta_min:
                    spacing_violations += 1

    return {
        "one_hot":  one_hot_violations,
        "capacity": capacity_violations,
        "spacing":  spacing_violations,
    }


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

def print_qubo_summary(qubo: QUBOProblem, data: AMCSData) -> None:
    """Print a concise summary of the QUBO dimensions and penalty weights."""
    pw = qubo.penalties
    print("=" * 60)
    print(f"QUBO  type={qubo.check_type}  day={qubo.current_day}")
    print(f"  Scheduling vars : {qubo.n_sched}")
    print(f"  Slack vars      : {qubo.n_slack}")
    print(f"  Total variables : {qubo.n_vars}")
    print(f"  Q shape         : {qubo.Q.shape}  "
          f"nnz={int(np.count_nonzero(qubo.Q))}")
    print(f"  max_U           : {pw.max_U:.2f}")
    print(f"  lam1 (one-hot)  : {pw.lambda_1:.2e}")
    print(f"  lam2 (capacity) : {pw.lambda_2:.2e}")
    if qubo.check_type == "C":
        print(f"  lam3 (spacing)  : {pw.lambda_3:.2e}")
    print(f"  beta (blackout) : {pw.beta:.2e}")
    print(f"  Checks in QUBO  : {len(qubo.windows)}")
    for wi in qubo.windows:
        print(f"    {wi.tail:<14} k={wi.k}  "
              f"due={wi.due_day}  valid={len(wi.valid_days)} days  "
              f"dur={wi.duration}d")
    print("=" * 60)


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

    path = sys.argv[1] if len(sys.argv) > 1 else "Scheduling_Input_2017.xlsx"
    data = load_amcs_data(path)

    states      = init_runtime_states(data)
    committed_c = init_committed_from_c_initial(data)
    committed_a = []
    config      = RollingConfig()

    prep = run_preprocessing(
        data, states, current_day=0,
        committed_c=committed_c,
        committed_a=committed_a,
        config=config,
    )

    # --- Stage 1: C-check QUBO ---
    c_qubo = build_c_qubo(
        prep.c_windows, committed_c, data, current_day=0, config=config
    )
    print_qubo_summary(c_qubo, data)

    # Quick sanity: all-zeros solution energy (nothing scheduled)
    x_zero = np.zeros(c_qubo.n_vars, dtype=np.int8)
    print(f"\n  E(all-zeros) = {evaluate_energy(c_qubo, x_zero):.4f}  "
          f"(expected: 0)")

    # Greedy solution: for each check window, pick the day closest to due_day
    x_greedy = np.zeros(c_qubo.n_vars, dtype=np.int8)
    for wi in c_qubo.windows:
        t_best = min(wi.valid_days, key=lambda t: abs(t - wi.due_day))
        x_greedy[c_qubo.var_index[(wi.tail, wi.k, t_best)]] = 1
    E_greedy = evaluate_energy(c_qubo, x_greedy)
    print(f"  E(greedy)    = {E_greedy:.4f}")
    viol = constraint_violations(c_qubo, x_greedy, data)
    print(f"  Violations   = {viol}")

    # --- Stage 2: A-check QUBO ---
    a_qubo = build_a_qubo(
        prep.a_windows, committed_c, committed_a, data, current_day=0, config=config
    )
    print()
    print_qubo_summary(a_qubo, data)

    x_greedy_a = np.zeros(a_qubo.n_vars, dtype=np.int8)
    for wi in a_qubo.windows:
        t_best = min(wi.valid_days, key=lambda t: abs(t - wi.due_day))
        x_greedy_a[a_qubo.var_index[(wi.tail, wi.k, t_best)]] = 1
    E_greedy_a = evaluate_energy(a_qubo, x_greedy_a)
    print(f"\n  E(greedy A)  = {E_greedy_a:.4f}")
    viol_a = constraint_violations(a_qubo, x_greedy_a, data)
    print(f"  Violations   = {viol_a}")

    # Decode and show committed checks
    print("\n  Decoded C-check decisions (greedy):")
    for cc in decode_solution(c_qubo, x_greedy, data):
        d = day_to_date(cc.start_day, data.params.begin_day)
        print(f"    {cc.tail:<14} k={cc.k}  day={cc.start_day} ({d})  "
              f"dur={cc.duration}d  ends={cc.end_day}")
