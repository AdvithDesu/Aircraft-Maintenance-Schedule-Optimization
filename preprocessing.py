"""
preprocessing.py
================
Section 2 of the AMCS Two-Stage QUBO Formulation:
  2.1  Predicting due dates  (forward accumulation of FH/FC, month-by-month)
  2.2  Due-date uncertainty
  2.3  Constructing maintenance windows
  2.4  C-check duration lookup  (delegated to AMCSData.get_c_duration)
  2.5  A-into-C merging
  2.6  Aircraft grounding mask
  2.7  Filtering active checks per rolling solve

All day indices are 0-based integers relative to the planning horizon start
(Day 0 = 25 Sep 2017).

Public API
----------
init_runtime_states(data)            -> Dict[str, AircraftRuntimeState]
run_preprocessing(data, states,
                  current_day,
                  committed_c, committed_a,
                  config)            -> PreprocessingResult
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from data_utils import AMCSData, day_to_date

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

Z_ALPHA: float = 2.0        # Confidence parameter (~95% coverage for windows)
W_C_UTIL: int  = 10         # Early-scheduling buffer for C-checks (days)
W_A_UTIL: int  = 5          # Early-scheduling buffer for A-checks (days)
DELTA_MERGE: int = 30       # A-into-C merge lookahead window (days)
ROLLING_HORIZON: int = 60   # Rolling horizon H (days)
COMMIT_WINDOW: int = 7      # How many days of decisions to lock per QUBO solve (days)
LA: int = 1                 # A-check duration (fixed 1 day)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RollingConfig:
    """Tunable parameters for the rolling-horizon preprocessing."""
    z_alpha:         float = Z_ALPHA
    w_c_util:        int   = W_C_UTIL
    w_a_util:        int   = W_A_UTIL
    delta_merge:     int   = DELTA_MERGE
    rolling_horizon: int   = ROLLING_HORIZON
    commit_window:   int   = COMMIT_WINDOW


@dataclass
class AircraftRuntimeState:
    """
    Live state of one aircraft at the current day of the rolling simulation.

    Initialised from AMCSData (C_INITIAL + A_INITIAL) at Day 0 and updated
    each simulated day by the executor.

    next_c_k / next_a_k
        Index of the NEXT check to be scheduled (into c_check_codes for C,
        into the regular A-check sequence for A).  Incremented each time a
        check *completes* (not when it starts).
    """
    tail:  str
    fleet: str

    # ---- C-check accumulators (since last completed C-check) ---------------
    fh_c:     float  # flight hours accumulated
    fc_c:     float  # flight cycles accumulated
    dy_c:     float  # calendar days elapsed
    next_c_k: int    # next uncommitted C-check index (0-based in c_check_codes)

    # ---- A-check accumulators (since last completed A-check) ---------------
    fh_a:     float
    fc_a:     float
    dy_a:     float
    next_a_k: int    # next uncommitted A-check index (0-based in the sequence)

    # ---- Maintenance status flags ------------------------------------------
    in_c_check:     bool = False  # currently in hangar for a C-check
    c_check_end_day: int = -1     # day when ongoing C-check finishes (exclusive)
    in_a_check:     bool = False
    a_check_end_day: int = -1


@dataclass
class CommittedCheck:
    """
    A check that has been locked in by a previous rolling solve (or pre-committed
    from the initial data via C_INITIAL.c_start).
    """
    tail:       str
    k:          int    # 0-based check index (in c_check_codes for C; in A sequence for A)
    check_type: str    # 'C' or 'A'
    start_day:  int    # absolute day index t
    duration:   int    # number of days the aircraft is in the hangar

    @property
    def end_day(self) -> int:
        """First day AFTER the check completes (exclusive end)."""
        return self.start_day + self.duration


@dataclass
class DueDateResult:
    """
    Due-date computation result for one check instance (aircraft i, check k, type c).
    All day values are absolute indices (0-based from begin_day).
    """
    tail:           str
    k:              int
    check_type:     str    # 'C' or 'A'

    d_abs:          int    # binding due day (integer, min of FH/FC/DY)
    d_fh_abs:       float  # FH-based due day
    d_fc_abs:       float  # FC-based due day
    d_dy_abs:       float  # DY-based due day (deterministic)

    binding_metric: str    # 'FH', 'FC', or 'DY'
    sigma_d:        float  # due-date std-dev (days), 0 for DY binding


@dataclass
class CheckWindowInfo:
    """
    Complete preprocessing output for one check instance: due date + maintenance window.

    valid_days
        Sorted list of candidate QUBO start days within [t_early, t_late]
        where the hangar has available capacity (capacity[t] > 0) and,
        for A-checks, the aircraft is not grounded by a C-check.

    Note on blackout soft-penalty:
        Days in B_C/B_A that still have capacity > 0 (due to MORE_*_SLOTS
        overrides) ARE included in valid_days.  The QUBO builder applies
        the soft penalty beta_C / beta_A to those days; they are NOT
        hard-removed here.
    """
    tail:       str
    k:          int
    check_type: str   # 'C' or 'A'

    due_day:    int   # binding due day (absolute)
    sigma_d:    float

    t_early:    int   # earliest candidate start day (after clamping + buffer)
    t_late:     int   # latest  candidate start day

    valid_days: List[int]   # sorted; subset of [t_early, t_late] with capacity > 0
    duration:   int         # expected maintenance duration (days)
    check_code: Optional[float] = None   # e.g. 12.1  (C-checks only)


@dataclass
class PreprocessingResult:
    """Full output of one rolling-horizon preprocessing run."""
    current_day: int

    # Active check windows for the C-check QUBO (Stage 1)
    c_windows: List[CheckWindowInfo]

    # Active check windows for the A-check QUBO (Stage 2),
    # after excluding merged instances
    a_windows: List[CheckWindowInfo]

    # (tail, k) pairs absorbed into scheduled C-checks (Section 2.5)
    merged_a: Set[Tuple[str, int]]

    # G_i(t): grounding mask from committed C-checks (Section 2.6)
    # Dict: tail -> set of grounded day indices
    grounding_mask: Dict[str, Set[int]]

    # Full due-date chains (used by the executor for airworthiness checks)
    due_dates_c: Dict[str, List[DueDateResult]]
    due_dates_a: Dict[str, List[DueDateResult]]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _month_idx(t: int, begin_day: datetime.date) -> int:
    """Return 0-based month index (0=Jan … 11=Dec) for absolute day index t."""
    return day_to_date(t, begin_day).month - 1


def _forward_accumulate(
    remaining: float,
    rates:     np.ndarray,   # shape (12,), monthly daily rates
    start_day: int,
    begin_day: datetime.date,
    T:         int,
) -> float:
    """
    Forward-accumulate a daily utilisation rate month-by-month until
    `remaining` is consumed (Section 2.1 — month-varying utilisation).

    Returns the number of days from `start_day` until the limit is reached.
    Returns float('inf') if the horizon T is exhausted before that.

    Processing is done month-by-month (O(T/30) instead of O(T)) for speed.
    """
    if remaining <= 0.0:
        return 0.0   # already at or past the limit

    accumulated = 0.0
    d = start_day
    current_date = day_to_date(d, begin_day)

    while d < T:
        month_idx = current_date.month - 1
        rate = float(rates[month_idx])

        # Days remaining in the current calendar month
        if current_date.month == 12:
            next_month = datetime.date(current_date.year + 1, 1, 1)
        else:
            next_month = datetime.date(current_date.year, current_date.month + 1, 1)

        days_in_chunk = min((next_month - current_date).days, T - d)

        if rate <= 0.0:
            # No utilisation in this month — just advance
            d += days_in_chunk
            current_date = day_to_date(d, begin_day)
            continue

        chunk_total = rate * days_in_chunk

        if accumulated + chunk_total >= remaining:
            # Limit is reached within this chunk
            days_needed = (remaining - accumulated) / rate
            return float(d - start_day) + days_needed

        accumulated += chunk_total
        d += days_in_chunk
        current_date = day_to_date(d, begin_day)

    return float("inf")   # horizon exhausted without reaching the limit


def _compute_sigma_d(
    fh_std_arr: np.ndarray,  # σ^FH_i(m) for all months
    dfh_arr:    np.ndarray,  # µ^FH_i(m)
    dfc_arr:    np.ndarray,  # µ^FC_i(m)
    d_fh_rel:   float,       # FH-based due date (days from accumulation start)
    d_fc_rel:   float,       # FC-based due date
    binding:    str,         # 'FH', 'FC', or 'DY'
    month_idx:  int,
) -> float:
    """
    Delta-method approximation of due-date uncertainty (Section 2.2).

        sigma_d,FH ≈ (sigma^FH_i / mu^FH_i) · sqrt(d^FH_i,k,c)
        sigma_d,FC ≈ (sigma^FH_i / mu^FH_i) · sqrt(d^FC_i,k,c)
            (same CV assumed for FC, since no FC_STD is available in the dataset)
        sigma_d,DY = 0  (deterministic)

    Returns sigma_d for the binding metric.
    """
    if binding == "DY":
        return 0.0

    mu_fh  = float(dfh_arr[month_idx])
    sig_fh = float(fh_std_arr[month_idx])

    if mu_fh <= 0.0:
        return 0.0

    cv = sig_fh / mu_fh   # coefficient of variation (same for FH and FC)

    if binding == "FH":
        d_rel = max(0.0, d_fh_rel)
    else:  # 'FC'
        d_rel = max(0.0, d_fc_rel)

    return cv * math.sqrt(d_rel)


def _build_window(
    tail:        str,
    k:           int,
    check_type:  str,
    due_day:     int,
    sigma_d:     float,
    duration:    int,
    check_code:  Optional[float],
    current_day: int,
    horizon_end: int,   # current_day + H
    capacity_arr: np.ndarray,  # c_capacity or a_capacity
    grounding:   Set[int],     # grounded day indices (empty for C-checks)
    config:      RollingConfig,
) -> CheckWindowInfo:
    """
    Build the maintenance window for one check instance (Section 2.3).

    t_early = d - z_alpha · sigma_d - W_util
    t_late  = d + z_alpha · sigma_d

    Clamping:
      t_early >= current_day
      t_late  <= horizon_end   (current_day + H)

    valid_days = {t in [t_early, t_late] : capacity[t] > 0 AND t not in grounding}
    """
    w_util = config.w_c_util if check_type == "C" else config.w_a_util

    t_early_raw = due_day - config.z_alpha * sigma_d - w_util
    t_late_raw  = due_day + config.z_alpha * sigma_d

    t_early = max(current_day, int(math.floor(t_early_raw)))
    t_late  = min(horizon_end, int(math.ceil(t_late_raw)))

    T = len(capacity_arr)
    valid_days: List[int] = []
    for t in range(t_early, t_late + 1):
        if t < 0 or t >= T:
            continue
        if capacity_arr[t] > 0 and t not in grounding:
            valid_days.append(t)

    return CheckWindowInfo(
        tail=tail,
        k=k,
        check_type=check_type,
        due_day=due_day,
        sigma_d=sigma_d,
        t_early=t_early,
        t_late=t_late,
        valid_days=valid_days,
        duration=duration,
        check_code=check_code,
    )


# ---------------------------------------------------------------------------
# Grounding mask (Section 2.6)
# ---------------------------------------------------------------------------

def build_grounding_mask(
    committed_c: List[CommittedCheck],
    aircraft:    List[str],
    T:           int,
) -> Dict[str, Set[int]]:
    """
    Build G_i(t): the set of grounded days per aircraft from committed C-checks.

    G_i(t) = 1 iff aircraft i is in the hangar for a committed C-check on day t.
    A C-check starting on day s with duration L occupies days s, s+1, ..., s+L-1.

    Returns dict: tail -> Set[day index].
    """
    mask: Dict[str, Set[int]] = {tail: set() for tail in aircraft}

    for cc in committed_c:
        if cc.check_type != "C":
            continue
        grounded_days = set(range(cc.start_day, min(cc.end_day, T)))
        mask[cc.tail].update(grounded_days)

    return mask


# ---------------------------------------------------------------------------
# Due-date computation (Sections 2.1 – 2.2) — per aircraft, chained
# ---------------------------------------------------------------------------

def _c_check_due_dates(
    tail:        str,
    state:       AircraftRuntimeState,
    data:        AMCSData,
    current_day: int,
    committed_c: List[CommittedCheck],
) -> List[DueDateResult]:
    """
    Compute due-date info for all upcoming C-check instances of one aircraft.

    For k = next_c_k (first uncommitted check):
        accumulated counters come from `state`.

    For k > next_c_k:
        counters reset to 0 after the previous check completes.
        If check k-1 is committed, chain through its actual end_day.
        Otherwise, chain through its predicted end_day (due_day + duration).

    Committed checks in committed_c are SKIPPED (not re-scheduled) but
    their actual end_day is used to anchor the next check's accumulation.
    """
    params    = data.params
    begin_day = params.begin_day
    T         = params.horizon_days

    cs = data.c_initial[tail]

    # Index set of committed C-check k-values for this aircraft
    committed_map: Dict[int, CommittedCheck] = {
        cc.k: cc
        for cc in committed_c
        if cc.tail == tail and cc.check_type == "C"
    }

    results: List[DueDateResult] = []

    # Accumulated state at the start of the accumulation window
    accum_fh      = state.fh_c
    accum_fc      = state.fc_c
    accum_dy      = state.dy_c
    accum_from    = current_day  # absolute day from which forward accumulation begins

    for k in range(state.next_c_k, len(data.c_check_codes[tail])):
        # Effective limits:
        # k == next_c_k  →  use the actual limits from C_INITIAL
        #                   (incorporates TOLU used in the prior check)
        # k >  next_c_k  →  TOLU = 0 assumed (fresh check, full tolerance available)
        if k == state.next_c_k:
            lim_dy = cs.eff_limit_dy
            lim_fh = cs.eff_limit_fh
            lim_fc = cs.eff_limit_fc
        else:
            lim_dy = cs.ci_dy + cs.tol_dy
            lim_fh = cs.ci_fh + cs.tol_fh
            lim_fc = cs.ci_fc + cs.tol_fc

        remaining_fh = lim_fh - accum_fh
        remaining_fc = lim_fc - accum_fc
        remaining_dy = lim_dy - accum_dy

        # ---- Forward-accumulate to find each due date ----------------------
        d_fh_rel = _forward_accumulate(
            remaining_fh, data.dfh[tail], accum_from, begin_day, T
        )
        d_fc_rel = _forward_accumulate(
            remaining_fc, data.dfc[tail], accum_from, begin_day, T
        )
        d_dy_rel = max(0.0, float(remaining_dy))

        d_fh_abs = accum_from + d_fh_rel
        d_fc_abs = accum_from + d_fc_rel
        d_dy_abs = accum_from + d_dy_rel

        # ---- Binding metric (Section 2.1) ----------------------------------
        d_abs_float = min(d_fh_abs, d_fc_abs, d_dy_abs)
        if d_fh_abs <= d_fc_abs and d_fh_abs <= d_dy_abs:
            binding = "FH"
        elif d_fc_abs <= d_dy_abs:
            binding = "FC"
        else:
            binding = "DY"

        # ---- Due-date uncertainty (Section 2.2) ----------------------------
        month = _month_idx(max(0, accum_from), begin_day)
        sigma_d = _compute_sigma_d(
            data.fh_std[tail], data.dfh[tail], data.dfc[tail],
            d_fh_rel, d_fc_rel, binding, month,
        )

        results.append(DueDateResult(
            tail=tail,
            k=k,
            check_type="C",
            d_abs=int(math.floor(d_abs_float)),
            d_fh_abs=d_fh_abs,
            d_fc_abs=d_fc_abs,
            d_dy_abs=d_dy_abs,
            binding_metric=binding,
            sigma_d=sigma_d,
        ))

        # ---- Chain to next check -------------------------------------------
        # Determine when this check completes so accumulators can be reset.
        duration = data.get_c_duration(tail, k)

        if k in committed_map:
            # Known actual completion: use it
            next_accum_from = committed_map[k].end_day
        else:
            # Estimate: starts at due day, completes duration days later
            next_accum_from = int(math.floor(d_abs_float)) + duration

        # If the next accumulation start is already beyond the horizon, stop
        if next_accum_from >= T:
            break

        # Reset accumulators for k+1
        accum_fh   = 0.0
        accum_fc   = 0.0
        accum_dy   = 0.0
        accum_from = next_accum_from

    return results


def _a_check_due_dates(
    tail:        str,
    state:       AircraftRuntimeState,
    data:        AMCSData,
    current_day: int,
    committed_c: List[CommittedCheck],
    committed_a: List[CommittedCheck],
    max_instances: int = 10,
) -> List[DueDateResult]:
    """
    Compute due-date info for upcoming A-check instances of one aircraft.

    A-checks are a regular recurring sequence (no pre-defined code list).
    Counters reset when an A-check OR a C-check completes (Section 5.3).

    Counter-reset events that affect the A-check chain:
      1. Committed A-check at day t_A:  A-counters reset to 0 at t_A + L^A.
      2. Committed C-check at day t_C:  BOTH counters reset to 0 at t_C + L^C.

    We track the cumulative FH/FC/DY from the most recent reset event.
    max_instances caps the chain to avoid unbounded computation.
    """
    params    = data.params
    begin_day = params.begin_day
    T         = params.horizon_days

    as_ = data.a_initial[tail]

    # Build a sorted timeline of all counter-reset events for this aircraft
    # Each event: (reset_day, 'A' or 'C')
    reset_events: List[Tuple[int, str]] = []

    for cc in committed_c:
        if cc.tail == tail and cc.check_type == "C":
            reset_events.append((cc.end_day, "C"))

    for ca in committed_a:
        if ca.tail == tail and ca.check_type == "A":
            reset_events.append((ca.end_day, "A"))

    reset_events.sort(key=lambda x: x[0])

    # Committed A-check k-values (to skip — already locked in)
    committed_a_map: Dict[int, CommittedCheck] = {
        ca.k: ca
        for ca in committed_a
        if ca.tail == tail and ca.check_type == "A"
    }

    results: List[DueDateResult] = []

    accum_fh   = state.fh_a
    accum_fc   = state.fc_a
    accum_dy   = state.dy_a
    accum_from = current_day

    # Effective limits (TOLU = 0 assumed for all future checks since all
    # current TOLU values in the dataset are 0)
    lim_dy = as_.eff_limit_dy
    lim_fh = as_.eff_limit_fh
    lim_fc = as_.eff_limit_fc

    k = state.next_a_k

    for _ in range(max_instances):
        if accum_from >= T:
            break

        # ---- Check for a counter-reset event before the due date -----------
        # If a C-check completes between accum_from and the A-check due date,
        # the A-counters reset at that C-check completion, potentially
        # deferring this A-check (or merging it, handled in Section 2.5).
        # We advance the accumulation start to the latest reset before d_abs.
        # First compute tentative due dates
        remaining_fh = lim_fh - accum_fh
        remaining_fc = lim_fc - accum_fc
        remaining_dy = lim_dy - accum_dy

        d_fh_rel = _forward_accumulate(
            remaining_fh, data.dfh[tail], accum_from, begin_day, T
        )
        d_fc_rel = _forward_accumulate(
            remaining_fc, data.dfc[tail], accum_from, begin_day, T
        )
        d_dy_rel = max(0.0, float(remaining_dy))

        d_fh_abs = accum_from + d_fh_rel
        d_fc_abs = accum_from + d_fc_rel
        d_dy_abs = accum_from + d_dy_rel
        d_abs_float = min(d_fh_abs, d_fc_abs, d_dy_abs)

        # Look for a C-check that completes between accum_from and d_abs
        # (and resets A-check counters before we reach the A-check due date)
        reset_applied = False
        for reset_day, reset_type in reset_events:
            if accum_from < reset_day <= d_abs_float:
                if reset_type == "C":
                    # C-check completion resets BOTH A and C counters
                    accum_fh   = 0.0
                    accum_fc   = 0.0
                    accum_dy   = 0.0
                    accum_from = reset_day
                    reset_applied = True
                    break
                # Note: A-check completion events are already tracked via
                # committed_a_map — they advance k, handled below.

        if reset_applied:
            # Re-compute with reset accumulators and the same k
            continue

        # ---- Binding metric and uncertainty --------------------------------
        if d_fh_abs <= d_fc_abs and d_fh_abs <= d_dy_abs:
            binding = "FH"
        elif d_fc_abs <= d_dy_abs:
            binding = "FC"
        else:
            binding = "DY"

        month = _month_idx(max(0, accum_from), begin_day)
        sigma_d = _compute_sigma_d(
            data.fh_std[tail], data.dfh[tail], data.dfc[tail],
            d_fh_rel, d_fc_rel, binding, month,
        )

        results.append(DueDateResult(
            tail=tail,
            k=k,
            check_type="A",
            d_abs=int(math.floor(d_abs_float)),
            d_fh_abs=d_fh_abs,
            d_fc_abs=d_fc_abs,
            d_dy_abs=d_dy_abs,
            binding_metric=binding,
            sigma_d=sigma_d,
        ))

        # ---- Chain to next A-check -----------------------------------------
        if k in committed_a_map:
            next_accum_from = committed_a_map[k].end_day
        else:
            next_accum_from = int(math.floor(d_abs_float)) + LA

        accum_fh   = 0.0
        accum_fc   = 0.0
        accum_dy   = 0.0
        accum_from = next_accum_from

        # Reset to full interval for all future checks (TOLU = 0)
        lim_dy = as_.ci_dy + as_.tol_dy
        lim_fh = as_.ci_fh + as_.tol_fh
        lim_fc = as_.ci_fc + as_.tol_fc

        k += 1

    return results


# ---------------------------------------------------------------------------
# A-into-C merging (Section 2.5)
# ---------------------------------------------------------------------------

def identify_merged_a_checks(
    a_due_results: Dict[str, List[DueDateResult]],
    committed_c:   List[CommittedCheck],
    config:        RollingConfig,
) -> Set[Tuple[str, int]]:
    """
    Identify A-check instances that will be absorbed by a scheduled C-check
    (Section 2.5).

    Merge condition: aircraft i has a committed C-check starting on t_C
    with duration L^C, AND the A-check due day d^A satisfies:

        t_C - delta_merge  <=  d^A  <=  t_C + L^C

    Returns a set of (tail, k) pairs to be excluded from the A-check QUBO.
    """
    merged: Set[Tuple[str, int]] = set()

    # Group committed C-checks by aircraft tail
    c_by_tail: Dict[str, List[CommittedCheck]] = {}
    for cc in committed_c:
        if cc.check_type == "C":
            c_by_tail.setdefault(cc.tail, []).append(cc)

    for tail, a_results in a_due_results.items():
        c_checks = c_by_tail.get(tail, [])
        if not c_checks:
            continue

        for a_res in a_results:
            d_a = a_res.d_abs
            for cc in c_checks:
                merge_lo = cc.start_day - config.delta_merge
                merge_hi = cc.start_day + cc.duration   # = cc.end_day
                if merge_lo <= d_a <= merge_hi:
                    merged.add((tail, a_res.k))
                    break   # one merge per A-check is enough

    return merged


# ---------------------------------------------------------------------------
# Active-set construction (Section 2.7) + window building
# ---------------------------------------------------------------------------

def _compute_c_windows(
    data:        AMCSData,
    states:      Dict[str, AircraftRuntimeState],
    current_day: int,
    committed_c: List[CommittedCheck],
    config:      RollingConfig,
    due_results: Dict[str, List[DueDateResult]],
) -> List[CheckWindowInfo]:
    """
    Build C-check maintenance windows for all active instances (Section 2.3).

    Active = window [t_early, t_late] overlaps [current_day, current_day + H].
    Committed checks (already in committed_c) are skipped.
    """
    horizon_end  = current_day + config.rolling_horizon
    capacity_arr = data.capacity.c_capacity
    windows: List[CheckWindowInfo] = []

    # k-values already committed per tail
    committed_k: Dict[str, Set[int]] = {}
    for cc in committed_c:
        if cc.check_type == "C":
            committed_k.setdefault(cc.tail, set()).add(cc.k)

    for tail in data.aircraft:
        for dr in due_results.get(tail, []):
            if dr.k in committed_k.get(tail, set()):
                continue   # already committed — don't re-schedule

            duration  = data.get_c_duration(tail, dr.k)
            code      = data.c_check_codes[tail][dr.k]

            wi = _build_window(
                tail=tail, k=dr.k, check_type="C",
                due_day=dr.d_abs, sigma_d=dr.sigma_d,
                duration=duration, check_code=code,
                current_day=current_day, horizon_end=horizon_end,
                capacity_arr=capacity_arr, grounding=set(),
                config=config,
            )

            # Section 2.7: include only if window overlaps the rolling horizon
            if wi.valid_days:
                windows.append(wi)

    return windows


def _compute_a_windows(
    data:          AMCSData,
    states:        Dict[str, AircraftRuntimeState],
    current_day:   int,
    committed_c:   List[CommittedCheck],
    committed_a:   List[CommittedCheck],
    merged_a:      Set[Tuple[str, int]],
    grounding_mask: Dict[str, Set[int]],
    config:        RollingConfig,
    due_results:   Dict[str, List[DueDateResult]],
) -> List[CheckWindowInfo]:
    """
    Build A-check maintenance windows for all active, non-merged instances.

    Exclusions (Section 2.5 and 2.6):
      - Instances in `merged_a`  (absorbed into a C-check)
      - Instances already in `committed_a` (already scheduled)
      - Days where aircraft is grounded by a C-check (via grounding_mask)
      - Days with zero A-check hangar capacity
    """
    horizon_end  = current_day + config.rolling_horizon
    capacity_arr = data.capacity.a_capacity
    windows: List[CheckWindowInfo] = []

    committed_k_a: Dict[str, Set[int]] = {}
    for ca in committed_a:
        if ca.check_type == "A":
            committed_k_a.setdefault(ca.tail, set()).add(ca.k)

    for tail in data.aircraft:
        grounding = grounding_mask.get(tail, set())

        for dr in due_results.get(tail, []):
            if (tail, dr.k) in merged_a:
                continue   # merged into C-check
            if dr.k in committed_k_a.get(tail, set()):
                continue   # already committed

            wi = _build_window(
                tail=tail, k=dr.k, check_type="A",
                due_day=dr.d_abs, sigma_d=dr.sigma_d,
                duration=LA, check_code=None,
                current_day=current_day, horizon_end=horizon_end,
                capacity_arr=capacity_arr, grounding=grounding,
                config=config,
            )

            if wi.valid_days:
                windows.append(wi)

    return windows


# ---------------------------------------------------------------------------
# State initialisation
# ---------------------------------------------------------------------------

def init_runtime_states(data: AMCSData) -> Dict[str, AircraftRuntimeState]:
    """
    Initialise AircraftRuntimeState for every aircraft from AMCSData at Day 0.

    For aircraft that have a pre-committed C-check in C_INITIAL (c_start != -1),
    the corresponding CommittedCheck is NOT created here — the caller is
    responsible for populating committed_c from the initial data if needed.
    """
    states: Dict[str, AircraftRuntimeState] = {}

    for tail in data.aircraft:
        cs  = data.c_initial[tail]
        as_ = data.a_initial[tail]

        states[tail] = AircraftRuntimeState(
            tail=tail,
            fleet=cs.fleet,
            # C-check counters from C_INITIAL
            fh_c=cs.fh_c,
            fc_c=cs.fc_c,
            dy_c=cs.dy_c,
            next_c_k=0,
            # A-check counters from A_INITIAL
            fh_a=as_.fh_a,
            fc_a=as_.fc_a,
            dy_a=as_.dy_a,
            next_a_k=0,
        )

    return states


def init_committed_from_c_initial(data: AMCSData) -> List[CommittedCheck]:
    """
    Build the initial committed_c list from pre-committed C-checks stored in
    C_INITIAL (c_start != -1 indicates a check already locked in at Day 0).
    """
    committed: List[CommittedCheck] = []

    for tail in data.aircraft:
        cs = data.c_initial[tail]
        if cs.c_start != -1 and cs.c_end != -1:
            duration = cs.c_end - cs.c_start
            committed.append(CommittedCheck(
                tail=tail,
                k=0,            # Day-0 pre-committed check is always k=0
                check_type="C",
                start_day=cs.c_start,
                duration=max(1, duration),
            ))

    return committed


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_preprocessing(
    data:        AMCSData,
    states:      Dict[str, AircraftRuntimeState],
    current_day: int,
    committed_c: List[CommittedCheck],
    committed_a: List[CommittedCheck],
    config:      Optional[RollingConfig] = None,
) -> PreprocessingResult:
    """
    Execute the full Section-2 preprocessing for one rolling-horizon solve.

    Steps (following Section 5.1 of the algorithm):
      Step 3  – Predict C-check due dates  (2.1 + 2.2)
      Step 4  – Build C-check windows      (2.3)
      Step 8  – A-into-C merging           (2.5)
      Step 9  – Predict A-check due dates  (2.1 + 2.2)
      Step 10 – Build A-check windows      (2.3 + 2.6)

    Parameters
    ----------
    data        : AMCSData from data_utils.load_amcs_data()
    states      : per-aircraft runtime state dict (from init_runtime_states()
                  or updated by the executor)
    current_day : 0-based day index of the current solve (t_now)
    committed_c : C-checks committed by previous solves (locked in)
    committed_a : A-checks committed by previous solves
    config      : rolling-horizon tuning parameters (defaults if None)

    Returns
    -------
    PreprocessingResult with all window info and auxiliary structures.
    """
    if config is None:
        config = RollingConfig()

    T = data.params.horizon_days

    # ---- Step 3: Predict C-check due dates ---------------------------------
    due_dates_c: Dict[str, List[DueDateResult]] = {}
    for tail in data.aircraft:
        due_dates_c[tail] = _c_check_due_dates(
            tail, states[tail], data, current_day, committed_c
        )

    # ---- Step 4: Build C-check windows + filter active (Section 2.7) ------
    c_windows = _compute_c_windows(
        data, states, current_day, committed_c, config, due_dates_c
    )

    # ---- Step 6a: Build grounding mask from committed C-checks (2.6) -------
    grounding_mask = build_grounding_mask(committed_c, data.aircraft, T)

    # ---- Step 8: A-into-C merging ------------------------------------------
    # First compute A-check due dates tentatively to identify merge candidates
    due_dates_a_tentative: Dict[str, List[DueDateResult]] = {}
    for tail in data.aircraft:
        due_dates_a_tentative[tail] = _a_check_due_dates(
            tail, states[tail], data, current_day, committed_c, committed_a
        )

    merged_a = identify_merged_a_checks(due_dates_a_tentative, committed_c, config)

    # ---- Step 9: Final A-check due dates (same as tentative, kept separate
    #              to allow future re-computation post-merge if needed) -------
    due_dates_a = due_dates_a_tentative   # no re-computation needed here

    # ---- Step 10: Build A-check windows ------------------------------------
    a_windows = _compute_a_windows(
        data, states, current_day, committed_c, committed_a,
        merged_a, grounding_mask, config, due_dates_a
    )

    return PreprocessingResult(
        current_day=current_day,
        c_windows=c_windows,
        a_windows=a_windows,
        merged_a=merged_a,
        grounding_mask=grounding_mask,
        due_dates_c=due_dates_c,
        due_dates_a=due_dates_a,
    )


# ---------------------------------------------------------------------------
# Diagnostic printout
# ---------------------------------------------------------------------------

def print_preprocessing_summary(result: PreprocessingResult, data: AMCSData) -> None:
    """Quick human-readable summary of one preprocessing run."""
    p = data.params
    horizon_end = result.current_day + ROLLING_HORIZON
    cur_date    = day_to_date(result.current_day, p.begin_day)
    end_date    = day_to_date(min(horizon_end, p.horizon_days - 1), p.begin_day)

    print("=" * 65)
    print(f"Preprocessing  Day {result.current_day} ({cur_date})  "
          f"horizon [{result.current_day}, {horizon_end}] ({cur_date} to {end_date})")
    print("=" * 65)

    print(f"\n  Active C-check windows  : {len(result.c_windows)}")
    for wi in result.c_windows:
        due_date = day_to_date(wi.due_day, p.begin_day)
        print(f"    {wi.tail:<14} k={wi.k}  code={wi.check_code}  "
              f"due={wi.due_day} ({due_date})  "
              f"window=[{wi.t_early},{wi.t_late}]  "
              f"valid={len(wi.valid_days)} days  "
              f"duration={wi.duration}d")

    print(f"\n  Active A-check windows  : {len(result.a_windows)}")
    for wi in result.a_windows:
        due_date = day_to_date(wi.due_day, p.begin_day)
        print(f"    {wi.tail:<14} k={wi.k}  "
              f"due={wi.due_day} ({due_date})  "
              f"window=[{wi.t_early},{wi.t_late}]  "
              f"valid={len(wi.valid_days)} days")

    print(f"\n  Merged A-checks         : {len(result.merged_a)}")
    for tail, k in sorted(result.merged_a):
        print(f"    {tail}  k={k}")

    grounded_aircraft = [t for t, g in result.grounding_mask.items() if g]
    print(f"\n  Grounded aircraft       : {len(grounded_aircraft)}")
    for tail in grounded_aircraft:
        days = sorted(result.grounding_mask[tail])
        print(f"    {tail}  grounded days [{days[0]}, {days[-1]}]  "
              f"({len(days)} days)")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from data_utils import load_amcs_data

    path = sys.argv[1] if len(sys.argv) > 1 else "Scheduling_Input_2017.xlsx"
    data = load_amcs_data(path)

    states      = init_runtime_states(data)
    committed_c = init_committed_from_c_initial(data)
    committed_a: List[CommittedCheck] = []

    # --- Day 0 solve --------------------------------------------------------
    config = RollingConfig()
    result = run_preprocessing(data, states, current_day=0,
                               committed_c=committed_c,
                               committed_a=committed_a,
                               config=config)
    print_preprocessing_summary(result, data)

    # --- Peek at Aircraft-2 due dates (closest C-check to window edge) ------
    print("\nAircraft-2  C-check due-date chain:")
    for dr in result.due_dates_c.get("Aircraft-2", []):
        date = day_to_date(dr.d_abs, data.params.begin_day)
        print(f"  k={dr.k}  d={dr.d_abs} ({date})  "
              f"binding={dr.binding_metric}  sigma={dr.sigma_d:.2f}d")

    print("\nAircraft-5  A-check due-date chain (first 4):")
    for dr in result.due_dates_a.get("Aircraft-5", [])[:4]:
        date = day_to_date(dr.d_abs, data.params.begin_day)
        print(f"  k={dr.k}  d={dr.d_abs} ({date})  "
              f"binding={dr.binding_metric}  sigma={dr.sigma_d:.2f}d")
