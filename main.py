"""
main.py
=======
Rolling-horizon orchestrator for the AMCS Two-Stage QUBO solver.

Implements Section 5.1 of the formulation:
  H  = 60-day rolling horizon
  delta = 7-day commit window  (decisions locked per solve)
  ~157 solves over 3-year horizon (1096 days / 7 days)

Pipeline per solve
------------------
  1. run_preprocessing  -> C/A maintenance windows
  2. build_c_qubo       -> Stage-1 QUBO (C-checks)
  3. solve_stage        -> SA + repair -> committed C-checks
  4. build_a_qubo       -> Stage-2 QUBO (A-checks, grounded by new C-plan)
  5. solve_stage        -> SA + repair -> committed A-checks
  6. Commit checks whose start_day falls in [current_day, current_day+delta)
  7. advance_day x delta -> update aircraft states with stochastic simulation

Output
------
  run_rolling_horizon()   -> RollingResult
  print_kpi_report()      -> full KPI dashboard to stdout
  print_schedule_table()  -> per-aircraft schedule to stdout
  verify_schedule()       -> constraint audit dict
  export_schedule_csv()   -> write schedule to CSV file

Usage
-----
  python main.py [path_to_excel] [--seed N] [--verbose 0-3] [--csv output.csv]
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data_utils import AMCSData, day_to_date, load_amcs_data
from preprocessing import (
    AircraftRuntimeState, CommittedCheck,
    RollingConfig,
    init_committed_from_c_initial,
    run_preprocessing,
)
from qubo_builder import (
    QUBOProblem,
    build_a_qubo, build_c_qubo,
    constraint_violations,
)
from simulator import DayResult, advance_day, init_sim_states
from solver import SAResult, solve_stage


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ScheduledCheck:
    """
    A committed check enriched with scheduling-quality metadata.
    Created when a check is locked into the commit window.
    """
    tail:         str
    k:            int
    check_type:   str    # 'C' or 'A'
    start_day:    int
    duration:     int
    due_day:      int    # due date from preprocessing window
    t_early:      int    # earliest valid start day
    t_late:       int    # latest  valid start day
    solve_idx:    int    # rolling solve that committed this check

    @property
    def end_day(self) -> int:
        return self.start_day + self.duration

    @property
    def on_time(self) -> bool:
        """True if check started on or before its due date."""
        return self.start_day <= self.due_day

    @property
    def days_early(self) -> int:
        return max(0, self.due_day - self.start_day)

    @property
    def days_overdue(self) -> int:
        return max(0, self.start_day - self.due_day)


@dataclass
class SolveLog:
    """Record of one rolling-horizon solve iteration."""
    solve_idx:    int
    current_day:  int
    commit_end:   int

    # Preprocessing
    n_c_windows:  int
    n_a_windows:  int
    n_merged_a:   int

    # Stage 1
    c_sa:         Optional[SAResult]
    c_raw_viol:   Dict[str, int]    # constraint violations on raw SA bits
    n_c_locked:   int               # checks locked this solve

    # Stage 2
    a_sa:         Optional[SAResult]
    a_raw_viol:   Dict[str, int]
    n_a_locked:   int

    # Newly locked checks (within commit window)
    new_c:        List[ScheduledCheck]
    new_a:        List[ScheduledCheck]

    @property
    def c_elapsed(self) -> float:
        return self.c_sa.elapsed_s if self.c_sa else 0.0

    @property
    def a_elapsed(self) -> float:
        return self.a_sa.elapsed_s if self.a_sa else 0.0

    @property
    def total_elapsed(self) -> float:
        return self.c_elapsed + self.a_elapsed


@dataclass
class KPISummary:
    """Section 7 KPIs computed from the full rolling-horizon run."""
    total_days:          int
    total_solves:        int
    n_aircraft:          int

    # Check counts
    total_c_committed:   int
    total_a_committed:   int

    # Airworthiness
    total_aw_viol_days:  int          # (aircraft, day) pairs with a violation
    aw_by_aircraft:      Dict[str, int]  # tail -> violation-day count

    # Capacity utilisation  (fraction of available slots that were occupied)
    c_hangar_util:       float
    a_hangar_util:       float

    # Schedule adherence
    c_on_time:           int          # C-checks started <= due_day
    c_overdue:           int
    a_on_time:           int
    a_overdue:           int

    # Solver performance
    total_solver_s:      float
    avg_c_solver_s:      float
    avg_a_solver_s:      float
    total_elapsed_s:     float


@dataclass
class RollingResult:
    """Complete output of run_rolling_horizon()."""
    scheduled_c:     List[ScheduledCheck]
    scheduled_a:     List[ScheduledCheck]
    locked_c:        List[CommittedCheck]   # raw committed list (for simulation)
    locked_a:        List[CommittedCheck]
    day_results:     List[DayResult]
    solve_logs:      List[SolveLog]
    final_states:    Dict[str, AircraftRuntimeState]
    total_elapsed_s: float
    data:            AMCSData               # retained for KPI / reporting


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_scheduled(
    cc:        CommittedCheck,
    windows:   List,         # List[CheckWindowInfo]
    solve_idx: int,
) -> Optional[ScheduledCheck]:
    """Match a CommittedCheck to its CheckWindowInfo and build ScheduledCheck."""
    wi = next((w for w in windows if w.tail == cc.tail and w.k == cc.k), None)
    if wi is None:
        return None
    return ScheduledCheck(
        tail       = cc.tail,
        k          = cc.k,
        check_type = cc.check_type,
        start_day  = cc.start_day,
        duration   = cc.duration,
        due_day    = wi.due_day,
        t_early    = wi.t_early,
        t_late     = wi.t_late,
        solve_idx  = solve_idx,
    )


def _print_solve_line(log: SolveLog, data: AMCSData) -> None:
    """One-liner progress output per solve."""
    d = day_to_date(log.current_day, data.params.begin_day)
    c_viol_str = (f"viol={log.c_raw_viol}" if any(log.c_raw_viol.values()) else "ok")
    a_viol_str = (f"viol={log.a_raw_viol}" if any(log.a_raw_viol.values()) else "ok")
    print(
        f"  Solve {log.solve_idx:>3d}  day={log.current_day:>4d} ({d})"
        f"  C:{log.n_c_windows}win->{log.n_c_locked}lock({c_viol_str})"
        f"  A:{log.n_a_windows}win->{log.n_a_locked}lock({a_viol_str})"
        f"  t={log.total_elapsed:.2f}s"
    )


def _print_solve_detail(log: SolveLog, data: AMCSData) -> None:
    """Detailed per-solve output (verbose >= 2)."""
    _print_solve_line(log, data)
    if log.c_sa:
        print(f"         C-SA: E {log.c_sa.init_energy:.3e} -> {log.c_sa.best_energy:.3e}"
              f"  iter={log.c_sa.n_iter:,}  acc={log.c_sa.n_accepted:,}"
              f"  impr={log.c_sa.n_improved}")
    if log.a_sa:
        print(f"         A-SA: E {log.a_sa.init_energy:.3e} -> {log.a_sa.best_energy:.3e}"
              f"  iter={log.a_sa.n_iter:,}  acc={log.a_sa.n_accepted:,}"
              f"  impr={log.a_sa.n_improved}")
    for sc in log.new_c:
        tag = "ON-TIME" if sc.on_time else f"LATE+{sc.days_overdue}d"
        print(f"         [C] {sc.tail:<14} k={sc.k}  "
              f"day={sc.start_day}-{sc.end_day-1}  due={sc.due_day}  {tag}")
    for sa in log.new_a:
        tag = "ON-TIME" if sa.on_time else f"LATE+{sa.days_overdue}d"
        print(f"         [A] {sa.tail:<14} k={sa.k}  "
              f"day={sa.start_day}  due={sa.due_day}  {tag}")


# ---------------------------------------------------------------------------
# Core: rolling-horizon solver
# ---------------------------------------------------------------------------

def run_rolling_horizon(
    data:        AMCSData,
    config:      Optional[RollingConfig] = None,
    sa_c_kwargs: Optional[Dict] = None,
    sa_a_kwargs: Optional[Dict] = None,
    seed:        int  = 42,
    verbose:     int  = 1,
) -> RollingResult:
    """
    Execute the full rolling-horizon QUBO solve + simulation loop.

    Parameters
    ----------
    data        : AMCSData from load_amcs_data()
    config      : rolling-horizon parameters (H, delta, z_alpha, etc.)
    sa_c_kwargs : extra kwargs for simulated_annealing() in the C-check stage
    sa_a_kwargs : extra kwargs for simulated_annealing() in the A-check stage
    seed        : numpy random seed (for reproducibility)
    verbose     : 0=silent, 1=one line/solve, 2=SA details, 3=daily simulation

    Returns
    -------
    RollingResult with all scheduled checks, day results, and solver logs.
    """
    if config is None:
        config = RollingConfig()

    sa_c_kwargs = sa_c_kwargs or dict(
        max_iter=100_000, alpha=0.9995, frozen_limit=15_000
    )
    sa_a_kwargs = sa_a_kwargs or dict(
        max_iter=200_000, alpha=0.9995, frozen_limit=20_000
    )

    rng = np.random.default_rng(seed)
    T   = data.params.horizon_days
    H   = config.rolling_horizon
    dlt = config.commit_window

    # --- Initialise committed lists and states ----------------------------
    locked_c: List[CommittedCheck] = init_committed_from_c_initial(data)
    locked_a: List[CommittedCheck] = []
    states    = init_sim_states(data, locked_c, start_day=0)

    scheduled_c: List[ScheduledCheck] = []
    scheduled_a: List[ScheduledCheck] = []
    solve_logs:  List[SolveLog]       = []
    day_results: List[DayResult]      = []

    t_wall_start = time.perf_counter()
    current_day  = 0
    solve_idx    = 0

    if verbose >= 1:
        d0 = day_to_date(0, data.params.begin_day)
        dT = day_to_date(T - 1, data.params.begin_day)
        print("=" * 72)
        print(f"AMCS Rolling-Horizon Solver")
        print(f"  Horizon : {d0} to {dT} ({T} days)")
        print(f"  Aircraft: {len(data.aircraft)}")
        print(f"  H={H}d  delta={dlt}d  seed={seed}")
        print("=" * 72)

    # ======================================================================
    # Main rolling loop
    # ======================================================================
    while current_day < T:
        commit_end = min(current_day + dlt, T)

        # ---- Step 1: Preprocessing ----------------------------------------
        prep = run_preprocessing(
            data, states, current_day, locked_c, locked_a, config
        )

        # ---- Step 2-3: Stage 1 — C-check QUBO ----------------------------
        c_sa    = None
        c_viol  = {"one_hot": 0, "capacity": 0, "spacing": 0}
        c_new: List[CommittedCheck] = []

        if prep.c_windows:
            c_qubo = build_c_qubo(
                prep.c_windows, locked_c, data, current_day, config
            )
            c_new, c_sa = solve_stage(c_qubo, data, sa_kwargs=sa_c_kwargs, rng=rng)
            c_viol = constraint_violations(c_qubo, c_sa.best_bits, data)

        # Full C-plan for Stage 2 (tentative beyond commit window)
        all_c_for_stage2 = locked_c + c_new

        # ---- Step 4-5: Stage 2 — A-check QUBO ----------------------------
        a_sa    = None
        a_viol  = {"one_hot": 0, "capacity": 0, "spacing": 0}
        a_new: List[CommittedCheck] = []

        if prep.a_windows:
            a_qubo = build_a_qubo(
                prep.a_windows, all_c_for_stage2, locked_a,
                data, current_day, config
            )
            a_new, a_sa = solve_stage(a_qubo, data, sa_kwargs=sa_a_kwargs, rng=rng)
            a_viol = constraint_violations(a_qubo, a_sa.best_bits, data)

        # ---- Step 6: Commit checks within [current_day, commit_end) -------
        new_c_commits = [cc for cc in c_new
                         if current_day <= cc.start_day < commit_end]
        new_a_commits = [ca for ca in a_new
                         if current_day <= ca.start_day < commit_end]

        # Build rich ScheduledCheck records (paired with window info)
        new_sc: List[ScheduledCheck] = []
        for cc in new_c_commits:
            sc = _make_scheduled(cc, prep.c_windows, solve_idx)
            if sc:
                new_sc.append(sc)

        new_sa_sched: List[ScheduledCheck] = []
        for ca in new_a_commits:
            sa_s = _make_scheduled(ca, prep.a_windows, solve_idx)
            if sa_s:
                new_sa_sched.append(sa_s)

        locked_c = locked_c + new_c_commits
        locked_a = locked_a + new_a_commits
        scheduled_c.extend(new_sc)
        scheduled_a.extend(new_sa_sched)

        # ---- Logging -------------------------------------------------------
        log = SolveLog(
            solve_idx   = solve_idx,
            current_day = current_day,
            commit_end  = commit_end,
            n_c_windows = len(prep.c_windows),
            n_a_windows = len(prep.a_windows),
            n_merged_a  = len(prep.merged_a),
            c_sa        = c_sa,
            c_raw_viol  = c_viol,
            n_c_locked  = len(new_c_commits),
            a_sa        = a_sa,
            a_raw_viol  = a_viol,
            n_a_locked  = len(new_a_commits),
            new_c       = new_sc,
            new_a       = new_sa_sched,
        )
        solve_logs.append(log)

        if verbose == 1:
            _print_solve_line(log, data)
        elif verbose >= 2:
            _print_solve_detail(log, data)

        # ---- Step 7: Simulate commit-window days --------------------------
        for day in range(current_day, commit_end):
            result = advance_day(states, locked_c, locked_a, day, data, rng)
            day_results.append(result)
            if verbose >= 3 and (result.events or result.violations):
                from simulator import print_day_summary
                print_day_summary(result, data)

        current_day = commit_end
        solve_idx  += 1

    total_elapsed = time.perf_counter() - t_wall_start

    if verbose >= 1:
        print("=" * 72)
        print(f"Done.  {solve_idx} solves, {T} days simulated, "
              f"{total_elapsed:.1f}s total.")
        print("=" * 72)

    return RollingResult(
        scheduled_c     = scheduled_c,
        scheduled_a     = scheduled_a,
        locked_c        = locked_c,
        locked_a        = locked_a,
        day_results     = day_results,
        solve_logs      = solve_logs,
        final_states    = states,
        total_elapsed_s = total_elapsed,
        data            = data,
    )


# ---------------------------------------------------------------------------
# KPI computation
# ---------------------------------------------------------------------------

def compute_kpis(result: RollingResult) -> KPISummary:
    """
    Compute Section 7 KPIs from a completed rolling-horizon run.
    """
    data = result.data
    T    = data.params.horizon_days
    M_C  = data.params.max_c_check
    M_A  = data.params.max_a_check
    n_ac = len(data.aircraft)

    # ---- Airworthiness violations -----------------------------------------
    total_aw      = 0
    aw_by_aircraft: Dict[str, int] = {tail: 0 for tail in data.aircraft}
    for dr in result.day_results:
        seen_tails = set()
        for v in dr.violations:
            if v.tail not in seen_tails:
                total_aw += 1
                aw_by_aircraft[v.tail] = aw_by_aircraft.get(v.tail, 0) + 1
                seen_tails.add(v.tail)

    # ---- Hangar utilisation -----------------------------------------------
    c_occupied = np.zeros(T, dtype=int)
    for cc in result.locked_c:
        t0 = max(0, cc.start_day)
        t1 = min(T, cc.end_day)
        if t0 < t1:
            c_occupied[t0:t1] += 1

    a_occupied = np.zeros(T, dtype=int)
    for ca in result.locked_a:
        t0 = max(0, ca.start_day)
        t1 = min(T, ca.end_day)
        if t0 < t1:
            a_occupied[t0:t1] += 1

    c_available  = int(np.sum(data.capacity.c_capacity))
    a_available  = int(np.sum(data.capacity.a_capacity))
    c_util = int(np.sum(c_occupied)) / max(c_available, 1)
    a_util = int(np.sum(a_occupied)) / max(a_available, 1)

    # ---- Schedule adherence -----------------------------------------------
    c_on   = sum(1 for s in result.scheduled_c if s.on_time)
    c_late = len(result.scheduled_c) - c_on
    a_on   = sum(1 for s in result.scheduled_a if s.on_time)
    a_late = len(result.scheduled_a) - a_on

    # ---- Solver timing ----------------------------------------------------
    c_times = [lg.c_elapsed for lg in result.solve_logs if lg.c_sa]
    a_times = [lg.a_elapsed for lg in result.solve_logs if lg.a_sa]
    total_solver = sum(c_times) + sum(a_times)
    avg_c = float(np.mean(c_times)) if c_times else 0.0
    avg_a = float(np.mean(a_times)) if a_times else 0.0

    return KPISummary(
        total_days          = T,
        total_solves        = len(result.solve_logs),
        n_aircraft          = n_ac,
        total_c_committed   = len(result.scheduled_c),
        total_a_committed   = len(result.scheduled_a),
        total_aw_viol_days  = total_aw,
        aw_by_aircraft      = aw_by_aircraft,
        c_hangar_util       = c_util,
        a_hangar_util       = a_util,
        c_on_time           = c_on,
        c_overdue           = c_late,
        a_on_time           = a_on,
        a_overdue           = a_late,
        total_solver_s      = total_solver,
        avg_c_solver_s      = avg_c,
        avg_a_solver_s      = avg_a,
        total_elapsed_s     = result.total_elapsed_s,
    )


# ---------------------------------------------------------------------------
# Schedule verification
# ---------------------------------------------------------------------------

def verify_schedule(result: RollingResult) -> Dict[str, Any]:
    """
    Audit the committed schedule for hard-constraint violations.

    Checks
    ------
    capacity_c  : days where more than M_C C-checks are active simultaneously
    capacity_a  : days where more than M_A A-checks are active simultaneously
    spacing     : pairs of C-check starts within delta_min days of each other
    duplicates  : (tail, k) pairs committed more than once
    aw_viol     : total airworthiness violation-days (from simulation)

    Returns a dict of violation counts and details.
    """
    data    = result.data
    T       = data.params.horizon_days
    M_C     = data.params.max_c_check
    M_A     = data.params.max_a_check
    dmin    = data.params.start_day_interval

    # ---- Duplicate check --------------------------------------------------
    c_keys = [(cc.tail, cc.k) for cc in result.locked_c]
    a_keys = [(ca.tail, ca.k) for ca in result.locked_a]
    dup_c = len(c_keys) - len(set(c_keys))
    dup_a = len(a_keys) - len(set(a_keys))

    # ---- Capacity ---------------------------------------------------------
    c_load = np.zeros(T, dtype=int)
    for cc in result.locked_c:
        t0 = max(0, cc.start_day)
        t1 = min(T, cc.end_day)
        if t0 < t1:
            c_load[t0:t1] += 1

    a_load = np.zeros(T, dtype=int)
    for ca in result.locked_a:
        t0 = max(0, ca.start_day)
        t1 = min(T, ca.end_day)
        if t0 < t1:
            a_load[t0:t1] += 1

    cap_viol_c_days = [
        (t, int(c_load[t]), int(data.capacity.c_capacity[t]))
        for t in range(T)
        if c_load[t] > data.capacity.c_capacity[t] and data.capacity.c_capacity[t] > 0
    ]
    cap_viol_c_nom  = [
        (t, int(c_load[t]))
        for t in range(T)
        if c_load[t] > M_C
    ]
    cap_viol_a_days = [
        (t, int(a_load[t]), int(data.capacity.a_capacity[t]))
        for t in range(T)
        if a_load[t] > data.capacity.a_capacity[t] and data.capacity.a_capacity[t] > 0
    ]
    cap_viol_a_nom  = [
        (t, int(a_load[t]))
        for t in range(T)
        if a_load[t] > M_A
    ]

    # ---- Spacing ----------------------------------------------------------
    c_starts = sorted(cc.start_day for cc in result.locked_c)
    spacing_viols = [
        (c_starts[i], c_starts[i+1])
        for i in range(len(c_starts) - 1)
        if 0 < c_starts[i+1] - c_starts[i] < dmin
    ]

    # ---- Airworthiness from simulation ------------------------------------
    aw_viol_days = sum(
        1 for dr in result.day_results
        for v in dr.violations
    )

    return {
        "duplicate_c":         dup_c,
        "duplicate_a":         dup_a,
        "capacity_c_over_nominal": len(cap_viol_c_nom),
        "capacity_c_details":  cap_viol_c_nom[:10],   # first 10
        "capacity_a_over_nominal": len(cap_viol_a_nom),
        "capacity_a_details":  cap_viol_a_nom[:10],
        "spacing_violations":  len(spacing_viols),
        "spacing_details":     spacing_viols[:10],
        "aw_violation_events": aw_viol_days,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_kpi_report(result: RollingResult) -> None:
    """Print a full KPI dashboard to stdout."""
    kpi  = compute_kpis(result)
    data = result.data
    begin = data.params.begin_day

    print()
    print("=" * 72)
    print("KPI REPORT — AMCS Rolling-Horizon Solution")
    print("=" * 72)

    print(f"\n{'OVERVIEW':}")
    print(f"  Horizon          : {kpi.total_days} days")
    print(f"  Aircraft         : {kpi.n_aircraft}")
    print(f"  Solves           : {kpi.total_solves}")
    print(f"  Wall time        : {kpi.total_elapsed_s:.1f}s  "
          f"(solver {kpi.total_solver_s:.1f}s)")

    print(f"\n{'CHECKS COMMITTED':}")
    print(f"  C-checks         : {kpi.total_c_committed}")
    print(f"  A-checks         : {kpi.total_a_committed}")
    total_merged = sum(lg.n_merged_a for lg in result.solve_logs)
    print(f"  Merged A-into-C  : {total_merged}")

    print(f"\n{'SCHEDULE ADHERENCE':}")
    c_total = kpi.c_on_time + kpi.c_overdue
    a_total = kpi.a_on_time + kpi.a_overdue
    c_pct = 100 * kpi.c_on_time / max(c_total, 1)
    a_pct = 100 * kpi.a_on_time / max(a_total, 1)
    print(f"  C on-time        : {kpi.c_on_time}/{c_total} ({c_pct:.1f}%)")
    print(f"  C overdue        : {kpi.c_overdue}  "
          f"(started after due date)")
    print(f"  A on-time        : {kpi.a_on_time}/{a_total} ({a_pct:.1f}%)")
    print(f"  A overdue        : {kpi.a_overdue}")

    print(f"\n{'AIRWORTHINESS':}")
    aw_rate = 100 * (1 - kpi.total_aw_viol_days /
                     max(kpi.total_days * kpi.n_aircraft, 1))
    print(f"  Compliance rate  : {aw_rate:.2f}%")
    print(f"  Violation-days   : {kpi.total_aw_viol_days}  "
          f"(aircraft x day pairs)")
    worst = sorted(kpi.aw_by_aircraft.items(), key=lambda x: -x[1])[:5]
    if worst and worst[0][1] > 0:
        print(f"  Worst aircraft   : "
              + ", ".join(f"{t}({n}d)" for t, n in worst if n > 0))

    print(f"\n{'HANGAR UTILISATION':}")
    print(f"  C-hangar         : {100*kpi.c_hangar_util:.1f}%  "
          f"(of total available slot-days)")
    print(f"  A-hangar         : {100*kpi.a_hangar_util:.1f}%")

    print(f"\n{'SOLVER TIMING':}")
    print(f"  Avg C-SA time    : {kpi.avg_c_solver_s*1000:.1f}ms per solve")
    print(f"  Avg A-SA time    : {kpi.avg_a_solver_s*1000:.1f}ms per solve")

    # ---- Constraint audit ------------------------------------------------
    print(f"\n{'CONSTRAINT AUDIT':}")
    audit = verify_schedule(result)
    print(f"  Duplicate C      : {audit['duplicate_c']}")
    print(f"  Duplicate A      : {audit['duplicate_a']}")
    print(f"  Cap. C violations: {audit['capacity_c_over_nominal']}")
    print(f"  Cap. A violations: {audit['capacity_a_over_nominal']}")
    print(f"  Spacing violations: {audit['spacing_violations']}")
    print(f"  AW events (sim)  : {audit['aw_violation_events']}")
    if audit["capacity_c_details"]:
        for t, load in audit["capacity_c_details"][:5]:
            print(f"    C cap day {t} ({day_to_date(t, begin)}): "
                  f"load={load} > M_C={data.params.max_c_check}")
    if audit["spacing_details"]:
        for s1, s2 in audit["spacing_details"][:5]:
            print(f"    Spacing: starts {s1} and {s2} "
                  f"(gap={s2-s1} < delta_min={data.params.start_day_interval})")

    print("=" * 72)


def print_schedule_table(result: RollingResult) -> None:
    """
    Print a per-aircraft schedule table showing all committed checks,
    their windows, and on-time status.
    """
    data  = result.data
    begin = data.params.begin_day

    # Index scheduled checks by tail
    c_by_tail: Dict[str, List[ScheduledCheck]] = {t: [] for t in data.aircraft}
    for sc in result.scheduled_c:
        c_by_tail[sc.tail].append(sc)

    a_by_tail: Dict[str, List[ScheduledCheck]] = {t: [] for t in data.aircraft}
    for sa in result.scheduled_a:
        a_by_tail[sa.tail].append(sa)

    print()
    print("=" * 100)
    print("SCHEDULE TABLE — per aircraft")
    print(f"{'Tail':<14}  {'Type':3}  {'k':>2}  "
          f"{'Start':>5}  {'End':>5}  {'Due':>5}  "
          f"{'Window':^13}  {'Status':^10}  {'Date'}")
    print("-" * 100)

    for tail in data.aircraft:
        first = True

        # C-checks
        for sc in sorted(c_by_tail[tail], key=lambda x: x.k):
            tag    = "ON-TIME" if sc.on_time else f"LATE+{sc.days_overdue:2d}d"
            d_str  = str(day_to_date(sc.start_day, begin))
            w_str  = f"[{sc.t_early},{sc.t_late}]"
            prefix = tail if first else " " * 14
            print(f"  {prefix:<14}  C    {sc.k:>2d}  "
                  f"{sc.start_day:>5d}  {sc.end_day-1:>5d}  {sc.due_day:>5d}  "
                  f"{w_str:^13}  {tag:^10}  {d_str}")
            first = False

        # A-checks (condensed: one line with all start days)
        a_sorted = sorted(a_by_tail[tail], key=lambda x: x.k)
        if a_sorted:
            prefix  = tail if first else " " * 14
            a_parts = []
            for sa in a_sorted:
                ot = "" if sa.on_time else f"(L+{sa.days_overdue})"
                a_parts.append(f"k{sa.k}:d{sa.start_day}{ot}")
            print(f"  {prefix:<14}  A  [{len(a_sorted)} checks]  "
                  + "  ".join(a_parts))
            first = False

        if first:
            print(f"  {tail:<14}  (no checks scheduled)")

    print("=" * 100)


def export_schedule_csv(result: RollingResult, filepath: str) -> None:
    """
    Write all scheduled checks to a CSV file for external analysis.

    Columns: tail, check_type, k, start_day, end_day, duration, due_day,
             t_early, t_late, on_time, days_overdue, solve_idx, start_date
    """
    data  = result.data
    begin = data.params.begin_day
    rows  = []

    for sc in result.scheduled_c + result.scheduled_a:
        rows.append({
            "tail":        sc.tail,
            "check_type":  sc.check_type,
            "k":           sc.k,
            "start_day":   sc.start_day,
            "end_day":     sc.end_day,
            "duration":    sc.duration,
            "due_day":     sc.due_day,
            "t_early":     sc.t_early,
            "t_late":      sc.t_late,
            "on_time":     sc.on_time,
            "days_early":  sc.days_early,
            "days_overdue": sc.days_overdue,
            "solve_idx":   sc.solve_idx,
            "start_date":  str(day_to_date(sc.start_day, begin)),
        })

    rows.sort(key=lambda r: (r["tail"], r["check_type"], r["k"]))

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSchedule exported -> {filepath}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Solve-log summary table
# ---------------------------------------------------------------------------

def print_solve_log_table(result: RollingResult) -> None:
    """Print a compact table of every solve iteration."""
    data = result.data
    begin = data.params.begin_day

    print()
    print("=" * 95)
    print("SOLVE LOG")
    print(f"  {'Solve':>5}  {'Day':>5}  {'Date':<12}  "
          f"{'C-win':>5}  {'C-lock':>6}  {'A-win':>5}  {'A-lock':>6}  "
          f"{'C-E_init':>12}  {'C-E_best':>12}  "
          f"{'A-E_init':>12}  {'A-E_best':>12}  "
          f"{'t(s)':>6}")
    print("-" * 95)

    for lg in result.solve_logs:
        d = str(day_to_date(lg.current_day, begin))
        c_ei = f"{lg.c_sa.init_energy:.3e}" if lg.c_sa else "        -"
        c_eb = f"{lg.c_sa.best_energy:.3e}" if lg.c_sa else "        -"
        a_ei = f"{lg.a_sa.init_energy:.3e}" if lg.a_sa else "        -"
        a_eb = f"{lg.a_sa.best_energy:.3e}" if lg.a_sa else "        -"
        print(f"  {lg.solve_idx:>5d}  {lg.current_day:>5d}  {d:<12}  "
              f"{lg.n_c_windows:>5d}  {lg.n_c_locked:>6d}  "
              f"{lg.n_a_windows:>5d}  {lg.n_a_locked:>6d}  "
              f"{c_ei:>12}  {c_eb:>12}  "
              f"{a_ei:>12}  {a_eb:>12}  "
              f"{lg.total_elapsed:>6.2f}")

    print("=" * 95)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AMCS Two-Stage QUBO Rolling-Horizon Solver"
    )
    parser.add_argument(
        "filepath",
        nargs="?",
        default="Scheduling_Input_2017.xlsx",
        help="Path to Scheduling_Input_2017.xlsx",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--verbose", type=int, default=1, choices=[0, 1, 2, 3],
        help="Verbosity: 0=silent, 1=per-solve, 2=SA details, 3=daily sim",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Export schedule to CSV file",
    )
    parser.add_argument(
        "--log-table", action="store_true",
        help="Print full solve-log table after the run",
    )
    parser.add_argument(
        "--no-schedule-table", action="store_true",
        help="Skip printing the per-aircraft schedule table",
    )
    args = parser.parse_args()

    # Load data
    if args.verbose >= 1:
        print(f"Loading data from {args.filepath} ...")
    data = load_amcs_data(args.filepath)

    # Run rolling horizon
    result = run_rolling_horizon(
        data    = data,
        seed    = args.seed,
        verbose = args.verbose,
    )

    # Reports
    print_kpi_report(result)

    if not args.no_schedule_table:
        print_schedule_table(result)

    if args.log_table:
        print_solve_log_table(result)

    if args.csv:
        export_schedule_csv(result, args.csv)


if __name__ == "__main__":
    main()
