"""
simulator.py
============
Section 5.3 of the AMCS Two-Stage QUBO Formulation: daily simulation.

Responsibilities
----------------
1. Daily utilisation sampling  – FH and FC drawn from N(µ, σ²) and clamped to
   [FH_MIN, FH_MAX]; FC scaled proportionally from sampled FH.
2. Check execution             – start/complete C-checks and A-checks each day,
   sampling stochastic C-check durations on start, resetting counters on
   completion per the MRO counter-reset rules.
3. Airworthiness monitoring    – flag any aircraft whose accumulated FH, FC, or
   calendar days (DY) exceed the effective limit for their current check interval.

Counter-reset rules (Section 5.3)
----------------------------------
  C-check completion: reset fh_c, fc_c, dy_c  AND  fh_a, fc_a, dy_a to 0.
  A-check completion: reset fh_a, fc_a, dy_a to 0 only.
  DY always increments by 1 per calendar day (flying or grounded).
  FH / FC only increment when the aircraft is not in the hangar.

Public API
----------
simulate_daily_utilization(tail, day, data, rng)     -> Tuple[float, float]
init_sim_states(data, committed_c, start_day)        -> Dict[str, AircraftRuntimeState]
advance_day(states, committed_c, committed_a,
            day, data, rng)                          -> DayResult
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from data_utils import AMCSData, day_to_date
from preprocessing import AircraftRuntimeState, CommittedCheck, init_runtime_states


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CheckEvent:
    """
    One check start or completion event on a simulation day.

    event_type : 'start' or 'complete'
    actual_dur : sampled duration (days) — set only on C-check 'start' events.
                 None for A-checks and for 'complete' events.
    """
    tail:       str
    k:          int
    check_type: str             # 'C' or 'A'
    event_type: str             # 'start' | 'complete'
    day:        int
    actual_dur: Optional[int] = None


@dataclass
class AirworthinessViolation:
    """
    An accumulator that has crossed its effective limit on a given day.

    metric  : 'FH', 'FC', or 'DY'
    current : value of the accumulator at end-of-day
    limit   : effective limit that was breached
    """
    tail:       str
    day:        int
    check_type: str     # 'C' or 'A' (which check interval was exceeded)
    metric:     str     # 'FH' | 'FC' | 'DY'
    current:    float
    limit:      float

    @property
    def overrun(self) -> float:
        return self.current - self.limit


@dataclass
class DayResult:
    """
    Complete output of advance_day() for one calendar day.

    fh_flown / fc_flown
        Per-aircraft flight hours / cycles accumulated today.
        0.0 for any aircraft in the hangar.

    events
        Check start and completion events, ordered: completions first,
        then starts (matching the intra-day processing order).

    violations
        Airworthiness breaches detected at end-of-day.
    """
    day:        int
    events:     List[CheckEvent]
    fh_flown:   Dict[str, float]
    fc_flown:   Dict[str, float]
    violations: List[AirworthinessViolation]

    @property
    def grounded(self) -> Set[str]:
        """Set of tail strings that were in the hangar today (FH = 0)."""
        return {tail for tail, fh in self.fh_flown.items() if fh == 0.0}

    @property
    def total_fh(self) -> float:
        return sum(self.fh_flown.values())

    @property
    def total_fc(self) -> float:
        return sum(self.fc_flown.values())


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _sample_c_duration(
    tail: str,
    k:    int,
    data: AMCSData,
    rng:  np.random.Generator,
) -> int:
    """
    Sample an actual C-check turnaround time from the empirical distribution.

    Draws uniformly from the historical TAT samples stored in STOCHASTIC.
    Falls back to get_c_duration() (mean or C_ELAPSED_TIME) if sampling
    would produce a non-positive value.
    """
    dist = data.get_stochastic_duration(tail, k)
    dur  = int(dist.samples[rng.integers(len(dist.samples))])
    return max(1, dur)


def _check_limits(
    tail:       str,
    k:          int,
    check_type: str,
    data:       AMCSData,
) -> Tuple[float, float, float]:
    """
    Return (lim_dy, lim_fh, lim_fc) for check k of the given type.

    k == 0 (first check since Day 0): uses effective limits which
        already account for any tolerance pre-consumed (TOLU != 0).
    k > 0 (subsequent checks):        full interval + tolerance (TOLU = 0).
    """
    if check_type == "C":
        cs = data.c_initial[tail]
        if k == 0:
            return cs.eff_limit_dy, cs.eff_limit_fh, cs.eff_limit_fc
        return (cs.ci_dy + cs.tol_dy,
                cs.ci_fh + cs.tol_fh,
                cs.ci_fc + cs.tol_fc)
    else:
        as_ = data.a_initial[tail]
        if k == 0:
            return as_.eff_limit_dy, as_.eff_limit_fh, as_.eff_limit_fc
        return (as_.ci_dy + as_.tol_dy,
                as_.ci_fh + as_.tol_fh,
                as_.ci_fc + as_.tol_fc)


def _collect_violations(
    states: Dict[str, AircraftRuntimeState],
    day:    int,
    data:   AMCSData,
) -> List[AirworthinessViolation]:
    """
    Check every aircraft's end-of-day accumulators against their current limits.

    An aircraft in the hangar still accumulates DY, so DY violations can occur
    even for grounded aircraft.  FH / FC violations can only accumulate while
    flying, so they are less likely but possible if a check was deferred.
    """
    violations: List[AirworthinessViolation] = []

    for tail, st in states.items():
        # C-check limits for current check index
        lim_dy_c, lim_fh_c, lim_fc_c = _check_limits(tail, st.next_c_k, "C", data)

        if st.fh_c > lim_fh_c:
            violations.append(AirworthinessViolation(
                tail=tail, day=day, check_type="C", metric="FH",
                current=st.fh_c, limit=lim_fh_c))
        if st.fc_c > lim_fc_c:
            violations.append(AirworthinessViolation(
                tail=tail, day=day, check_type="C", metric="FC",
                current=st.fc_c, limit=lim_fc_c))
        if st.dy_c > lim_dy_c:
            violations.append(AirworthinessViolation(
                tail=tail, day=day, check_type="C", metric="DY",
                current=st.dy_c, limit=lim_dy_c))

        # A-check limits for current check index
        lim_dy_a, lim_fh_a, lim_fc_a = _check_limits(tail, st.next_a_k, "A", data)

        if st.fh_a > lim_fh_a:
            violations.append(AirworthinessViolation(
                tail=tail, day=day, check_type="A", metric="FH",
                current=st.fh_a, limit=lim_fh_a))
        if st.fc_a > lim_fc_a:
            violations.append(AirworthinessViolation(
                tail=tail, day=day, check_type="A", metric="FC",
                current=st.fc_a, limit=lim_fc_a))
        if st.dy_a > lim_dy_a:
            violations.append(AirworthinessViolation(
                tail=tail, day=day, check_type="A", metric="DY",
                current=st.dy_a, limit=lim_dy_a))

    return violations


# ---------------------------------------------------------------------------
# Public: daily utilisation sampler
# ---------------------------------------------------------------------------

def simulate_daily_utilization(
    tail: str,
    day:  int,
    data: AMCSData,
    rng:  np.random.Generator,
) -> Tuple[float, float]:
    """
    Sample one flying day's FH and FC for a single aircraft.

    FH ~ Normal(µ^FH_i(m), σ^FH_i(m)²)  clamped to [FH_MIN_i(m), FH_MAX_i(m)]
    FC = FH * (µ^FC_i(m) / µ^FH_i(m))   (proportional scaling; FC_STD unavailable)

    If µ^FH = 0 for the month, returns (0.0, µ^FC) as a safe fallback.

    Parameters
    ----------
    tail : aircraft tail string
    day  : 0-based absolute day index (used to determine the calendar month)
    data : AMCSData
    rng  : numpy Generator

    Returns
    -------
    (fh_today, fc_today) — both non-negative floats
    """
    m      = day_to_date(day, data.params.begin_day).month - 1   # 0=Jan
    mu_fh  = float(data.dfh[tail][m])
    sig_fh = float(data.fh_std[tail][m])
    fh_min = float(data.fh_min[tail][m])
    fh_max = float(data.fh_max[tail][m])
    mu_fc  = float(data.dfc[tail][m])

    if mu_fh <= 0.0:
        return 0.0, max(0.0, mu_fc)

    # Sample and clamp FH
    fh_raw = float(rng.normal(mu_fh, sig_fh))
    fh     = float(np.clip(fh_raw, fh_min, fh_max))

    # Scale FC proportionally
    fc = fh * (mu_fc / mu_fh)

    return fh, max(0.0, fc)


# ---------------------------------------------------------------------------
# Public: state initialisation for simulation
# ---------------------------------------------------------------------------

def init_sim_states(
    data:        AMCSData,
    committed_c: List[CommittedCheck],
    start_day:   int = 0,
) -> Dict[str, AircraftRuntimeState]:
    """
    Initialise AircraftRuntimeState for every aircraft, ready for simulation
    starting at `start_day`.

    Extends preprocessing.init_runtime_states() by marking aircraft that are
    already in a committed C-check at `start_day` as in_c_check = True and
    setting c_check_end_day to the committed check's end_day.

    This correctly handles pre-committed checks from C_INITIAL (c_start != -1)
    that straddle the planning start.

    Parameters
    ----------
    data        : AMCSData
    committed_c : committed C-check list (from init_committed_from_c_initial
                  or accumulated by the rolling solver)
    start_day   : first day of the simulation (default 0)

    Returns
    -------
    Dict mapping tail -> AircraftRuntimeState
    """
    states = init_runtime_states(data)

    for cc in committed_c:
        if cc.check_type != "C":
            continue
        # If this check is in progress at start_day (covers start_day):
        if cc.start_day <= start_day < cc.end_day:
            st = states[cc.tail]
            st.in_c_check      = True
            st.c_check_end_day = cc.end_day
            # Note: accumulators (fh_c etc.) are left at the C_INITIAL values.
            # They represent how much was accumulated when the check was triggered
            # and will be zeroed when the check completes.

    return states


# ---------------------------------------------------------------------------
# Public: daily simulation step
# ---------------------------------------------------------------------------

def advance_day(
    states:      Dict[str, AircraftRuntimeState],
    committed_c: List[CommittedCheck],
    committed_a: List[CommittedCheck],
    day:         int,
    data:        AMCSData,
    rng:         np.random.Generator,
) -> DayResult:
    """
    Advance the simulation by one day, in-place updating all aircraft states.

    Processing order per aircraft (Section 5.3)
    -------------------------------------------
    1. Complete C-check  if in_c_check AND day >= c_check_end_day.
       → Reset fh_c, fc_c, dy_c, fh_a, fc_a, dy_a to 0.
       → Increment next_c_k.

    2. Complete A-check  if in_a_check AND day >= a_check_end_day.
       → Reset fh_a, fc_a, dy_a to 0.
       → Increment next_a_k.

    3. Start C-check     if a committed C-check has start_day == day
       AND the aircraft is not already in_c_check.
       → Sample actual duration from STOCHASTIC distribution.
       → Set c_check_end_day = day + actual_dur.

    4. Start A-check     if a committed A-check has start_day == day
       AND the aircraft is not in_c_check or in_a_check.
       → Set a_check_end_day = day + 1  (LA = 1).

    5. Accumulate FH/FC  if not in_c_check AND not in_a_check.
       → Sample from simulate_daily_utilization().
       → Add to fh_c, fc_c, fh_a, fc_a.

    6. Accumulate DY     always (calendar days count in the hangar too).
       → dy_c += 1, dy_a += 1.

    7. Airworthiness     check end-of-day accumulators against effective limits.

    Parameters
    ----------
    states      : per-aircraft runtime state dict (mutated in-place)
    committed_c : all committed C-checks (past, present, future)
    committed_a : all committed A-checks
    day         : 0-based current day index
    data        : AMCSData
    rng         : numpy Generator

    Returns
    -------
    DayResult with events, utilisation, and any airworthiness violations.
    """
    # Build fast lookup: tail -> committed check starting today
    c_starting: Dict[str, CommittedCheck] = {
        cc.tail: cc for cc in committed_c
        if cc.check_type == "C" and cc.start_day == day
    }
    a_starting: Dict[str, CommittedCheck] = {
        ca.tail: ca for ca in committed_a
        if ca.check_type == "A" and ca.start_day == day
    }

    events:   List[CheckEvent]  = []
    fh_flown: Dict[str, float]  = {}
    fc_flown: Dict[str, float]  = {}

    for tail in data.aircraft:
        st = states[tail]

        # ------------------------------------------------------------------
        # Step 1: Complete C-check
        # ------------------------------------------------------------------
        if st.in_c_check and day >= st.c_check_end_day:
            events.append(CheckEvent(
                tail=tail, k=st.next_c_k,
                check_type="C", event_type="complete", day=day,
            ))
            # Reset all counters — both C and A intervals restart after C-check
            st.fh_c = 0.0;  st.fc_c = 0.0;  st.dy_c = 0.0
            st.fh_a = 0.0;  st.fc_a = 0.0;  st.dy_a = 0.0
            st.next_c_k       += 1
            st.in_c_check      = False
            st.c_check_end_day = -1

        # ------------------------------------------------------------------
        # Step 2: Complete A-check
        # ------------------------------------------------------------------
        if st.in_a_check and day >= st.a_check_end_day:
            events.append(CheckEvent(
                tail=tail, k=st.next_a_k,
                check_type="A", event_type="complete", day=day,
            ))
            st.fh_a = 0.0;  st.fc_a = 0.0;  st.dy_a = 0.0
            st.next_a_k       += 1
            st.in_a_check      = False
            st.a_check_end_day = -1

        # ------------------------------------------------------------------
        # Step 3: Start C-check
        # ------------------------------------------------------------------
        if tail in c_starting and not st.in_c_check:
            cc         = c_starting[tail]
            actual_dur = _sample_c_duration(tail, cc.k, data, rng)

            st.in_c_check      = True
            st.c_check_end_day = day + actual_dur

            events.append(CheckEvent(
                tail=tail, k=cc.k,
                check_type="C", event_type="start", day=day,
                actual_dur=actual_dur,
            ))

        # ------------------------------------------------------------------
        # Step 4: Start A-check
        # ------------------------------------------------------------------
        if tail in a_starting and not st.in_c_check and not st.in_a_check:
            ca = a_starting[tail]

            st.in_a_check      = True
            st.a_check_end_day = day + 1    # LA = 1

            events.append(CheckEvent(
                tail=tail, k=ca.k,
                check_type="A", event_type="start", day=day,
            ))

        # ------------------------------------------------------------------
        # Step 5: Accumulate FH / FC (flying aircraft only)
        # ------------------------------------------------------------------
        if not st.in_c_check and not st.in_a_check:
            fh, fc = simulate_daily_utilization(tail, day, data, rng)
            st.fh_c += fh;  st.fc_c += fc
            st.fh_a += fh;  st.fc_a += fc
            fh_flown[tail] = fh
            fc_flown[tail] = fc
        else:
            fh_flown[tail] = 0.0
            fc_flown[tail] = 0.0

        # ------------------------------------------------------------------
        # Step 6: Calendar days (always)
        # ------------------------------------------------------------------
        st.dy_c += 1.0
        st.dy_a += 1.0

    # Step 7: Airworthiness check across all aircraft
    violations = _collect_violations(states, day, data)

    return DayResult(
        day        = day,
        events     = events,
        fh_flown   = fh_flown,
        fc_flown   = fc_flown,
        violations = violations,
    )


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

def print_day_summary(result: DayResult, data: AMCSData) -> None:
    """Print a concise summary of one simulated day."""
    d = day_to_date(result.day, data.params.begin_day)
    n = len(data.aircraft)

    flying    = n - len(result.grounded)
    grounded  = len(result.grounded)
    n_starts  = sum(1 for e in result.events if e.event_type == "start")
    n_comps   = sum(1 for e in result.events if e.event_type == "complete")

    print(f"Day {result.day:4d}  ({d})  "
          f"flying={flying:2d}  grounded={grounded:2d}  "
          f"fleet_FH={result.total_fh:6.1f}  "
          f"starts={n_starts}  completes={n_comps}  "
          f"aw_violations={len(result.violations)}")

    for ev in result.events:
        dur_str = (f"  sampled_dur={ev.actual_dur}d"
                   if ev.actual_dur is not None else "")
        print(f"  {ev.event_type.upper():8s}  {ev.tail:<14} k={ev.k}"
              f"  type={ev.check_type}{dur_str}")

    for v in result.violations:
        print(f"  AIRWORTHINESS  {v.tail:<14} {v.check_type}-{v.metric}"
              f"  {v.current:.1f} > {v.limit:.1f}  "
              f"(+{v.overrun:.1f})")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from data_utils import load_amcs_data
    from preprocessing import (
        init_committed_from_c_initial, run_preprocessing, RollingConfig,
    )
    from qubo_builder import build_c_qubo, build_a_qubo
    from solver import solve_stage

    path = sys.argv[1] if len(sys.argv) > 1 else "Scheduling_Input_2017.xlsx"
    data = load_amcs_data(path)

    rng    = np.random.default_rng(42)
    config = RollingConfig()

    # --- Solve Day 0 --------------------------------------------------------
    committed_c = init_committed_from_c_initial(data)
    committed_a: List[CommittedCheck] = []

    states = init_sim_states(data, committed_c, start_day=0)

    prep = run_preprocessing(
        data, states, current_day=0,
        committed_c=committed_c, committed_a=committed_a, config=config,
    )

    c_qubo = build_c_qubo(prep.c_windows, committed_c, data, current_day=0)
    c_committed, _ = solve_stage(c_qubo, data, rng=rng)
    committed_c = committed_c + c_committed

    a_qubo = build_a_qubo(
        prep.a_windows, committed_c, committed_a, data, current_day=0
    )
    a_committed, _ = solve_stage(a_qubo, data, rng=rng)
    committed_a = committed_a + a_committed

    print(f"Committed C-checks after Day-0 solve: {len(committed_c)}")
    print(f"Committed A-checks after Day-0 solve: {len(committed_a)}")

    # --- Simulate first 75 days -----------------------------------------------
    print("\nSimulating Day 0 to 74 (first 75 days):")
    print("-" * 72)

    total_aw = 0
    for day in range(75):
        result = advance_day(states, committed_c, committed_a, day, data, rng)
        if result.events or result.violations:
            print_day_summary(result, data)
        total_aw += len(result.violations)

    print("-" * 72)
    print(f"\nTotal airworthiness violation-days over 75 days: {total_aw}")

    # --- State snapshot at Day 75 ---------------------------------------------
    print("\nState snapshot at Day 75 (first 5 aircraft):")
    for tail in data.aircraft[:5]:
        st = states[tail]
        c_lim_dy, c_lim_fh, c_lim_fc = _check_limits(tail, st.next_c_k, "C", data)
        a_lim_dy, a_lim_fh, a_lim_fc = _check_limits(tail, st.next_a_k, "A", data)
        in_h = "IN-C" if st.in_c_check else ("IN-A" if st.in_a_check else "FLY ")
        print(
            f"  {tail:<14} [{in_h}]  "
            f"C: dy={st.dy_c:5.0f}/{c_lim_dy:.0f}  "
            f"fh={st.fh_c:6.1f}/{c_lim_fh:.0f}  |  "
            f"A: dy={st.dy_a:5.0f}/{a_lim_dy:.0f}  "
            f"fh={st.fh_a:6.1f}/{a_lim_fh:.0f}  "
            f"[next_c_k={st.next_c_k} next_a_k={st.next_a_k}]"
        )
