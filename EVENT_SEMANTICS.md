# Event semantics

This project treats event behavior as versioned numerical semantics.

## Version

Current event semantics version: 1.0.0

## Dose events

A DoseEvent is an instantaneous bolus applied to a specific state index defined by the PK model.

## Time handling

Let t0 be the simulation start time and t1 be the end time.

Dose events are partitioned into:

1. Events with time == t0  
   These are applied by adding their amounts into the initial condition.

2. Events with time in (t0, t1]  
   These are applied using a preset-time callback.

Events with time < t0 are invalid. Events with time > t1 are ignored.

## Duplicate times

If multiple dose events share the same time, their amounts are summed and applied as one event.

This rule applies both at t0 and within (t0, t1].

## Ordering

If the input dose schedule is nondecreasing in time, the engine behavior is deterministic.

The engine does not preserve ordering among events with identical time because they are summed.

## Rationale

Summing same-time events prevents ambiguous ordering, reduces sensitivity to input ordering,
and avoids callback index ambiguity.
