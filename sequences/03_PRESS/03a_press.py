"""
03a_press_bare.py — the bare PRESS timing skeleton: two refocusing
pulses instead of one.

Physics: PRESS = 90 (slice-select axis 1) -> 180 (slice-select axis 2)
-> 180 (slice-select axis 3). Three mutually orthogonal slices; ONLY
their mutual intersection -- a single voxel -- experiences all three
pulses and produces correctly-refocused signal. Tissue outside that
intersection gets hit by some but not all pulses and (helped by
crushers, added in 03b) gets spoiled rather than contributing signal.
This is the same localization principle behind sLASER, just using
simple 180s instead of adiabatic full-passage pulses.

Standard PRESS timing, with two independently-settable echo times:
    90        at t = 0
    180 (#1)  at t = TE1/2
    180 (#2)  at t = TE1 + TE2/2
    acquisition (final echo) at t = TE1 + TE2 = TE

Why this works: the FIRST 180 creates an echo at t=TE1 (never directly
sampled -- it's an intermediate step). That echo's magnetization then
free-precesses and gets refocused a SECOND time by the second 180,
producing the final, acquired echo at TE1+TE2. Each refocusing follows
exactly the same TE/2-symmetric rule you already verified in
02a_spin_echo.py -- PRESS is that same rule, applied twice in series.

This is a "bare" version -- hard pulses, no slice selection -- to
isolate and verify the TWO-refocusing timing skeleton before adding
three-axis slice selection (03b) and crushers.
"""

import numpy as np
import pypulseq as pp

system = pp.Opts(
    max_grad=28, grad_unit="mT/m",
    max_slew=150, slew_unit="T/m/s",
    rf_ringdown_time=20e-6,
    rf_dead_time=100e-6,
    adc_dead_time=10e-6,
)

# ---------------------------------------------------------------------
# 1. RF pulses: one 90, two 180s. All hard pulses -- bare version.
# ---------------------------------------------------------------------
rf90 = pp.make_block_pulse(flip_angle=np.pi / 2, duration=1e-3, system=system)
rf180a = pp.make_block_pulse(flip_angle=np.pi, duration=1e-3, system=system)
rf180b = pp.make_block_pulse(flip_angle=np.pi, duration=1e-3, system=system)

# ---------------------------------------------------------------------
# 2. Timing targets
# ---------------------------------------------------------------------
te1 = 15e-3  # first echo time (90 to first echo, via 180a)
te2 = 15e-3  # second echo time (first echo to final echo, via 180b)
te_total = te1 + te2

adc = pp.make_adc(num_samples=512, duration=6.4e-3, system=system)

seq = pp.Sequence(system=system)

# --- block: 90 ---
seq.add_block(rf90)
rf90_center = rf90.delay + rf90.shape_dur / 2

# --- delay until 180a's center lands at TE1/2 ---
time_after_90 = pp.calc_duration(rf90)
target_180a_center = te1 / 2
delay1 = target_180a_center - (time_after_90 - rf90_center) - (rf180a.delay + rf180a.shape_dur / 2)
seq.add_block(pp.make_delay(round(delay1, 5)))

# --- block: 180a ---
seq.add_block(rf180a)
time_after_180a_block = pp.calc_duration(rf180a)

# --- delay until 180b's center lands at TE1 + TE2/2 (measured from
#     rf90's center, per the PRESS timing convention) ---
time_so_far = time_after_90 + delay1 + time_after_180a_block
target_180b_center = te1 + te2 / 2
delay2 = rf90_center + target_180b_center - time_so_far - (rf180b.delay + rf180b.shape_dur / 2)
seq.add_block(pp.make_delay(round(delay2, 5)))

# --- block: 180b ---
seq.add_block(rf180b)
time_after_180b_block = pp.calc_duration(rf180b)

# --- delay until ADC center lands at TE1 + TE2 (final echo, measured
#     from rf90's center) ---
time_so_far_2 = time_so_far + delay2 + time_after_180b_block
adc_center_offset = adc.delay + (adc.num_samples * adc.dwell) / 2
delay3 = rf90_center + te_total - time_so_far_2 - adc_center_offset
seq.add_block(pp.make_delay(round(delay3, 5)))

# --- block: readout, centered on the final (second) echo ---
seq.add_block(adc)

# ---------------------------------------------------------------------
# 3. Timing check + explicit verification of both refocusing points
# ---------------------------------------------------------------------
ok, error_report = seq.check_timing()
print("Timing check passed:", ok)
if not ok:
    print(error_report)

seq.set_definition("Name", "press_bare")
seq.write("press_bare.seq")
print("Wrote press_bare.seq")
print(f"Target TE1: {te1*1e3:.1f} ms, TE2: {te2*1e3:.1f} ms, "
      f"total TE: {te_total*1e3:.1f} ms")

actual_180a_center = time_after_90 + delay1 + rf180a.delay + rf180a.shape_dur / 2 - rf90_center
actual_180b_center = (
    time_after_90 + delay1 + time_after_180a_block + delay2
    + rf180b.delay + rf180b.shape_dur / 2 - rf90_center
)
print(f"180a center (from 90 center): {actual_180a_center:.5f} s "
      f"(target TE1/2 = {te1/2:.5f} s)")
print(f"180b center (from 90 center): {actual_180b_center:.5f} s "
      f"(target TE1 + TE2/2 = {te1 + te2/2:.5f} s)")