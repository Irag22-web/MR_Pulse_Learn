"""
03b_press_localized.py — three-axis slice-selective PRESS: the real
voxel-localization sequence.

New concepts vs. 03a_press_bare.py:
  - Each pulse slice-selects on a DIFFERENT gradient axis:
        90  (excitation) -> slice-select on z
        180 (#1)         -> slice-select on x
        180 (#2)         -> slice-select on y
    Only the mutual intersection of all three slices -- a rectangular
    voxel -- experiences correct excitation AND both refocusing pulses.
    Tissue outside that intersection is only hit by 1 or 2 of the three
    pulses and (with crushers) doesn't survive to contribute signal.
    This is the actual PRESS/sLASER localization mechanism -- same
    principle you already use experimentally, non-adiabatic version.
  - Crushers around EACH 180, same reasoning as 02b_slice_selective_se:
    spoil signal from imperfectly-refocused spins so only the correctly
    localized voxel's signal survives cleanly.
  - The 90 still needs its rephasing lobe (undoes phase spread from its
    own slice-select gradient). Neither 180 needs one (self-refocusing,
    same reasoning as 02b).

Timing: identical two-refocusing-point structure as 03a (180a at TE1/2,
180b at TE1+TE2/2, echo at TE1+TE2) -- just with real pulse/gradient
durations now, so TE1/TE2 need to be longer to leave room for them.
"""

import numpy as np
import pypulseq as pp

system = pp.Opts(
    max_grad=28, grad_unit="mT/m",
    max_slew=150, slew_unit="T/m/s",
    rf_ringdown_time=20e-6,
    rf_dead_time=100e-6,
    adc_dead_time=10e-6,
    B0=3,
)

voxel_size = 20e-3  # 20 mm cubic voxel -- typical SVS dimensions
te1 = 20e-3
te2 = 20e-3
te_total = te1 + te2

# ---------------------------------------------------------------------
# 90 excitation: slice-select on z
# ---------------------------------------------------------------------
rf90, gz90, gz90_reph = pp.make_sinc_pulse(
    flip_angle=np.pi / 2, duration=3e-3, slice_thickness=voxel_size,
    apodization=0.5, time_bw_product=4, system=system,
    return_gz=True, use="excitation",
)
gz90.channel = "z"
gz90_reph.channel = "z"

# ---------------------------------------------------------------------
# 180 #1: slice-select on x (note: make_sinc_pulse builds gradients on
# the z channel by default -- we relabel .channel to place them on the
# axis we actually want, a standard pypulseq pattern for orthogonal
# multi-axis localization).
# ---------------------------------------------------------------------
rf180a, gx180a, _unused = pp.make_sinc_pulse(
    flip_angle=np.pi, duration=3e-3, slice_thickness=voxel_size,
    apodization=0.5, time_bw_product=4, system=system,
    return_gz=True, use="refocusing",
)
gx180a.channel = "x"

# ---------------------------------------------------------------------
# 180 #2: slice-select on y
# ---------------------------------------------------------------------
rf180b, gy180b, _unused2 = pp.make_sinc_pulse(
    flip_angle=np.pi, duration=3e-3, slice_thickness=voxel_size,
    apodization=0.5, time_bw_product=4, system=system,
    return_gz=True, use="refocusing",
)
gy180b.channel = "y"

# ---------------------------------------------------------------------
# Crushers: one pair per refocusing pulse, placed on that pulse's OWN
# slice-select axis (a simplification -- real PRESS implementations
# often crush on multiple axes simultaneously for more robust spoiling,
# a natural extension noted here rather than built in).
# ---------------------------------------------------------------------
crush_x = pp.make_trapezoid(channel="x", area=gx180a.area * 2, system=system)
crush_y = pp.make_trapezoid(channel="y", area=gy180b.area * 2, system=system)

adc = pp.make_adc(num_samples=512, duration=6.4e-3, system=system)

# ---------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------
seq = pp.Sequence(system=system)

# --- 90 + slice-select-z, then rephase ---
seq.add_block(rf90, gz90)
seq.add_block(gz90_reph)
block1_duration = pp.calc_duration(rf90, gz90) + pp.calc_duration(gz90_reph)
rf90_center = rf90.delay + rf90.shape_dur / 2

# --- crusher before 180a ---
seq.add_block(crush_x)

# --- delay to place 180a's center at TE1/2 ---
time_before_180a = block1_duration + pp.calc_duration(crush_x)
target_180a_center = te1 / 2
delay1 = target_180a_center - (time_before_180a - rf90_center) - (rf180a.delay + rf180a.shape_dur / 2)
seq.add_block(pp.make_delay(round(delay1, 5)))

# --- 180a + slice-select-x ---
seq.add_block(rf180a, gx180a)

# --- crusher after 180a ---
seq.add_block(crush_x)

time_so_far = time_before_180a + delay1 + pp.calc_duration(rf180a, gx180a) + pp.calc_duration(crush_x)

# --- crusher before 180b ---
seq.add_block(crush_y)
time_so_far += pp.calc_duration(crush_y)

# --- delay to place 180b's center at TE1 + TE2/2 ---
target_180b_center = te1 + te2 / 2
delay2 = rf90_center + target_180b_center - time_so_far - (rf180b.delay + rf180b.shape_dur / 2)
seq.add_block(pp.make_delay(round(delay2, 5)))

# --- 180b + slice-select-y ---
seq.add_block(rf180b, gy180b)

# --- crusher after 180b ---
seq.add_block(crush_y)

time_so_far_2 = time_so_far + delay2 + pp.calc_duration(rf180b, gy180b) + pp.calc_duration(crush_y)

# --- delay to place ADC center at the final echo (TE1+TE2) ---
adc_center_offset = adc.delay + (adc.num_samples * adc.dwell) / 2
delay3 = rf90_center + te_total - time_so_far_2 - adc_center_offset
seq.add_block(pp.make_delay(round(delay3, 5)))

seq.add_block(adc)

# ---------------------------------------------------------------------
# Timing check + explicit numeric verification of both refocusing points
# (given a real bug was caught here in 03a, verify numerically again
# rather than trusting the arithmetic by inspection alone)
# ---------------------------------------------------------------------
ok, error_report = seq.check_timing()
print("Timing check passed:", ok)
if not ok:
    print(error_report)

seq.set_definition("Name", "press_localized")
seq.write("press_localized.seq")
print("Wrote press_localized.seq")
print(f"Voxel size: {voxel_size*1e3:.0f} mm cubic")
print(f"Target TE1: {te1*1e3:.1f} ms, TE2: {te2*1e3:.1f} ms, total TE: {te_total*1e3:.1f} ms")

actual_180a_center = time_before_180a + delay1 + rf180a.delay + rf180a.shape_dur / 2 - rf90_center
actual_180b_center = (
    time_before_180a + delay1 + pp.calc_duration(rf180a, gx180a) + pp.calc_duration(crush_x)
    + pp.calc_duration(crush_y) + delay2 + rf180b.delay + rf180b.shape_dur / 2 - rf90_center
)
print(f"180a center (from 90 center): {actual_180a_center:.5f} s "
      f"(target TE1/2 = {te1/2:.5f} s)")
print(f"180b center (from 90 center): {actual_180b_center:.5f} s "
      f"(target TE1 + TE2/2 = {te1 + te2/2:.5f} s)")