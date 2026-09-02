"""
02b_slice_selective_se.py — adding slice selection + crushers to the
bare spin-echo.

New concepts vs. 02a_spin_echo.py:
  - Both the 90 and 180 are now sinc pulses with slice-select gradients,
    selecting the SAME slice.
  - The 90 still needs its rephasing lobe (same reason as 01b: undo the
    phase spread accumulated during the slice-select gradient).
  - The 180 does NOT get a separate rephasing lobe. A slice-select
    gradient that is symmetric around a 180 pulse self-refocuses: the
    first half dephases spins, the pulse flips their phase, and the
    second half (now experienced as "unwinding") reconverges them. This
    matches pypulseq's own reference examples, which explicitly discard
    the rephase gradient returned for use='refocusing' pulses.
  - Crusher gradients, placed symmetrically before and after the 180.
    Real RF pulses are imperfect -- some spins get flipped by slightly
    more or less than exactly 180 degrees. Those imperfectly-refocused
    spins would otherwise contribute unwanted stray signal (stimulated
    echoes / residual FID) contaminating the real echo. A spin that WAS
    properly refocused experiences the crusher twice with cancelling
    net effect (no penalty). A spin that wasn't gets spoiled (dephased
    into the noise floor) instead of contaminating your data.

Still "bare" in the sense that there's no frequency/phase encoding --
this is a single-voxel-style acquisition (excite a slice, refocus,
read out), which is actually the closest thing so far in this repo to
the PRESS localization physics (PRESS uses this SAME 90-180 refocusing
idea, twice, along two more axes, to localize a 3D voxel instead of a
2D slice).
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

slice_thickness = 5e-3
te = 30e-3  # longer than 02a's bare version -- slice-select pulses and
            # crushers take real time, and TE must accommodate all of it

# ---------------------------------------------------------------------
# 90 excitation: sinc pulse + slice-select gradient + rephasing lobe
# ---------------------------------------------------------------------
rf90, gz90, gz90_reph = pp.make_sinc_pulse(
    flip_angle=np.pi / 2, duration=3e-3, slice_thickness=slice_thickness,
    apodization=0.5, time_bw_product=4, system=system,
    return_gz=True, use="excitation",
)

# ---------------------------------------------------------------------
# 180 refocusing: sinc pulse + slice-select gradient. We deliberately
# discard the third return value (the rephase gradient) -- see
# docstring above for why it isn't needed here.
# ---------------------------------------------------------------------
rf180, gz180, _unused_rephase = pp.make_sinc_pulse(
    flip_angle=np.pi, duration=3e-3, slice_thickness=slice_thickness,
    apodization=0.5, time_bw_product=4, system=system,
    return_gz=True, use="refocusing",
)

# ---------------------------------------------------------------------
# Crusher gradients: equal area, same polarity, placed before and after
# the 180. Placed on the same (z) axis as the slice-select gradient for
# simplicity. Area chosen somewhat arbitrarily larger than the slice
# select gradient's own area -- in practice this is tuned based on
# expected off-resonance/pulse imperfection, not a universal formula.
# ---------------------------------------------------------------------
gz_crush = pp.make_trapezoid(channel="z", area=gz90.area * 2, system=system)

adc = pp.make_adc(num_samples=512, duration=6.4e-3, system=system)

# ---------------------------------------------------------------------
# Assemble, computing delays to hit TE/2 (180 center) and TE (echo/ADC
# center) exactly, same logic as 02a but now accounting for the
# rephase lobe, crushers, and slice-select gradient durations.
# ---------------------------------------------------------------------
seq = pp.Sequence(system=system)

seq.add_block(rf90, gz90)
seq.add_block(gz90_reph)
block1_duration = pp.calc_duration(rf90, gz90) + pp.calc_duration(gz90_reph)
rf90_center = rf90.delay + rf90.shape_dur / 2  # measured from start of block 1

seq.add_block(gz_crush)

target_180_center = te / 2
time_before_180 = block1_duration + pp.calc_duration(gz_crush)
delay_needed = target_180_center - rf90_center - time_before_180 - (rf180.delay + rf180.shape_dur / 2)
seq.add_block(pp.make_delay(round(delay_needed, 5)))

seq.add_block(rf180, gz180)
seq.add_block(gz_crush)  # second crusher, identical area -- see docstring

time_so_far = (
    block1_duration
    + pp.calc_duration(gz_crush)
    + max(delay_needed, 0)
    + pp.calc_duration(rf180, gz180)
    + pp.calc_duration(gz_crush)
)
adc_center_offset = adc.delay + (adc.num_samples * adc.dwell) / 2
delay2 = te - rf90_center - time_so_far - adc_center_offset
seq.add_block(pp.make_delay(round(delay2, 5)))

seq.add_block(adc)

ok, error_report = seq.check_timing()
print("Timing check passed:", ok)
if not ok:
    print(error_report)

seq.set_definition("Name", "slice_selective_se")
seq.write("slice_selective_se.seq")
print("Wrote slice_selective_se.seq")
print(f"Target TE: {te*1e3:.1f} ms")

# Explicit numeric verification: where did the 180 pulse's center
# actually land, relative to the 90's center? Should equal TE/2.
rf180_center_actual = rf90_center + time_before_180 + max(delay_needed, 0) + rf180.delay + rf180.shape_dur / 2
print(f"180 pulse center (from 90 center): {rf180_center_actual:.5f} s "
      f"(should equal TE/2 = {te/2:.5f} s)")

# Verify 180 landed at TE/2 by re-reading the file's block timings
seq_check = pp.Sequence()
seq_check.read("slice_selective_se.seq")
print("(Re-load + check_timing on the written file, for good measure:)")
ok2, _ = seq_check.check_timing()
print("Re-loaded timing check passed:", ok2)