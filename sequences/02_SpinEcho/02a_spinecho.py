"""
02a_spin_echo.py — the bare spin-echo: adding a 180 refocusing pulse.

Physics vs. the FID/GRE sequences: after the 90, spins dephase due to
local field inhomogeneity (some precess faster, some slower). A 180
pulse at TE/2 flips every spin's phase -- the fast ones are now behind,
the slow ones ahead. Since each spin's own precession rate is unchanged,
they realign exactly at time TE: the "echo." This undoes STATIC field
inhomogeneity dephasing only -- true molecular-level relaxation can't be
refocused, so spin-echo signal decays with T2 (slower), not T2* (faster,
what a bare FID/GRE decays with).

This is a "bare" spin-echo -- hard pulses, no spatial encoding -- same
teaching philosophy as 01a_fid.py: isolate the new physics concept
(refocusing) before adding slice selection / imaging gradients on top,
which will happen in 02b/02c.

The one thing that makes spin-echo timing strict in a way FID/GRE
wasn't: the 180 pulse must sit at EXACTLY TE/2, symmetrically, or the
echo doesn't form centered in the readout window.
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
# 1. RF pulses: 90 excitation, 180 refocusing. Both hard pulses here --
#    no slice selection yet, matching the "bare" philosophy above.
# ---------------------------------------------------------------------
rf90 = pp.make_block_pulse(flip_angle=np.pi / 2, duration=1e-3, system=system)
rf180 = pp.make_block_pulse(flip_angle=np.pi, duration=1e-3, system=system)

# ---------------------------------------------------------------------
# 2. Timing. TE = time from center of 90 to the echo peak (center of
#    readout). The 180 sits at exactly TE/2 from the 90's center.
# ---------------------------------------------------------------------
te = 20e-3  # 20 ms -- longer than the FID's TE, to leave room for both
            # pulses and delays; real spin-echo TEs are chosen based on
            # the T2 of the tissue you want to weight for/against

adc = pp.make_adc(num_samples=512, duration=6.4e-3, system=system)

seq = pp.Sequence(system=system)

# --- block 1: 90 pulse ---
seq.add_block(rf90)
rf90_center = rf90.delay + rf90.shape_dur / 2

# --- delay until 180's center lands at TE/2 ---
rf180_center_target = te / 2
time_after_rf90_block = pp.calc_duration(rf90)
delay1 = rf180_center_target - (time_after_rf90_block - rf90_center) - (rf180.delay + rf180.shape_dur / 2)
seq.add_block(pp.make_delay(round(delay1, 5)))

# --- block 2: 180 pulse ---
seq.add_block(rf180)

# --- delay until ADC center lands at TE ---
time_after_rf180_block = pp.calc_duration(rf180)
adc_center_target = te
time_so_far = time_after_rf90_block + delay1 + time_after_rf180_block
adc_center_offset = adc.delay + (adc.num_samples * adc.dwell) / 2
delay2 = adc_center_target - time_so_far - adc_center_offset
seq.add_block(pp.make_delay(round(delay2, 5)))

# --- block 3: readout, centered on the echo ---
seq.add_block(adc)

# ---------------------------------------------------------------------
# 3. Timing check
# ---------------------------------------------------------------------
ok, error_report = seq.check_timing()
print("Timing check passed:", ok)
if not ok:
    print(error_report)

seq.set_definition("Name", "spin_echo")
seq.write("spin_echo.seq")
print("Wrote spin_echo.seq")
print(f"Target TE: {te*1e3:.1f} ms")
print(f"180 pulse center (from 90 center): "
      f"{(time_after_rf90_block + delay1 + rf180.delay + rf180.shape_dur/2) - rf90_center:.5f} s "
      f"(should equal TE/2 = {te/2:.5f} s)")