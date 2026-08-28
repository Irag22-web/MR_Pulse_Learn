"""
01a_fid.py — the bare FID: the smallest possible MR experiment.

Physics: excite the whole sensitive volume with a single RF pulse, then
listen to the free induction decay. No spatial encoding at all -- this
sequence cannot produce an image. Its only purpose here is to isolate
RF excitation + readout from everything else (slice selection, phase
encoding, frequency encoding) that we'll add in later sequences.
"""

import numpy as np
import pypulseq as pp

# ---------------------------------------------------------------------
# 1. System limits. These bound every event we create below.
# ---------------------------------------------------------------------
system = pp.Opts(
    max_grad=28, grad_unit="mT/m",
    max_slew=150, slew_unit="T/m/s",
    rf_ringdown_time=20e-6,
    rf_dead_time=100e-6,
    adc_dead_time=10e-6,
)

# ---------------------------------------------------------------------
# 2. RF pulse: 90 degrees, hard (rectangular) pulse, 1 ms long.
#    No spatial selectivity -- fine, since we're not slice-selecting.
# ---------------------------------------------------------------------
rf = pp.make_block_pulse(
    flip_angle=np.pi / 2,
    duration=1e-3,
    system=system,
)

# ---------------------------------------------------------------------
# 3. Timing: TE = time from pulse center to readout start.
# ---------------------------------------------------------------------
te = 5e-3  # 5 ms

# ---------------------------------------------------------------------
# 4. ADC (readout) event. bandwidth = 1/dwell = num_samples/duration
# ---------------------------------------------------------------------
adc = pp.make_adc(
    num_samples=512,
    duration=6.4e-3,   # -> dwell = 12.5 us -> bandwidth = 80 kHz
    system=system,
)

# ---------------------------------------------------------------------
# 5. Assemble into a Sequence object.
# ---------------------------------------------------------------------
seq = pp.Sequence(system=system)

rf_center_delay = rf.shape_dur / 2 + rf.delay
adc_start = te - (rf.shape_dur - rf_center_delay)

seq.add_block(rf)
seq.add_block(pp.make_delay(round(adc_start, 5)))
seq.add_block(adc)

# ---------------------------------------------------------------------
# 6. Timing check -- always run this before trusting a sequence.
# ---------------------------------------------------------------------
ok, error_report = seq.check_timing()
print("Timing check passed:", ok)
if not ok:
    print(error_report)

# ---------------------------------------------------------------------
# 7. Save + report
# ---------------------------------------------------------------------
seq.set_definition("Name", "fid")
seq.write("fid.seq")
print("Wrote fid.seq")
print(f"TE (pulse-center to ADC start) target = {te*1e3:.2f} ms")