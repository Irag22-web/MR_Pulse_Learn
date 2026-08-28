"""
01b_slice_selective.py — adding slice selection to the bare FID.

New concepts vs. 01a_fid.py:
  - sinc RF pulse instead of a hard pulse (narrower frequency content
    -> cleaner slice profile)
  - a slice-select gradient played concurrently with the RF pulse
  - a rephasing gradient lobe immediately after, to undo the phase
    spread accumulated across the slice during excitation

Still no frequency/phase encoding -- this excites a thin slab of tissue
and reads out the FID from that slab only, but still can't form an
image (that needs gradients during readout + repeated phase-encoded
acquisitions, which is the next step -> full GRE).
"""

import numpy as np
import pypulseq as pp

system = pp.Opts(
    max_grad=28, grad_unit="mT/m",
    max_slew=150, slew_unit="T/m/s",
    rf_ringdown_time=20e-6,
    rf_dead_time=100e-6,
    adc_dead_time=10e-6,
    B0=3,  # your lab's field strength -- matters now because slice
           # thickness/bandwidth tradeoffs depend on gamma*B0
)

slice_thickness = 5e-3  # 5 mm, a typical imaging slice

# make_sinc_pulse returns THREE things when you ask for a slice-select
# gradient: the RF event, the slice-select gradient, and the rephasing
# gradient -- pyPulseq computes the rephasing lobe's area for you.
rf, gz, gz_reph = pp.make_sinc_pulse(
    flip_angle=np.pi / 2,
    duration=3e-3,           # sinc pulses are longer than hard pulses
                              # -- narrower time-bandwidth product needs
                              # more time
    slice_thickness=slice_thickness,
    apodization=0.5,         # tapers the sinc's edges -> reduces ringing
                              # in the slice profile (Gibbs artifact)
    time_bw_product=4,       # standard value; higher = sharper slice
                              # edges but longer/higher-SAR pulse
    system=system,
    return_gz=True,
)

print(f"Slice-select gradient amplitude: {gz.amplitude:.1f} Hz/m")
print(f"Rephasing gradient area: {gz_reph.area:.2f} 1/m")
print(f"(Rephasing area should be close to -0.5x the excitation "
      f"gradient's flat-top area, by design)")

te = 8e-3  # slightly longer than before, since the pulse itself is longer
adc = pp.make_adc(num_samples=512, duration=6.4e-3, system=system)

seq = pp.Sequence(system=system)

# Order matters: RF+slice-select gradient happen together (pyPulseq
# handles this when you pass rf and gz as one block), THEN the
# rephasing lobe, THEN wait out the remaining time to TE, THEN read out.
seq.add_block(rf, gz)
seq.add_block(gz_reph)

# figure out how much delay is left before TE
time_so_far = pp.calc_duration(rf, gz) / 2 + pp.calc_duration(gz_reph)
# (dividing rf/gz duration by ~2 approximates pulse-center; good enough
#  for this teaching example -- we'll be more precise once we build
#  spin-echo, where exact TE symmetry actually matters for refocusing)
remaining_delay = te - time_so_far
seq.add_block(pp.make_delay(round(remaining_delay, 5)))
seq.add_block(adc)

ok, error_report = seq.check_timing()
print("Timing check passed:", ok)
if not ok:
    print(error_report)

seq.set_definition("Name", "slice_selective_fid")
seq.write("slice_selective_fid.seq")
print("Wrote slice_selective_fid.seq")