"""
01c_gre_2d.py — full 2D gradient-echo: adding frequency + phase encoding.

New concepts vs. 01b_slice_selective.py:
  - Gx (readout/frequency-encoding) gradient, on during the ADC window.
    While it's on, position along x maps to instantaneous frequency, so
    the collected signal is a 1D Fourier transform of the object along x.
  - a Gx "prephasing" lobe before readout, which walks k-space out to
    -kx_max so that by the time ADC starts, we sweep from -kx_max to
    +kx_max symmetrically through k-space center (needed so the echo
    -- the point of maximum signal -- lands mid-readout).
  - Gy (phase-encoding) gradient, briefly pulsed before readout, with a
    DIFFERENT amplitude each repetition. This doesn't change frequency;
    it accumulates a phase proportional to y-position that differs from
    one repetition to the next. Across many repetitions this builds the
    second dimension of k-space.
  - a repetition loop over phase-encode steps: one full k-space line
    per TR.

This script does NOT simulate the actual MR signal (that needs a Bloch
simulation over a numerical phantom -- next step). What it DOES let you
verify is that the gradient waveforms trace the k-space trajectory you
expect: a raster of horizontal lines filling a rectangular k-space grid.
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

# ---------------------------------------------------------------------
# Imaging parameters
# ---------------------------------------------------------------------
fov = 220e-3        # 220 mm field of view
Nx = 64              # readout (frequency-encode) matrix size
Ny = 64              # phase-encode matrix size
slice_thickness = 5e-3
te = 10e-3
tr = 20e-3           # keep short for a quick test sequence; real TR
                     # depends on desired T1-weighting, SAR, etc.

delta_k = 1 / fov    # k-space step size, same in both directions here

# ---------------------------------------------------------------------
# Slice-selective excitation (same as 01b, factored out for reuse)
# ---------------------------------------------------------------------
rf, gz, gz_reph = pp.make_sinc_pulse(
    flip_angle=np.pi / 2,
    duration=3e-3,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    system=system,
    return_gz=True,
)

# ---------------------------------------------------------------------
# Frequency-encoding (readout) gradient + its prephasing lobe.
#   flat_area = Nx * delta_k  -> the readout gradient must sweep exactly
#   Nx samples' worth of k-space during its flat top.
# ---------------------------------------------------------------------
gx = pp.make_trapezoid(
    channel="x",
    flat_area=Nx * delta_k,
    flat_time=3.2e-3,   # readout duration; combined with Nx sets bandwidth
    system=system,
)
adc = pp.make_adc(
    num_samples=Nx,
    duration=gx.flat_time,
    delay=gx.rise_time,
    system=system,
)
# prephasing lobe: walks k-space to -kx_max before the flat top starts.
# Area = half the flat-top area, opposite polarity (standard symmetric
# readout: rewind, then sweep all the way across through center).
gx_pre = pp.make_trapezoid(
    channel="x",
    area=-gx.area / 2,
    system=system,
)

# ---------------------------------------------------------------------
# Phase-encoding gradient: one trapezoid PER Ny step, amplitude varies.
# Precompute the full set of areas up front.
# ---------------------------------------------------------------------
phase_areas = (np.arange(Ny) - Ny / 2) * delta_k

# ---------------------------------------------------------------------
# Assemble the sequence: one full k-space line (one TR) per phase step.
# ---------------------------------------------------------------------
seq = pp.Sequence(system=system)

for i in range(Ny):
    seq.add_block(rf, gz)

    gy_pre = pp.make_trapezoid(channel="y", area=phase_areas[i], system=system)
    # gz_reph, gx_pre, and gy_pre all happen concurrently -- pyPulseq
    # will right-align them within the block by default, so they finish
    # together right as the readout gradient begins.
    seq.add_block(gz_reph, gx_pre, gy_pre)

    seq.add_block(gx, adc)

    # TR fill: pad remaining time so every repetition takes exactly `tr`
    time_so_far = (
        pp.calc_duration(rf, gz)
        + pp.calc_duration(gz_reph, gx_pre, gy_pre)
        + pp.calc_duration(gx, adc)
    )
    tr_delay = tr - time_so_far
    if tr_delay > 0:
        seq.add_block(pp.make_delay(round(tr_delay, 5)))

ok, error_report = seq.check_timing()
print("Timing check passed:", ok)
if not ok:
    print(error_report)

seq.set_definition("Name", "gre_2d")
seq.set_definition("FOV", [fov, fov, slice_thickness])
seq.write("gre_2d.seq")
print("Wrote gre_2d.seq")
print(f"Matrix: {Nx} x {Ny}, FOV: {fov*1e3:.0f} mm, TR: {tr*1e3:.1f} ms, "
      f"total scan time: {Ny*tr*1e3:.0f} ms")

# ---------------------------------------------------------------------
# k-space trajectory check -- this is the payoff. Confirms the gradient
# waveforms actually trace a rectangular raster through k-space, which
# is what makes this sequence capable of producing an image (unlike
# 01a/01b, which had no spatial encoding or only 1D slice selection).
# ---------------------------------------------------------------------
ktraj_adc, ktraj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()
print(f"k-space trajectory shape: {ktraj_adc.shape}  "
      f"(3 x total_adc_samples, rows = kx, ky, kz)")
print(f"kx range: [{ktraj_adc[0].min():.1f}, {ktraj_adc[0].max():.1f}] 1/m")
print(f"ky range: [{ktraj_adc[1].min():.1f}, {ktraj_adc[1].max():.1f}] 1/m")