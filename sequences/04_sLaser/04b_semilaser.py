"""
04b_semilaser.py — the full semi-LASER (sLASER) localization sequence:
slice-selective 90 (z) + a PAIR of adiabatic full-passage (AFP) pulses
on x + a PAIR of AFP pulses on y.

This is the real acquisition your Fleischer Lab protocols use (at 3T
and 7T), built here as its own PRESS-style progression: same voxel-
intersection localization logic as 03b_press_localized.py, but with
the simple 180s replaced by pairs of the B1-insensitive HS1 pulse
verified in 04a_adiabatic_pulse.py.

Why PAIRS, not single AFP pulses per axis: a single adiabatic
refocusing pulse leaves a residual, position-dependent phase error
(from the frequency sweep itself) that a single ordinary 180 doesn't
have. Using two AFP pulses of the same design on the same axis is the
standard way real sLASER sequences cancel that residual phase --
consult a pulse-design reference for the exact justification, since
deriving it from scratch is beyond a teaching implementation.

TIMING SIMPLIFICATION, stated explicitly: real sLASER implementations
carefully tune the exact gap between the two pulses in each pair based
on that pulse's specific phase-evolution properties. Here, each pair's
MIDPOINT is instead treated as the effective refocusing center -- the
same role TE1/2 and TE1+TE2/2 played for PRESS's single 180s. This is
a deliberate simplification for teaching clarity, not the literature-
exact spacing a real sequence would use.
"""

import numpy as np
import pypulseq as pp


def make_hs1_pulse(duration, bandwidth, peak_b1_hz, beta=5.3, system=None, delay=0.0):
    """HS1 adiabatic full-passage pulse -- see 04a_adiabatic_pulse.py
    for the full derivation and Bloch-simulated verification."""
    dt = system.rf_raster_time if system is not None else 1e-6
    n_t = int(round(duration / dt))
    t = np.arange(n_t) * dt
    tau = 2 * t / duration - 1
    amplitude_envelope = 1 / np.cosh(beta * tau)
    freq_sweep = -(bandwidth / 2) * np.tanh(beta * tau)
    phase = 2 * np.pi * np.cumsum(freq_sweep) * dt
    signal = peak_b1_hz * amplitude_envelope * np.exp(1j * phase)
    rf = pp.make_arbitrary_rf(
        signal=signal, flip_angle=2 * np.pi, bandwidth=bandwidth,
        no_signal_scaling=True, use="inversion", system=system, delay=delay,
    )
    return rf


def make_afp_slice_select(duration, bandwidth, peak_b1_hz, voxel_size, channel, system, beta=5.3):
    """Build one AFP pulse + its matched slice-select gradient, aligned
    so the RF plays entirely within the gradient's flat top."""
    g_amplitude = bandwidth / voxel_size  # Hz/m
    g = pp.make_trapezoid(channel=channel, amplitude=g_amplitude,
                           flat_time=duration, rise_time=200e-6, system=system)
    rf = make_hs1_pulse(duration, bandwidth, peak_b1_hz, beta=beta,
                         system=system, delay=g.rise_time)
    return rf, g


system = pp.Opts(
    max_grad=28, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s",
    rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=10e-6, B0=3,
)

voxel_size = 20e-3
afp_duration = 4e-3
afp_bandwidth = 4000     # Hz -- matches the verified adiabatic pulse
afp_peak_b1 = 4000       # Hz -- well above the adiabatic threshold found in 04a
te1 = 40e-3
te2 = 40e-3
te_total = te1 + te2

# ---------------------------------------------------------------------
# 90 excitation: slice-select on z (conventional sinc, same as PRESS)
# ---------------------------------------------------------------------
rf90, gz90, gz90_reph = pp.make_sinc_pulse(
    flip_angle=np.pi / 2, duration=3e-3, slice_thickness=voxel_size,
    apodization=0.5, time_bw_product=4, system=system,
    return_gz=True, use="excitation",
)
gz90.channel = "z"
gz90_reph.channel = "z"

# ---------------------------------------------------------------------
# AFP pair 1: x-axis
# ---------------------------------------------------------------------
rf_x1, gx1 = make_afp_slice_select(afp_duration, afp_bandwidth, afp_peak_b1, voxel_size, "x", system)
rf_x2, gx2 = make_afp_slice_select(afp_duration, afp_bandwidth, afp_peak_b1, voxel_size, "x", system)

# ---------------------------------------------------------------------
# AFP pair 2: y-axis
# ---------------------------------------------------------------------
rf_y1, gy1 = make_afp_slice_select(afp_duration, afp_bandwidth, afp_peak_b1, voxel_size, "y", system)
rf_y2, gy2 = make_afp_slice_select(afp_duration, afp_bandwidth, afp_peak_b1, voxel_size, "y", system)

# ---------------------------------------------------------------------
# Crushers: one pair around each AFP PAIR (not between the two pulses
# within a pair), on that pair's own axis.
# ---------------------------------------------------------------------
crush_x = pp.make_trapezoid(channel="x", area=gx1.area * 1.5, system=system)
crush_y = pp.make_trapezoid(channel="y", area=gy1.area * 1.5, system=system)

adc = pp.make_adc(num_samples=512, duration=6.4e-3, system=system)

# ---------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------
seq = pp.Sequence(system=system)

seq.add_block(rf90, gz90)
seq.add_block(gz90_reph)
block1_duration = pp.calc_duration(rf90, gz90) + pp.calc_duration(gz90_reph)
rf90_center = rf90.delay + rf90.shape_dur / 2

# --- crusher before x-pair ---
seq.add_block(crush_x)
time_before_pairx = block1_duration + pp.calc_duration(crush_x)

# --- delay so the X-PAIR MIDPOINT lands at TE1/2 ---
pair_x_internal_span = pp.calc_duration(rf_x1, gx1)  # time from AFP_x1 start to AFP_x2 start
# midpoint of the pair (relative to pair start) = half of (AFP_x1 duration + AFP_x2 duration)/... 
# defined here as: pair start -> AFP_x1 (duration D) -> AFP_x2 (duration D) -> pair end
# midpoint of the whole pair, from pair start, = D (i.e. exactly between the two pulses)
pair_x_midpoint_offset = pp.calc_duration(rf_x1, gx1)
target_pairx_mid = te1 / 2
delay1 = target_pairx_mid - (time_before_pairx - rf90_center) - pair_x_midpoint_offset
seq.add_block(pp.make_delay(round(delay1, 5)))

seq.add_block(rf_x1, gx1)
seq.add_block(rf_x2, gx2)
seq.add_block(crush_x)

time_so_far = (
    time_before_pairx + delay1
    + pp.calc_duration(rf_x1, gx1) + pp.calc_duration(rf_x2, gx2)
    + pp.calc_duration(crush_x)
)

# --- crusher before y-pair ---
seq.add_block(crush_y)
time_so_far += pp.calc_duration(crush_y)

# --- delay so the Y-PAIR MIDPOINT lands at TE1 + TE2/2 ---
pair_y_midpoint_offset = pp.calc_duration(rf_y1, gy1)
target_pairy_mid = te1 + te2 / 2
delay2 = rf90_center + target_pairy_mid - time_so_far - pair_y_midpoint_offset
seq.add_block(pp.make_delay(round(delay2, 5)))

seq.add_block(rf_y1, gy1)
seq.add_block(rf_y2, gy2)
seq.add_block(crush_y)

time_so_far_2 = (
    time_so_far + delay2
    + pp.calc_duration(rf_y1, gy1) + pp.calc_duration(rf_y2, gy2)
    + pp.calc_duration(crush_y)
)

# --- delay to place ADC center at the final echo (TE1+TE2) ---
adc_center_offset = adc.delay + (adc.num_samples * adc.dwell) / 2
delay3 = rf90_center + te_total - time_so_far_2 - adc_center_offset
seq.add_block(pp.make_delay(round(delay3, 5)))

seq.add_block(adc)

# ---------------------------------------------------------------------
# Timing check + explicit numeric verification of both pair midpoints
# ---------------------------------------------------------------------
ok, error_report = seq.check_timing()
print("Timing check passed:", ok)
if not ok:
    print(error_report)

seq.set_definition("Name", "semilaser")
seq.write("semilaser.seq")
print("Wrote semilaser.seq")
print(f"Voxel size: {voxel_size*1e3:.0f} mm cubic")
print(f"Target TE1: {te1*1e3:.1f} ms, TE2: {te2*1e3:.1f} ms, total TE: {te_total*1e3:.1f} ms")

actual_pairx_mid = time_before_pairx + delay1 + pair_x_midpoint_offset - rf90_center
actual_pairy_mid = (
    time_before_pairx + delay1
    + pp.calc_duration(rf_x1, gx1) + pp.calc_duration(rf_x2, gx2)
    + pp.calc_duration(crush_x) + pp.calc_duration(crush_y)
    + delay2 + pair_y_midpoint_offset - rf90_center
)
print(f"X-pair midpoint (from 90 center): {actual_pairx_mid:.5f} s "
      f"(target TE1/2 = {te1/2:.5f} s)")
print(f"Y-pair midpoint (from 90 center): {actual_pairy_mid:.5f} s "
      f"(target TE1 + TE2/2 = {te1 + te2/2:.5f} s)")