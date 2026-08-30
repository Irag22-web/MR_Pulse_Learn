"""
bloch_sim/slice_profile.py — validate slice selectivity via direct
numerical integration of the Bloch equation.

This does NOT use an idealized textbook RF pulse -- it reads the actual
sampled RF waveform and slice-select gradient amplitude out of a real
pypulseq sequence object, so what gets validated is the literal pulse
your .seq file contains, not a stand-in.

Method: isochromat (rotation-based) Bloch simulation.
  - Represent an array of independent "isochromats" at different z
    positions along the slice-select direction.
  - At each z, the gradient makes the *effective* off-resonance
    frequency different: delta_f(z) = Gz_amplitude * z  (Hz), since
    pypulseq stores gradient amplitude in Hz/m already (i.e. already
    multiplied by gamma).
  - Step through time in small increments dt (matching the RF's own
    1 us raster). At each step, the isochromat's magnetization vector
    is rotated (not just incremented -- an exact small-angle rotation)
    around the instantaneous effective field vector
    B_eff = (B1x(t), B1y(t), delta_f(z)), by angle 2*pi*|B_eff|*dt.
  - After stepping through the full RF+gradient window, whatever
    transverse magnetization (Mxy) remains at each z is the slice
    profile.

Relaxation (T1/T2) is neglected here -- a hard-pulse/rotation Bloch sim
of a few-millisecond RF pulse is a standard place to ignore relaxation,
since T1/T2 are typically >> pulse duration. This keeps the physics
focused on what a slice-select gradient + RF pulse actually does.
"""

import numpy as np


def rotate(M, axis, angle):
    """Rotate magnetization vector(s) M by `angle` (radians) around `axis`.
    Uses Rodrigues' rotation formula. M and axis are (..., 3) arrays;
    angle can be a scalar or match the leading shape of M.
    """
    axis_norm = axis / (np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-15)
    cos_a = np.cos(angle)[..., None]
    sin_a = np.sin(angle)[..., None]
    dot = np.sum(M * axis_norm, axis=-1, keepdims=True)
    cross = np.cross(axis_norm, M)
    return M * cos_a + cross * sin_a + axis_norm * dot * (1 - cos_a)


def simulate_slice_profile(rf, gz, z_range_m=(-10e-3, 10e-3), n_z=201):
    """
    Simulate slice profile for a pypulseq sinc RF pulse + slice-select
    gradient.

    Parameters
    ----------
    rf : pypulseq SimpleNamespace (from make_sinc_pulse)
    gz : pypulseq SimpleNamespace, the slice-select gradient
    z_range_m : (min, max) z positions to simulate, in meters
    n_z : number of z positions

    Returns
    -------
    z : (n_z,) array of z positions, meters
    mxy : (n_z,) array of |transverse magnetization| after the pulse
    mz : (n_z,) array of longitudinal magnetization after the pulse
    """
    dt = 1e-6  # matches pypulseq's default rf_raster_time
    b1 = rf.signal  # Hz, real-valued (sign encodes 0/180 phase toggle)
    n_t = len(b1)

    # Gradient amplitude during the RF: gz is a trapezoid whose flat-top
    # is time-aligned with the RF pulse by make_sinc_pulse. We use the
    # flat-top amplitude for the entire RF duration -- a standard
    # simplification (the ramps contribute negligibly for a pulse this
    # much longer than the ramp time).
    g_amplitude = gz.amplitude  # Hz/m

    z = np.linspace(z_range_m[0], z_range_m[1], n_z)
    off_resonance = g_amplitude * z  # Hz, per isochromat

    # M starts fully longitudinal: [0, 0, 1] for every isochromat
    M = np.zeros((n_z, 3))
    M[:, 2] = 1.0

    for i in range(n_t):
        b1x = b1[i]
        b1y = 0.0
        beff = np.stack([
            np.full(n_z, b1x),
            np.full(n_z, b1y),
            off_resonance,
        ], axis=-1)  # (n_z, 3), units Hz
        beff_mag = np.linalg.norm(beff, axis=-1)
        angle = 2 * np.pi * beff_mag * dt
        # avoid divide-by-zero in rotate() when beff_mag == 0 (only
        # possible if b1x==0 AND on-resonance -- won't happen mid-pulse)
        M = rotate(M, beff, angle)

    mxy = np.sqrt(M[:, 0] ** 2 + M[:, 1] ** 2)
    mz = M[:, 2]
    return z, mxy, mz


if __name__ == "__main__":
    import sys
    import os
    import numpy as np
    import pypulseq as pp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    system = pp.Opts(
        max_grad=28, grad_unit="mT/m",
        max_slew=150, slew_unit="T/m/s",
        rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=10e-6,
        B0=3,
    )
    slice_thickness = 5e-3
    rf, gz, gz_reph = pp.make_sinc_pulse(
        flip_angle=np.pi / 2, duration=3e-3,
        slice_thickness=slice_thickness, apodization=0.5,
        time_bw_product=4, system=system, return_gz=True,
    )

    z, mxy, mz = simulate_slice_profile(rf, gz, z_range_m=(-10e-3, 10e-3), n_z=201)

    plt.figure(figsize=(7, 4.5))
    plt.plot(z * 1e3, mxy, label="|Mxy| (transverse, i.e. excited signal)")
    plt.plot(z * 1e3, mz, label="Mz (longitudinal, remaining)")
    plt.axvspan(-slice_thickness / 2 * 1e3, slice_thickness / 2 * 1e3,
                color="gray", alpha=0.15, label=f"nominal slice ({slice_thickness*1e3:.0f} mm)")
    plt.xlabel("z position (mm)")
    plt.ylabel("Magnetization (normalized, M0=1)")
    plt.title("Bloch-simulated slice profile: sinc pulse, TBW=4, 90 deg")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("bloch_sim/slice_profile.jpg", dpi=130)
    print("Saved bloch_sim/slice_profile.jpg")
    print(f"Peak |Mxy|: {mxy.max():.3f} (should be close to 1.0 for a 90 deg pulse)")

    # quantify slice sharpness: where does Mxy cross half-max?
    half_max = mxy.max() / 2
    above = np.where(mxy >= half_max)[0]
    fwhm_mm = (z[above[-1]] - z[above[0]]) * 1e3
    print(f"FWHM of excited slice: {fwhm_mm:.2f} mm (nominal target: {slice_thickness*1e3:.1f} mm)")