"""
bloch_sim/press_voxel_profile.py — Bloch-simulate the actual PRESS
voxel: show that the intersection of the three orthogonal slice-select
pulses (90 on z, 180 on x, 180 on y) produces a spatially localized
voxel, using the REAL pulses from sequences/03_press/03b_press_localized.py.

Method: because the three slice-select gradients are on orthogonal
physical axes, each pulse's effect on magnetization depends ONLY on
position along its own axis. This lets us validate each axis
independently via 1D Bloch simulation, then combine them:

    voxel_profile(x, y, z) = profile_z(z) * profile_x(x) * profile_y(y)

This is the standard approach used in PRESS localization-profile
analysis in the literature (the "spatial response function"), not a
simplification unique to this teaching example.

Two different Bloch simulations are needed, because the 90 and the
180s do physically different jobs:
  - profile_z: the 90 converts LONGITUDINAL magnetization (Mz=1) into
    transverse signal. Simulated exactly as in slice_profile.py.
  - profile_x, profile_y: each 180 acts on magnetization that is
    ALREADY transverse (produced by the preceding 90/180). The relevant
    question isn't "does excitation happen here" but "does refocusing
    happen here" -- so we start these simulations with transverse
    magnetization (My=1) and measure how much comes back properly
    inverted (-My) as a function of position. This is a direct Bloch-
    equation probe of refocusing quality, not excitation quality.
"""

import numpy as np
import pypulseq as pp


def rotate(M, axis, angle):
    """Rodrigues' rotation formula, vectorized over leading dimensions."""
    axis_norm = axis / (np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-15)
    cos_a = np.cos(angle)[..., None]
    sin_a = np.sin(angle)[..., None]
    dot = np.sum(M * axis_norm, axis=-1, keepdims=True)
    cross = np.cross(axis_norm, M)
    return M * cos_a + cross * sin_a + axis_norm * dot * (1 - cos_a)


def simulate_profile(rf, g, positions, initial_M):
    """
    Generic Bloch simulation across a range of positions along a
    pulse's own slice-select gradient axis.

    rf, g       : pypulseq RF and (already-generated) slice-select gradient
    positions   : array of positions (m) along the gradient's axis
    initial_M   : (3,) starting magnetization, e.g. [0,0,1] or [0,1,0]

    Returns final (n_pos, 3) magnetization after the pulse.
    """
    dt = 1e-6
    b1 = rf.signal
    n_t = len(b1)
    g_amplitude = g.amplitude  # Hz/m
    off_resonance = g_amplitude * positions

    n_pos = len(positions)
    M = np.tile(np.array(initial_M, dtype=float), (n_pos, 1))

    for i in range(n_t):
        beff = np.stack([
            np.full(n_pos, b1[i]),
            np.zeros(n_pos),
            off_resonance,
        ], axis=-1)
        beff_mag = np.linalg.norm(beff, axis=-1)
        angle = 2 * np.pi * beff_mag * dt
        M = rotate(M, beff, angle)

    return M


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------------
    # Rebuild the EXACT same pulses used in 03b_press_localized.py, so
    # this validates the actual sequence, not a stand-in.
    # ------------------------------------------------------------------
    system = pp.Opts(
        max_grad=28, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s",
        rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=10e-6, B0=3,
    )
    voxel_size = 20e-3

    rf90, gz90, _ = pp.make_sinc_pulse(
        flip_angle=np.pi / 2, duration=3e-3, slice_thickness=voxel_size,
        apodization=0.5, time_bw_product=4, system=system,
        return_gz=True, use="excitation",
    )
    rf180a, gx180a, _ = pp.make_sinc_pulse(
        flip_angle=np.pi, duration=3e-3, slice_thickness=voxel_size,
        apodization=0.5, time_bw_product=4, system=system,
        return_gz=True, use="refocusing",
    )
    rf180b, gy180b, _ = pp.make_sinc_pulse(
        flip_angle=np.pi, duration=3e-3, slice_thickness=voxel_size,
        apodization=0.5, time_bw_product=4, system=system,
        return_gz=True, use="refocusing",
    )

    span = 20e-3  # simulate +/- 20 mm around isocenter
    n_pos = 201
    z = np.linspace(-span, span, n_pos)
    x = np.linspace(-span, span, n_pos)
    y = np.linspace(-span, span, n_pos)

    # --- z profile: 90 excitation, Mz=1 -> Mxy ---
    M_z = simulate_profile(rf90, gz90, z, initial_M=[0, 0, 1])
    profile_z = np.sqrt(M_z[:, 0] ** 2 + M_z[:, 1] ** 2)

    # --- x profile: 180a refocusing, My=1 -> -My recovered ---
    M_x = simulate_profile(rf180a, gx180a, x, initial_M=[0, 1, 0])
    profile_x = np.clip(-M_x[:, 1], 0, None)  # negative My = correctly inverted

    # --- y profile: 180b refocusing, My=1 -> -My recovered ---
    M_y = simulate_profile(rf180b, gy180b, y, initial_M=[0, 1, 0])
    profile_y = np.clip(-M_y[:, 1], 0, None)

    # ------------------------------------------------------------------
    # Combine: voxel signal is the product of all three 1D profiles.
    # Plot the three 1D profiles, plus a 2D XY cross-section (the
    # product of profile_x and profile_y) showing the voxel appearing
    # as the intersection of two orthogonal slabs.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    axes[0].plot(z * 1e3, profile_z)
    axes[0].axvspan(-voxel_size/2*1e3, voxel_size/2*1e3, color="gray", alpha=0.15)
    axes[0].set_title("z profile (90 excitation)")
    axes[0].set_xlabel("z (mm)")
    axes[0].set_ylabel("|Mxy|")

    axes[1].plot(x * 1e3, profile_x, color="tab:orange")
    axes[1].axvspan(-voxel_size/2*1e3, voxel_size/2*1e3, color="gray", alpha=0.15)
    axes[1].set_title("x profile (180 #1 refocusing)")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("Refocusing efficiency")

    axes[2].plot(y * 1e3, profile_y, color="tab:green")
    axes[2].axvspan(-voxel_size/2*1e3, voxel_size/2*1e3, color="gray", alpha=0.15)
    axes[2].set_title("y profile (180 #2 refocusing)")
    axes[2].set_xlabel("y (mm)")
    axes[2].set_ylabel("Refocusing efficiency")

    voxel_xy = np.outer(profile_y, profile_x)  # rows=y, cols=x
    im = axes[3].imshow(
        voxel_xy, extent=[x.min()*1e3, x.max()*1e3, y.min()*1e3, y.max()*1e3],
        origin="lower", cmap="viridis", aspect="equal",
    )
    axes[3].set_title("XY voxel cross-section\n(product of x and y profiles)")
    axes[3].set_xlabel("x (mm)")
    axes[3].set_ylabel("y (mm)")
    plt.colorbar(im, ax=axes[3], fraction=0.046)

    plt.tight_layout()
    plt.savefig("bloch_sim/press_voxel_profile.jpg", dpi=130)

    # ------------------------------------------------------------------
    # Quantify: FWHM of each profile, compared to the nominal voxel size
    # ------------------------------------------------------------------
    def fwhm(coord, profile):
        half_max = profile.max() / 2
        above = np.where(profile >= half_max)[0]
        return (coord[above[-1]] - coord[above[0]]) * 1e3

    print(f"Nominal voxel size: {voxel_size*1e3:.0f} mm cubic")
    print(f"z profile FWHM: {fwhm(z, profile_z):.2f} mm")
    print(f"x profile FWHM: {fwhm(x, profile_x):.2f} mm")
    print(f"y profile FWHM: {fwhm(y, profile_y):.2f} mm")
    print("Saved bloch_sim/press_voxel_profile.jpg")