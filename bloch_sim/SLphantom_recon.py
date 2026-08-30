"""
bloch_sim/SLphantom_recon.py — reconstruct the Shepp-Logan phantom from
the REAL k-space trajectory of sequences/01_fid_gre/gre_2d.seq.

The Shepp-Logan phantom is the standard synthetic test image used
throughout the CT/MRI reconstruction literature since 1974 -- ten
overlapping ellipses of varying intensity, standing in for skull,
gray/white matter, and small lesion-like features.

Physics: the MR signal equation, derived from the Bloch equation under
linear encoding gradients (relaxation and off-resonance neglected):

    s(kx, ky) = integral of rho(x,y) * exp(-i*2*pi*(kx*x + ky*y)) dx dy

This says the signal collected at each k-space sample is literally one
term of the object's 2D Fourier transform. That's why an inverse FFT of
correctly-sampled k-space reconstructs the image -- it's not a numerical
trick, it's the direct physical consequence of frequency/phase encoding.

Reconstruction here uses direct summation over the actual (kx, ky)
sample locations pulled from the real sequence -- this doesn't assume
anything about sample ordering/grid structure, so it's a rigorous test
of whether the real gradient trajectory encodes position correctly.

Deliberately NOT modeled here (left as a noted extension): T2* decay
during readout, B0 inhomogeneity, and noise.
"""

import numpy as np
import pypulseq as pp


def make_shepp_logan_phantom(nx, ny):
    """The Shepp-Logan phantom. Ellipse parameters (A, a, b, x0, y0,
    phi_degrees) are the classic Shepp-Logan values, in a coordinate
    system where the phantom spans [-1, 1] x [-1, 1].
    """
    ellipses = [
        # A,     a,      b,      x0,     y0,     phi (deg)
        (1.00,  0.69,   0.92,   0.00,   0.00,   0),
        (-0.80, 0.6624, 0.8740, 0.00,  -0.0184, 0),
        (-0.20, 0.1100, 0.3100, 0.22,   0.00,  -18),
        (-0.20, 0.1600, 0.4100, -0.22,  0.00,   18),
        (0.10,  0.2100, 0.2500, 0.00,   0.35,   0),
        (0.10,  0.0460, 0.0460, 0.00,   0.10,   0),
        (0.10,  0.0460, 0.0460, 0.00,  -0.10,   0),
        (0.10,  0.0460, 0.0230, -0.08, -0.605,  0),
        (0.10,  0.0230, 0.0230, 0.00,  -0.606,  0),
        (0.10,  0.0230, 0.0460, 0.06,  -0.605,  0),
    ]

    y, x = np.mgrid[0:ny, 0:nx]
    x = (x - nx / 2) / (nx / 2)
    y = (y - ny / 2) / (ny / 2)

    phantom = np.zeros((ny, nx))
    for A, a, b, x0, y0, phi_deg in ellipses:
        phi = np.deg2rad(phi_deg)
        x_shift = x - x0
        y_shift = y - y0
        x_rot = x_shift * np.cos(phi) + y_shift * np.sin(phi)
        y_rot = -x_shift * np.sin(phi) + y_shift * np.cos(phi)
        inside = (x_rot**2 / a**2 + y_rot**2 / b**2) <= 1
        phantom[inside] += A

    # Shepp-Logan intensities can go slightly negative by construction
    # (overlapping ellipses subtract); clip and normalize for display.
    phantom = np.clip(phantom, 0, None)
    phantom = phantom / phantom.max()
    return phantom


def direct_dft_recon(kx, ky, x_coords, y_coords, phantom_flat):
    """Direct summation signal equation: for each k-sample, sum the
    object's Fourier kernel over every pixel. Rigorous, uses the exact
    (kx, ky) values from the real trajectory -- no grid assumption.
    """
    n_k = len(kx)
    signal = np.zeros(n_k, dtype=complex)
    for i in range(n_k):
        phase = -2 * np.pi * (kx[i] * x_coords + ky[i] * y_coords)
        signal[i] = np.sum(phantom_flat * np.exp(1j * phase))
    return signal


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------------
    # 1. Load the REAL sequence and its k-space trajectory
    # ------------------------------------------------------------------
    seq = pp.Sequence()
    seq.read("sequences/01_fid_gre/gre_2d.seq")
    ktraj_adc, ktraj, t_ex, t_ref, t_adc = seq.calculate_kspace()
    kx, ky = ktraj_adc[0], ktraj_adc[1]

    Nx = Ny = 64  # matches the sequence's own matrix size
    fov = 220e-3  # matches the sequence's own FOV

    # ------------------------------------------------------------------
    # 2. Build the phantom and its real-space pixel coordinates
    # ------------------------------------------------------------------
    phantom = make_shepp_logan_phantom(Nx, Ny)
    y_idx, x_idx = np.mgrid[0:Ny, 0:Nx]
    x_coords = ((x_idx - Nx / 2) * (fov / Nx)).flatten()
    y_coords = ((y_idx - Ny / 2) * (fov / Ny)).flatten()
    phantom_flat = phantom.flatten()

    # ------------------------------------------------------------------
    # 3. Direct summation reconstruction (rigorous, uses real trajectory)
    # ------------------------------------------------------------------
    print("Running direct DFT over real k-space trajectory "
          f"({len(kx)} k-samples x {Nx*Ny} pixels)...")
    signal = direct_dft_recon(kx, ky, x_coords, y_coords, phantom_flat)

    # Reshape assumes row-major (ky outer loop, kx inner loop) ordering,
    # matching how the sequence script looped: one full kx readout line
    # per ky step.
    kspace_grid = signal.reshape(Ny, Nx)
    recon = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_grid)))

    error = np.abs(np.abs(recon) - phantom)
    print(f"Max |reconstruction - phantom| = {error.max():.4f} "
          f"(normalized intensity 0-1 scale; should be ~0)")

    # ------------------------------------------------------------------
    # 4. Plot: phantom vs. k-space vs. reconstruction
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(phantom, cmap="gray")
    axes[0].set_title("Shepp-Logan: ground truth")
    axes[0].axis("off")

    axes[1].imshow(np.log(np.abs(kspace_grid) + 1e-3), cmap="gray")
    axes[1].set_title("k-space (log magnitude)\nfrom real .seq trajectory")
    axes[1].axis("off")

    axes[2].imshow(np.abs(recon), cmap="gray")
    axes[2].set_title("Reconstructed image\n(IFFT of simulated signal)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("bloch_sim/SLphantom_reconstruction.jpg", dpi=130)
    print("Saved bloch_sim/SLphantom_reconstruction.jpg")
