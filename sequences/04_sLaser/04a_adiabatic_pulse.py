"""
04a_adiabatic_pulse.py — build a hyperbolic secant (HS1) adiabatic
full-passage pulse, and Bloch-simulate its inversion profile to verify
the property that makes it worth using: B1-INSENSITIVE inversion.

Why this is a different kind of pulse than anything built so far:
A conventional (sinc/hard) 180 achieves inversion by applying a FIXED
B1 amplitude for a duration precisely calibrated to produce exactly a
180 rotation. If actual B1 differs from the calibrated value (due to
B1 inhomogeneity -- a real, significant problem at 7T), the flip angle
comes out wrong and refocusing degrades. That's exactly the sidelobe
contamination seen in press_voxel_profile.py.

An ADIABATIC pulse works differently: it sweeps BOTH amplitude and
frequency slowly over the pulse duration. As long as the sweep is slow
compared to the precession rate around the effective field (the
"adiabatic condition"), the magnetization vector continuously tracks
the effective field and ends up inverted -- REGARDLESS of the exact
peak B1 amplitude, above some threshold. This B1-insensitivity is the
entire reason sLASER uses these instead of simple 180s.

HS1 pulse definition (Silver-Hoult / Baum, standard form):
    tau(t)   = 2*t/Tp - 1                      , ranges -1 to +1
    A(t)     = A_max * sech(beta * tau)          amplitude envelope
    freq(t)  = -(bandwidth/2) * tanh(beta * tau) instantaneous frequency sweep
    phase(t) = cumulative integral of 2*pi*freq(t) dt
    signal(t)= A(t) * exp(i * phase(t))          complex RF waveform, Hz

beta (dimensionless truncation factor) and the bandwidth/duration
together set how "adiabatic" the sweep is. Typical literature values:
beta ~ 5-6, giving a good tradeoff between sharp inversion profile and
achievable peak B1 on real hardware.
"""

import numpy as np
import pypulseq as pp


def make_hs1_pulse(duration, bandwidth, peak_b1_hz, beta=5.3, system=None):
    """
    Build an HS1 adiabatic full-passage RF pulse as a pypulseq arbitrary
    RF event (no_signal_scaling=True, since flip-angle-based scaling is
    not meaningful for adiabatic pulses).

    Parameters
    ----------
    duration    : pulse duration, s (adiabatic pulses are typically
                  longer than conventional pulses -- several ms)
    bandwidth   : total frequency sweep range, Hz
    peak_b1_hz  : peak RF amplitude, Hz (i.e. gamma*B1_max) -- must
                  exceed the adiabatic threshold for the sweep rate
                  chosen, or inversion will be incomplete/B1-sensitive
    beta        : dimensionless truncation factor (unitless), controls
                  how sharply the sech/tanh envelopes are truncated
    """
    dt = system.rf_raster_time if system is not None else 1e-6
    n_t = int(round(duration / dt))
    t = np.arange(n_t) * dt
    tau = 2 * t / duration - 1  # -1 to +1

    amplitude_envelope = 1 / np.cosh(beta * tau)  # sech(beta*tau)
    freq_sweep = -(bandwidth / 2) * np.tanh(beta * tau)  # Hz

    # integrate frequency to get instantaneous phase
    phase = 2 * np.pi * np.cumsum(freq_sweep) * dt

    signal = peak_b1_hz * amplitude_envelope * np.exp(1j * phase)

    rf = pp.make_arbitrary_rf(
        signal=signal,
        flip_angle=2 * np.pi,  # placeholder -- irrelevant, see no_signal_scaling
        bandwidth=bandwidth,
        no_signal_scaling=True,
        use="inversion",
        system=system,
    )
    return rf


def rotate(M, axis, angle):
    """Rodrigues' rotation formula, vectorized over leading dimensions."""
    axis_norm = axis / (np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-15)
    cos_a = np.cos(angle)[..., None]
    sin_a = np.sin(angle)[..., None]
    dot = np.sum(M * axis_norm, axis=-1, keepdims=True)
    cross = np.cross(axis_norm, M)
    return M * cos_a + cross * sin_a + axis_norm * dot * (1 - cos_a)


def simulate_inversion_vs_offresonance(rf, dt, off_resonances_hz, initial_M=(0, 0, 1)):
    """
    Bloch-simulate this pulse's effect across a range of STATIC
    off-resonance values (Hz) -- i.e. no gradient, just testing how the
    pulse behaves for spins at different frequencies. This is the
    correct way to test an adiabatic pulse's own frequency-sweep
    behavior in isolation, before combining it with a slice-select
    gradient (which converts position into an effective off-resonance,
    exactly like every other slice-selective pulse in this repo).
    """
    b1_real = rf.signal.real
    b1_imag = rf.signal.imag
    n_t = len(b1_real)
    n_f = len(off_resonances_hz)

    M = np.tile(np.array(initial_M, dtype=float), (n_f, 1))
    for i in range(n_t):
        beff = np.stack([
            np.full(n_f, b1_real[i]),
            np.full(n_f, b1_imag[i]),
            off_resonances_hz,
        ], axis=-1)
        beff_mag = np.linalg.norm(beff, axis=-1)
        angle = 2 * np.pi * beff_mag * dt
        M = rotate(M, beff, angle)
    return M


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    system = pp.Opts(
        max_grad=28, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s",
        rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=10e-6, B0=3,
    )

    duration = 4e-3   # 4 ms -- adiabatic pulses run longer than the 1-3ms
                      # pulses used so far
    bandwidth = 4000  # 4 kHz sweep range
    beta = 5.3

    # A parameter sweep (done separately, not shown here) found the
    # adiabatic threshold for these duration/bandwidth/beta settings
    # sits around 2000 Hz peak B1 -- below it, inversion fails outright;
    # above it, inversion is excellent AND stays flat even as B1 keeps
    # increasing further. That plateau IS the adiabatic signature.
    peak_b1_levels = {
        "100% (4000 Hz) -- well above threshold": 4000,
        "50% (2000 Hz) -- still above threshold": 2000,
        "25% (1000 Hz) -- BELOW threshold": 1000,
    }

    offres = np.linspace(-3000, 3000, 301)  # Hz
    results = {}
    for label, peak_b1 in peak_b1_levels.items():
        rf = make_hs1_pulse(duration, bandwidth, peak_b1, beta, system)
        M = simulate_inversion_vs_offresonance(
            rf, system.rf_raster_time, offres, initial_M=(0, 0, 1)
        )
        results[label] = M[:, 2]

    plt.figure(figsize=(8, 5))
    for label, mz in results.items():
        plt.plot(offres, mz, label=label)
    plt.axhline(-1, color="gray", linestyle=":", alpha=0.5, label="Perfect inversion (Mz = -1)")
    plt.xlabel("Off-resonance (Hz)")
    plt.ylabel("Mz after pulse")
    plt.title("HS1 adiabatic pulse: inversion holds above threshold, fails below it")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("sequences/04_sLaser/adiabatic_inversion_profile.jpg", dpi=130)

    passband = np.abs(offres) < bandwidth / 2 * 0.8
    for label, mz in results.items():
        print(f"Mean Mz within passband, {label}: {mz[passband].mean():.3f}")
    print("(the 100% and 50% cases should both be close to -1.0 despite a 2x")
    print(" difference in peak B1 -- that's the adiabatic B1-insensitivity property.")
    print(" The 25% case should fail badly -- proving this isn't just 'any pulse")
    print(" works', there's a real threshold that must be exceeded.)")
    print("Saved sequences/04_sLaser/adiabatic_inversion_profile.jpg")