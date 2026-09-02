"""
bloch_sim/refocusing_demo.py — proof that a 180 refocusing pulse undoes
static-inhomogeneity dephasing, and a direct visual of why spin-echo
signal decays with T2 (slow) while a bare FID/GRE decays with T2* (fast).

Method: represent local field inhomogeneity as a population of
isochromats, each with its own STATIC off-resonance frequency f_i drawn
from a Lorentzian distribution (a Lorentzian linewidth is the standard
model for T2'-type static dephasing -- its Fourier transform is exactly
an exponential decay, which is why an inhomogeneous bare FID looks
exponentially-decaying even though no "true" relaxation has occurred).

For isochromat i, accumulated phase before the 180 (at TE/2):
    phase_before = 2*pi*f_i*(TE/2)
A 180 pulse flips this phase. Accumulated phase after the 180, at any
time t > TE/2:
    phase(t) = -phase_before + 2*pi*f_i*(t - TE/2)
Substitute t = TE:
    phase(TE) = -2*pi*f_i*(TE/2) + 2*pi*f_i*(TE/2) = 0
...for EVERY f_i, regardless of its value. That's the whole proof: no
matter how much static dephasing occurred, the 180 exactly undoes it by
time TE.

We separately multiply in an exp(-t/T2) envelope to represent TRUE,
irreversible relaxation (random microscopic processes -- diffusion,
spin-spin interactions -- that no refocusing pulse can undo). This is
why the echo peak at TE is below the initial signal, even though the
reversible (static-inhomogeneity) part refocused perfectly.
"""

import numpy as np


def simulate_signal(f_offsets, te, t2, t_max, n_t=2000):
    """
    f_offsets : array of per-isochromat static off-resonance frequencies (Hz)
    te        : echo time (s) -- the 180 is applied at te/2
    t2        : true T2 (s), applied as an overall irreversible envelope
    t_max     : total simulated time (s)
    Returns t (n_t,), signal_with_refocusing (n_t,), signal_no_refocusing (n_t,)
    """
    t = np.linspace(0, t_max, n_t)

    # --- No refocusing (bare FID/GRE case): phase just accumulates ---
    phase_no_refocus = 2 * np.pi * np.outer(t, f_offsets)  # (n_t, n_iso)
    signal_no_refocus = np.abs(np.mean(np.exp(1j * phase_no_refocus), axis=1))
    signal_no_refocus *= np.exp(-t / t2)  # true relaxation still applies

    # --- With refocusing: phase flips at TE/2 ---
    # Before TE/2: normal accumulation, phase(t) = 2*pi*f_i*t
    # At TE/2: 180 pulse flips accumulated phase -> -phase(TE/2)
    # After TE/2: continues accumulating from that flipped starting point:
    #   phase(t) = -phase(TE/2) + 2*pi*f_i*(t - TE/2)
    # At t = TE this equals exactly 0 for every f_i -- the refocusing proof.
    phase_at_flip = 2 * np.pi * (te / 2) * f_offsets  # (n_iso,), phase(TE/2)
    phase_refocus = np.where(
        t[:, None] <= te / 2,
        2 * np.pi * np.outer(t, f_offsets),
        -phase_at_flip[None, :] + 2 * np.pi * np.outer(t - te / 2, f_offsets),
    )
    signal_refocus = np.abs(np.mean(np.exp(1j * phase_refocus), axis=1))
    signal_refocus *= np.exp(-t / t2)  # true relaxation still applies -- this
                                        # is what keeps the echo peak BELOW 1.0

    return t, signal_refocus, signal_no_refocus


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    np.random.seed(0)
    n_iso = 2000
    # Lorentzian-distributed static off-resonance frequencies -- standard
    # model for field-inhomogeneity-induced dephasing (T2' contribution)
    linewidth_hz = 40.0  # controls how fast the "no refocusing" case decays
    f_offsets = linewidth_hz * np.tan(np.pi * (np.random.rand(n_iso) - 0.5)) * 0.5

    te = 0.040       # 40 ms echo time
    t2_true = 0.080  # 80 ms -- the TRUE, irreversible T2
    t_max = 0.070    # simulate a bit past TE to show the post-echo decay

    t, signal_refocus, signal_no_refocus = simulate_signal(
        f_offsets, te=te, t2=t2_true, t_max=t_max
    )

    plt.figure(figsize=(8, 5))
    plt.plot(t * 1e3, signal_no_refocus, label="No refocusing (bare FID/GRE) -- decays with T2*", color="tab:red")
    plt.plot(t * 1e3, signal_refocus, label="With 180 refocusing (spin-echo) -- reforms at TE", color="tab:blue")
    plt.axvline(te / 2 * 1e3, color="gray", linestyle="--", alpha=0.6, label="180 pulse (TE/2)")
    plt.axvline(te * 1e3, color="black", linestyle=":", alpha=0.6, label="Echo (TE)")
    plt.plot(t * 1e3, np.exp(-t / t2_true), color="gray", linestyle="-.", alpha=0.5,
              label=f"True T2 envelope (T2={t2_true*1e3:.0f} ms)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Signal magnitude (normalized)")
    plt.title("Spin-echo refocusing: static dephasing is undone, true T2 decay is not")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("bloch_sim/refocusing_demo.jpg", dpi=130)

    echo_idx = np.argmin(np.abs(t - te))
    print(f"Signal at TE, WITH refocusing:    {signal_refocus[echo_idx]:.3f}")
    print(f"Signal at TE, WITHOUT refocusing: {signal_no_refocus[echo_idx]:.3f}")
    print(f"True T2 envelope value at TE:      {np.exp(-te/t2_true):.3f}")
    print("(the refocused signal at TE should closely match the T2 envelope --")
    print(" i.e. ALL that's left is true relaxation, static dephasing is fully undone)")
    print("Saved bloch_sim/refocusing_demo.jpg")