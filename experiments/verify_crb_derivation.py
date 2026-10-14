#!/usr/bin/env python
"""Numerically verify the Cramer-Rao derivation printed in the Paper B manuscript.

Section VI-A derives, from the FMCW beat model of Section III-B,

    I(nu)       = 8 pi^2 gamma_0 Ns Nc sum_n (n - nbar)^2            (manuscript eq. 6)
    sum_n (...) = m(m^2 - 1)/12                                      (manuscript eq. 7)
    sigma_theta >= lambda / (pi d cos theta) * sqrt(3 / (2 gamma_0 R))   (manuscript eq. 8)
    R           = Ns Nc m(m^2 - 1)                                   (manuscript eq. 9)

A factor error in a published bound is exactly the class of defect this project exists to
catch, and the first draft of eq. (8) was wrong by a factor of two. This builds the Fisher
information matrix numerically from the model, with the complex amplitude treated as an unknown
nuisance parameter, and compares. It asserts, so it fails loudly.

Usage: python experiments/verify_crb_derivation.py
"""
import sys
import os
import re

import numpy as np

TOL = 2e-3


def fim_sigma_nu_coupled(m, Ns, Nc, g0=1.0, nu=0.11, fR=0.23, fD=0.07):
    """sigma_nu when Doppler rides on the FAST-TIME beat as well as on slow time.

    The 2026-10-13 audit found the manuscript's eq. (1) omitting the fast-time Doppler term
    that its cited source carries, and asked whether the printed bearing bound survives the
    correction. It does, for a reason worth asserting rather than arguing: the map
    (f_R, f_D) -> (f_b, f_D) = (f_R + f_D, f_D) is invertible and leaves nu alone, so the
    two nuisance derivatives span the same subspace (span{k, k+l} = span{k, l}) and the
    information left in nu after eliminating them is unchanged. This builds the coupled FIM
    and asserts equality with the separable one.
    """
    n, k, l = np.meshgrid(np.arange(m), np.arange(Ns), np.arange(Nc), indexing="ij")
    n, k, l = n.ravel(), k.ravel(), l.ravel()
    ph = np.exp(2j * np.pi * ((fR + fD) * k + fD * l + nu * n))
    J = np.stack([2j * np.pi * k * ph, 2j * np.pi * (k + l) * ph,
                  2j * np.pi * n * ph, ph, 1j * ph])
    F = 2 * g0 * np.real(J.conj() @ J.T)
    return float(np.sqrt(np.linalg.inv(F)[2, 2]))


def fim_sigma_nu(m, Ns, Nc, g0=1.0, nu=0.11, fR=0.23, fD=0.07, s2=1.0):
    """sigma_nu from a numerically built FIM, alpha complex and unknown."""
    n = np.arange(m)[:, None, None]
    k = np.arange(Ns)[None, :, None]
    l = np.arange(Nc)[None, None, :]
    ph = np.exp(2j * np.pi * (fR * k + fD * l + nu * n))
    mu = np.sqrt(g0 * s2) * ph
    D = [((2j * np.pi * n) * mu).ravel(), ph.ravel(), (1j * ph).ravel()]
    I = np.array([[(2 / s2) * np.real(np.vdot(u, v)) for v in D] for u in D])
    return float(np.sqrt(np.linalg.inv(I)[0, 0]))


def main():
    cases = [(4, 16, 8), (4, 32, 16), (2, 16, 8), (8, 16, 8), (4, 8, 8), (12, 16, 8),
             (3, 24, 12), (6, 16, 16)]
    print(f"{'m':>3}{'Ns':>5}{'Nc':>5}{'numerical':>13}{'eq.(10) form':>14}{'ratio':>8}")
    worst = 0.0
    for m, Ns, Nc in cases:
        # eq. (7): the exact ULA second central moment
        moment = sum((n - (m - 1) / 2) ** 2 for n in range(m))
        assert abs(moment - m * (m * m - 1) / 12) < 1e-9, f"eq.(7) fails at m={m}"
        R = Ns * Nc * m * (m * m - 1)
        num = fim_sigma_nu(m, Ns, Nc)
        closed_nu = np.sqrt(3.0 / (2 * np.pi ** 2 * R))          # from eq. (6) and (7)
        lam, d, th = 1.0, 0.5, 0.0
        s_th_num = num / ((d / lam) * np.cos(th))
        s_th_eq8 = lam / (np.pi * d * np.cos(th)) * np.sqrt(3.0 / (2 * R))
        r = s_th_num / s_th_eq8
        worst = max(worst, abs(r - 1))
        print(f"{m:>3}{Ns:>5}{Nc:>5}{s_th_num:>13.6e}{s_th_eq8:>14.6e}{r:>8.4f}")
        assert abs(num / closed_nu - 1) < TOL, f"eq.(6)/(7) fail at m={m}, Ns={Ns}, Nc={Nc}"
        assert abs(r - 1) < TOL, f"eq.(10) fails at m={m}, Ns={Ns}, Nc={Nc}: ratio {r:.4f}"

    # m = 1 must give exactly zero information, hence an infinite bound
    assert sum((n - 0.0) ** 2 for n in range(1)) == 0.0
    assert 1 * (1 * 1 - 1) == 0
    print(f"\nm = 1: aperture moment is exactly 0, so the bearing bound is infinite, as the "
          f"manuscript states")
    print(f"worst deviation across {len(cases)} configurations: {worst:.2e} (tolerance {TOL})")
    worst = 0.0
    for m, Ns, Nc in ((2, 8, 8), (4, 8, 8), (4, 16, 8), (4, 8, 16),
                      (8, 8, 8), (8, 16, 16), (16, 8, 8), (3, 12, 7)):
        a, b = fim_sigma_nu(m, Ns, Nc), fim_sigma_nu_coupled(m, Ns, Nc)
        worst = max(worst, abs(b / a - 1))
    assert worst < 1e-9, f"the fast-time Doppler coupling moved sigma_nu by {worst:.2e}"
    print(f"carrying Doppler in fast time leaves sigma_nu unchanged, worst "
          f"relative deviation {worst:.2e} over 8 configurations")
    # Equation NUMBERS drift whenever an equation is added. This script named (8) to (11)
    # while the equations it checks had become (10) to (13), so it reported verifying the
    # static-shortlist construction. Resolve them from the manuscript's own .aux, and name
    # the labels when it is unavailable: a label cannot drift, a number can.
    labels = ["eq:fisher", "eq:moment", "eq:crb", "eq:product"]
    aux = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "paper_b", "PaperB.aux")
    nums = {}
    if os.path.exists(aux):
        txt = open(aux).read()
        for lab in labels:
            m = re.search(r"\\newlabel\{" + re.escape(lab) + r"\}\{\{([^}]*)\}", txt)
            if m:
                nums[lab] = m.group(1)
    shown = ", ".join(f"({nums[l]})" if l in nums else l for l in labels)
    print(f"{shown} all verified against a numerically built FIM")
    return 0


if __name__ == "__main__":
    sys.exit(main())
