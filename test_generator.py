"""Checks for generator.py against answers worked out by hand. Run: python test_generator.py"""
import itertools

import numpy as np

from generator import (BASELINES, EVENTS, NOISE_LEVELS, make_curve, peak_answers, peak_shape,
                       step_shape, tg_midpoint)

T = np.linspace(100, 200, 10001)  # fine grid, 100 to 200 °C, 0.01 K apart
RATE = 10 / 60  # K/s


def close_K(got, want):
    """Temperatures must agree within 0.05 K."""
    assert abs(got - want) <= 0.05, f"got {got:.3f}, want {want:.3f}"


def close_pct(got, want):
    """Values (enthalpies, slopes) must agree within 0.1 %."""
    assert abs(got - want) <= 0.001 * abs(want), f"got {got:.4f}, want {want:.4f}"


def test_triangle():
    # Rises in a straight line from 145 to a top of 1 W/g at 150, falls back to 0 at 157.
    # Onset = 145 (the leading edge is its own tangent), peak = 150, area = 1/2 × 12 K × 1 W/g = 6.
    tri = np.interp(T, [145, 150, 157], [0, 1, 0])
    for dev in (tri, -tri):  # exo and endo give the same answers
        a = peak_answers(T, dev)
        close_K(a["onset_C"], 145)
        close_K(a["peak_C"], 150)
        close_pct(a["enthalpy_J_per_g"], 6 / RATE)


def test_gaussian():
    # A Gaussian check curve: -0.8 exp(-((T - 156.5)/1.2)^2), so sigma = 1.2/sqrt(2).
    # Steepest point at 156.5 - sigma; its tangent meets 0 one more sigma lower: onset = 156.5 - 2 sigma.
    # Area = 0.8 × 1.2 × sqrt(pi).
    a = peak_answers(T, -0.8 * np.exp(-(((T - 156.5) / 1.2) ** 2)))
    close_K(a["onset_C"], 156.5 - 1.2 * np.sqrt(2))  # 154.80
    close_K(a["peak_C"], 156.5)
    close_pct(a["enthalpy_J_per_g"], 0.8 * 1.2 * np.sqrt(np.pi) / RATE)


def test_rectangle():
    # 0.5 W/g from 140 to 160: onset = 140, area = 0.5 × 20 = 10.
    a = peak_answers(T, np.where((T >= 140) & (T <= 160), 0.5, 0.0))
    close_K(a["onset_C"], 140)
    close_pct(a["enthalpy_J_per_g"], 10 / RATE)


def test_tg_step():
    # A step of 0.05 W/g centered at 132.3 °C: halfway between the two levels is at 132.3.
    close_K(tg_midpoint(T, -0.05 * step_shape(T, 132.3, 6)), 132.3)


def test_shapes():
    # Every peak has its top (1) at the center and the requested width at half height.
    for skew in (0, 0.3, 1.0):
        y = peak_shape(T, 150, 8, skew)
        assert abs(y.max() - 1) < 1e-3
        close_K(T[np.argmax(y)], 150)
        above = T[y >= 0.5]
        close_K(above[-1] - above[0], 8)
    # The step's tangent at its midpoint spans `width`: slope there = 1 / width.
    s = step_shape(T, 150, 6)
    close_pct(np.gradient(s, T).max(), 1 / 6)


def test_make_curve():
    # All 3 x 3 x 3 = 27 knob combinations; the counter i is used as the seed.
    for i, (noise, base, event) in enumerate(itertools.product(NOISE_LEVELS.keys(), BASELINES, EVENTS)):
        c = make_curve(noise, base, event, seed=i)
        T_c, p, t = c["T"], c["params"], c["truth"]
        # 100 K every 0.1 K, event temperature 60-300 °C.
        assert len(T_c) == 1001 and abs(T_c[-1] - T_c[0] - 100) < 1e-9
        assert np.allclose(np.diff(T_c), 0.1)
        assert 60 <= p["event_C"] <= 300
        assert T_c[0] < p["event_C"] < T_c[-1]
        # Exactly the right answers are filled in for the event type.
        if event == "glass_transition":
            assert t["onset_C"] is None and t["peak_C"] is None and t["enthalpy_J_per_g"] is None
            close_K(t["tg_mid_C"], p["event_C"])
        else:
            assert t["tg_mid_C"] is None
            close_K(t["peak_C"], p["event_C"])
            assert t["onset_C"] < t["peak_C"]
            assert t["enthalpy_J_per_g"] > 0
        assert t["direction"] == ("exo" if event == "crystallization" else "endo")
        # Noise: readers see clean + noise with the requested standard deviation.
        resid = c["noisy"] - c["clean"]
        assert abs(resid.std() / p["noise_sd"] - 1) < 0.15
        # The same seed gives the same curve.
        assert np.array_equal(make_curve(noise, base, event, seed=i)["noisy"], c["noisy"])


def test_bad_knob():
    # A misspelled knob level must stop with an error, not make a mislabelled curve.
    try:
        make_curve("medium", "flat", "melt", seed=0)
    except ValueError:
        return
    raise AssertionError("a misspelled event was accepted")


if __name__ == "__main__":
    tests = [test_triangle, test_gaussian, test_rectangle, test_tg_step, test_shapes, test_make_curve,
             test_bad_knob]
    for f in tests:
        f()
        print("PASS", f.__name__)
    print(f"All {len(tests)} checks passed.")
