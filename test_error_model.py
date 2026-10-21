"""Checks for error_model.py. Run: python test_error_model.py      (a minute or two: a few short sampler runs)

The fast checks use numbers worked out by hand. The sampler checks use simulated readings whose true bias,
knob effects and scatter are known, with quick-sized sampling (2 chains x 200 draws).
"""
import csv
import itertools
import json
import re
import tempfile
from pathlib import Path

import arviz as az
import numpy as np

import error_model
from error_model import (QUICK, beta_summary, error_range, fit_report, gross_and_missed, hdi, knob_levels,
                         mcse_of_hdi_low, mcse_of_share, median_abs_error, needs_longer_run, parse_args, prob_text,
                         reader_checks, worst_call, worst_condition, worst_text)
from score import CURVE_COLUMNS, ERROR_COLUMNS, KNOBS, write_csv

# The simulated reader: what the model should find again.
TRUE_BIAS = 1.5                                                   # K
TRUE_BASELINE = {"flat": -0.6, "sloped": -0.4, "curved": 1.0}     # K, adds up to zero
TRUE_SCATTER = 0.6                                                # K
TRUE_NOISE_WIDTH = {"low": 0.6, "medium": 1.0, "high": 1.6}       # the scatter is multiplied by these
OUTLIERS = {5: 14.0, 30: -12.0, 61: 18.0}                         # row number: added error, K


def simulated_design():
    """108 curves, every combination of noise, baseline, event, convention and units once (as make_test.py)."""
    rows = []
    for noise, baseline, event, convention, units in itertools.product(
            ("low", "medium", "high"), ("flat", "sloped", "curved"),
            ("melting", "crystallization", "glass_transition"), ("exo_up", "exo_down"), ("C_mW", "K_Wg")):
        style = ("plain", "printout", "report")[len(rows) % 3]
        rows.append({"name": f"sim_{len(rows) + 1:03d}", "noise": noise, "baseline": baseline, "event": event,
                     "convention": convention, "units": units, "style": style})
    return rows


def simulated_truths(design):
    """A true event for every simulated curve: temperatures 80-280 °C, ΔH 20-200 J/g."""
    truths = {}
    for i, row in enumerate(design):
        temp = 80 + 200 * i / (len(design) - 1)
        if row["event"] == "glass_transition":
            truths[row["name"]] = {"type": "glass_transition", "direction": "endo", "onset_C": None,
                                   "peak_C": None, "enthalpy_J_per_g": None, "tg_mid_C": temp}
        else:
            truths[row["name"]] = {"type": row["event"], "direction": "endo" if row["event"] == "melting" else "exo",
                                   "onset_C": temp - 4, "peak_C": temp, "enthalpy_J_per_g": 20 + i * 1.5,
                                   "tg_mid_C": None}
    return truths


def onset_row(row, error, reader="sim-reader"):
    """One row of the errors table: an onset reading of one simulated curve."""
    return {"reader": reader, "curve": row["name"], **{k: row[k] for k in KNOBS},
            "quantity": "onset_C", "error": error, "abs_error": abs(error), "status": "ok"}


def simulated_onset_rows(design, reader="sim-reader", seed=7):
    """Onset readings of the peak curves: error = bias + baseline effect + scatter, plus 3 wild readings.

    The curved baseline moves the errors up by 1 K: with the bias, readings there are 2.5 K too high on
    average, so baseline = curved is clearly the worst condition (high noise only widens the scatter).
    """
    rng = np.random.default_rng(seed)
    peaks = [row for row in design if row["event"] != "glass_transition"]
    rows = []
    for i, row in enumerate(peaks):
        error = (TRUE_BIAS + TRUE_BASELINE[row["baseline"]]
                 + TRUE_SCATTER * TRUE_NOISE_WIDTH[row["noise"]] * rng.normal() + OUTLIERS.get(i, 0.0))
        rows.append(onset_row(row, error, reader))
    return rows


def true_range_90(design, seed=11):
    """5th and 95th percentiles of the simulated reader's errors on a new curve, wild readings included at the
    rate they had (3 of 72): about [-0.09, 3.50] K (without them about [0.04, 3.15] K)."""
    rng = np.random.default_rng(seed)
    peaks = [row for row in design if row["event"] != "glass_transition"]
    pick = rng.integers(0, len(peaks), 400_000)
    centre = np.array([TRUE_BIAS + TRUE_BASELINE[row["baseline"]] for row in peaks])
    width = np.array([TRUE_SCATTER * TRUE_NOISE_WIDTH[row["noise"]] for row in peaks])
    errors = centre[pick] + width[pick] * rng.normal(size=pick.size)
    wild = rng.random(pick.size) < len(OUTLIERS) / len(peaks)
    errors[wild] += rng.choice(list(OUTLIERS.values()), size=wild.sum())
    return np.percentile(errors, [5, 95])


def test_median_abs_error_known_answers():
    # One design row, one draw: the median of |error| for a Student-t error with this centre, width and nu.
    one = lambda centre, width, nu: float(median_abs_error(np.array([[centre]]), np.array([[width]]),
                                                           np.array([nu]))[0])
    # Centred normal error (nu very large): half of |error| is below 0.6745 x width.
    assert abs(one(0.0, 1.0, 1e6) - 0.67449) < 1e-4
    assert abs(one(0.0, 2.0, 1e6) - 2 * 0.67449) < 2e-4
    # Centred Cauchy error (nu = 1): the median of |error| is exactly the width.
    assert abs(one(0.0, 1.0, 1.0) - 1.0) < 1e-6
    # A very narrow error far from 0: |error| is about the size of the centre, on either side.
    assert abs(one(3.0, 1e-9, 5.0) - 3.0) < 1e-5 and abs(one(-3.0, 1e-9, 5.0) - 3.0) < 1e-5
    # Against a million random draws, for shifted, heavy-tailed errors.
    rng = np.random.default_rng(1)
    for centre, width, nu in [(1.5, 0.7, 3.0), (0.3, 1.0, 1.2), (-2.0, 2.0, 30.0)]:
        draws = np.abs(centre + width * rng.standard_t(nu, size=1_000_000))
        assert abs(one(centre, width, nu) - np.median(draws)) < 0.005 * width, (centre, width, nu)
    # Several rows: a new reading lands on any of them. Narrow errors at -1, 2 and 3: the middle size is 2.
    rows = median_abs_error(np.array([[-1.0, 2.0, 3.0]]), np.full((1, 3), 1e-9), np.array([5.0]))
    assert abs(rows[0] - 2.0) < 1e-5
    # Two draws at once, each with its own nu: a normal and a Cauchy error, both centred with width 1.
    both = median_abs_error(np.zeros((2, 4)), np.ones((2, 4)), np.array([1e6, 1.0]))
    assert abs(both[0] - 0.67449) < 1e-4 and abs(both[1] - 1.0) < 1e-5


def test_error_range_known_answers():
    rng = np.random.default_rng(2)
    # Every row centred on 0 with width 1 and a normal error: 90 % of errors within ±1.645.
    zeros = np.zeros((4000, 10))
    r = error_range(zeros, np.ones_like(zeros), np.full(4000, 1e6), rng)
    assert abs(r["low"] + 1.645) < 0.03 and abs(r["high"] - 1.645) < 0.03, r
    # No spread at all: the range is the 5th and 95th percentiles of the row centres (0, 0.01, ... 1).
    centres = np.tile(np.linspace(0, 1, 101), (4000, 1))
    r = error_range(centres, np.full_like(centres, 1e-9), np.full(4000, 5.0), rng)
    assert abs(r["low"] - 0.05) < 0.011 and abs(r["high"] - 0.95) < 0.011, r


def hand_made_posterior(draw_kinds):
    """A posterior made by hand: 9 design rows (3 baselines x 3 noise levels), errors with (almost) no spread.

    Each draw kind gives a baseline shift B and a noise shift N; the centre of a row is B[baseline] + N[noise].
    With no spread, a reading on a row is off by exactly the size of its centre, so the typical (median) |error|
    at a knob level is the middle one of the sizes of its 3 rows.
    """
    kinds = {"curved_worst": ({"flat": 0, "sloped": 0, "curved": 3}, {"low": 0, "medium": 0, "high": 2.7}),
             "high_worst": ({"flat": 0, "sloped": 0, "curved": 0}, {"low": 0, "medium": 0, "high": 30})}
    design = [{"baseline": b, "noise": n} for b in ("flat", "sloped", "curved") for n in ("low", "medium", "high")]
    location = np.array([[kinds[k][0][row["baseline"]] + kinds[k][1][row["noise"]] for row in design]
                         for k in draw_kinds], dtype=float)
    levels = {"noise": ["low", "medium", "high"], "baseline": ["flat", "sloped", "curved"]}
    return location, np.full_like(location, 1e-9), np.full(len(draw_kinds), 5.0), levels, design


def test_worst_condition_hand_made():
    # Row sizes: "curved_worst" draws: flat 0, 0, 2.7; sloped 0, 0, 2.7; curved 3, 3, 5.7 (low, medium, high).
    #   Typical |error|: curved 3 (the worst), high noise 2.7 (of 2.7, 2.7, 5.7), all 9 rows 2.7. Curved adds 0.3.
    # "high_worst" draws: every baseline 0, 0, 30. Typical |error|: high noise 30, curved 0, all 9 rows 0.
    # In 60 % of the draws curved is the worst level, in 40 % high noise. On average over the draws high noise
    # is larger (0.6 x 2.7 + 0.4 x 30 = 13.6 against 1.8), but the named level must be the most likely one.
    w = worst_condition(*hand_made_posterior(["curved_worst"] * 60 + ["high_worst"] * 40))
    assert (w["knob"], w["level"]) == ("baseline", "curved") and abs(w["prob_worst"] - 0.6) < 1e-12, w
    assert abs(w["all_levels"]["noise=high"]["prob_worst"] - 0.4) < 1e-12
    assert w["all_levels"]["noise=high"]["typical_abs_error"] > w["all_levels"]["baseline=curved"]["typical_abs_error"]
    # What curved adds: 0.3 in 60 % of draws and 0 in the rest: mean 0.18, 94 % interval [0, 0.3].
    assert abs(w["adds"]["mean"] - 0.18) < 1e-4 and abs(w["adds"]["low"]) < 1e-4 and abs(w["adds"]["high"] - 0.3) < 1e-4
    assert abs(w["typical_abs_error"]["mean"] - 0.6 * 3) < 1e-4
    assert abs(w["typical_abs_error_overall"]["mean"] - 0.6 * 2.7) < 1e-4
    assert w["n_levels"] == 6 and not w["clear"]  # what it adds may be 0: not a clear worst condition
    # One chain of 60 curved-worst draws, then 40 high-worst ones: so few changes that the sampling errors are
    # large, and the call is borderline, not none clear: a rerun could well give another answer.
    assert w["mcse_prob"] > 0.1 and w["mcse_adds_low"] > 0 and w["call"] == "borderline", w
    # Curved worst in every draw: named, adds 0.3 for sure, so it is clear. Nothing varies: no sampling error.
    w = worst_condition(*hand_made_posterior(["curved_worst"] * 50))
    assert (w["knob"], w["level"], w["prob_worst"], w["call"], w["clear"]) == ("baseline", "curved", 1.0, "clear",
                                                                               True), w
    assert abs(w["adds"]["mean"] - 0.3) < 1e-4 and w["mcse_prob"] == 0 and w["mcse_adds_low"] == 0
    # Half and half: no level is more likely than not to be the worst, and what it adds may be 0 in every
    # bootstrap resample too, so none is clear.
    w = worst_condition(*hand_made_posterior(["curved_worst", "high_worst"] * 50))
    assert w["prob_worst"] == 0.5 and w["call"] == "none clear" and not w["clear"], w
    # The draws come chains end to end: 2 chains of 50 here. The MCSEs see each chain apart, so 2 chains that
    # each switch once (curved worst, then high noise) give a larger sampling error than the same draws
    # alternating, and the same number of draws.
    kinds = (["curved_worst"] * 30 + ["high_worst"] * 20) * 2
    apart = worst_condition(*hand_made_posterior(kinds), chains=2)
    mixed = worst_condition(*hand_made_posterior(["curved_worst", "high_worst", "curved_worst", "curved_worst",
                                                  "high_worst"] * 20), chains=2)
    assert apart["prob_worst"] == mixed["prob_worst"] == 0.6 and apart["mcse_prob"] > 2 * mixed["mcse_prob"]


def test_worst_call_rule():
    # The rule: probability above 0.50 and the lower end of what it adds above 0, each with 2 MCSE to spare for
    # clear; within 2 MCSE, borderline; otherwise none clear.
    assert error_model.CLEAR_PROB == 0.5 and error_model.CALL_MARGIN == 2
    assert worst_call(0.70, 0.01, 0.05, 0.01) == "clear"
    assert worst_call(0.515, 0.01, 0.05, 0.01) == "borderline"     # 0.515 - 0.02 is not above 0.50
    assert worst_call(0.70, 0.01, 0.015, 0.01) == "borderline"     # lower end: 0.015 - 0.02 < 0 < 0.015 + 0.02
    assert worst_call(0.485, 0.01, -0.015, 0.01) == "borderline"   # both miss the rule, but only within 2 MCSE
    assert worst_call(0.475, 0.01, 0.05, 0.01) == "none clear"     # 0.475 + 0.02 is not above 0.50
    assert worst_call(0.70, 0.01, -0.025, 0.01) == "none clear"    # -0.025 + 0.02 is not above 0
    assert worst_call(0.99, 0.0, 0.0, 0.0) == "none clear"         # adds exactly 0 in every draw: nothing to name
    # The margin replaces the earlier rounding to 2 decimals: a probability that prints as 0.50 is not clear even
    # with the MCSE of 4000 independent draws (sqrt(0.25 / 4000) = 0.008), the smallest that is realistic.
    assert worst_call(np.float64(0.5049), 0.0079, 1.0, 0.0) == "borderline"
    assert worst_call(0.53, 0.0079, 1.0, 0.0) == "clear" and worst_call(0.53, 0.0079, 1.0, 0.0, margin=4) == "borderline"
    # An MCSE that could not be worked out counts as unbounded: at best borderline.
    assert worst_call(0.99, np.nan, 1.0, 0.0) == "borderline" and worst_call(0.99, 0.0, 1.0, np.nan) == "borderline"
    assert type(worst_call(np.float64(0.9), np.float64(0.01), np.float64(0.1), 0.0)) is str  # plain text for the JSON


def test_calls_from_hand_made_draws():
    # Indicator arrays ("this level is the worst" in each draw) and draws of what it adds, 4 chains x 1000, made by
    # hand, through the same MCSEs and rule as worst_condition.
    rng = np.random.default_rng(12)

    def share(p):
        """Exactly p x 4000 ones, in random order: independent draws."""
        x = np.zeros(4000)
        x[:round(p * x.size)] = 1
        rng.shuffle(x)
        return x.reshape(4, 1000)

    def adds_starting_at(low):
        """Draws of what the level adds (spread 0.1), shifted so that their 94 % HDI starts exactly at low."""
        x = rng.normal(0.3, 0.1, size=(4, 1000))
        return x + (low - hdi(x.ravel())[0])

    def call(indicator, adds):
        return worst_call(indicator.mean(), mcse_of_share(indicator), hdi(adds.ravel())[0], mcse_of_hdi_low(adds, rng))

    sure = adds_starting_at(0.1)    # the MCSE of its lower end is about 0.007, far below 0.1
    assert abs(hdi(sure.ravel())[0] - 0.1) < 1e-12
    assert call(share(0.70), sure) == "clear"
    assert call(share(0.51), sure) == "borderline"         # MCSE about 0.008: 0.51 is within 2 MCSE of 0.50
    assert call(share(0.45), sure) == "none clear"
    assert call(share(0.70), adds_starting_at(0.005)) == "borderline"
    assert call(share(0.70), adds_starting_at(-0.1)) == "none clear"
    # The same 2120 ones of 4000 (0.53): clear when the draws are independent, borderline when each chain has
    # its ones in one run (the chains then disagree, and are worth only a few independent draws).
    x = share(0.53)
    assert call(x, sure) == "clear"
    assert call(-np.sort(-x, axis=1), sure) == "borderline"


def test_mcse_of_share_known_answers():
    rng = np.random.default_rng(13)
    # Independent draws: the MCSE of a share p of N draws is sqrt(p (1 - p) / N).
    for p in (0.5, 0.2, 0.03):
        found = np.mean([mcse_of_share(rng.random((4, 1000)) < p) for _ in range(20)])
        assert abs(found / np.sqrt(p * (1 - p) / 4000) - 1) < 0.05, (p, found)
    # 400 independent draws, each repeated 10 times in a row: 4000 draws worth only 400, so about sqrt(10) = 3.2
    # times the MCSE of 4000 independent draws, and about the MCSE of the 400 draws themselves.
    x = rng.random((4, 100)) < 0.5
    repeated = mcse_of_share(np.repeat(x, 10, axis=1))
    assert 2.5 < repeated / np.sqrt(0.25 / 4000) < 4.0 and 0.8 < repeated / mcse_of_share(x) < 1.25, repeated
    # The same in every draw: 0, not nan.
    assert mcse_of_share(np.ones((4, 1000))) == 0.0 and mcse_of_share(np.zeros((2, 50))) == 0.0


def ar1_gamma(sets, phi, rng):
    """Draws from a skewed posterior (gamma, shape 3), 4 chains x 1000, `sets` independent sets of them. Each draw
    is phi x the one before plus new noise (on a normal scale, then mapped to the gamma): phi = 0 gives
    independent draws, phi = 0.9 about one effective draw in 19, as from a slowly mixing sampler."""
    from scipy import stats
    z = rng.normal(size=(sets, 4, 1000))
    for t in range(1, 1000):
        z[:, :, t] = phi * z[:, :, t - 1] + np.sqrt(1 - phi ** 2) * z[:, :, t]
    return stats.gamma.ppf(stats.norm.cdf(z), 3)


def test_mcse_of_hdi_low():
    rng = np.random.default_rng(14)
    assert mcse_of_hdi_low(np.full((4, 100), 0.3), rng) == 0.0  # nothing varies: no sampling error
    # A chain no longer than a block (2 x 50 draws) is still cut into blocks: the resamples differ, so the MCSE is
    # not 0 (with one block of 50 per chain, every resample was the chain itself and the MCSE came out 2e-16).
    assert mcse_of_hdi_low(rng.normal(size=(2, 50)), rng) > 0.05
    # Against the truth: the standard deviation of the HDI's lower end over 300 independent sets of draws.
    found = {}
    for phi in (0.0, 0.9):
        sets = ar1_gamma(300, phi, rng)
        lows = [hdi(s.ravel())[0] for s in sets]
        true = np.std(lows, ddof=1)
        boot = np.mean([mcse_of_hdi_low(s, rng) for s in sets[:20]])
        assert 0.75 < boot / true < 1.33, (phi, boot, true)
        # ArviZ's quantile MCSE at the HDI's tail probability is too small (by about 1.4 to 2 times): it treats the
        # lower end as a fixed quantile, but the HDI also chooses the quantile, and that choice moves too.
        quantile = np.mean([float(az.mcse(s, method="quantile", prob=np.mean(s <= low)))
                            for s, low in zip(sets[:20], lows)])
        assert quantile < 0.8 * true, (phi, quantile, true)
        found[phi] = boot
    assert found[0.9] > 1.2 * found[0.0]  # correlated draws: a larger sampling error from the same number of draws


def test_prob_text():
    # A probability that would print as 1.00 is printed "> 0.99", one that would print as 0.00 "< 0.01"; the rest
    # with 2 decimals, as before.
    assert prob_text(1.0) == "> 0.99" and prob_text(np.float64(0.9975)) == "> 0.99" and prob_text(0.996) == "> 0.99"
    assert prob_text(0.994) == "0.99" and prob_text(0.50325) == "0.50" and prob_text(0.71) == "0.71"
    assert prob_text(0.0) == "< 0.01" and prob_text(0.004) == "< 0.01" and prob_text(0.006) == "0.01"
    # With 3 decimals (the numbers a clear or borderline call was judged with): the same, one decimal further.
    assert prob_text(0.9996, 3) == "> 0.999" and prob_text(0.9994, 3) == "0.999" and prob_text(0.4848, 3) == "0.485"
    assert prob_text(0.0004, 3) == "< 0.001" and prob_text(-0.0, 3) == "< 0.001"
    # In a worst-condition line: the hand-made posterior where curved is the worst level in every draw.
    w = worst_condition(*hand_made_posterior(["curved_worst"] * 50))
    seen = {"median_abs_error_at_worst": 3.0, "median_abs_error": 2.7}
    line = error_model.worst_text(w, 1, " K", seen)
    assert "probability > 0.999 that it is the worst" in line and "1.00" not in line, line
    assert line.startswith("- **Worst condition**: baseline = curved. Typical |error| 3.0 K there"), line
    assert ("sampling error below 0.001; lower end of what it adds 0.300 K, sampling error below 0.001 K)"
            in line), line


def test_worst_text_three_calls():
    w = {"knob": "noise", "level": "high", "prob_worst": 0.512, "mcse_prob": 0.0094, "n_levels": 15,
         "adds": {"mean": 0.171, "low": 0.004, "high": 0.352}, "mcse_adds_low": 0.0121,
         "typical_abs_error": {"mean": 0.9}, "typical_abs_error_overall": {"mean": 0.73}}
    seen = {"median_abs_error_at_worst": 0.95, "median_abs_error": 0.7}
    line = error_model.worst_text({**w, "call": "borderline", "clear": False}, 1, " K", seen)
    assert line == ("- **Worst condition**: borderline. noise = high has probability 0.512 (sampling error ± 0.009) "
                    "of being the worst of the 15 knob levels and adds +0.17 K [0.00, 0.35] to the typical |error| "
                    "(lower end 0.004 K, sampling error ± 0.012 K); a rerun can change this call"), line
    line = error_model.worst_text({**w, "call": "none clear", "clear": False}, 1, " K", seen)
    assert line.startswith("- **Worst condition**: none clear. Most likely noise = high (probability 0.51"), line
    clear = {**w, "call": "clear", "clear": True, "prob_worst": 0.81, "adds": {**w["adds"], "low": 0.054}}
    line = error_model.worst_text(clear, 1, " K", seen)
    assert line.startswith("- **Worst condition**: noise = high. Typical |error| 0.9 K there against 0.7 K"), line
    assert "probability 0.810 that it is the worst of the 15 knob levels, sampling error ± 0.009; lower end of " \
           "what it adds 0.054 K, sampling error ± 0.012 K)" in line, line


def printed_call(line):
    """The call worked out from the numbers a worst-condition line prints, as a reader would: the probability,
    its sampling error, the lower end of what the level adds and its sampling error ("below x" taken as x)."""
    number = r"[<>]? ?(−?[0-9.]+)"
    found = re.search(rf"probability {number}.*?sampling error (?:± |below )([0-9.]+)\)?;? .*?lower end "
                      rf"(?:of what it adds )?{number} K, sampling error (?:± |below )([0-9.]+) K", line)
    p, mp, low, ml = (float(x.replace("−", "-")) for x in found.groups())
    return worst_call(p, mp, low, ml)


def test_printed_numbers_give_the_call():
    seen = {"median_abs_error_at_worst": 0.95, "median_abs_error": 0.7}

    def line_for(p, mp, low, ml, d=1):
        w = {"knob": "baseline", "level": "flat", "prob_worst": p, "mcse_prob": mp, "n_levels": 15,
             "adds": {"mean": low + 0.15, "low": low, "high": low + 0.3}, "mcse_adds_low": ml,
             "typical_abs_error": {"mean": 0.9}, "typical_abs_error_overall": {"mean": 0.73}}
        w["call"] = worst_call(p, mp, low, ml)
        return w["call"], worst_text(w, d, " K", seen)

    # claude-sonnet-5-5 onset_C, seed 2026: borderline, as 0.48475 + 2 x 0.00913 = 0.503 is above 0.50. Printed
    # with 2 decimals (0.48 and 0.01) it read as none clear: 0.48 + 2 x 0.01 = 0.50 is not above 0.50.
    call, line = line_for(0.48475, 0.00913381, 0.0206325, 0.00704944)
    assert call == "borderline" and worst_call(0.48, 0.01, 0.02, 0.01) == "none clear"
    assert "probability 0.485 (sampling error ± 0.009)" in line and "(lower end 0.021 K, sampling error ± 0.007 K)" \
        in line and printed_call(line) == "borderline", line
    # code-claude-opus-5-5-v2 onset_C, seed 2027: clear, as 0.0386 - 2 x 0.0174 = 0.004 is above 0. Printed with
    # 2 decimals (0.04 and 0.02) it read as borderline: 0.04 - 2 x 0.02 = 0 is not above 0.
    call, line = line_for(0.5925, 0.00853551, 0.0385739, 0.0173841)
    assert call == "clear" and printed_call(line) == "clear" and "0.039 K, sampling error ± 0.017 K" in line, line
    # Where 3 decimals are not enough either, the line gets more: 0.5181 - 2 x 0.0090 = 0.5001 is clear, but
    # 0.518 - 2 x 0.009 = 0.500 is not.
    call, line = line_for(0.5181, 0.0090, 0.3, 0.01)
    assert call == "clear" and "probability 0.5181 that" in line and printed_call(line) == "clear", line
    # Many numbers close to the line: the printed numbers always give the call.
    rng = np.random.default_rng(15)
    calls = set()
    for _ in range(2000):
        mp, ml = rng.uniform(0.002, 0.02), rng.uniform(0.0005, 0.05)
        p = 0.5 + rng.choice([-2, 2]) * mp + rng.normal(0, 0.003)
        low = rng.choice([-2, 2]) * ml + rng.normal(0, 0.003)
        call, line = line_for(p, mp, low, ml, d=int(rng.integers(1, 3)))
        if call != "none clear":
            calls.add(call)
            assert printed_call(line) == call, (p, mp, low, ml, line)
    assert calls == {"clear", "borderline"}


def test_needs_longer_run():
    check = {"gate3_pass": False, "divergences": 0, "stuck": False}
    assert needs_longer_run(check)                                # missed Gate 3 on R-hat or ESS: run longer
    assert not needs_longer_run({**check, "gate3_pass": True})    # passed: nothing to do
    assert not needs_longer_run({**check, "divergences": 2})      # divergences: more draws do not cure them
    assert not needs_longer_run({**check, "stuck": True})         # a stuck chain gets stuck again with the same seed


def test_beta_summary():
    # 0 of 72: Beta(1, 73). Mean 1/74; the narrowest 94 % interval starts at 0 and ends where 94 % is reached.
    b = beta_summary(0, 72)
    assert abs(b["mean"] - 1 / 74) < 1e-12 and b["low"] < 1e-6
    assert abs(b["high"] - (1 - 0.06 ** (1 / 73))) < 1e-5  # Beta(1, n) cumulative: 1 - (1 - x)^n
    # 72 of 72 is the mirror image.
    b = beta_summary(72, 72)
    assert abs(b["mean"] - 73 / 74) < 1e-12 and b["high"] > 1 - 1e-6 and abs(b["low"] - 0.06 ** (1 / 73)) < 1e-5
    # 36 of 72: a symmetric curve, so the interval is centred on one half.
    b = beta_summary(36, 72)
    assert abs(b["mean"] - 0.5) < 1e-12 and abs((b["low"] + b["high"]) - 1) < 1e-6
    assert b["k"] == 36 and b["n"] == 72
    assert beta_summary(0, 0)["mean"] is None
    # The text gives the measured share first: 108 of 108 is 100 %, not the Beta mean of 99.1 %.
    assert error_model.share_text(beta_summary(108, 108)).startswith("100 %; true share above 97.5 %")
    assert error_model.share_text(beta_summary(0, 72)).startswith("0 %; true share below 3.8 %")
    assert error_model.share_text(beta_summary(100, 108)).startswith("92.6 %; 94 % interval")


def test_knob_levels_left_out_when_single_or_not_for_tg():
    rows = [{"noise": n, "baseline": "flat", "event": e, "convention": "exo_up", "units": u, "style": "plain"}
            for n, e, u in itertools.product(("high", "low"), ("melting", "crystallization"), ("C_mW", "K_Wg"))]
    levels = knob_levels(rows, "peak_C")
    # baseline, convention and style have one level here: left out. Levels come in the fixed order.
    assert levels == {"event": ["melting", "crystallization"], "noise": ["low", "high"], "units": ["C_mW", "K_Wg"]}
    # Tg never uses the event knob (every Tg curve is a glass transition).
    assert "event" not in knob_levels(rows, "tg_mid_C")


def test_gross_and_missed():
    base = {"reader": "r", "curve": "c"}
    rows = [{**base, "status": "ok", "error": "10.0"},      # exactly 10 K: not gross
            {**base, "status": "ok", "error": "-10.5"},     # gross
            {**base, "status": "ok", "error": "0.2"},
            {**base, "status": "missed", "error": ""},
            {**base, "status": "no_answer", "error": ""},
            {**base, "status": "extra_value", "error": ""}]  # no true value: in neither count
    gross, missed = gross_and_missed(rows, "onset_C")
    assert (gross["k"], gross["n"], gross["limit"]) == (1, 3, 10)
    assert abs(gross["mean"] - 2 / 5) < 1e-12
    assert missed == {"k": 2, "n": 5}
    # ΔH: gross beyond 50 % of the true value, from the rel_error_pct column.
    dh = [{**base, "status": "ok", "error": "99", "rel_error_pct": "49.9"},
          {**base, "status": "ok", "error": "1", "rel_error_pct": "-50.1"}]
    gross, missed = gross_and_missed(dh, "enthalpy_J_per_g")
    assert (gross["k"], gross["n"], gross["limit"]) == (1, 2, 50) and missed == {"k": 0, "n": 2}


def test_reader_checks_direction_breakdown():
    def curve(convention, event, type_ok, direction_ok, reported_type, reported_direction):
        return {"reader": "r", "convention": convention, "event": event, "type_ok": type_ok,
                "direction_ok": direction_ok, "reported_type": reported_type, "reported_direction": reported_direction}
    rows = [curve("exo_down", "crystallization", "1", "0", "crystallization", "endo"),
            curve("exo_down", "crystallization", "1", "0", "crystallization", "endo"),
            curve("exo_down", "melting", "1", "1", "melting", "endo"),
            curve("exo_up", "crystallization", "0", "1", "melting", "exo"),
            curve("exo_up", "melting", "", "", "", ""),                        # no event reported: not counted
            curve("exo_up", "melting", "0", "0", "", ""),                      # type and direction null
            curve("exo_down", "glass_transition", "1", "0", "glass_transition", "")]  # null: not wrong
    checks = reader_checks(rows)
    assert checks["curves"] == 7 and checks["with_event"] == 6 and checks["no_type_no_direction"] == 1
    assert (checks["type_right"]["k"], checks["type_right"]["n"]) == (4, 5)
    assert (checks["direction_right"]["k"], checks["direction_right"]["n"]) == (2, 4)
    cells = checks["direction_wrong_by_convention_and_event"]
    assert list(cells) == ["exo_up", "exo_down"]
    assert cells["exo_down"]["crystallization"] == {"k": 2, "n": 2}
    assert cells["exo_down"]["melting"] == {"k": 0, "n": 1}
    assert cells["exo_up"]["melting"] == {"k": 0, "n": 0}               # blank directions are not wrong ones
    assert cells["exo_down"]["glass_transition"] == {"k": 0, "n": 0}
    # The null replies by true event, out of the replies with an event (the one with no event is left out).
    assert checks["no_type_no_direction_by_event"] == {
        "melting": {"k": 1, "n": 2}, "crystallization": {"k": 0, "n": 3}, "glass_transition": {"k": 0, "n": 1}}
    md = "\n".join(error_model.reader_lines("r", {"checks": checks, "quantities": {}}))
    assert ("1 of the 6 replies with an event gave neither a type nor a direction (null): counted neither right nor "
            "wrong. By true event, null on 1 of the 2 melting, 0 of the 3 crystallization and 0 of the 1 glass "
            "transition curves.") in md, md


def test_missing_numpy_calls_and_reader_note():
    # A code reader's script that calls a NumPy function this NumPy does not have is named in its note; a call
    # that is only a fallback for another NumPy (on a line that asks hasattr(np, ...)) is not.
    absent = "not_in_any_numpy"
    v1 = f"import numpy as np\narea = float(np.trapz(y, x))\nb = np.{absent} (y)\npeak = np.argmax(y)\n"
    v2 = (f"import numpy as np\n"
          f"area = float(np.trapezoid(y, x)) if hasattr(np, \"trapezoid\") else float(np.{absent}(y, x))\n")
    saved = error_model.READERS
    with tempfile.TemporaryDirectory() as tmp:
        error_model.READERS = Path(tmp)
        try:
            assert error_model.script_of("code-some-model-1") == Path(tmp) / "code_some_model_1.py"
            assert error_model.script_of("code-some-model-1-v2") == Path(tmp) / "code_some_model_1_v2.py"
            assert error_model.script_of("standard") is None and error_model.missing_numpy_calls(None) == []
            (Path(tmp) / "code_some_model_1.py").write_text(v1, encoding="utf-8")
            (Path(tmp) / "code_some_model_1_v2.py").write_text(v2, encoding="utf-8")
            expected = sorted(name for name in ("trapz", absent) if not hasattr(np, name))
            assert error_model.missing_numpy_calls(error_model.script_of("code-some-model-1")) == expected
            assert error_model.missing_numpy_calls(error_model.script_of("code-some-model-1-v2")) == []
            note = error_model.reader_note("code-some-model-1")
            assert f"The script calls {' and '.join('np.' + n for n in expected)}, which NumPy" in note, note
            assert "the likely reason for any null reply" in note
            note = error_model.reader_note("code-some-model-1-v2")
            assert "does not have" not in note and note.endswith("It reads the CSV numbers, not the image."), note
            assert "does not have" not in error_model.reader_note("code-no-script-here")
            # With the .usage.json files that write_code_reader.py saves: prompt v2 named the versions (made-up
            # ones here, so the note must read them, not know them) and was asked after the v1 script, which
            # fails here. The run files say what the script ran on.
            usage = lambda version, when, **more: json.dumps({"prompt_version": f"code-{version}",
                                                              "timestamp_utc": when, **more})
            (Path(tmp) / "code_some_model_1.usage.json").write_text(usage("v1", "2026-01-01T10:00:00+00:00"),
                                                                    encoding="utf-8")
            named = {"python": "3.99", "numpy": "9.8", "scipy": "7.6"}
            (Path(tmp) / "code_some_model_1_v2.usage.json").write_text(
                usage("v2", "2026-01-01T11:00:00+00:00", versions_in_prompt=named), encoding="utf-8")
            replies = Path(tmp) / "replies"
            replies.mkdir()
            for i in range(3):
                (replies / f"test_{i:03d}.run.json").write_text(json.dumps(
                    {"stop_reason": "exit_0", "python": "3.99.1", "numpy": "9.8.7", "scipy": "7.6.5"}), encoding="utf-8")
            note = error_model.reader_note("code-some-model-1-v2", replies)
            assert note.startswith("One Python script, written once by some-model-1 from prompt v2"), note
            assert ("It reads the CSV numbers, not the image. Prompt v2 was asked after the v1 script failed: it adds "
                    "only the line naming Python 3.99, NumPy 9.8 and SciPy 7.6, and does not mention the failure. "
                    "The script ran on Python 3.99.1, NumPy 9.8.7 and SciPy 7.6.5, as recorded in its run files.") in note, note
            assert "does not have" not in note
            # No run files given: nothing said about them. The v1 note says nothing about versions.
            assert "run files" not in error_model.reader_note("code-some-model-1-v2")
            assert "Prompt v" not in error_model.reader_note("code-some-model-1", replies)
            # If the v1 script did not fail, or was asked for after v2, the note does not say v2 followed a failure.
            (Path(tmp) / "code_some_model_1.usage.json").write_text(usage("v1", "2026-01-01T12:00:00+00:00"),
                                                                    encoding="utf-8")
            note = error_model.reader_note("code-some-model-1-v2")
            assert "failed" not in note and note.endswith(
                "Prompt v2 adds only the line naming Python 3.99, NumPy 9.8 and SciPy 7.6."), note
        finally:
            error_model.READERS = saved


def test_parse_args():
    assert parse_args([]) == (False, error_model.DEFAULT_SEED)
    assert parse_args(["test", "--quick", "--seed", "11"]) == (True, 11)
    for bad in (["dev"], ["--seed"], ["--seed", "x"], ["--fast"]):
        try:
            parse_args(bad)
        except SystemExit as stop:
            assert str(stop) == error_model.USAGE
        else:
            raise AssertionError(f"{bad} was accepted")


def test_parameter_recovery():
    # Simulated onset errors with a known bias, a known baseline shift, noise-dependent scatter and 3 wild
    # readings. Quick-sized sampling is enough to find them again.
    design = simulated_design()
    rows = simulated_onset_rows(design)
    rep = fit_report("sim-reader", "onset_C", rows, design, simulated_truths(design), QUICK, seed=3)
    assert rep["status"] == "fitted" and rep["n_readings"] == 72
    for field in ("bias", "range_90", "observed", "gross", "missed", "worst_condition", "tested_range", "diagnostics",
                  "knob_effects", "width_factor", "scatter", "nu"):
        assert field in rep, field
    assert abs(rep["bias"]["mean"] - TRUE_BIAS) < 0.3, rep["bias"]
    # The curved baseline adds 1 K: found within 0.4 K (24 readings per level, scatter about 0.6 K, and the
    # model pulls small effects a little towards 0). The outliers make the tails heavy (small nu).
    baseline = {level: effect["mean"] for level, effect in rep["knob_effects"]["baseline"].items()}
    assert abs(baseline["curved"] - TRUE_BASELINE["curved"]) < 0.4, baseline
    assert baseline["flat"] < 0 < baseline["curved"] and max(baseline, key=baseline.get) == "curved", baseline
    noise_width = rep["width_factor"]["noise"]
    assert noise_width["high"]["mean"] > noise_width["low"]["mean"]
    assert rep["nu"]["mean"] < 10
    # The worst condition comes from a shift in the centre, despite 3 gross misreads: curved, and clearly so.
    worst = rep["worst_condition"]
    assert (worst["knob"], worst["level"], worst["call"], worst["clear"]) == ("baseline", "curved", "clear", True), worst
    assert worst["prob_worst"] > 0.9 and worst["adds"]["low"] > 2 * worst["mcse_adds_low"] > 0
    assert 0 <= worst["mcse_prob"] < 0.05  # 0 when curved is the worst level in every draw
    # 3 wild readings beyond 10 K out of 72; none missed.
    assert (rep["gross"]["k"], rep["gross"]["n"]) == (3, 72) and rep["missed"] == {"k": 0, "n": 72}
    # 90 % range: close to the true 5th and 95th percentiles of the reader (about [-0.09, 3.50] K). Within 0.7 K:
    # the heavy tails the model needs for the 3 wild readings widen the range a little (about [-0.6, 3.6] K).
    low, high = true_range_90(design)
    assert abs(rep["range_90"]["low"] - low) < 0.7 and abs(rep["range_90"]["high"] - high) < 0.7, rep["range_90"]
    tested = rep["tested_range"]
    assert tested["knob_levels"]["event"] == ["melting", "crystallization"]
    assert tested["event_temperature_C"][0] >= 80 and tested["event_temperature_C"][1] <= 280
    assert "says nothing" in tested["sentence"] and "synthetic" in tested["sentence"]
    for key in ("max_rhat", "min_ess_bulk", "divergences", "gate3_pass", "stuck", "seconds"):
        assert key in rep["diagnostics"], key


def test_worst_condition_from_width_alone():
    # A curved baseline triples the scatter but moves nothing: the readings there are much further off,
    # so baseline = curved is the worst condition, found through the width terms alone.
    design = simulated_design()
    rng = np.random.default_rng(21)
    peaks = [row for row in design if row["event"] != "glass_transition"]
    rows = [onset_row(row, 0.2 + 0.3 * (3 if row["baseline"] == "curved" else 1) * rng.normal()) for row in peaks]
    rep = fit_report("sim-wide", "onset_C", rows, design, simulated_truths(design), QUICK, seed=4)
    worst = rep["worst_condition"]
    assert (worst["knob"], worst["level"], worst["call"], worst["clear"]) == ("baseline", "curved", "clear", True), worst
    assert rep["width_factor"]["baseline"]["curved"]["low"] > 1  # the scatter there is surely wider
    assert all(abs(effect["mean"]) < 0.2 for effect in rep["knob_effects"]["baseline"].values())


def test_many_exact_zeros_cannot_squeeze_the_width():
    # A code-like reader whose reading is exactly right on 20 of 36 curves and a little off on the rest.
    # Without the width floor, the model's density keeps growing as scatter goes to 0 (each exact zero adds
    # about log(1 / scatter), more than the other readings lose when nu is small), and the sampler gets stuck
    # there. With the floor it levels off.
    design = simulated_design()
    rng = np.random.default_rng(5)
    glass = [row for row in design if row["event"] == "glass_transition"]
    errors = [0.0 if i % 9 < 5 else round(0.05 * rng.standard_t(2), 2) or 0.01 for i in range(len(glass))]
    assert sum(e == 0 for e in errors) == 20
    model = error_model.build_model(errors, glass, knob_levels(glass, "tg_mid_C"), "K")
    density = model.compile_logp(jacobian=False)  # the model density itself, in the units of scatter and nu
    point = model.initial_point()
    at = lambda scatter, nu: float(density({**point, "scatter_log__": np.log(scatter), "nu_log__": np.log(nu)}))
    # From scatter 1e-6 K down to 1e-12 K, with very heavy tails (nu 0.2): at most a tiny change. Without the
    # floor the density would rise by 232 (log units) over the same step, and keep rising.
    assert abs(at(1e-12, 0.2) - at(1e-6, 0.2)) < 0.01
    # The same for any tail weight: no run-away as the width shrinks.
    for nu in (0.05, 0.5, 1.0, 5.0):
        assert abs(at(1e-14, nu) - at(1e-7, nu)) < 0.01, nu


def test_longer_run_when_gate3_missed():
    # 2 chains x 150 draws can never reach a bulk ESS of 400, so the fit is sampled again with the longer
    # setting, and the error report keeps what the first run gave.
    design = simulated_design()
    settings = {"draws": 150, "tune": 300, "chains": 2, "longer_draws": (200,)}
    rep = fit_report("sim-reader", "onset_C", simulated_onset_rows(design), design, simulated_truths(design),
                   settings, seed=3)
    d = rep["diagnostics"]
    assert d["draws"] == 200 and d["first_try"]["draws"] == 150 and not d["first_try"]["gate3_pass"], d
    assert "first run (2 × 150 draws)" in "\n".join(error_model.report_lines(rep))
    # Without a longer setting (the quick run), nothing is run again.
    rep = fit_report("sim-reader", "onset_C", simulated_onset_rows(design), design, simulated_truths(design),
                   {"draws": 150, "tune": 300, "chains": 2}, seed=3)
    assert rep["diagnostics"]["draws"] == 150 and rep["diagnostics"]["first_try"] is None


def test_main_quick_writes_outputs():
    # Two simulated readers through main(quick=True): one with onset readings to fit, one with too few.
    design = simulated_design()
    truths = simulated_truths(design)
    with tempfile.TemporaryDirectory() as tmp:
        data, results = Path(tmp) / "data", Path(tmp) / "results"
        (data / "test").mkdir(parents=True)
        with open(data / "test" / "design.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(design[0]))
            writer.writeheader()
            writer.writerows(design)
        for name, truth in truths.items():
            (data / "test" / f"{name}.truth.json").write_text(json.dumps({"events": [truth]}), encoding="utf-8")
        errors = simulated_onset_rows(design, reader="good")
        few = simulated_onset_rows(design, reader="sparse")
        for row in few[5:]:
            row.update(status="missed", error=None, abs_error=None)
        # "good" also gave an onset for 3 glass transitions, which have none: score.py marks them extra_value.
        glass = [row for row in design if row["event"] == "glass_transition"]
        errors += [{**onset_row(row, 0.0, "good"), "error": None, "abs_error": None, "reading": 150.0,
                    "status": "extra_value"} for row in glass[:3]]
        curves = []
        for reader in ("good", "sparse"):
            for row in design:
                right = not (reader == "sparse" and row["event"] == "melting")  # sparse reads every melting as exo
                curves.append({"reader": reader, "curve": row["name"], **{k: row[k] for k in KNOBS}, "n_events": 1,
                               "type_ok": 1, "direction_ok": int(right), "reported_type": row["event"],
                               "reported_direction": truths[row["name"]]["direction"] if right else "exo"})
        write_csv(results / "test_errors.csv", ERROR_COLUMNS, errors + few)
        write_csv(results / "test_curves.csv", CURVE_COLUMNS, curves)
        report = error_model.main(quick=True, seed=5, data_dir=data, results_dir=results)

        written = sorted(p.name for p in results.iterdir())
        assert written == ["quick_error_reports.json", "quick_error_reports.md", "quick_model_diagnostics.csv",
                           "test_curves.csv", "test_errors.csv"], written  # real error reports untouched
        saved = json.loads((results / "quick_error_reports.json").read_text(encoding="utf-8"))
        assert saved["settings"]["quick"] is True and saved["settings"]["chains"] == 2
        assert saved["settings"]["clear_prob"] == 0.5 and saved["settings"]["call_margin_mcse"] == 2
        assert saved["settings"]["mcse_adds_low_bootstrap"] == {"block_draws": 50, "resamples": 400}
        assert "clear_prob_rounded_to" not in saved["settings"]
        good = saved["readers"]["good"]["quantities"]["onset_C"]
        w = good["worst_condition"]
        assert w["call"] in ("clear", "borderline", "none clear") and w["clear"] == (w["call"] == "clear"), w
        assert w["mcse_prob"] >= 0 and w["mcse_adds_low"] >= 0
        assert good["status"] == "fitted" and abs(good["bias"]["mean"] - TRUE_BIAS) < 0.3
        sparse = saved["readers"]["sparse"]
        assert sparse["quantities"]["onset_C"]["status"].startswith("no error report")
        assert sparse["quantities"]["onset_C"]["missed"] == {"k": 67, "n": 72}
        assert sparse["checks"]["direction_wrong_by_convention_and_event"]["exo_up"]["melting"] == {"k": 18, "n": 18}
        assert saved["gate3"]["fits"] == 1 and report["gate3"]["fits"] == 1
        md = (results / "quick_error_reports.md").read_text(encoding="utf-8")
        assert "not for reporting" in md and "## good" in md and "## sparse" in md
        for words in ("**Bias**", "**90 % of readings**", "**Gross misreads**", "**Worst condition**",
                      "**Tested range**", "*Diagnostics*", "says nothing about real instrument curves",
                      "94 % credible interval", "Typical |error|", "probability above 0.50",
                      "both with 2 sampling errors to spare", "\"borderline\" and a rerun can change it",
                      "Monte Carlo standard error", "neither the median nor the mean", "# Error reports",
                      "How to read an error report:", "**Bias** (centre of the errors)",
                      "This error report says nothing about real instrument curves"):
            assert words in md, words
        assert "as printed (to 2 decimals" not in md  # the margin replaced the rounding
        assert "close to the median" not in md  # the bias is the model's centre, not the median of the readings
        assert "typical error" not in md.lower() and "valid range" not in md.lower()
        assert "calibration" not in md.lower() and saved["about"].startswith("Error reports from the Bayesian")
        # The onsets given for glass transitions: counted, not scored, and said once for the reader that gave them.
        assert saved["readers"]["good"]["onset_on_glass_transitions"] == {"k": 3, "n": 36}
        assert saved["readers"]["sparse"]["onset_on_glass_transitions"] == {"k": 0, "n": 36}
        line = ("Also gave an onset on 3 of the 36 glass transitions (not scored: a glass transition has no onset in "
                "this test).")
        assert md.count(line) == 1 and md.index(line) < md.index("## sparse"), md
        assert good["n_readings"] == 72 and good["missed"] == {"k": 0, "n": 72}  # the extra values are in neither
        with open(results / "quick_model_diagnostics.csv", newline="", encoding="utf-8") as f:
            diag = list(csv.DictReader(f))
        assert len(diag) == 1 and diag[0]["reader"] == "good" and diag[0]["chains"] == "2"


if __name__ == "__main__":
    tests = [test_median_abs_error_known_answers, test_error_range_known_answers, test_worst_condition_hand_made,
             test_worst_call_rule, test_calls_from_hand_made_draws, test_mcse_of_share_known_answers,
             test_mcse_of_hdi_low, test_prob_text, test_worst_text_three_calls, test_printed_numbers_give_the_call,
             test_needs_longer_run,
             test_beta_summary,
             test_knob_levels_left_out_when_single_or_not_for_tg,
             test_gross_and_missed, test_reader_checks_direction_breakdown, test_missing_numpy_calls_and_reader_note,
             test_parse_args,
             test_parameter_recovery, test_worst_condition_from_width_alone, test_many_exact_zeros_cannot_squeeze_the_width,
             test_longer_run_when_gate3_missed, test_main_quick_writes_outputs]
    for f in tests:
        f()
        print("PASS", f.__name__)
    print(f"All {len(tests)} checks passed.")
