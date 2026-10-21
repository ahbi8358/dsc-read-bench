"""Bayesian error model: turns each reader's scored errors into an error report.

Usage: python error_model.py [test] [--quick] [--seed N]      (run from the repo folder, after score.py test)

  --quick   2 chains x 200 draws instead of 4 x 1000, to check that the script runs. Its numbers are not
            for reporting, and it writes quick_error_reports.* so the real error reports are never overwritten.
  --seed N  base random seed (default 2026). Rerunning with the same seed on the same computer gives the same
            numbers, except that the very first run, while PyTensor compiles the models, can give a slightly
            different set of draws; another computer, or another seed, gives other draws too. The differences are
            of the size of the sampling noise, well inside the intervals, but can move a close worst-condition
            call: that is why such a call is reported as borderline (see "worst condition" below).

Input:  results/test_errors.csv and results/test_curves.csv (written by score.py), data/test/design.csv and
        data/test/*.truth.json (the frozen test set), and readers/code_*.py (the code readers' scripts, only to
        say if one calls a NumPy function that this NumPy does not have), with their .usage.json files and the
        data/test_replies/code-*/*.run.json files (only to say which prompt each code reader was asked and which
        Python, NumPy and SciPy versions its script ran on).
Output: results/error_reports.json (every number), results/error_reports.md (one readable block per reader),
        results/model_diagnostics.csv (one row per fit). A short table and the Gate 3 line are printed at the end.

The model. One separate fit for every reader and quantity with at least 10 values read:
  onset_C, peak_C and tg_mid_C (error in K) and enthalpy_J_per_g (error in % of the true ΔH).
  For reading i:   error_i ~ StudentT(nu, centre_i, width_i)
                   centre_i = mu + sum over knobs of b_knob[level of reading i]
                   width_i  = sqrt((scatter * exp(sum over knobs of g_knob[level of reading i]))^2 + r^2)
  - mu is the bias: the centre of the errors, that is where the error distribution is centred, averaged over the
    balanced test design (the knob effects b add up to zero). It is the model's own estimate, neither the median nor
    the mean of the readings, and it can differ from both (the error report prints both next to it): a Student-t
    gives wild readings less weight than a plain average does, and the model counts readings more where they
    scatter less. Gross misreads are also counted separately.
  - b_knob[level] is how much one knob level moves the errors up or down (knobs: noise, baseline, plot
    convention, units, plot style, and for peaks also melting vs crystallization).
  - scatter is the spread of the errors before the knob effects, and exp(g_knob[level]) widens or narrows it for
    one knob level. So a knob level can make the readings scatter more without moving them up or down.
  - r = 0.01 / sqrt(12) (K, or % for ΔH) is a floor under the width: readings carry at most 2 decimals, so an
    error much smaller than 0.01 cannot be measured. Without it, many readings with an error of exactly 0 would
    make the model's density grow without limit as the width shrinks towards 0, and the sampler gets stuck there.
  - A Student-t curve looks like a bell curve but has heavier tails: nu (degrees of freedom) sets how often
    a wild reading turns up. Small nu means many; large nu means almost a normal (Gaussian) bell curve.
  Priors (what is assumed before seeing the data), with P = 5 K for temperatures and 20 % for ΔH:
    mu ~ Normal(0, P), scatter ~ HalfNormal(P), nu ~ Gamma(2, 0.1),
    b_knob ~ ZeroSumNormal(sigma=sd_knob) with sd_knob ~ HalfNormal(P/2), g_knob ~ ZeroSumNormal(0.5).
  (ZeroSumNormal: the effects of a knob's levels add up to zero. The b effects are sampled in a partly
  non-centered form that keeps the sampler free of divergences; see build_model.)
  A knob with only one level among the readings is left out. PyMC's NUTS sampler draws thousands of plausible
  sets of these numbers (the posterior); every number in the error reports is computed from those draws.

What an error report says, for one reader and one quantity:
  bias            posterior mean of mu (the centre of the errors) with its 94 % credible interval, and the observed
                  median and mean error of the readings next to it
  90 % range      5th and 95th percentiles of the error expected on a new curve drawn from the test design,
                  with the 5th and 95th percentiles of the actual readings next to it as a check
  gross misreads  share of readings with |error| > 10 K (> 50 % for ΔH); missed values are counted apart
  worst condition the knob level most likely to have the largest typical |error| (the model's median |error|),
                  how much it adds and how sure that is. The rule: it is the worst level with probability above
                  0.50 and the 94 % interval of what it adds starts above 0. Both numbers come from a finite set
                  of draws, so each has a Monte Carlo standard error (MCSE, "sampling error"): how much it would
                  move in a rerun with another seed or on another computer. The call is
                    "clear"       if the rule holds with 2 MCSE to spare (probability - 2 MCSE > 0.50 and lower
                                  end - 2 MCSE > 0); only then is the level named as the worst condition,
                    "borderline"  if not, but the rule holds within 2 MCSE (probability + 2 MCSE > 0.50 and lower
                                  end + 2 MCSE > 0): a rerun can change the call,
                    "none clear"  otherwise.
                  A clear call with little to spare can also turn borderline in a rerun; to turn none clear, a
                  number would have to move by some 4 MCSE or more between the two runs.
                  MCSE of the probability: ArviZ's MCSE of a mean (az.mcse, method "mean"), the mean of the
                  indicator "this level is the worst" over the draws, each chain in its own order, so that the
                  correlation between successive draws counts. MCSE of the lower end of the interval: a block
                  bootstrap of the draws (see mcse_of_hdi_low; ArviZ's quantile MCSE misses part of it).
                  A clear or borderline line prints the four numbers its call was judged with, to as many decimals
                  as it takes for the printed numbers to give the same call (see call_numbers).
  tested range    the knob levels, temperatures and ΔH values the test set covers; nothing is said outside them
  diagnostics     max R-hat, min bulk ESS, divergences. Gate 3: R-hat <= 1.01, ESS >= 400, 0 divergences.
                  A fit that misses Gate 3 on R-hat or ESS alone (no divergences, no stuck chain) is sampled
                  again with 2000, then 4000 draws per chain; its diagnostics line then says what the first run
                  gave. A stuck chain (R-hat above 1.05 or bulk ESS below 50) is reported and not run longer:
                  with the same seed, a longer run gets stuck in the same way.
A 94 % credible interval: given these readings and the model, there is a 94 % probability that the true value
lies inside it (the narrowest such range, also called the highest-density interval or HDI; 94 % is ArviZ's
default, not a special number).
"""
import csv
import itertools
import json
import logging
import platform
import re
import sys
import time
import zlib
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
from scipy import optimize, special, stats

from score import GROSS_K, QUANTITIES

# ---------------------------------------------------------------- settings
DEFAULT_SEED = 2026
FULL = {"draws": 1000, "tune": 1000, "chains": 4}   # the real run
QUICK = {"draws": 200, "tune": 200, "chains": 2}    # --quick: for testing only
# A full-run fit that misses Gate 3 only on R-hat or ESS (no divergences) is sampled again, longer:
# more draws give more effective samples and a sharper R-hat. Divergences are never retried, because
# a longer run does not cure them. Set to () to switch this off.
LONGER_DRAWS = (2000, 4000)
# A chain is stuck when the chains disagree badly (R-hat above STUCK_RHAT) or the draws are worth almost
# nothing (bulk ESS below STUCK_ESS). The same seed would get stuck again, so such a fit is not run longer.
STUCK_RHAT = 1.05
STUCK_ESS = 50
TARGET_ACCEPT = 0.95  # smaller sampler steps than the default 0.8: slower, but fewer divergences
HDI_PROB = 0.94       # credible intervals hold 94 % of the posterior (the 90 % range is a different thing)
MIN_READINGS = 10     # fewer values read than this: no fit, no error report
CLEAR_PROB = 0.5      # a worst condition is named only above this probability (and if what it adds is above 0),
CALL_MARGIN = 2       # with this many Monte Carlo standard errors (MCSE) to spare; within them it is borderline
BOOT_BLOCK = 50       # MCSE of the lower end of an HDI: block bootstrap, blocks of 50 successive draws of a chain,
BOOT_RESAMPLES = 400  # 400 resamples (see mcse_of_hdi_low)

UNIT = {"onset_C": "K", "peak_C": "K", "tg_mid_C": "K", "enthalpy_J_per_g": "%"}
PRIOR_SCALE = {"K": 5.0, "%": 20.0}       # P in the model above
# Smallest error that can be told from 0: readings carry at most 2 decimals (0.01 K; for ΔH, 0.01 % is about
# what 0.01 J/g is on a ΔH of 100 J/g). Rounding to a step h adds an error of spread h / sqrt(12).
RESOLUTION = {"K": 0.01, "%": 0.01}
GROSS_LIMIT = {"K": GROSS_K, "%": 50.0}   # gross misread: |error| above 10 K, or above 50 % of the true ΔH
NAMES = {"onset_C": "Onset temperature", "peak_C": "Peak temperature",
         "enthalpy_J_per_g": "Enthalpy ΔH", "tg_mid_C": "Glass transition Tg (midpoint)"}

PEAK_EVENTS = ("melting", "crystallization")
GLASS_EVENTS = ("glass_transition",)
COMMON_KNOBS = ("noise", "baseline", "convention", "units", "style")
# Order in which knob levels are listed (any level not in this list goes at the end, in A-Z order).
LEVEL_ORDER = ["low", "medium", "high", "flat", "sloped", "curved", "melting", "crystallization",
               "glass_transition", "exo_up", "exo_down", "C_mW", "K_Wg", "plain", "printout", "report"]

PREDICT_PER_DRAW = 25  # new curves imagined per posterior draw for the 90 % range (more = smoother percentiles)

GATE_RHAT = 1.01
GATE_ESS = 400

SCOPE = ("Every curve here is synthetic (generator.py: one event per curve, heated at 10 K/min, peaks 2–15 K wide "
         "at half height, glass-transition steps 3–10 K wide, Gaussian noise, no overlapping peaks). This "
         "error report says nothing about real instrument curves, or about curves outside these knob levels, "
         "temperatures and ΔH values.")

USAGE = "Usage: python error_model.py [test] [--quick] [--seed N]"
READERS = Path("readers")  # where write_code_reader.py saves the code readers' scripts


# ---------------------------------------------------------------- reading the inputs
def read_csv(path):
    """All rows of a CSV file, as a list of dicts (every value is text)."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_truths(folder):
    """{curve name: true event} from the *.truth.json files of the test set."""
    truths = {}
    for path in sorted(Path(folder).glob("*.truth.json")):
        truths[path.name[:-len(".truth.json")]] = json.loads(path.read_text(encoding="utf-8"))["events"][0]
    return truths


def events_with(quantity):
    """The event types that have this quantity: Tg for glass transitions, the rest for peaks."""
    return GLASS_EVENTS if quantity == "tg_mid_C" else PEAK_EVENTS


def knobs_for(quantity):
    """The knobs that may move this quantity's errors. Melting vs crystallization only matters for peaks."""
    return COMMON_KNOBS if quantity == "tg_mid_C" else ("event", *COMMON_KNOBS)


def ordered(levels):
    """Knob levels in a fixed, readable order (low, medium, high; flat, sloped, curved; ...)."""
    known = [x for x in LEVEL_ORDER if x in levels]
    return known + sorted(set(levels) - set(known))


def error_of(row, quantity):
    """The error of one reading: in K for temperatures, in % of the true value for ΔH."""
    return float(row["rel_error_pct"] if UNIT[quantity] == "%" else row["error"])


# ---------------------------------------------------------------- the model
def knob_levels(rows, quantity):
    """{knob: levels seen in these readings}, leaving out knobs with only one level (nothing to compare)."""
    levels = {}
    for knob in knobs_for(quantity):
        seen = ordered({row[knob] for row in rows})
        if len(seen) >= 2:
            levels[knob] = seen
    return levels


def level_codes(rows, knob, levels):
    """Each reading's level of one knob, as a position in `levels` (0, 1, 2, ...)."""
    return np.array([levels.index(row[knob]) for row in rows])


def width_floor(unit):
    """r in the model: the spread that rounding readings to RESOLUTION adds (step / sqrt(12))."""
    return RESOLUTION[unit] / np.sqrt(12)


def effect_scale(errors, n_levels, unit):
    """Roughly how precisely the readings pin down one knob level's effect: spread / sqrt(readings per level).

    The spread is the median absolute deviation x 1.4826, a standard deviation that one wild reading cannot
    inflate. Readings exactly at the median are left out of it: a reader that is often exactly right would
    otherwise get a spread of 0. It is never below the width floor r.
    Used only to choose the sampler's coordinates (see build_model); it does not change the model.
    """
    errors = np.asarray(errors, dtype=float)
    deviations = np.abs(errors - np.median(errors))
    if np.any(deviations > 0):
        deviations = deviations[deviations > 0]
    spread = 1.4826 * np.median(deviations)
    return max(spread, width_floor(unit)) / np.sqrt(len(errors) / n_levels)


def build_model(errors, rows, levels, unit):
    """The PyMC model for one reader and one quantity (see the top of this file)."""
    P = PRIOR_SCALE[unit]
    with pm.Model(coords=levels) as model:  # coords name the levels, so the results are labelled
        mu = pm.Normal("mu", mu=0, sigma=P)             # bias: the centre of the errors over the test design
        scatter = pm.HalfNormal("scatter", sigma=P)     # spread of the errors before the knob effects
        nu = pm.Gamma("nu", alpha=2, beta=0.1)          # tail weight: small nu = more wild readings
        location, log_width = mu, 0
        for knob, these_levels in levels.items():
            codes = level_codes(rows, knob, these_levels)
            sd_knob = pm.HalfNormal(f"sd_{knob}", sigma=P / 2)  # how big this knob's effects tend to be
            # The model is b_knob ~ ZeroSumNormal(sigma=sd_knob): one effect per level, adding up to zero,
            # so mu stays the centre of the errors over the test design. The sampler does not walk in b itself
            # but in z = b * stretch / sd_knob, a halfway form between the two textbook choices:
            #   "centered" (walk in b) gets stuck when the effects are tiny and sd_knob shrinks towards 0;
            #   "non-centered" (walk in b / sd_knob) gets stuck when the readings pin the effects down
            #   much more tightly than the prior on sd_knob (the case here: 0.1 K effects against a prior
            #   P/2 of 2.5 K for temperatures, or 10 % for ΔH).
            # c is the scale the readings can resolve, and stretch = sqrt(1 + (sd_knob / c)^2):
            #   if sd_knob is much larger than c, z is about b / c (we walk in b itself, in steps of c);
            #   if sd_knob is much smaller than c, z is about b / sd_knob (the textbook non-centered form);
            #   in between it blends the two.
            # It is the same model either way: z ~ ZeroSumNormal(sigma=stretch) gives b ~ ZeroSumNormal(sd_knob).
            c = effect_scale(errors, len(these_levels), unit)
            stretch = pm.math.sqrt(1 + (sd_knob / c) ** 2)
            z_knob = pm.ZeroSumNormal(f"z_{knob}", sigma=stretch, dims=knob)
            b_knob = pm.Deterministic(f"b_{knob}", z_knob * sd_knob / stretch, dims=knob)
            # Every knob level can also widen or narrow the spread: scatter is multiplied by exp(g_knob[level]).
            g_knob = pm.ZeroSumNormal(f"g_{knob}", sigma=0.5, dims=knob)
            location = location + b_knob[codes]
            log_width = log_width + g_knob[codes]
        width = pm.math.sqrt((scatter * pm.math.exp(log_width)) ** 2 + width_floor(unit) ** 2)
        pm.StudentT("error", nu=nu, mu=location, sigma=width, observed=np.asarray(errors, dtype=float))
    return model


def fit_seed(seed, reader, quantity):
    """A fixed seed for one fit, made from the base seed and the fit's name.

    So one fit gives the same numbers whichever other readers are in the table (when rerun on the same
    computer; the very first run, while PyTensor compiles, can give slightly different draws: see the top).
    """
    return (seed * 100_003 + zlib.crc32(f"{reader}/{quantity}".encode())) % 2**32


def sample(model, settings, seed):
    """Run the NUTS sampler. cores=1 runs the chains one after the other (avoids multiprocessing trouble on macOS)."""
    logging.getLogger("pymc").setLevel(logging.CRITICAL)  # quiet PyMC's own messages; diagnostics() reports instead
    with model:
        return pm.sample(draws=settings["draws"], tune=settings["tune"], chains=settings["chains"], cores=1,
                         target_accept=TARGET_ACCEPT, random_seed=seed, progressbar=False,
                         compute_convergence_checks=False)  # our own checks are in diagnostics()


def draws_of(idata, name):
    """All posterior draws of one variable, chains put end to end: shape (draws,) or (draws, levels)."""
    values = np.asarray(idata.posterior[name].values)  # shape (chain, draw, ...)
    return values.reshape(values.shape[0] * values.shape[1], *values.shape[2:])


def diagnostics(idata):
    """Max R-hat, min bulk ESS and the number of divergences, over every number in the posterior.

    R-hat compares the chains: near 1.00 means they all found the same answer.
    Bulk ESS is how many independent draws the correlated draws are worth.
    A divergence is a step where the sampler lost its way; even one means a result may be biased.
    """
    worst_rhat, worst_rhat_name, lowest_ess, lowest_ess_name = 0.0, "", np.inf, ""
    for name in idata.posterior.data_vars:
        values = np.asarray(idata.posterior[name].values)
        columns = values.reshape(values.shape[0], values.shape[1], -1)  # one column per level
        for j in range(columns.shape[2]):
            rhat = float(az.rhat(columns[:, :, j]))
            ess = float(az.ess(columns[:, :, j], method="bulk"))
            if rhat > worst_rhat:
                worst_rhat, worst_rhat_name = rhat, name
            if ess < lowest_ess:
                lowest_ess, lowest_ess_name = ess, name
    divergences = int(np.asarray(idata.sample_stats["diverging"].values).sum())
    passed = worst_rhat <= GATE_RHAT and lowest_ess >= GATE_ESS and divergences == 0
    stuck = worst_rhat > STUCK_RHAT or lowest_ess < STUCK_ESS
    return {"max_rhat": worst_rhat, "max_rhat_variable": worst_rhat_name, "min_ess_bulk": lowest_ess,
            "min_ess_variable": lowest_ess_name, "divergences": divergences, "gate3_pass": passed, "stuck": stuck}


def needs_longer_run(check):
    """True when a longer run can help: Gate 3 missed on R-hat or ESS only, with no divergence and no stuck chain."""
    return not check["gate3_pass"] and check["divergences"] == 0 and not check["stuck"]


# ---------------------------------------------------------------- from posterior draws to error report numbers
def hdi(values):
    """94 % highest-density interval of some draws: the narrowest range holding 94 % of them."""
    low, high = np.asarray(az.hdi(np.asarray(values, dtype=float), prob=HDI_PROB)).ravel()
    return float(low), float(high)


def summary_of(values):
    """Posterior mean and 94 % HDI of some draws."""
    low, high = hdi(values)
    return {"mean": float(np.mean(values)), "low": low, "high": high}


def cell_location_and_width(post, levels, design_rows, unit):
    """Centre and width of the error distribution for every posterior draw and every design row.

    Returns two arrays of shape (draws, rows), worked out as in build_model. A level seen in the design but not
    among the readings (possible only if a reader missed every curve at that level) gets an effect of 0, the
    average.
    """
    n_draws, n_rows = len(post["mu"]), len(design_rows)
    location = np.repeat(post["mu"][:, None], n_rows, axis=1)
    log_width = np.zeros((n_draws, n_rows))
    for knob, these_levels in levels.items():
        where = [these_levels.index(row[knob]) if row[knob] in these_levels else -1 for row in design_rows]
        # One extra column of zeros at the end, for levels the readings never had (position -1).
        effect = np.hstack([post[f"b_{knob}"], np.zeros((n_draws, 1))])
        widen = np.hstack([post[f"g_{knob}"], np.zeros((n_draws, 1))])
        location += effect[:, where]
        log_width += widen[:, where]
    width = np.sqrt((post["scatter"][:, None] * np.exp(log_width)) ** 2 + width_floor(unit) ** 2)
    return location, width


def median_abs_error(location, width, nu):
    """Median size |error| of a new reading on one of these design rows, picked at random: half are smaller.

    location and width: shape (draws, rows); nu: shape (draws,). Returns one median per draw, shape (draws,).
    m is the median when the share of errors between -m and +m, averaged over the rows, is one half:
      average over rows of [F((m - centre) / width) - F((-m - centre) / width)] = 0.5   (F = t cumulative)
    Each row has half its errors within |centre| + t75 x width (t75 = 75th percentile of a t variable) and no
    more than half within t75 x width, so m lies between the smallest and the largest of these; it is found by
    halving that gap 20 times (bisection). Unlike the average |error|, it is finite for every nu, so a few
    heavy-tailed draws cannot run away with it, and it can be compared with the median |error| of the readings.
    """
    nu = np.broadcast_to(np.reshape(nu, (-1, 1)), np.shape(location))
    t75 = special.stdtrit(nu, 0.75)
    low = np.min(t75 * width, axis=1)
    high = np.max(np.abs(location) + t75 * width, axis=1)
    for _ in range(20):
        middle = (low + high) / 2
        m = middle[:, None]
        share_within = np.mean(special.stdtr(nu, (m - location) / width) - special.stdtr(nu, (-m - location) / width),
                               axis=1)
        too_small = share_within < 0.5
        low = np.where(too_small, middle, low)
        high = np.where(too_small, high, middle)
    return (low + high) / 2


def error_range(location, width, nu, rng):
    """5th and 95th percentiles of the error on a new curve drawn from the test design.

    For each posterior draw, PREDICT_PER_DRAW design rows are picked at random (so the knob mix is the
    real one) and one Student-t error is drawn for each. 90 % of readings are expected in this range.
    """
    n_draws, n_rows = location.shape
    pick = rng.integers(0, n_rows, size=(n_draws, PREDICT_PER_DRAW))
    t = rng.standard_t(np.repeat(nu[:, None], PREDICT_PER_DRAW, axis=1))
    errors = np.take_along_axis(location, pick, axis=1) + np.take_along_axis(width, pick, axis=1) * t
    low, high = np.percentile(errors, [5, 95])
    return {"low": float(low), "high": float(high)}


def mcse_of_share(indicator):
    """Monte Carlo standard error (MCSE) of a share of posterior draws, such as prob_worst: how much the share
    would move if the sampler were run again with another seed (the draws are a finite sample).

    indicator: 1 for a draw where the thing holds, 0 where it does not, shape (chains, draws), each chain in the
    order it was drawn. The share is the mean of the indicator, so this is ArviZ's MCSE of a mean,
    az.mcse(method="mean"): the indicator's standard deviation / sqrt(ESS), where the effective sample size ESS
    counts how many independent draws the chains are worth, given that successive draws of a chain are
    correlated. For independent draws it is sqrt(p (1 - p) / N). 0 when the indicator is the same in every draw
    (some ArviZ versions give nan there): no draw disagrees, so nothing measures a sampling error.
    """
    indicator = np.asarray(indicator, dtype=float)
    if np.all(indicator == indicator.flat[0]):
        return 0.0
    return float(az.mcse(indicator, method="mean"))


def mcse_of_hdi_low(values, rng):
    """MCSE of the lower end of the 94 % HDI of some draws, shape (chains, draws), each chain in its own order.

    Block bootstrap: each chain is rebuilt from blocks of successive draws of that same chain, picked at random
    with replacement, the HDI is worked out again on all chains, and this is done BOOT_RESAMPLES times; the MCSE
    is the standard deviation of the HDI's lower end over the resamples. Blocks keep the correlation between
    successive draws: Gate 3 asks for at least 400 effective draws, so 4 chains x 1000 draws have at most 10
    draws per effective draw, and a block of BOOT_BLOCK = 50 draws spans 5 of them. A fit sampled again with
    4000 draws per chain can have up to 40; on simulated draws like that (4 x 4000, 40 draws per effective
    draw) blocks of 50 still gave 0.93 of the real spread of the lower end, blocks of 200 gave 0.96. A chain
    of fewer than 200 draws (the quick checks) is cut into at least 4 blocks, so that the resamples differ.
    rng: a seeded NumPy generator, so a rerun gives the same MCSE.
    Why not ArviZ's quantile MCSE at the HDI's tail probability (az.mcse(method="quantile")): it treats the
    lower end as a fixed quantile, but the HDI also chooses which quantile (the narrowest range), and that choice
    moves with the draws too. On simulated draws (4 chains x 1000) the lower end varied 1.4 to 2 times more
    than the quantile MCSE says, and 0.9 to 1.2 times what this bootstrap says (test_mcse_of_hdi_low).
    """
    values = np.asarray(values, dtype=float)
    if np.all(values == values.flat[0]):
        return 0.0
    chains, n = values.shape
    block = max(1, min(BOOT_BLOCK, n // 4))
    blocks_per_chain = -(-n // block)  # rounded up; the last block is cut to length n
    lows = []
    for _ in range(BOOT_RESAMPLES):
        starts = rng.integers(0, n - block + 1, size=(chains, blocks_per_chain))
        picked = (starts[:, :, None] + np.arange(block)).reshape(chains, -1)[:, :n]
        lows.append(hdi(np.take_along_axis(values, picked, axis=1).ravel())[0])
    return float(np.std(lows, ddof=1))


def worst_call(prob_worst, mcse_prob, adds_low, mcse_low, margin=CALL_MARGIN):
    """The worst-condition call: "clear", "borderline" or "none clear".

    The rule: a level is the worst with probability above CLEAR_PROB, and the 94 % HDI of what it adds starts
    above 0. Both numbers come from a finite set of draws, so each has a sampling error (MCSE).
      clear:      the rule holds with `margin` MCSEs to spare: prob_worst - margin x mcse_prob > CLEAR_PROB and
                  adds_low - margin x mcse_low > 0.
      borderline: not clear, but the rule could hold within `margin` MCSEs: prob_worst + margin x mcse_prob >
                  CLEAR_PROB and adds_low + margin x mcse_low > 0. A rerun (another seed, another computer) can
                  turn it into either of the other two calls.
      none clear: otherwise.
    The margin replaces the earlier rounding of the probability to 2 decimals as printed: a probability printed
    as 0.50 (below 0.505) could be clear only with an MCSE below 0.0025, which takes some 40 000 effective draws,
    ten times the 4000 drawn (independent draws give sqrt(0.25 / 4000) = 0.008).
    An MCSE that cannot be worked out (nan) counts as unbounded: never clear, at best borderline.
    """
    def spare(mcse):
        mcse = float(mcse)
        return margin * mcse if np.isfinite(mcse) else np.inf
    p, low = float(prob_worst), float(adds_low)
    if p - spare(mcse_prob) > CLEAR_PROB and low - spare(mcse_low) > 0:
        return "clear"
    if p + spare(mcse_prob) > CLEAR_PROB and low + spare(mcse_low) > 0:
        return "borderline"
    return "none clear"


def worst_condition(location, width, nu, levels, design_rows, chains=1, rng=None):
    """The knob level most likely to have the largest typical |error|, the other knobs mixed as in the design.

    For each posterior draw: the typical |error| of a new reading at each knob level, that is the median
    |error| of a reading on a design row at that level (see median_abs_error), and the same over all rows.
    The named level is the one that is largest in the most draws (prob_worst); for it the typical |error|
    there, over the whole design, and what it adds (the difference) are given as posterior means with 94 %
    HDIs. call: see worst_call, with the sampling errors of prob_worst (mcse_prob, see mcse_of_share) and of the
    lower end of what it adds (mcse_adds_low, see mcse_of_hdi_low); clear is true when call is "clear".
    The draws come chains end to end (as draws_of gives them): chains says how many, so that the MCSEs see each
    chain in its own order. rng: the seeded generator for the bootstrap (a fixed one if not given).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    overall = median_abs_error(location, width, nu)  # (draws,)
    names, per_level = [], []
    for knob, these_levels in levels.items():
        for level in these_levels:
            rows_here = [i for i, row in enumerate(design_rows) if row[knob] == level]
            if rows_here:
                names.append((knob, level))
                per_level.append(median_abs_error(location[:, rows_here], width[:, rows_here], nu))
    if not names:
        return None
    per_level = np.column_stack(per_level)          # (draws, levels)
    worst_in_draw = np.argmax(per_level, axis=1)
    prob = np.bincount(worst_in_draw, minlength=len(names)) / len(per_level)
    top = int(np.argmax(prob))
    adds_draws = per_level[:, top] - overall
    adds = summary_of(adds_draws)
    mcse_prob = mcse_of_share((worst_in_draw == top).reshape(chains, -1))
    mcse_low = mcse_of_hdi_low(adds_draws.reshape(chains, -1), rng)
    call = worst_call(prob[top], mcse_prob, adds["low"], mcse_low)
    all_levels = {f"{knob}={level}": {"typical_abs_error": float(np.mean(per_level[:, i])),
                                      "adds": float(np.mean(per_level[:, i] - overall)),
                                      "prob_worst": float(prob[i])}
                  for i, (knob, level) in enumerate(names)}
    return {"knob": names[top][0], "level": names[top][1], "prob_worst": float(prob[top]),
            "mcse_prob": mcse_prob, "call": call, "clear": call == "clear", "adds": adds,
            "mcse_adds_low": mcse_low,
            "typical_abs_error": summary_of(per_level[:, top]), "typical_abs_error_overall": summary_of(overall),
            "n_levels": len(names), "all_levels": all_levels}


def observed_numbers(errors, rows, worst):
    """What the readings themselves show, printed next to the model's numbers as a check."""
    errors = np.asarray(errors, dtype=float)
    seen = {"mean_error": float(np.mean(errors)), "median_error": float(np.median(errors)),
            "p5": float(np.percentile(errors, 5)), "p95": float(np.percentile(errors, 95)),
            "median_abs_error": float(np.median(np.abs(errors)))}
    if worst:
        here = [abs(e) for e, row in zip(errors, rows) if row[worst["knob"]] == worst["level"]]
        seen["median_abs_error_at_worst"] = float(np.median(here)) if here else None
    return seen


def beta_summary(k, n):
    """k successes out of n: Beta(1+k, 1+n-k) posterior mean and its 94 % highest-density interval.

    Beta(1+k, 1+n-k) is what a flat prior on the true share becomes after seeing k of n. No sampler needed.
    """
    if n == 0:
        return {"k": 0, "n": 0, "mean": None, "low": None, "high": None}
    dist = stats.beta(1 + k, 1 + n - k)
    # The narrowest interval holding 94 %: slide the 6 % left outside between the two tails.
    width = lambda p: dist.ppf(p + HDI_PROB) - dist.ppf(p)
    best = optimize.minimize_scalar(width, bounds=(0, 1 - HDI_PROB), method="bounded",
                                    options={"xatol": 1e-9})
    p = best.x
    # The ends of the slide are not always tried by the search, so try them too.
    for end in (0.0, 1 - HDI_PROB):
        if width(end) < width(p):
            p = end
    return {"k": int(k), "n": int(n), "mean": float(dist.mean()),
            "low": float(dist.ppf(p)), "high": float(dist.ppf(p + HDI_PROB))}


def gross_and_missed(rows, quantity):
    """Gross misreads among the values read, and values missed, for one reader and one quantity."""
    unit = UNIT[quantity]
    ok = [row for row in rows if row["status"] == "ok"]
    true_values = [row for row in rows if row["status"] in ("ok", "missed", "no_answer")]
    gross = sum(abs(error_of(row, quantity)) > GROSS_LIMIT[unit] for row in ok)
    gross_share = {"limit": GROSS_LIMIT[unit], **beta_summary(gross, len(ok))}
    missed = {"k": len(true_values) - len(ok), "n": len(true_values)}  # missed or no answer, of all true values
    return gross_share, missed


def tested_range(rows, levels_all, truths):
    """What the test set covered for these readings: knob levels, event temperatures and ΔH values."""
    curves = [truths[row["curve"]] for row in rows if row["curve"] in truths]
    temps = [c["tg_mid_C"] if c["type"] == "glass_transition" else c["peak_C"] for c in curves]
    enthalpies = [c["enthalpy_J_per_g"] for c in curves if c.get("enthalpy_J_per_g") is not None]
    span = lambda xs: [float(min(xs)), float(max(xs))] if xs else None
    return {"knob_levels": levels_all, "event_temperature_C": span(temps), "enthalpy_J_per_g": span(enthalpies),
            "design_temperature_C": [60, 300], "sentence": SCOPE}


def fit_report(reader, quantity, rows, design, truths, settings, seed):
    """Fit the model for one reader and one quantity and work out its error report. Returns a dict."""
    unit = UNIT[quantity]
    ok = [row for row in rows if row["status"] == "ok"]
    gross, missed = gross_and_missed(rows, quantity)
    result = {"quantity": quantity, "unit": unit, "n_readings": len(ok), "gross": gross, "missed": missed}
    if len(ok) < MIN_READINGS:
        result["status"] = f"no error report: {len(ok)} values read, fewer than {MIN_READINGS}"
        return result
    errors = [error_of(row, quantity) for row in ok]
    levels = knob_levels(ok, quantity)
    levels_all = {knob: ordered({row[knob] for row in ok}) for knob in knobs_for(quantity)}
    this_seed = fit_seed(seed, reader, quantity)
    started = time.perf_counter()
    model = build_model(errors, ok, levels, unit)
    idata = sample(model, settings, this_seed)
    check = diagnostics(idata)
    first_try = {**check, "draws": settings["draws"]}
    for draws in settings.get("longer_draws", ()):
        if not needs_longer_run(check):
            break
        idata = sample(model, {**settings, "draws": draws}, this_seed)
        check = diagnostics(idata)
    draws_used = idata.posterior.sizes["draw"]
    seconds = time.perf_counter() - started

    post = {name: draws_of(idata, name) for name in idata.posterior.data_vars}
    design_rows = [row for row in design if row["event"] in events_with(quantity)]
    location, width = cell_location_and_width(post, levels, design_rows, unit)
    rng = np.random.default_rng(this_seed)
    # The bootstrap in worst_condition gets its own generator, so the 90 % range draws the same numbers as before.
    worst = worst_condition(location, width, post["nu"], levels, design_rows, chains=idata.posterior.sizes["chain"],
                            rng=np.random.default_rng([this_seed, 1]))
    result.update({
        "status": "fitted",
        "knobs_in_model": list(levels),
        "bias": summary_of(post["mu"]),
        "range_90": error_range(location, width, post["nu"], rng),
        "observed": observed_numbers(errors, ok, worst),
        "scatter": summary_of(post["scatter"]),
        "nu": summary_of(post["nu"]),
        "knob_effects": {knob: {level: summary_of(post[f"b_{knob}"][:, i]) for i, level in enumerate(lv)}
                         for knob, lv in levels.items()},
        "width_factor": {knob: {level: summary_of(np.exp(post[f"g_{knob}"][:, i])) for i, level in enumerate(lv)}
                         for knob, lv in levels.items()},
        "worst_condition": worst,
        "tested_range": tested_range(ok, levels_all, truths),
        "diagnostics": {**check, "draws": int(draws_used), "tune": settings["tune"], "chains": settings["chains"],
                        "seed": this_seed, "seconds": seconds,
                        "first_try": None if draws_used == settings["draws"] else first_try},
    })
    return result


# ---------------------------------------------------------------- checks that need no sampler
def reader_checks(curve_rows):
    """Event type and direction (endo/exo) right, out of the curves where the reader gave one, and direction
    errors counted by plot convention and true event type.

    A reply whose event has no type (or no direction: null or blank) gave no answer to that question, so it
    is counted neither right nor wrong. Such replies are counted apart, in all and by true event type.
    """
    with_event = [row for row in curve_rows if row.get("type_ok", "") != ""]  # score.py: blank = no event at all
    typed = [row for row in curve_rows if row.get("reported_type", "") != ""]
    directed = [row for row in curve_rows if row.get("reported_direction", "") != ""]
    blank = [row for row in with_event if row.get("reported_type", "") == "" and row.get("reported_direction", "") == ""]
    events = ordered({row["event"] for row in curve_rows})
    blank_by_event = {event: {"k": sum(row["event"] == event for row in blank),
                              "n": sum(row["event"] == event for row in with_event)} for event in events}
    by_cell = {}
    for convention in ordered({row["convention"] for row in curve_rows}):
        by_cell[convention] = {}
        for event in events:
            here = [row for row in directed if row["convention"] == convention and row["event"] == event]
            by_cell[convention][event] = {"k": sum(row["direction_ok"] == "0" for row in here), "n": len(here)}
    return {"curves": len(curve_rows), "with_event": len(with_event), "no_type_no_direction": len(blank),
            "no_type_no_direction_by_event": blank_by_event,
            "type_right": beta_summary(sum(row["type_ok"] == "1" for row in typed), len(typed)),
            "direction_right": beta_summary(sum(row["direction_ok"] == "1" for row in directed), len(directed)),
            "direction_wrong_by_convention_and_event": by_cell}


def onsets_on_glass(errors, design, reader):
    """How many glass transitions one reader gave an onset for, of all glass transitions in the design. A glass
    transition has no onset in this test, so score.py marks such a value extra_value and does not score it."""
    k = sum(row["reader"] == reader and row["quantity"] == "onset_C" and row["status"] == "extra_value"
            and row["event"] in GLASS_EVENTS for row in errors)
    return {"k": k, "n": sum(row["event"] in GLASS_EVENTS for row in design)}


# ---------------------------------------------------------------- writing the results
def tidy(x, digits):
    """x rounded, with -0.00 turned into 0.00."""
    return round(x, digits) + 0.0


def signed(x, digits):
    return f"{tidy(x, digits):+.{digits}f}".replace("-", "−")


def plain(x, digits):
    return f"{tidy(x, digits):.{digits}f}".replace("-", "−")


def digits_for(rep):
    """2 decimals for a reader whose errors are below 1 K (or 1 %), else 1."""
    r = rep["range_90"]
    return 2 if max(abs(r["low"]), abs(r["high"])) < 1 else 1


def share_text(b):
    """The measured share k of n, then where the true share lies (Beta posterior, 94 %)."""
    if b["k"] == 0:
        return f"0 %; true share below {100 * b['high']:.1f} % with 94 % probability"
    if b["k"] == b["n"]:
        return f"100 %; true share above {100 * b['low']:.1f} % with 94 % probability"
    return f"{100 * b['k'] / b['n']:.1f} %; 94 % interval {100 * b['low']:.1f}–{100 * b['high']:.1f} %"


def prob_text(p, digits=2):
    """A probability as the error reports print it: 2 decimals (or `digits`), except that one that would print
    as 1.00 is printed "> 0.99" and one that would print as 0.00 is printed "< 0.01" (no share of posterior
    draws ever shows that it is exactly 1 or 0). The JSON file keeps the raw value."""
    shown = f"{p:.{digits}f}"
    if float(shown) == 1:
        return f"> {1 - 10 ** -digits:.{digits}f}"
    if float(shown) == 0:
        return f"< {10 ** -digits:.{digits}f}"
    return shown


def tested_range_text(v):
    parts = [f"{knob} {'/'.join(levels)}" for knob, levels in v["knob_levels"].items()]
    if v["event_temperature_C"]:
        low, high = v["event_temperature_C"]
        parts.append(f"event temperatures {low:.0f}–{high:.0f} °C (generator range 60–300 °C)")
    if v["enthalpy_J_per_g"]:
        low, high = v["enthalpy_J_per_g"]
        parts.append(f"ΔH {low:.0f}–{high:.0f} J/g")
    return " · ".join(parts)


def diagnostics_notes(diag):
    """For a fit that was sampled again, or got stuck: what happened."""
    notes = ""
    first = diag["first_try"]
    if first:
        notes += (f"; sampled again with more draws because the first run ({diag['chains']} × {first['draws']} "
                  f"draws) gave max R-hat {first['max_rhat']:.3f}, min bulk ESS {first['min_ess_bulk']:.0f}")
    if diag["stuck"]:
        notes += (f"; a chain got stuck (R-hat above {STUCK_RHAT} or bulk ESS below {STUCK_ESS}), so these numbers "
                  f"are not reliable; not run longer, because the same seed gets stuck the same way")
    return notes


def sampling_text(mcse, digits):
    """A sampling error (MCSE) as printed: "± 0.01", or "below 0.01" when it would print as 0."""
    shown = plain(mcse, digits)
    if float(shown.replace("−", "-")) == 0:
        return f"below {10 ** -digits:.{digits}f}"
    return f"± {shown}"


def read_back(text):
    """The values a reader can take a printed number to be: "below 0.001" is anything from 0 to 0.001, and
    "> 0.999" or "< 0.001" is taken as the number shown."""
    number = float(text.split()[-1].replace("−", "-"))
    return (0.0, number) if text.startswith("below") else (number,)


def call_numbers(w, a):
    """The four numbers a worst-condition call was judged with, as printed: the probability, its sampling error
    (MCSE), the lower end of what the level adds and its MCSE. The probability and its MCSE get 3 decimals, the
    lower end and its MCSE a + 1 (one more than the interval of what it adds), or more where rounding would put
    a number on the wrong side of the line: the printed numbers, put through the rule (worst_call), give the
    printed call. (With 2 decimals they did not always: a probability of 0.4848 with an MCSE of 0.0091 is
    borderline, as 0.4848 + 2 x 0.0091 = 0.503, but prints as 0.48 and 0.01, and 0.48 + 2 x 0.01 = 0.50.)"""
    for extra in range(6):
        dp, dl = 3 + extra, a + 1 + extra
        shown = (prob_text(w["prob_worst"], dp), sampling_text(w["mcse_prob"], dp),
                 plain(w["adds"]["low"], dl), sampling_text(w["mcse_adds_low"], dl))
        if all(worst_call(*values) == w["call"] for values in itertools.product(*map(read_back, shown))):
            break
    return shown


def worst_text(w, d, u, seen):
    """The worst-condition line of one error report. What it adds gets one more decimal, so that its interval
    can be seen to be above 0 or not. A clear or borderline call also gives the four numbers that the call was
    judged with (see worst_call), to enough decimals that they give the same call (see call_numbers)."""
    a = d + 1
    adds = f"{signed(w['adds']['mean'], a)}{u} [{plain(w['adds']['low'], a)}, {plain(w['adds']['high'], a)}]"
    if w["call"] == "none clear":
        return (f"- **Worst condition**: none clear. Most likely {w['knob']} = {w['level']} (probability "
                f"{prob_text(w['prob_worst'])} that it is the worst of the {w['n_levels']} knob levels; adds "
                f"{adds} to the typical |error|)")
    prob, mcse_prob, low, mcse_low = call_numbers(w, a)
    if w["call"] == "borderline":
        return (f"- **Worst condition**: borderline. {w['knob']} = {w['level']} has probability {prob} (sampling "
                f"error {mcse_prob}) of being the worst of the {w['n_levels']} knob levels and adds {adds} to the "
                f"typical |error| (lower end {low}{u}, sampling error {mcse_low}{u}); a rerun can change this call")
    return (f"- **Worst condition**: {w['knob']} = {w['level']}. Typical |error| {plain(w['typical_abs_error']['mean'], d)}"
            f"{u} there against {plain(w['typical_abs_error_overall']['mean'], d)}{u} over the test design (adds "
            f"{adds}; probability {prob} that it is the worst of the {w['n_levels']} knob levels, sampling error "
            f"{mcse_prob}; lower end of what it adds {low}{u}, sampling error {mcse_low}{u}). Observed median "
            f"|error|: {plain(seen['median_abs_error_at_worst'], d)}{u} there, {plain(seen['median_abs_error'], d)}"
            f"{u} over all readings")


def report_lines(rep):
    """The Markdown lines of one quantity's error report."""
    q = rep["quantity"]
    u = " K" if rep["unit"] == "K" else " %"  # ΔH errors are in % of the true ΔH
    lines = [f"### {NAMES[q]} ({q}) · {rep['n_readings']} values read"]
    g, m = rep["gross"], rep["missed"]
    gross_text = (f"**Gross misreads** (|error| > {g['limit']:g}{u}): {g['k']} of {g['n']} readings "
                  f"({share_text(g)})" if g["n"] else "**Gross misreads**: no readings")
    if rep["status"] != "fitted":
        lines += [f"- {rep['status']}.", f"- {gross_text}; missed {m['k']} of {m['n']}", ""]
        return lines
    d = digits_for(rep)
    b, r, w, diag, seen = rep["bias"], rep["range_90"], rep["worst_condition"], rep["diagnostics"], rep["observed"]
    lines += [
        f"- **Bias** (centre of the errors) {signed(b['mean'], d)}{u} [{plain(b['low'], d)}, {plain(b['high'], d)}] "
        f"(observed median {signed(seen['median_error'], d)}{u}, mean {signed(seen['mean_error'], d)}{u})",
        f"- **90 % of readings** on a new curve from this test design are expected within [{signed(r['low'], d)}, "
        f"{signed(r['high'], d)}]{u} of the right answer (observed 5th and 95th percentiles: {signed(seen['p5'], d)}, "
        f"{signed(seen['p95'], d)}{u})",
        f"- {gross_text}; missed {m['k']} of {m['n']}",
    ]
    if w:
        lines.append(worst_text(w, d, u, seen))
    lines += [
        f"- **Tested range**: {tested_range_text(rep['tested_range'])}",
        f"- *Diagnostics*: {diag['chains']} chains × {diag['draws']} draws, max R-hat {diag['max_rhat']:.3f}, "
        f"min bulk ESS {diag['min_ess_bulk']:.0f}, {diag['divergences']} divergences "
        f"(Gate 3 {'pass' if diag['gate3_pass'] else 'FAIL'})" + diagnostics_notes(diag),
        "",
    ]
    return lines


def script_of(reader):
    """The script of a code reader, where write_code_reader.py saves it: readers/code_<model>.py for prompt v1,
    readers/code_<model>_v2.py for prompt v2 (the dashes of the model id become underscores). None for a reader
    that is not a code reader."""
    if not reader.startswith("code-"):
        return None
    model, suffix = reader[len("code-"):], ""
    match = re.fullmatch(r"(.+)-(v\d+)", model)
    if match:
        model, suffix = match.group(1), "_" + match.group(2)
    return READERS / ("code_" + model.replace("-", "_") + suffix + ".py")


def missing_numpy_calls(script):
    """The NumPy functions that a script calls (as np.<name>(...)) and this NumPy does not have, e.g. ["trapz"]
    under NumPy 2.5. [] if there is none, or if there is no such script here. A call on a line that asks
    hasattr(np, ...) is a fallback for another NumPy, such as np.trapezoid(...) if hasattr(np, "trapezoid")
    else np.trapz(...), and is not counted (make_figure.py uses the same rule)."""
    if script is None or not Path(script).exists():
        return []
    lines = [line for line in Path(script).read_text(encoding="utf-8").splitlines()
             if not re.search(r"\bhasattr\(\s*np\s*,", line)]
    called = set(re.findall(r"\bnp\.([A-Za-z_]\w*)\s*\(", "\n".join(lines)))
    return sorted(name for name in called if not hasattr(np, name))


def usage_of(reader):
    """What write_code_reader.py saved about how a code reader's script was asked for (its .usage.json next to the
    script: prompt_version, and for prompt v2 versions_in_prompt), or {} if there is no such file."""
    script = script_of(reader)
    path = script.with_suffix(".usage.json") if script else None
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def versions_text(versions, names=("python", "numpy", "scipy")):
    """ "Python 3.14, NumPy 2.5 and SciPy 1.18" from {"python": "3.14", "numpy": "2.5", "scipy": "1.18"}."""
    shown = {"python": "Python", "numpy": "NumPy", "scipy": "SciPy"}
    parts = [f"{shown[name]} {versions[name]}" for name in names if versions.get(name)]
    if not parts:
        return ""
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def run_versions(replies):
    """The Python, NumPy and SciPy versions a code reader's script ran on, as run_code_reader.py recorded them in
    the *.run.json files of its replies folder: a list of the different version sets found ([] if none)."""
    found = []
    for path in sorted(Path(replies).glob("*.run.json")) if replies else []:
        run = json.loads(path.read_text(encoding="utf-8"))
        versions = {name: run[name] for name in ("python", "numpy", "scipy") if run.get(name)}
        if versions and versions not in found:
            found.append(versions)
    return found


def reader_note(reader, replies=None):
    """One line on what a reader is and what it sees, from its name ("" for a name it does not know). For a code
    reader whose script calls a NumPy function that this NumPy does not have, it also names that function, the
    likely reason for any reply that gave null. For a code reader whose prompt named the versions it would run
    with (prompt v2), it says what that prompt added, and the versions the script ran on, read from the
    *.run.json files in replies (that reader's replies folder), if given."""
    if reader == "standard":
        return ("A fixed, non-AI algorithm (read_standard.py) that reads the CSV numbers. It was designed knowing how "
                "the curves are made (generator.py), so it is a reference for what a careful standard analysis "
                "achieves, not a blind competitor.")
    if reader.startswith("code-"):
        model, version = reader[len("code-"):], "v1"
        match = re.fullmatch(r"(.+)-(v\d+)", model)
        if match:
            model, version = match.group(1), match.group(2)
        usage = usage_of(reader)
        version = str(usage.get("prompt_version", version)).removeprefix("code-")
        note = (f"One Python script, written once by {model} from prompt {version} (write_code_reader.py) and run "
                f"unchanged on every curve (run_code_reader.py). It reads the CSV numbers, not the image.")
        missing = missing_numpy_calls(script_of(reader))
        if missing:
            names = " and ".join(f"np.{name}" for name in missing)
            numpy_here = ".".join(np.__version__.split(".")[:2])
            note += (f" The script calls {names}, which NumPy {numpy_here} does not have. Where it calls "
                     f"{'it' if len(missing) == 1 else 'one'}, the script stops with an error, or gives null if it "
                     f"catches the error: the likely reason for any null reply.")
        named = usage.get("versions_in_prompt")
        if isinstance(named, dict) and versions_text(named):
            first = f"code-{model}"  # the same model's script from prompt v1
            failed = first != reader and bool(missing_numpy_calls(script_of(first)))
            asked_first = usage_of(first).get("timestamp_utc", "")
            if failed and asked_first and usage.get("timestamp_utc", "") < asked_first:
                failed = False  # the v1 script was asked for after this one, so this prompt did not follow it
            if failed:
                note += (f" Prompt {version} was asked after the v1 script failed: it adds only the line naming "
                         f"{versions_text(named)}, and does not mention the failure.")
            else:
                note += f" Prompt {version} adds only the line naming {versions_text(named)}."
            ran = run_versions(replies)
            if ran:
                note += (f" The script ran on {' or '.join(versions_text(v) for v in ran)}, as recorded in its run "
                         f"files.")
        return note
    if reader.startswith("claude-"):
        return "A vision model that sees only the PNG image of each curve (read_vision.py)."
    return ""


def reader_lines(reader, block):
    """The Markdown block of one reader."""
    checks = block["checks"]
    t, dr = checks["type_right"], checks["direction_right"]
    lines = [f"## {reader}", ""]
    note = block.get("about", reader_note(reader))
    if note:
        lines += [f"*{note}*", ""]
    if checks["with_event"]:
        type_text = (f"event type right {t['k']} of the {t['n']} replies that gave a type ({share_text(t)})"
                     if t["n"] else "no reply gave an event type")
        direction_text = (f"direction (endo/exo) right {dr['k']} of the {dr['n']} that gave a direction "
                          f"({share_text(dr)})" if dr["n"] else "no reply gave a direction")
        lines.append(f"{type_text[0].upper()}{type_text[1:]} · {direction_text}; {checks['curves']} curves read.")
        if checks["no_type_no_direction"]:
            by_event = [f"{c['k']} of the {c['n']} {event.replace('_', ' ')}"
                        for event, c in checks["no_type_no_direction_by_event"].items()]
            by_event = by_event[0] if len(by_event) == 1 else ", ".join(by_event[:-1]) + " and " + by_event[-1]
            lines.append(f"{checks['no_type_no_direction']} of the {checks['with_event']} replies with an event gave "
                         f"neither a type nor a direction (null): counted neither right nor wrong. By true event, "
                         f"null on {by_event} curves.")
        if checks["curves"] > checks["with_event"]:
            lines.append(f"{checks['curves'] - checks['with_event']} replies reported no event.")
        extra = block.get("onset_on_glass_transitions")
        if extra and extra["k"]:
            lines.append(f"Also gave an onset on {extra['k']} of the {extra['n']} glass transitions (not scored: a "
                         f"glass transition has no onset in this test).")
        lines.append("")
    else:
        lines += [f"No event reported on any of the {checks['curves']} curves.", ""]
    cells = checks["direction_wrong_by_convention_and_event"]
    events = ordered({e for row in cells.values() for e in row})
    lines += ["Direction wrong, by plot convention and true event (wrong of those that gave a direction):", "",
              "| convention | " + " | ".join(events) + " |", "|---" * (len(events) + 1) + "|"]
    for convention, row in cells.items():
        lines.append(f"| {convention} | " + " | ".join(f"{row[e]['k']} of {row[e]['n']}" for e in events) + " |")
    lines.append("")
    for q in QUANTITIES:
        if q in block["quantities"]:
            lines += report_lines(block["quantities"][q])
    lines += [SCOPE, ""]
    return lines


def gate3(readers):
    """Gate 3 counts every fit: each needs R-hat <= 1.01, bulk ESS >= 400 and 0 divergences."""
    fits = [(reader, q, c) for reader, block in readers.items() for q, c in block["quantities"].items()
            if c["status"] == "fitted"]
    passing = sum(c["diagnostics"]["gate3_pass"] for _, _, c in fits)
    longer = [f"{reader} {q} ({c['diagnostics']['draws']} draws)" for reader, q, c in fits
              if c["diagnostics"]["first_try"]]
    stuck = [f"{reader} {q}" for reader, q, c in fits if c["diagnostics"]["stuck"]]
    return {"fits": len(fits), "passing": passing, "passed": bool(fits) and passing == len(fits),
            "run_longer": longer, "stuck": stuck,
            "rule": f"every fit: R-hat <= {GATE_RHAT}, bulk ESS >= {GATE_ESS}, 0 divergences"}


def gate3_line(gate, quick):
    verdict = "PASS" if gate["passed"] else "FAIL"
    line = (f"Gate 3: {gate['passing']} of {gate['fits']} fits have R-hat <= {GATE_RHAT}, bulk ESS >= {GATE_ESS} "
            f"and 0 divergences: {verdict}")
    if gate["run_longer"]:
        line += f" (sampled again with more draws after missing it at first: {', '.join(gate['run_longer'])})"
    if gate["stuck"]:
        line += f" (a chain got stuck in: {', '.join(gate['stuck'])})"
    return line + (" (quick run: not a real Gate 3 check)" if quick else "")


def write_markdown(path, report):
    s = report["settings"]
    lines = ["# Error reports", ""]
    if s["quick"]:
        lines += ["**QUICK RUN (2 chains × 200 draws): these numbers are for testing only, not for reporting.**", ""]
    lines += [
        f"Test set: {s['curves']} synthetic DSC-style curves. One Bayesian error model per reader and quantity "
        f"(PyMC NUTS, {s['chains']} chains × {s['draws']} draws after {s['tune']} tuning steps, base seed "
        f"{s['seed']}; a fit that misses Gate 3 on R-hat or ESS alone is sampled again with more draws, and its "
        f"diagnostics line says so).",
        "",
        "How to read an error report:",
        "",
        "- Error = reading − right answer, so + means read too high. Temperatures in K; ΔH in % of the true ΔH.",
        "- **Bias** is the centre of the errors: where the error distribution is centred, averaged over the balanced "
        "test design. It is the model's own estimate, neither the median nor the mean of the readings, and it can "
        "differ from both (both are printed next to it): the model gives wild readings less weight than a plain "
        "average does, and counts readings more where they scatter less. Gross misreads are also counted separately.",
        "- Square brackets after a bias or an \"adds\" hold a 94 % credible interval: given these readings and the "
        "model, there is a 94 % probability that the true value lies inside (the narrowest such range; 94 % is "
        "ArviZ's default, not a special number).",
        "- The **90 % range** is different: it is where a new reading on a new curve from this test design is "
        "expected to land 9 times in 10. The model's error curve is symmetric around each knob combination, so for "
        "a reader whose errors lean to one side one end can be too far out and the other too close. The 5th and "
        "95th percentiles of the actual readings are printed next to it as a check.",
        "- Shares (event type, direction, gross misreads) give the measured k of n. Where the true share lies comes "
        "from Beta(1+k, 1+n−k), a flat prior updated with k of n, so 0 of n is not taken as a true share of 0 %.",
        "- **Typical |error|** is the model's median |error| for a new reading at one knob level, the other knobs "
        "mixed as in the test design: half the readings are expected closer to the right answer than this. Next to "
        "it, the median |error| of the actual readings is printed as a check. A knob level is named as the **worst "
        f"condition** only when the model gives it a probability above {CLEAR_PROB:.2f} of being the worst knob "
        f"level and the 94 % interval of what it adds is above 0, both with {CALL_MARGIN} sampling errors to spare. "
        "The sampling error (Monte Carlo standard error) says how much a number would move if the sampler were run "
        "again with another seed or on another computer: the model's numbers come from a finite set of draws. When "
        f"the rule holds or fails only within {CALL_MARGIN} sampling errors, the call is \"borderline\" and a rerun "
        "can change it; otherwise it says \"none clear\". A clear call with little to spare can also turn "
        f"borderline in a rerun; to turn none clear, a number would have to move by some {2 * CALL_MARGIN} "
        "sampling errors or more between the two runs. A clear or borderline call prints the probability, the "
        "lower end of what the level adds and their sampling errors with enough decimals to give the same call.",
        "- Readers differ in what they see: the vision models see only the PNG image; the code readers and standard "
        "read the CSV numbers. Each block below starts with what the reader is.",
        "", gate3_line(report["gate3"], s["quick"]) + ".", ""]
    for reader, block in report["readers"].items():
        lines += reader_lines(reader, block)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


DIAG_COLUMNS = ["reader", "quantity", "n_readings", "knobs_in_model", "chains", "draws", "tune", "seed",
                "first_run_draws", "first_run_max_rhat", "first_run_min_ess_bulk",
                "max_rhat", "max_rhat_variable", "min_ess_bulk", "min_ess_variable", "divergences",
                "stuck", "gate3_pass", "seconds"]


def write_diagnostics(path, readers):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DIAG_COLUMNS)
        writer.writeheader()
        for reader, block in readers.items():
            for q, c in block["quantities"].items():
                if c["status"] != "fitted":
                    continue
                d = c["diagnostics"]
                writer.writerow({"reader": reader, "quantity": q, "n_readings": c["n_readings"],
                                 "knobs_in_model": " ".join(c["knobs_in_model"]), "chains": d["chains"],
                                 "draws": d["draws"], "tune": d["tune"], "seed": d["seed"],
                                 "max_rhat": round(d["max_rhat"], 4), "max_rhat_variable": d["max_rhat_variable"],
                                 "min_ess_bulk": round(d["min_ess_bulk"]), "min_ess_variable": d["min_ess_variable"],
                                 "divergences": d["divergences"], "stuck": int(d["stuck"]),
                                 "gate3_pass": int(d["gate3_pass"]), "seconds": round(d["seconds"], 1),
                                 # filled only for a fit that was sampled again with more draws
                                 "first_run_draws": d["first_try"]["draws"] if d["first_try"] else "",
                                 "first_run_max_rhat": round(d["first_try"]["max_rhat"], 4) if d["first_try"] else "",
                                 "first_run_min_ess_bulk": round(d["first_try"]["min_ess_bulk"]) if d["first_try"] else ""})


def rounded(x):
    """Numbers in the JSON file rounded to 6 significant digits, so it stays readable."""
    if isinstance(x, float):
        return float(f"{x:.6g}") if np.isfinite(x) else None
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v) for v in x]
    return x


def print_table(readers):
    wide = max([len("reader")] + [len(reader) for reader in readers]) + 2  # e.g. code-claude-opus-5-5-v2 fits
    print(f"{'reader':<{wide}}{'quantity':<18}{'n':>4}  {'bias [94 % HDI]':<26}{'90 % within':<20}{'gross':>8}  "
          f"{'worst condition (adds)':<62}{'R-hat':>7}{'ESS':>7}{'div':>5}")
    for reader, block in readers.items():
        for q in QUANTITIES:
            c = block["quantities"].get(q)
            if c is None:
                continue
            if c["status"] != "fitted":
                print(f"{reader:<{wide}}{q:<18}{c['n_readings']:>4}  {c['status']}")
                continue
            d, unit = digits_for(c), c["unit"]
            b, r, w, g, dg = c["bias"], c["range_90"], c["worst_condition"], c["gross"], c["diagnostics"]
            bias = f"{tidy(b['mean'], d):+.{d}f} [{tidy(b['low'], d):.{d}f}, {tidy(b['high'], d):.{d}f}] {unit}"
            within = f"[{tidy(r['low'], d):+.{d}f}, {tidy(r['high'], d):+.{d}f}] {unit}"
            worst = "-"
            if w and w["call"] == "clear":
                worst = f"{w['knob']}={w['level']} ({tidy(w['adds']['mean'], d):+.{d}f})"
            elif w:  # borderline or none clear: the most likely level, its probability, the lower end of what it
                # adds and their MCSEs, as the .md file prints them for a borderline call (see call_numbers)
                p, mp, low, ml = (x.replace("± ", "±").replace("below ", "±<").replace("> ", ">").replace("< ", "<")
                                  for x in call_numbers(w, d + 1))
                worst = f"{w['call']} ({w['knob']}={w['level']} p{p}{mp} low{low}{ml})"
            print(f"{reader:<{wide}}{q:<18}{c['n_readings']:>4}  {bias:<26}{within:<20}{g['k']:>3} of {g['n']:<2}  "
                  f"{worst:<62}{dg['max_rhat']:>7.3f}{dg['min_ess_bulk']:>7.0f}{dg['divergences']:>5}")


def main(quick=False, seed=DEFAULT_SEED, split="test", data_dir=Path("data"), results_dir=Path("results")):
    """Fit every reader and quantity, write the three output files, print the table and Gate 3. Returns the report."""
    started = time.perf_counter()
    settings = QUICK if quick else {**FULL, "longer_draws": LONGER_DRAWS}
    if quick:
        print("QUICK RUN: 2 chains x 200 draws, to test the script only. These numbers are not for reporting.\n")
    data_dir, results_dir = Path(data_dir), Path(results_dir)
    errors = read_csv(results_dir / f"{split}_errors.csv")
    curves = read_csv(results_dir / f"{split}_curves.csv")
    design = read_csv(data_dir / split / "design.csv")
    truths = load_truths(data_dir / split)

    names = sorted({row["reader"] for row in errors} | {row["reader"] for row in curves})
    readers = {reader: {"about": reader_note(reader, data_dir / f"{split}_replies" / reader),
                        "checks": reader_checks([row for row in curves if row["reader"] == reader]),
                        "onset_on_glass_transitions": onsets_on_glass(errors, design, reader),
                        "quantities": {}} for reader in names}
    jobs = [(reader, q) for reader in names for q in QUANTITIES
            if any(row["reader"] == reader and row["quantity"] == q and row["status"] != "extra_value"
                   for row in errors)]
    for number, (reader, q) in enumerate(jobs, start=1):
        rows = [row for row in errors if row["reader"] == reader and row["quantity"] == q]
        n_ok = sum(row["status"] == "ok" for row in rows)
        print(f"[{number}/{len(jobs)}] {reader} {q}: {n_ok} values read", end="", flush=True)
        rep = fit_report(reader, q, rows, design, truths, settings, seed)
        if rep["status"] == "fitted":
            d = rep["diagnostics"]
            print(f", {d['seconds']:.0f} s, max R-hat {d['max_rhat']:.3f}, min bulk ESS {d['min_ess_bulk']:.0f}, "
                  f"{d['divergences']} divergences" + (f" ({d['draws']} draws per chain)" if d["first_try"] else "")
                  + (", a chain got stuck" if d["stuck"] else ""))
        else:
            print(f": {rep['status']}")
        readers[reader]["quantities"][q] = rep

    gate = gate3(readers)
    runtime = time.perf_counter() - started
    report = {
        "about": "Error reports from the Bayesian error model in error_model.py. Error = reading - right "
                 "answer; temperatures in K, ΔH in % of the true ΔH. bias = the centre of the errors: the centre "
                 "of the model's error distribution over the test design, which gives wild readings less weight "
                 "than a plain average does; it can differ from both the median and the mean of the readings "
                 "(observed), and gross misreads are also counted separately under gross. [low, high] pairs are "
                 "94 % credible intervals (highest density): given the readings and the model, a 94 % probability "
                 "that the true value is inside. range_90 is different: where a new reading on a new curve from "
                 "the test design is expected to land 9 times in 10. observed = the same numbers worked out from "
                 "the readings themselves, as a check. worst_condition.call is \"clear\" when prob_worst - m x "
                 "mcse_prob is above settings.clear_prob and adds.low - m x mcse_adds_low is above 0, with m = "
                 "settings.call_margin_mcse; \"borderline\" when it is not clear but both hold with + in place of "
                 "-, so that a rerun can change the call; \"none clear\" otherwise. worst_condition.clear is true "
                 "when call is \"clear\". mcse_prob and mcse_adds_low are Monte Carlo standard errors (sampling "
                 "errors): mcse_prob is ArviZ's MCSE of the mean of the indicator that the named level is the "
                 "worst, chains kept apart; mcse_adds_low is the standard deviation of adds.low over a block "
                 "bootstrap of the draws (settings.mcse_adds_low_bootstrap). The .md file prints "
                 "probabilities with 2 decimals, and the four numbers a clear or borderline call was judged with "
                 "(prob_worst, mcse_prob, adds.low, mcse_adds_low) with 3 or more, enough for the printed numbers "
                 "to give the printed call; a probability that would round to 1 is printed as > 0.99 (> 0.999 with "
                 "3 decimals), one that would round to 0 as < 0.01; this file keeps the raw values. "
                 "onset_on_glass_transitions = onsets given for glass transitions, which have no onset in "
                 "this test (not scored).",
        "settings": {"split": split, "quick": quick, "seed": seed, **settings, "target_accept": TARGET_ACCEPT,
                     "hdi_prob": HDI_PROB, "min_readings": MIN_READINGS, "clear_prob": CLEAR_PROB,
                     "call_margin_mcse": CALL_MARGIN,
                     "mcse_adds_low_bootstrap": {"block_draws": BOOT_BLOCK, "resamples": BOOT_RESAMPLES},
                     "resolution": RESOLUTION, "curves": len(design),
                     "python": platform.python_version(), "pymc": pm.__version__, "arviz": az.__version__,
                     "numpy": np.__version__},
        "gate3": gate, "runtime_s": runtime, "readers": readers}
    prefix = "quick_" if quick else ""
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"{prefix}error_reports.json"
    md_path = results_dir / f"{prefix}error_reports.md"
    csv_path = results_dir / f"{prefix}model_diagnostics.csv"
    json_path.write_text(json.dumps(rounded(report), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(md_path, report)
    write_diagnostics(csv_path, readers)

    print()
    print_table(readers)
    print(f"\nWrote {json_path}, {md_path} and {csv_path}. Run time {runtime:.0f} s.")
    if quick:
        print("QUICK RUN: these numbers are not for reporting. Run without --quick for the error reports.")
    print(gate3_line(gate, quick))
    return report


def parse_args(args):
    """(quick, seed) from the command line, or stop with the usage line."""
    quick, seed, rest = False, DEFAULT_SEED, list(args)
    while rest:
        arg = rest.pop(0)
        if arg == "test":
            continue
        if arg == "--quick":
            quick = True
        elif arg == "--seed" and rest and rest[0].isdigit():
            seed = int(rest.pop(0))
        else:
            raise SystemExit(USAGE)
    return quick, seed


if __name__ == "__main__":
    quick, seed = parse_args(sys.argv[1:])
    main(quick=quick, seed=seed)
