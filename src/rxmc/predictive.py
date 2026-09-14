r"""Posterior predictives on a grid the data were never measured on.

:func:`rxmc.diagnostics.predictive_draws` draws the posterior predictive at the
*measured* points, where every term of the error model is defined, the reported
point-by-point errors included.  Forecasting at new ``x`` is a different
question, and this module answers it:

* :func:`grid_draws` — the general tool.  For each posterior row the model is
  evaluated on the grid, every selected covariance term is re-evaluated there
  from its own definition, and one correlated draw is taken from the sum.  Any
  term that is a function of the :class:`~rxmc.terms.TermContext` travels:
  inferred noise, :func:`~rxmc.terms.noise_fraction`,
  :func:`~rxmc.terms.model_error`, normalisation, offset and systematic modes,
  a parametric ``Term(fn, params, kind="matrix")`` and a Gaussian-process
  :func:`~rxmc.terms.kernel`.
* :func:`gp_predictive_draws` — the same draws for a kernel term, with the
  option of conditioning the discrepancy on the observed residuals.
* :func:`gp_posterior_predictive` — that conditioning on bare arrays, and
  :func:`predictive_band` — percentiles over any draws.

All three draw functions share one convention with
:func:`~rxmc.diagnostics.predictive_draws`: a percentile band of shape
``(len(levels), n_points)`` by default, the draws themselves with
``return_draws=True``.

What cannot go on a new grid
----------------------------
A term that is an *array*, one number per measured point, has no value at an
``x`` that was never measured: the reported statistical errors a constraint
builds by default, a fixed ``Term(array)``, a normalisation or offset whose
``magnitude=`` is per point, or a function that closes over the measured rows.
Drawing it on a grid would be inventing the error of a measurement nobody made,
so both grid functions raise and name the term.  The remedies are to draw at
the data with :func:`~rxmc.diagnostics.predictive_draws`, to say which terms a
draw carries with ``terms=`` (``[]`` or ``model_only=True`` for the model
alone), or to declare the experimental error as a term that is a function of
``x`` — :func:`~rxmc.terms.noise` with ``statistical=False`` — so it is defined
everywhere the model is.

Which terms a draw carries is a choice of object, not a detail.  The model
alone, the model plus a discrepancy, and the model plus discrepancy plus
experimental error answer different questions, and only the last predicts a
*measurement*.  A term that belongs to one experiment (its normalisation, say)
drawn on a grid means "a future measurement by that experiment".

Two predictives for a kernel
----------------------------
A kernel term declares a *mean-zero* discrepancy, so the likelihood is the
marginal ``y ~ N(ym(theta), Sigma(theta))`` and inference in ``theta`` has the
discrepancy integrated out.  The matching predictive draws correlated
departures from the inferred covariance around the model's own prediction:

.. math::

    y_* = y_m(x_*; \theta) + \delta, \qquad
    \delta \sim \mathcal N\big(0,\ K_{**}(\theta)\big)

which says where and by how much *the model* fails.  That is what
:func:`grid_draws` gives.  :func:`gp_predictive_draws` with ``conditioned=True``
instead conditions the discrepancy on the observed residuals,

.. math::

    \delta \mid r \sim \mathcal N\big(K_{*t}(K_{tt} + N)^{-1} r,\
    K_{**} - K_{*t}(K_{tt} + N)^{-1}K_{t*}\big),

which is data-driven regression on top of the model: it interpolates the
residuals rather than describing the model's error.

Kernels are duck-typed as in :func:`~rxmc.terms.kernel`: sklearn-style objects
with ``clone_with_theta``, ``__call__`` and ``diag``, ``theta`` in sklearn's
log space.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .diagnostics import _psd_factor, _rows
from .problem import Problem, _per_point
from .terms import KernelTerm, TermContext, as_2d

__all__ = [
    "grid_draws",
    "gp_predictive_draws",
    "gp_posterior_predictive",
    "predictive_band",
]


# ----------------------------------------------------------------------------
# GP conditioning
# ----------------------------------------------------------------------------


def _train_noise_matrix(train_noise_var, n) -> np.ndarray:
    if train_noise_var is None:
        return np.zeros((n, n))
    v = np.asarray(train_noise_var, dtype=float)
    if v.ndim == 0:
        return float(v) * np.eye(n)
    if v.ndim == 1:
        return np.diag(v)
    return v


def _condition(Ktt, Kst, r, N, jitter):
    """``(mean, v)`` with ``mean = Kst (Ktt + N)^-1 r`` and ``v = L^-1 Kst^T``.

    Callers form the posterior covariance ``Kss - v^T v`` or its diagonal
    ``diag(Kss) - sum(v**2, 0)``.
    """
    n = len(r)
    L = sla.cholesky(Ktt + N + jitter * np.eye(n), lower=True)
    alpha = sla.cho_solve((L, True), r)
    v = sla.solve_triangular(L, Kst.T, lower=True)
    return Kst @ alpha, v


def gp_posterior_predictive(
    kernel, theta, X_train, residuals, X_pred, *, train_noise_var=None, jitter=1e-10
):
    r"""Posterior mean and covariance of a GP discrepancy at ``X_pred``.

    Conditions a zero-mean GP with covariance ``kernel`` (rebuilt at ``theta``)
    on the observed ``residuals`` at ``X_train``:

    .. math::

        \bar f_* = K_{*t} (K_{tt} + N)^{-1} r, \qquad
        \mathrm{cov}_* = K_{**} - K_{*t} (K_{tt} + N)^{-1} K_{t*}

    Parameters
    ----------
    kernel : sklearn-style kernel
    theta : array_like
        Kernel hyperparameters in sklearn log-theta space.
    X_train, X_pred : array_like
        Training and prediction inputs (1-D promoted to a column).
    residuals : array_like, shape (n_train,)
        Observed minus model mean at ``X_train``.
    train_noise_var : float, array_like or matrix, optional
        Training-noise covariance ``N``: scalar, per-point vector, or full.
    jitter : float, optional
        Nugget added to the training covariance.

    Returns
    -------
    mean : np.ndarray, shape (n_pred,)
    cov : np.ndarray, shape (n_pred, n_pred)
    """
    k = kernel.clone_with_theta(np.asarray(theta, dtype=float))
    Xt, Xp = as_2d(X_train), as_2d(X_pred)
    r = np.asarray(residuals, dtype=float)
    Ktt, Kst, Kss = k(Xt), k(Xp, Xt), k(Xp)
    N = _train_noise_matrix(train_noise_var, len(r))
    mean, v = _condition(Ktt, Kst, r, N, jitter)
    return mean, Kss - v.T @ v


def predictive_band(draws, levels=(16, 50, 84)) -> np.ndarray:
    """Percentile band over ``draws`` of shape ``(n_draws, n_points)``.

    Returns ``(len(levels), n_points)``.
    """
    return np.percentile(np.asarray(draws, dtype=float), levels, axis=0)


# ----------------------------------------------------------------------------
# Draws on a new grid
# ----------------------------------------------------------------------------


def _locate(problem: Problem, term: KernelTerm):
    """The compiled constraint holding ``term`` and its covariance entry."""
    if not isinstance(term, KernelTerm):
        raise TypeError(
            "term must be the KernelTerm returned by rxmc.terms.kernel, got "
            f"{term!r}"
        )
    for c in problem.constraints:
        for e in c.covariance.entries:
            if e.term is term:
                return c, e
        if any(t is term for t in c.source.terms):
            raise ValueError(
                "the kernel term's support is fully masked in this problem: it has "
                "no training rows to condition on"
            )
    raise ValueError("the kernel term is not part of any constraint of the problem")


def _same_space(a, b) -> bool:
    """One space: the same object, or parameter-free wrappers of one callable
    (``space=np.log`` on each comparison wraps it anew each time)."""
    return a is b or (not a.params and not b.params and a.fn is b.fn)


def _one_space(constraint, rows):
    spaces = []
    for comp, o in zip(constraint.comparisons, constraint.offsets):
        if np.any((rows >= o.start) & (rows < o.stop)):
            if not any(_same_space(comp.space, s) for s in spaces):
                spaces.append(comp.space)
    if len(spaces) != 1:
        raise ValueError(
            "the comparisons a grid draw spans must share one comparison space "
            f"to predict in it; found {[s.name for s in spaces]}.  Pass "
            "comparison= to predict in the space of one of them"
        )
    return spaces[0]


def _term_name(t) -> str:
    """A term in a message: its kind, its parameters and the data it sits on."""
    names = ", ".join(p.name for p in t.params)
    what = f"{t.kind} term({names})" if names else f"fixed {t.kind} term"
    label = getattr(getattr(t.on, "data", None), "label", None)
    return f"{what} on {label!r}" if label else what


def _meets(c, t, rows) -> bool:
    """Whether term ``t`` has an active entry on any of ``rows``."""
    return any(
        e.term is t and np.intersect1d(e.rows, rows).size for e in c.covariance.entries
    )


def _grid_terms(c, terms, rows) -> list:
    """The terms a grid draw carries, or a ``ValueError`` saying why it cannot.

    ``terms=None`` takes every declared term with an active entry on ``rows``,
    and refuses when the constraint's reported statistical errors are among
    them.  An explicit selection is honoured as given.  Either way, a term
    whose ``fn`` is an array is defined only at the measured rows.
    """
    if terms is None:
        stat = [t for t in c.statistical_terms if _meets(c, t, rows)]
        if stat:
            raise ValueError(
                "the constraint's covariance includes the reported statistical "
                f"errors ({[_term_name(t) for t in stat]}), one number per measured "
                "point, and they have no value at an x that was never measured.  "
                "Draw at the measured points with rxmc.diagnostics.predictive_draws; "
                "or choose the terms a grid draw carries with terms=[...] "
                "(model_only=True for the model alone); or declare the experimental "
                "error as a term that is a function of x, e.g. rxmc.terms.noise "
                "with statistical=False, so it is defined everywhere the model is"
            )
        chosen = [t for t in c.source.terms if _meets(c, t, rows)]
    else:
        chosen = list(terms)
        for t in chosen:
            if not any(t is u for u in c.source.terms):
                raise ValueError(
                    f"{t!r} is not a term of this constraint; terms= selects among "
                    "the terms it was declared with (the reported statistical "
                    "errors cannot be drawn on a new grid at all)"
                )
    arrays = [t for t in chosen if not callable(t.fn)]
    if arrays:
        raise ValueError(
            f"{[_term_name(t) for t in arrays]}: an array-valued term has one value "
            "per measured point and no value at a new x.  Leave it out of terms=, "
            "or write it as a function of the TermContext (c.x, c.ym) so it can be "
            "evaluated anywhere"
        )
    return chosen


def _grid_piece(t, x_pred, mu, meta_p, values) -> np.ndarray:
    """One term's contribution to the covariance at the prediction points.

    ``y`` and ``ym`` are both the model's prediction there: at a point that was
    never measured the only meaning a term's ``c.y`` can carry is the model.
    """
    try:
        v = np.asarray(t.value(x_pred, mu, mu, *values, meta=meta_p), dtype=float)
    except Exception as err:
        raise ValueError(
            f"the {_term_name(t)} could not be evaluated at the prediction points "
            f"({err}); a term closing over the measured rows is defined only "
            "there.  Leave it out of terms=, or write it as a function of the "
            "TermContext"
        ) from err
    if t.kind == "diag":
        return np.diag(v**2)
    if t.kind == "mode":
        return np.outer(v, v)
    return v


def _draw_on_grid(
    problem,
    c,
    space,
    chosen,
    predictor,
    x_pred,
    samples,
    *,
    model_only,
    joint,
    physical,
    n_rep,
    rng,
    levels,
    return_draws,
    noise_std=0.0,
    condition=None,
):
    """The loop both grid functions share; ``condition(theta, mu)`` returns the
    GP posterior ``(mean, v)`` to shift the draw and subtract ``v^T v``."""
    if physical and space.inverse is None:
        raise ValueError(f"comparison space {space.name!r} has no inverse")
    x_pred = np.asarray(x_pred)
    n_pred = x_pred.shape[0]
    cols_pred = problem.columns(predictor.params)
    gathers = [(t, problem.columns(t.params)) for t in chosen]
    # the prediction points carry the metadata the predictor was bound with
    meta_p = (
        None
        if predictor.meta is None
        else {k: _per_point(v, n_pred) for k, v in predictor.meta.items()}
    )
    back = space.inverse if physical else (lambda y: y)
    n = samples.shape[0]

    if model_only:
        out = np.empty((n, n_pred))
        for i, theta in enumerate(samples):
            out[i] = back(space(predictor(*theta[cols_pred])))
        return out if return_draws else predictive_band(out, levels)

    out = np.empty((n * n_rep, n_pred))
    for i, theta in enumerate(samples):
        mu = space(predictor(*theta[cols_pred]))
        C = np.zeros((n_pred, n_pred))
        for t, g in gathers:
            C += _grid_piece(t, x_pred, mu, meta_p, tuple(theta[g]))
        f_mean = 0.0
        if condition is not None:
            f_mean, v = condition(theta, mu)
            C -= v.T @ v
        C[np.diag_indices(n_pred)] += float(noise_std) ** 2
        z = rng.standard_normal((n_rep, n_pred))
        if joint:
            z = z @ _psd_factor(0.5 * (C + C.T)).T
        else:
            z *= np.sqrt(np.clip(np.diag(C), 0.0, None))
        z *= c.likelihood.predictive_scale(rng, n_rep, *theta[c.like_gather])[:, None]
        out[i * n_rep : (i + 1) * n_rep] = back(mu + f_mean + z)
    return out if return_draws else predictive_band(out, levels)


def grid_draws(
    problem: Problem,
    predictor,
    x_pred,
    samples,
    constraint: int = 0,
    *,
    comparison=None,
    terms=None,
    model_only: bool = False,
    joint: bool = True,
    physical: bool = False,
    n_rep: int = 1,
    rng=None,
    levels=(16, 50, 84),
    return_draws: bool = False,
) -> np.ndarray:
    r"""Posterior-predictive band, or draws, on a grid the data were not measured on.

    For each posterior row ``theta`` the predictor gives the model on
    ``x_pred`` in comparison space, every selected term of the constraint is
    evaluated there into one covariance ``C(\theta)``, and ``n_rep`` correlated
    draws

    .. math::

        y_* = y_m(x_*; \theta) + s\,L(\theta) z, \qquad
        L L^T = C(\theta), \quad z \sim \mathcal N(0, I)

    are taken, with ``s`` the likelihood's
    :meth:`~rxmc.likelihood.Likelihood.predictive_scale` (1 for a Gaussian).
    This is the grid counterpart of
    :func:`~rxmc.diagnostics.predictive_draws`, and at the measured points,
    with every term a function, the two draw from the same distribution.

    A term is carried by evaluating its own definition at the new points, with
    the model's prediction standing in for ``c.y`` and the predictor's
    ``meta`` for ``c.meta``; so only terms that are functions of the
    :class:`~rxmc.terms.TermContext` can be carried.  The reported statistical
    errors and any array-valued term raise (module docstring).

    Because each draw is a whole correlated curve, a functional summary — a
    simultaneous band, an extremum, an integral over a region — is well posed
    on ``return_draws=True``.

    Parameters
    ----------
    problem : Problem
        The compiled problem the samples come from.
    predictor : Predictor
        The model bound to ``x_pred`` (``model.bind(x_pred, meta)``).  Its
        parameters must be columns of the problem; it need not be the model of
        a comparison (a bare physics model under a comparison's correction is
        fine).
    x_pred : array_like
        The prediction grid, in the raw coordinates the terms receive.
    samples : array_like, shape (n, problem.ndim)
        Rows in ``problem.names`` order: a posterior chain, or
        ``problem.sample_prior(n)`` for the prior predictive.
    constraint : int, optional
        Index into ``problem.constraints`` whose error model is drawn.
    comparison : Comparison, optional
        The comparison whose experiment the grid stands for: its comparison
        space is used, and ``terms=None`` takes only the terms that apply to it.
        Required when the constraint holds several comparisons and
        ``terms=None``.
    terms : sequence of Term, optional
        The declared terms a draw carries, matched by identity; ``[]`` carries
        none.  Default: every term (see ``comparison``), refusing if the
        reported statistical errors are among them.
    model_only : bool, optional
        The model's own curves, with no error model and no covariance built.
    joint : bool, optional
        Draw whole correlated curves.  ``False`` keeps only the diagonal of
        ``C`` and draws each point independently, for a very large grid.
    physical : bool, optional
        Map every draw back through the comparison space's inverse before
        taking percentiles (``space=log`` gives a band in physical units).
    n_rep : int, optional
        Draws per row (ignored when ``model_only``).
    rng : numpy.random.Generator or seed, optional
    levels : sequence of float, optional
        Percentiles of the band.
    return_draws : bool, optional
        Return the draws instead of the band.

    Returns
    -------
    np.ndarray
        ``(len(levels), len(x_pred))``; with ``return_draws`` the draws,
        ``(n * n_rep, len(x_pred))``, or ``(n, len(x_pred))`` when
        ``model_only``.
    """
    rng = np.random.default_rng(rng)
    samples = _rows(samples, problem.ndim)
    c = problem.constraints[constraint]
    if comparison is None:
        if terms is None and not model_only and len(c.comparisons) > 1:
            raise ValueError(
                f"the constraint holds {len(c.comparisons)} comparisons "
                f"({c.labels}); pass comparison= to say which experiment the grid "
                "stands for, or terms= to choose the terms a draw carries"
            )
        rows = np.arange(c.offsets[-1].stop)
    else:
        i = next((k for k, u in enumerate(c.comparisons) if u is comparison), None)
        if i is None:
            raise ValueError(
                f"{comparison!r} is not a comparison of constraint {constraint}"
            )
        rows = np.arange(c.offsets[i].start, c.offsets[i].stop)
    space = _one_space(c, rows)
    chosen = [] if model_only else _grid_terms(c, terms, rows)
    return _draw_on_grid(
        problem, c, space, chosen, predictor, x_pred, samples,
        model_only=model_only, joint=joint, physical=physical, n_rep=n_rep,
        rng=rng, levels=levels, return_draws=return_draws,
    )  # fmt: skip


def gp_predictive_draws(
    problem: Problem,
    term: KernelTerm,
    predictor,
    x_pred,
    samples,
    *,
    terms=None,
    conditioned: bool = False,
    joint: bool = True,
    noise_std: float = 0.0,
    train_noise_var=None,
    physical: bool = False,
    n_rep: int = 1,
    rng=None,
    levels=(16, 50, 84),
    return_draws: bool = False,
) -> np.ndarray:
    r"""Grid draws for a Gaussian-process discrepancy, optionally conditioned.

    With ``conditioned=False`` this is :func:`grid_draws` on the constraint
    and comparison space the kernel term lives in: correlated departures
    about the model's own prediction, which is the predictive that matches a
    mean-zero discrepancy (module docstring).  ``conditioned=True`` instead
    conditions the discrepancy on the residuals of the kernel's training rows,

    .. math::

        y_* = y_m(x_*; \theta) + \bar f_* + L(\theta) z, \qquad
        L L^T = C(\theta) - K_{*t}(K_{tt} + N)^{-1}K_{t*} + \sigma^2 I

    with ``\bar f_* = K_{*t}(K_{tt} + N)^{-1} r``: GP regression on top of
    the model, which interpolates the residuals.

    Parameters
    ----------
    problem : Problem
        The compiled problem the chain was sampled from.
    term : KernelTerm
        The discrepancy term, as declared in one of the problem's constraints.
    predictor : Predictor
        The model bound to ``x_pred`` (``model.bind(x_pred, meta)``); the
        ``meta`` it was bound with is what a callable amplitude's
        ``c.meta(key)`` reads at ``x_pred``.
    x_pred : array_like
        The prediction grid (raw coordinates; the term's ``coords`` transform
        is applied for the kernel).
    samples : array_like, shape (n, problem.ndim)
        Chain rows in ``problem.names`` order.
    terms : sequence of Term, optional
        The terms a draw carries, including ``term`` itself.  Default: every
        term of the constraint that meets the kernel's rows, under the rules
        of :func:`grid_draws`.  ``[term]`` alone gives model plus discrepancy;
        adding the experimental terms gives what data is compared against.
        A selected term applies to the whole of ``x_pred``.
    conditioned : bool, optional
        Condition the discrepancy on the observed residuals instead of drawing
        it mean-zero.  Default ``False``.
    joint : bool, optional
        Draw whole correlated curves; ``False`` draws each point independently.
    noise_std : float, optional
        Extra observation noise at the prediction points, in comparison space.
    train_noise_var : float, array_like or matrix, optional
        Conditioning noise on the training rows; ``conditioned=True`` only.
        Default: everything in the constraint's covariance at those rows
        *except* the kernel term itself, which is the exact GP regression noise
        for the declared error model.
    physical : bool, optional
        Map every draw back through the comparison space's inverse.
    n_rep : int, optional
        Draws per chain row.
    rng : numpy.random.Generator or seed, optional
    levels : sequence of float, optional
        Percentiles of the band.
    return_draws : bool, optional
        Return the ``(n * n_rep, len(x_pred))`` draws instead of the band.

    Returns
    -------
    np.ndarray
        ``(len(levels), len(x_pred))``, or the draws when ``return_draws``.
    """
    rng = np.random.default_rng(rng)
    samples = _rows(samples, problem.ndim)
    if train_noise_var is not None and not conditioned:
        raise ValueError(
            "train_noise_var is the noise the GP regression conditions on and "
            "applies only with conditioned=True"
        )
    c, entry = _locate(problem, term)
    rows = entry.rows
    keep = entry.keep  # the active rows of the term's support
    pos = entry.pos[keep]  # their positions in the active stack
    space = _one_space(c, rows)
    chosen = _grid_terms(c, terms, rows)
    if not any(t is term for t in chosen):
        raise ValueError(
            "terms= must include the kernel term: it is the discrepancy these "
            "draws predict"
        )
    cols_term = problem.columns(term.params)
    nk, n_fn = term.n_kernel, term._n_fn_params
    x_pred = np.asarray(x_pred)
    meta = None if c.meta is None else {k: v[rows] for k, v in c.meta.items()}
    meta_p = (
        None
        if predictor.meta is None
        else {k: _per_point(v, x_pred.shape[0]) for k, v in predictor.meta.items()}
    )

    def condition(theta, mu):
        values = tuple(theta[cols_term])
        ym = c.ym(theta)
        # the term's own block on its rows, exactly as the likelihood saw it
        K_full = term.value(
            c.x[rows], c.y[rows], ym[rows], *values,
            meta=meta, segments=entry.segments, labels=entry.labels,
        )  # fmt: skip
        Ktt = K_full[np.ix_(keep, keep)]
        r = (c.y - ym)[rows][keep]
        if train_noise_var is None:
            N = c.covariance.matrix(ym, theta)[np.ix_(pos, pos)] - Ktt
        else:
            N = _train_noise_matrix(train_noise_var, len(r))
        # kernel and amplitude at the training and prediction coordinates
        k = (
            term.kernel.clone_with_theta(np.asarray(values[:nk], dtype=float))
            if nk
            else term.kernel
        )
        ctx_t = term.context(
            c.x[rows], c.y[rows], ym[rows], meta, *values,
            segments=entry.segments, labels=entry.labels,
        )  # fmt: skip
        ctx_p = TermContext(
            x=term.coords(x_pred, *values[n_fn:]), y=mu, ym=mu, _meta=meta_p
        )
        a_t, a_p = _amplitudes(term, ctx_t, ctx_p, values[nk:n_fn])
        Kst = np.outer(a_p, a_t[keep]) * k(as_2d(ctx_p.x), as_2d(ctx_t.x)[keep])
        return _condition(Ktt, Kst, r, N, 0.0)

    return _draw_on_grid(
        problem, c, space, chosen, predictor, x_pred, samples,
        model_only=False, joint=joint, physical=physical, n_rep=n_rep, rng=rng,
        levels=levels, return_draws=return_draws, noise_std=noise_std,
        condition=condition if conditioned else None,
    )  # fmt: skip


def _amplitudes(term, ctx_t, ctx_p, values):
    a = term.amplitude
    if a is None:
        return np.ones(len(ctx_t)), np.ones(len(ctx_p))
    if callable(a):
        return (
            np.broadcast_to(np.asarray(a(ctx_t, *values), dtype=float), (len(ctx_t),)),
            np.broadcast_to(np.asarray(a(ctx_p, *values), dtype=float), (len(ctx_p),)),
        )
    a = np.asarray(a, dtype=float)
    if a.ndim == 0:
        return np.full(len(ctx_t), float(a)), np.full(len(ctx_p), float(a))
    raise ValueError(
        "a kernel amplitude given as an array is defined only on the training "
        "rows; use a callable amplitude(c, ...) to predict at new points"
    )
