"""Empirical-Bayes hierarchical model for pass-through.

    rho_hat_i ~ N(rho_i, se_i^2)
    rho_i     = x_i' b + eta_episode(i) + u_i,   u_i ~ N(0, tau^2),   eta ~ N(0, sigma^2)

Coefficients have fixed Gaussian priors, so effects that the data barely identify (a cut,
seen in only one episode) stay near zero instead of being read off one country.
tau and sigma are set by penalised marginal likelihood, with a Gamma(2, 1/scale) penalty
that keeps them off zero (few episodes cannot rule out a small sigma).

With (b, eta) as one Gaussian vector the data covariance is diagonal, so the fit is closed
form given (tau, sigma).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

SE_FLOOR = 0.05
INTERCEPT_PRIOR = (0.5, 1.0)     # mean, sd on the rho scale
EFFECT_PRIOR_SD = 0.5
GAMMA_SCALE = 0.5


@dataclass
class Fit:
    columns: tuple[str, ...]
    episodes: tuple[str, ...]
    tau: float
    sigma: float
    theta: np.ndarray            # coefficients, then episode effects
    cov: np.ndarray

    @property
    def n_coef(self) -> int:
        return len(self.columns)

    def coefficients(self) -> dict[str, tuple[float, float]]:
        sd = np.sqrt(np.diag(self.cov))
        return {c: (float(self.theta[i]), float(sd[i])) for i, c in enumerate(self.columns)}

    def episode_effects(self) -> dict[str, float]:
        return {e: float(self.theta[self.n_coef + i]) for i, e in enumerate(self.episodes)}

    def _design(self, X: np.ndarray, episode: str | None) -> np.ndarray:
        Z = np.zeros((len(X), len(self.episodes)))
        if episode is not None:
            Z[:, self.episodes.index(episode)] = 1.0
        return np.hstack([X, Z])

    def predict(self, X: np.ndarray, episode: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Predictive (mean, sd) of the true rho. episode=None means a new, unseen episode."""
        W = self._design(X, episode)
        var = np.einsum("ij,jk,ik->i", W, self.cov, W) + self.tau ** 2
        if episode is None:
            var = var + self.sigma ** 2
        return W @ self.theta, np.sqrt(var)

    def sample_new_episode(self, X: np.ndarray, draws: int, seed: int) -> np.ndarray:
        """draws x items of rho for a new episode: items share b and one episode effect."""
        rng = np.random.default_rng(seed)
        W = self._design(X, None)
        theta = rng.multivariate_normal(self.theta, self.cov, size=draws, method="cholesky")
        eta = rng.normal(0, self.sigma, size=(draws, 1))
        u = rng.normal(0, self.tau, size=(draws, len(X)))
        return theta @ W.T + eta + u


def _prior(columns, n_episodes: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros(len(columns) + n_episodes)
    var = np.full(len(columns) + n_episodes, EFFECT_PRIOR_SD ** 2)
    mean[0] = INTERCEPT_PRIOR[0]
    var[0] = INTERCEPT_PRIOR[1] ** 2
    var[len(columns):] = sigma ** 2
    return mean, var


def _posterior(W, y, se2, tau, sigma, columns, n_ep):
    m, p = _prior(columns, n_ep, sigma)
    d = se2 + tau ** 2
    r = y - W @ m
    A = (W.T / d) @ W + np.diag(1 / p)
    c = cho_factor(A)
    b = (W.T / d) @ r
    shift = cho_solve(c, b)
    logdet = 2 * np.log(np.diag(c[0])).sum()
    ll = -0.5 * (np.log(d).sum() + np.log(p).sum() + logdet + (r ** 2 / d).sum() - b @ shift)
    return m + shift, c, ll


def fit(X: np.ndarray, y: np.ndarray, se: np.ndarray, episodes: list[str], columns: tuple[str, ...]) -> Fit:
    names = tuple(sorted(set(episodes)))
    Z = np.zeros((len(y), len(names)))
    for j, e in enumerate(names):
        Z[np.array(episodes) == e, j] = 1.0
    W = np.hstack([X, Z])
    se2 = np.maximum(se, SE_FLOOR) ** 2

    def objective(log_params):
        tau, sigma = np.exp(log_params)
        _, _, ll = _posterior(W, y, se2, tau, sigma, columns, len(names))
        penalty = np.log(tau) - tau / GAMMA_SCALE + np.log(sigma) - sigma / GAMMA_SCALE
        return -(ll + penalty)

    best = minimize(objective, np.log([0.4, 0.4]), method="Nelder-Mead",
                    options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 400})
    tau, sigma = map(float, np.exp(best.x))
    theta, c, _ = _posterior(W, y, se2, tau, sigma, columns, len(names))
    cov = cho_solve(c, np.eye(W.shape[1]))
    return Fit(columns, names, tau, sigma, theta, cov)
