from math import log, sqrt, exp
import numpy as np
import warnings
from scipy.stats import norm
from typing import Union, Optional

n = norm.pdf
N = norm.cdf

### BLACK SCHOLES HELPERS ###
def blackScholesCallPriceDelta(S: float, K: float, r: float, q: float, sigma: float, tau: float) -> tuple[float, float]:
    """
        Calculates Black Scholes price and delta.

        S: spot price
        K: strike price
        r: risk-free interest rate
        q: dividend yield (continuously compounded)
        sigma: volatility (annualized SD(asset returns))
        tau: time to maturity
    """
    if tau <= 0:
        price = max(S - K, 0)
        delta = 1.0 if S > K else (0.5 if S == K else 0)
        return float(price), float(delta)

    if sigma <= 0:
        forward = S * exp((r - q) * tau)
        price = exp(-r * tau) * max(forward - K, 0)
        delta = exp(-q * tau) * (1.0 if forward > K else (0.5 if forward == K else 0))
        return float(price), float(delta)

    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt(tau))
    d2 = d1 - (sigma * sqrt(tau))
    price = S * exp(-q * tau) * N(d1) - K * exp(-r * tau) * N(d2)
    delta = exp(-q * tau) * N(d1)
    return price, delta

def nextPriceGBM(Scurr: Union[float, np.ndarray], mu: float, sigma: float, dt: float, rng: np.random.Generator) -> Union[float, np.ndarray]:
    """
        Generates and returns the next price under Geometric Brownian Motion (GBM) dynamics, given current price and time step.

        Scurr: current spot price
        mu: mean return of the stock over time (drift)
        sigma: volatility (annualized SD(asset returns))
        dt: time step
        rng: random number generator
    """
    Z = rng.standard_normal(size=np.shape(Scurr))
    Snext = Scurr * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return Snext

def brownianPricePathsSimulation(paths: int, periods: int, S0: float, mu: float, sigma: float, dt: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
        Monte Carlo simulation of underyling's price paths.

        paths: number of distinct paths to simulate
        periods: number of periods of simulation
        S0: starting underlying price for all paths
        mu: mean return of the stock over time (drift)
        sigma: volatility (annualized SD(asset returns))
        dt: time step
    """
    if rng is None: 
        rng = np.random.default_rng()

    pricePaths = np.zeros((paths, periods))
    pricePaths[:, 0] = S0
    for t in range(periods - 1):
        pricePaths[:, t+1] = nextPriceGBM(pricePaths[:, t], mu, sigma, dt, rng)
    return pricePaths