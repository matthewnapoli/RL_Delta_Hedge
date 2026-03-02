import numpy as np
from hedgingEnvironment import hedgingEnvironment
from typing import Literal, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

DEFAULTREGIMETRANSITION = np.array([[0.93, 0.07, 0.00],   # abundant -> abundant/normal
                                    [0.04, 0.92, 0.04],   # normal   -> abundant/normal/stressed
                                    [0.00, 0.10, 0.90],   # stressed -> normal/stressed
                                   ], dtype=float)

#### ENVIRONMENT GENERATOR ####
@dataclass(frozen=False)
class EnvSpec:
    """
        Specification for building a hedging environment

        Attributes correspond to the environment parameters and choices of transaction cost regime
        Two modes are supported:
            - "paper" for constant transaction costs
            - "stochastic" for regime‑switching transaction costs
    """
    # toggle
    kappaMode: Literal["paper", "stochastic"] = "paper"

    # paper grid controls


    # MAKE REALISTIC
    maturity: timedelta = timedelta(days=250)
    rebalancingFrequency: timedelta = timedelta(days=5)

    one_day = timedelta(days=1)

    if maturity % one_day != timedelta(0):
        raise ValueError("maturity must be a multiple of 1 day")

    if rebalancingFrequency % one_day != timedelta(0):
        raise ValueError("rebalancingFrequency must be a multiple of 1 day")

    if 10 * rebalancingFrequency > maturity:
        raise ValueError("maturity must be at least 10 times rebalancing frequency to have enough steps for learning")
    
    steps = maturity // rebalancingFrequency

    # shared parameters
    S0: float = 100
    K: float = 100
    r: float = 0.0
    q: float = 0.0
    mu: float = 0.05
    sigma: float = 0.20
    sigmaValuation: Optional[float] = None  # if None, uses sigma (simulation sigma)
    Hmin: float = 0.0
    Hmax: float = 1.5

    # paper (constant kappa)
    kappaPaper: float = 0.01

    # stochastic kappa (regime-based)
    kappaLevelsStoch: Tuple[float, float, float] = (0.002, 0.01, 0.025)
    PStoch: Optional[np.ndarray] = None
    startFromStationary: bool = True
    burnin: int = 50
    startingRegime: int = 1

    data: Any = None

def makeEnvironment(spec: EnvSpec, seed: int = 0) -> hedgingEnvironment:
    """
        Factory to create a HedgingEnvironment from a specification.

        Single environment factory
        - spec.kappaMode = "paper": constant kappa, no regime switching (P = I), kappaLevels=[k,k,k]
        - spec.kappaMode = "stochastic": regime switching (P = spec.PStoch or env default), kappaLevels=spec.kappaLevelsStoch
    """
    rng = np.random.default_rng(seed)

    if spec.kappaMode == "paper":
        kappaLevels = [spec.kappaPaper, spec.kappaPaper, spec.kappaPaper]
        P = np.eye(3)
        startFromStationary = False
        burnin = 0
        startingRegime = 1 # regime is permanently "normal"; regime plots will show only regime 1 — expected
    elif spec.kappaMode == "stochastic":
        kappaLevels = list(spec.kappaLevelsStoch)
        P = spec.PStoch if spec.PStoch is not None else DEFAULTREGIMETRANSITION
        startFromStationary = spec.startFromStationary
        burnin = int(spec.burnin)
        startingRegime = int(spec.startingRegime)
    else:
        raise ValueError("kappaMode must be 'paper' or 'stochastic'")

    env = hedgingEnvironment(S0=spec.S0, K=spec.K, T=spec.maturity, steps=spec.steps, r=spec.r, q=spec.q, mu=spec.mu, sigma=spec.sigma,
                             sigmaValuation=spec.sigmaValuation, options=1, Hmin=spec.Hmin, Hmax=spec.Hmax,
                             kappaLevels=kappaLevels, P=P, startingRegime=startingRegime, rng=rng,
                             startFromStationary=startFromStationary, burnin=burnin)
    return env
