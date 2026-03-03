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
    """

    # MAKE MORE REALISTIC
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

    # transaction cost parameters
    trnsCostFunc: Optional[callable] = lambda x: x*0.01 # linear transaction cost function (1% of trade size)

    data: Any = None

def makeEnvironment(spec: EnvSpec, seed: int = 0) -> hedgingEnvironment:
    """
        Factory to create a HedgingEnvironment from a specification.

        Single environment factory
        - spec.trnscostMode = "static": constant trnscost, no regime switching (P = I), trnscostLevels=[k,k,k]
        - spec.trnscostMode = "stochastic": regime switching (P = spec.PStoch or env default), trnscostLevels=spec.trnscostLevelsStoch
    """
    rng = np.random.default_rng(seed)

    env = hedgingEnvironment(S0=spec.S0, K=spec.K, T=spec.maturity, steps=spec.steps, r=spec.r, q=spec.q, mu=spec.mu, 
                             sigma=spec.sigma, sigmaValuation=spec.sigmaValuation, options=1, Hmin=spec.Hmin, 
                             Hmax=spec.Hmax, trnsCostFunc=spec.trnsCostFunc, rng=rng)
    return env
