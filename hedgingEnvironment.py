import numpy as np
import pandas as pd
from datetime import timedelta
from helpers import blackScholesCallPriceDelta, nextPriceGBM
from typing import Optional, Callable

class hedgingEnvironment:

    def __init__(
        self,
        S0: float = 100.0,
        K: float = 100.0,
        expiration: float = 10 / 252,
        steps: float = 10.0,
        r: float = 0.0,
        q: float = 0.0,
        mu: float = 0.0,
        sigma: float = 0.2,
        sigmaValuation: Optional[float] = None,
        options: int = 1,
        Hmin: Optional[float] = None,
        Hmax: Optional[float] = None,
        trnsCostFunc: Optional[Callable[[float], float]] = None,
        deltaCutoff: float = 0.5,
        deltaPenaltyWeight: float = 0.05,
        deltaPenaltyPower: int = 2,
        deltaCostFunc: Optional[Callable[[float], float]] = None,
        rng=None,
    ):
        maturity: timedelta = timedelta(days=250)
        rebalancingFrequency: timedelta = timedelta(days=5)
        one_day = timedelta(days=1)

        if maturity % one_day != timedelta(0):
            raise ValueError("maturity must be a multiple of 1 day")

        if rebalancingFrequency % one_day != timedelta(0):
            raise ValueError("rebalancingFrequency must be a multiple of 1 day")

        if 10 * rebalancingFrequency > maturity:
            raise ValueError(
                "maturity must be at least 10 times rebalancing frequency to have enough steps for learning"
            )

        steps = maturity // rebalancingFrequency

        # price variables
        self.S0 = float(S0)
        self.K = float(K)
        self.expiration = float(expiration)
        self.N = int(steps)
        self.r = float(r)
        self.q = float(q)
        self.mu = float(mu)
        self.sigmaSimulation = float(sigma)
        self.sigmaValuation = float(sigma) if sigmaValuation is None else float(sigmaValuation)
        self.optionPrice0 = blackScholesCallPriceDelta(
            self.S0, self.K, self.r, self.q, self.sigmaValuation, self.expiration
        )[0]
        self.options = int(options)
        self.dt = self.expiration / self.N
        self.rng = rng if rng is not None else np.random.default_rng()

        # position constraints
        self.Hmin = -float(self.options) if Hmin is None else float(Hmin)
        self.Hmax = float(self.options) if Hmax is None else float(Hmax)

        # transaction cost
        self.defaultTransCost = 0.01
        self.transCostFunc = (
            trnsCostFunc
            if trnsCostFunc is not None
            else lambda moneyTraded: self.defaultTransCost * moneyTraded
        )

        # delta penalty
        self.deltaCutoff = float(deltaCutoff)
        self.deltaPenaltyWeight = float(deltaPenaltyWeight)
        self.deltaPenaltyPower = int(deltaPenaltyPower)

        if deltaCostFunc is not None:
            self.deltaPenaltyFunc = deltaCostFunc
        else:
            if self.deltaPenaltyPower == 1:
                self.deltaPenaltyFunc = lambda excessDelta: self.deltaPenaltyWeight * abs(excessDelta)
            elif self.deltaPenaltyPower == 2:
                self.deltaPenaltyFunc = lambda excessDelta: self.deltaPenaltyWeight * (excessDelta ** 2)
            else:
                raise ValueError("deltaPenaltyPower must be 1 or 2")

        # state variables
        self.i = 0
        self.S = 0.0
        self.H = 0.0
        self.V = 0.0
        self.L = 1

    def tau(self):
        """
        Returns remaining time-to-maturity tau_i = max(T - i*dt, 0).
        """
        return max(self.expiration - self.i * self.dt, 0.0)

    def stateVector(self):
        """
        Returns the current observable state as a NumPy array.
        [H, S, tau]
        """
        return np.array([self.H, self.S, self.tau()], dtype=np.float32)

    def reset(self):
        """
        Resets the episode to time i=0.
        Returns: (stateVector, initialReward)
        """
        self.i = 0
        self.S = float(np.asarray(self.S0).item())
        self.H = 0.0

        callPrice, _ = blackScholesCallPriceDelta(
            self.S, self.K, self.r, self.q, self.sigmaValuation, self.tau()
        )
        self.V = -self.options * callPrice
        initialReward = 0.0
        return self.stateVector(), float(initialReward)

    def step(self, Hnext):
        """
        One environment step.
        """
        if self.Hmin is not None and self.Hmax is not None:
            Hnext = float(np.clip(Hnext, self.Hmin, self.Hmax))

        # 1) simulate next price
        Snext = nextPriceGBM(self.S, self.mu, self.sigmaSimulation, self.dt, self.rng)
        Snext = float(np.asarray(Snext).item())
        spotExec = Snext
        tauNext = max(self.expiration - (self.i + 1) * self.dt, 0.0)

        # 2) transaction cost from adjusting hedge
        tradedH = Hnext - self.H
        moneyTraded = abs(spotExec * tradedH)
        transactionCost = float(self.transCostFunc(moneyTraded))

        # 3) revalue option at next time step
        nextPrice, nextDelta = blackScholesCallPriceDelta(
            Snext, self.K, self.r, self.q, self.sigmaValuation, tauNext
        )
        Vnext = -self.options * nextPrice

        # 4) delta penalty: only penalize excess outside the band
        netDelta = self.options * nextDelta - Hnext
        excessDelta = max(abs(netDelta) - self.deltaCutoff * self.options, 0.0)
        deltaCost = float(self.deltaPenaltyFunc(excessDelta))

        # 5) reward
        reward = (Vnext - self.V) + self.H * (Snext - self.S) - transactionCost - deltaCost

        self.i += 1
        self.S = float(Snext)
        self.H = float(Hnext)
        self.V = float(Vnext)

        done = self.i >= self.N
        terminalTC = 0.0

        if done:
            terminalTC = float(self.transCostFunc(abs(spotExec * self.H)))
            reward -= terminalTC

        info = {
            "Reward": float(reward),
            "TransactionCost": float(transactionCost),
            "DeltaPenalty": float(deltaCost),
            "NetDelta": float(netDelta),
            "ExcessDelta": float(excessDelta),
            "TerminalTransactionCost": float(terminalTC),
            "TotalTransactionCost": float(transactionCost + terminalTC),
            "SpotExec": float(spotExec),
            "DeltaTraded": float(tradedH),
            "MoneyTraded": float(moneyTraded),
        }
        return self.stateVector(), float(reward), done, info

    def seed(self, seed=None):
        """
        Sets/overwrites the environment RNG.
        """
        if seed is None:
            return
        if isinstance(seed, (int, np.integer)):
            self.rng = np.random.default_rng(int(seed))
            return
        if isinstance(seed, np.random.Generator):
            self.rng = seed
            return
        raise TypeError("seed must be None, an int, or a np.random.Generator.")


### BASELINE TRADING POLICIES ###
def unpackState(state):
    """
    State is [H, S, tau].
    Returns: (H, S, tau)
    """
    H, S, tau = float(state[0]), float(state[1]), float(state[2])
    return H, S, tau


def policyDeltaHedge(env, state):
    """
    Practitioner delta hedge: Htarget = options * DeltaCall(S, tau)
    """
    _, S, tau = unpackState(state)
    _, delta = blackScholesCallPriceDelta(S, env.K, env.r, env.q, env.sigmaValuation, tau)
    Htarget = env.options * delta
    return float(np.clip(Htarget, env.Hmin, env.Hmax))


def policyDeltaHedgeWithBand(band=0.25):
    """
    Delta hedge with an inaction band.
    """
    def policy(env, state):
        Hcurr, S, tau = unpackState(state)
        _, delta = blackScholesCallPriceDelta(S, env.K, env.r, env.q, env.sigmaValuation, tau)
        Htarget = env.options * delta
        if abs(Htarget - Hcurr) <= band:
            return float(Hcurr)
        return float(np.clip(Htarget, env.Hmin, env.Hmax))

    return policy


def policyNoTrading(env, state):
    """
    No-trade policy: keep the current hedge position.
    """
    return float(state[0])


def runEpisode(env, policyFunction, seed=0, reward_scaling_factor=1):
    """
    Runs one episode and returns summary bookkeeping.
    """
    env.rng = np.random.default_rng(seed)
    state, r0 = env.reset()

    totalReward = float(r0)
    totalTransactionCost = 0.0
    turnover = 0.0

    done = False
    while not done:
        Hprev = float(state[0])
        Hnext = float(policyFunction(env, state))

        nextState, reward, done, info = env.step(Hnext)
        totalReward += float(reward) * reward_scaling_factor

        tc = float(info.get("TotalTransactionCost", info.get("TransactionCost", 0.0)))
        totalTransactionCost += tc

        Hused = float(nextState[0])
        turnover += abs(Hused - Hprev)

        state = nextState

    terminalTurnover = abs(float(state[0]))
    turnover += terminalTurnover

    return {
        "totalReward": float(totalReward),
        "terminalTurnover": float(terminalTurnover),
        "turnover": float(turnover),
        "totalTransactionCost": float(totalTransactionCost),
    }


### POLICY EVALUATION ###
def evaluatePolicy(env, policyFunction, episodes=300, c=1.5, baseSeed=0):
    """
    Evaluate a policy over many episodes and return summary statistics.
    """
    rows = []

    for ep in range(episodes):
        env.rng = np.random.default_rng(baseSeed + ep)
        out = runEpisode(env, policyFunction, seed=baseSeed + ep)
        rows.append(out)

    df = pd.DataFrame(rows)
    C = -df["totalReward"].to_numpy(dtype=float)
    meanC = float(np.mean(C))
    stdC = float(np.std(C, ddof=1)) if len(C) > 1 else 0.0
    risk_adj_Y = meanC + c * stdC

    summary = {
        "episodes": int(episodes),
        "meanCostPct": meanC,
        "stdCostPct": stdC,
        "Y(mean+c*std)": risk_adj_Y,
        "meanTransactionCost": float(df["totalTransactionCost"].mean()),
        "meanTurnover": float(df["turnover"].mean()),
    }
    return summary, df


def preprocessState(env, state):
    """
    Normalize the raw state for neural network input.

    Raw state: [H, S, tau]
    Return normalized features:
        H -> scaled by max(|Hmin|,|Hmax|)
        S -> x = log(S/K)
        tau -> scaled time-to-maturity: tau / expiration
    """
    H, S, tau = float(state[0]), float(state[1]), float(state[2])
    hedgeScale = max(abs(env.Hmin), abs(env.Hmax), 1e-8)
    normalizedH = H / hedgeScale
    x = np.log(max(S, 1e-12) / max(env.K, 1e-12))
    normalizedTau = tau / max(env.expiration, 1e-12)
    return np.array([normalizedH, x, normalizedTau], dtype=np.float32)


def scaleActionToHedge(env, u):
    """
    Maps actor's scaled action u in [-1,1] -> hedge position H in [Hmin, Hmax]
    """
    u = float(np.clip(u, -1, 1))
    H = 0.5 * (u + 1.0) * (env.Hmax - env.Hmin) + env.Hmin
    return float(H)


def scaleHedgeToAction(env, H):
    """
    Maps hedge position H in [Hmin, Hmax] -> scaled action u in [-1,1]
    """
    H = float(np.clip(H, env.Hmin, env.Hmax))
    u = 2.0 * (H - env.Hmin) / (env.Hmax - env.Hmin) - 1.0
    return float(u)