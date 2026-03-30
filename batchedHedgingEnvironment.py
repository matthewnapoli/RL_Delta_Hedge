import numpy as np
import pandas as pd
from typing import Optional, Callable
from helpers import blackScholesCallPriceDelta


class batchedHedgingEnvironment:
    def __init__(
        self,
        batchSize: int,
        S0: float = 100.0,
        K: float = 100.0,
        expiration: float = 10 / 252,
        steps: int = 10,
        r: float = 0.0,
        q: float = 0.0,
        mu: float = 0.0,
        sigma: float = 0.2,
        sigmaValuation: Optional[float] = None,
        options: int = 1,
        Hmin: Optional[float] = None,
        Hmax: Optional[float] = None,
        trnsCostFunc: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        deltaCutoff: float = 0.5,
        deltaPenaltyWeight: float = 0.05,
        deltaPenaltyPower: int = 2,
        deltaCostFunc: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        rng=None,
    ):
        """
        Initializes a batched delta-hedging environment that runs multiple episodes in parallel.

        Args:
            batchSize: Number of parallel episodes.
            S0: Initial spot price.
            K: Option strike price.
            expiration: Option maturity in years.
            steps: Number of hedge decisions in each episode.
            r: Continuously compounded risk-free rate.
            q: Continuously compounded dividend yield.
            mu: Drift used in spot simulation.
            sigma: Volatility used in spot simulation.
            sigmaValuation: Volatility used for option pricing; if None, uses sigma.
            options: Number of options being hedged per episode.
            Hmin: Minimum allowable hedge position; if None, defaults to -options.
            Hmax: Maximum allowable hedge position; if None, defaults to options.
            trnsCostFunc: Function mapping traded notional array to transaction-cost array.
            deltaCutoff: No-penalty band for net delta, expressed per option.
            deltaPenaltyWeight: Scale applied to the delta penalty.
            deltaPenaltyPower: Penalty power; 1 for linear, 2 for quadratic.
            deltaCostFunc: Custom function mapping excess net delta array to penalty array.
            rng: Random number generator used for price simulation.
        """
        self.batchSize = int(batchSize)

        self.S0 = float(S0)
        self.K = float(K)
        self.expiration = float(expiration)
        self.N = int(steps)
        self.r = float(r)
        self.q = float(q)
        self.mu = float(mu)
        self.sigmaSimulation = float(sigma)
        self.sigmaValuation = float(sigma) if sigmaValuation is None else float(sigmaValuation)
        self.options = int(options)
        self.dt = self.expiration / self.N
        self.rng = rng if rng is not None else np.random.default_rng()

        self.Hmin = -float(self.options) if Hmin is None else float(Hmin)
        self.Hmax = float(self.options) if Hmax is None else float(Hmax)

        self.defaultTransCost = 0.01
        self.transCostFunc = (
            trnsCostFunc
            if trnsCostFunc is not None
            else lambda moneyTraded: self.defaultTransCost * np.asarray(moneyTraded, dtype=float)
        )

        self.deltaCutoff = float(deltaCutoff)
        self.deltaPenaltyWeight = float(deltaPenaltyWeight)
        self.deltaPenaltyPower = int(deltaPenaltyPower)

        if deltaCostFunc is not None:
            self.deltaPenaltyFunc = deltaCostFunc
        else:
            if self.deltaPenaltyPower == 1:
                self.deltaPenaltyFunc = lambda excessDelta: self.deltaPenaltyWeight * np.abs(np.asarray(excessDelta, dtype=float))
            elif self.deltaPenaltyPower == 2:
                self.deltaPenaltyFunc = lambda excessDelta: self.deltaPenaltyWeight * (np.asarray(excessDelta, dtype=float) ** 2)
            else:
                raise ValueError("deltaPenaltyPower must be 1 or 2")

        self.i = 0
        self.S = np.zeros(self.batchSize, dtype=float)
        self.H = np.zeros(self.batchSize, dtype=float)
        self.V = np.zeros(self.batchSize, dtype=float)

    def _bs_price_delta_batch(self, S, tau):
        """
        Computes Black-Scholes prices and deltas elementwise for batched inputs.

        Args:
            S: Spot-price array.
            tau: Scalar remaining maturity or array of remaining maturities.

        Returns:
            price: Array of option prices.
            delta: Array of option deltas.
        """
        S = np.asarray(S, dtype=float)
        if isinstance(tau, (int, float, np.integer, np.floating)):
            tau_arr = np.full_like(S, float(tau), dtype=float)
        else:
            tau_arr = np.asarray(tau, dtype=float)

        price = np.empty_like(S, dtype=float)
        delta = np.empty_like(S, dtype=float)

        for idx in np.ndindex(S.shape):
            p, d = blackScholesCallPriceDelta(
                float(S[idx]),
                self.K,
                self.r,
                self.q,
                self.sigmaValuation,
                float(tau_arr[idx]),
            )
            price[idx] = float(p)
            delta[idx] = float(d)

        return price, delta

    def tau(self):
        """
        Returns the remaining time to maturity at the current step.

        Returns:
            Remaining time to maturity in years.
        """
        return max(self.expiration - self.i * self.dt, 0.0)

    def stateVector(self):
        """
        Returns the current observable batched state.

        Returns:
            NumPy array of shape (batchSize, 3) with columns [H, S, tau].
        """
        tau_col = np.full(self.batchSize, self.tau(), dtype=np.float32)
        return np.column_stack(
            [
                self.H.astype(np.float32),
                self.S.astype(np.float32),
                tau_col,
            ]
        )

    def reset(self):
        """
        Resets all parallel episodes to time 0.

        Returns:
            stateVector: Initial batched state of shape (batchSize, 3).
            initialReward: Zero array of shape (batchSize,).
        """
        self.i = 0
        self.S = np.full(self.batchSize, self.S0, dtype=float)
        self.H = np.zeros(self.batchSize, dtype=float)

        callPrice, _ = self._bs_price_delta_batch(self.S, self.tau())
        self.V = -self.options * callPrice

        initialReward = np.zeros(self.batchSize, dtype=float)
        return self.stateVector(), initialReward

    def step(self, Hnext):
        """
        Advances all parallel episodes by one step using a batched hedge vector.

        Args:
            Hnext: Array of next hedge positions of shape (batchSize,).

        Returns:
            nextState: Next batched state of shape (batchSize, 3).
            reward: Reward array of shape (batchSize,).
            done: Boolean indicating whether all episodes reached maturity.
            info: Dictionary of batched diagnostics.
        """
        Hnext = np.asarray(Hnext, dtype=float)
        if Hnext.shape != (self.batchSize,):
            raise ValueError(f"Hnext must have shape ({self.batchSize},)")

        Hnext = np.clip(Hnext, self.Hmin, self.Hmax)

        z = self.rng.standard_normal(self.batchSize)
        Snext = self.S * np.exp(
            (self.mu - 0.5 * self.sigmaSimulation**2) * self.dt
            + self.sigmaSimulation * np.sqrt(self.dt) * z
        )
        spotExec = Snext
        tauNext = max(self.expiration - (self.i + 1) * self.dt, 0.0)

        tradedH = Hnext - self.H
        moneyTraded = np.abs(spotExec * tradedH)
        transactionCost = np.asarray(self.transCostFunc(moneyTraded), dtype=float)

        nextPrice, nextDelta = self._bs_price_delta_batch(Snext, tauNext)
        Vnext = -self.options * nextPrice

        netDelta = self.options * nextDelta - Hnext
        excessDelta = np.maximum(np.abs(netDelta) - self.deltaCutoff * self.options, 0.0)
        deltaCost = np.asarray(self.deltaPenaltyFunc(excessDelta), dtype=float)

        reward = (Vnext - self.V) + self.H * (Snext - self.S) - transactionCost - deltaCost

        self.i += 1
        self.S = Snext
        self.H = Hnext
        self.V = Vnext

        done = self.i >= self.N
        terminalTC = np.zeros(self.batchSize, dtype=float)

        if done:
            terminalTC = np.asarray(self.transCostFunc(np.abs(spotExec * self.H)), dtype=float)
            reward = reward - terminalTC

        info = {
            "Reward": reward.copy(),
            "TransactionCost": transactionCost.copy(),
            "DeltaPenalty": deltaCost.copy(),
            "NetDelta": netDelta.copy(),
            "ExcessDelta": excessDelta.copy(),
            "TerminalTransactionCost": terminalTC.copy(),
            "TotalTransactionCost": (transactionCost + terminalTC).copy(),
            "SpotExec": spotExec.copy(),
            "DeltaTraded": tradedH.copy(),
            "MoneyTraded": moneyTraded.copy(),
        }
        return self.stateVector(), reward.copy(), done, info

    def seed(self, seed=None):
        """
        Sets or replaces the random number generator used by the environment.

        Args:
            seed: Integer seed or NumPy Generator; if None, leaves the RNG unchanged.

        Returns:
            None.
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


def preprocessBatchedState(env, state):
    """
    Normalizes a batched raw state matrix for neural-network input.

    Args:
        env: Batched hedging environment containing scaling parameters.
        state: Raw state array of shape (batchSize, 3) with columns [H, S, tau].

    Returns:
        NumPy array of shape (batchSize, 3) with columns
        [normalizedH, logMoneyness, normalizedTau].
    """
    state = np.asarray(state, dtype=float)
    H = state[:, 0]
    S = state[:, 1]
    tau = state[:, 2]

    hedgeScale = max(abs(env.Hmin), abs(env.Hmax), 1e-8)
    normalizedH = H / hedgeScale
    x = np.log(np.maximum(S, 1e-12) / max(env.K, 1e-12))
    normalizedTau = tau / max(env.expiration, 1e-12)

    return np.column_stack([normalizedH, x, normalizedTau]).astype(np.float32)


def scaleActionsToHedges(env, u):
    """
    Maps batched scaled actions in [-1, 1] to hedge positions in [Hmin, Hmax].

    Args:
        env: Batched hedging environment containing hedge bounds.
        u: Array of scaled actions.

    Returns:
        Array of hedge positions.
    """
    u = np.clip(np.asarray(u, dtype=float), -1.0, 1.0)
    H = 0.5 * (u + 1.0) * (env.Hmax - env.Hmin) + env.Hmin
    return H.astype(float)


def scaleHedgesToActions(env, H):
    """
    Maps batched hedge positions in [Hmin, Hmax] to scaled actions in [-1, 1].

    Args:
        env: Batched hedging environment containing hedge bounds.
        H: Array of hedge positions.

    Returns:
        Array of scaled actions.
    """
    H = np.clip(np.asarray(H, dtype=float), env.Hmin, env.Hmax)
    u = 2.0 * (H - env.Hmin) / (env.Hmax - env.Hmin) - 1.0
    return u.astype(float)


def policyNoTradingBatch(env, state):
    """
    Returns a batched no-trade policy that keeps current hedge positions unchanged.

    Args:
        env: Batched hedging environment.
        state: Batched state array of shape (batchSize, 3).

    Returns:
        Array of hedge positions of shape (batchSize,).
    """
    state = np.asarray(state, dtype=float)
    return state[:, 0].astype(float)


def policyDeltaHedgeBatch(env, state):
    """
    Computes the Black-Scholes delta hedge target for each state in a batched state matrix.

    Args:
        env: Batched hedging environment containing contract and model parameters.
        state: Batched state array of shape (batchSize, 3).

    Returns:
        Array of clipped target hedge positions of shape (batchSize,).
    """
    state = np.asarray(state, dtype=float)
    S = state[:, 1]
    tau = state[:, 2]
    _, delta = env._bs_price_delta_batch(S, tau)
    Htarget = env.options * delta
    return np.clip(Htarget, env.Hmin, env.Hmax).astype(float)


def policyDeltaHedgeWithBandBatch(band=0.25):
    """
    Builds a batched delta-hedging policy with an inaction band around the target hedge.

    Args:
        band: Maximum allowed deviation from the target hedge before rebalancing.

    Returns:
        Policy function mapping (env, state) to an array of hedge positions.
    """
    def policy(env, state):
        state = np.asarray(state, dtype=float)
        Hcurr = state[:, 0]
        S = state[:, 1]
        tau = state[:, 2]

        _, delta = env._bs_price_delta_batch(S, tau)
        Htarget = env.options * delta
        keep_mask = np.abs(Htarget - Hcurr) <= band

        Hnext = np.where(keep_mask, Hcurr, Htarget)
        return np.clip(Hnext, env.Hmin, env.Hmax).astype(float)

    return policy


def runBatchedEpisode(env, policyFunction, seed=0, reward_scaling_factor=1.0):
    """
    Runs one full batched episode and records per-path summary metrics.

    Args:
        env: Batched hedging environment.
        policyFunction: Function mapping (env, state) to next hedge vector.
        seed: Random seed used to initialize the episode RNG.
        reward_scaling_factor: Multiplier applied to each step reward before aggregation.

    Returns:
        Dictionary containing per-path arrays for total reward, terminal turnover,
        total turnover, and total transaction cost.
    """
    env.rng = np.random.default_rng(seed)
    state, r0 = env.reset()

    totalReward = np.asarray(r0, dtype=float)
    totalTransactionCost = np.zeros(env.batchSize, dtype=float)
    turnover = np.zeros(env.batchSize, dtype=float)

    done = False
    while not done:
        Hprev = state[:, 0].astype(float)
        Hnext = np.asarray(policyFunction(env, state), dtype=float)

        nextState, reward, done, info = env.step(Hnext)
        totalReward += np.asarray(reward, dtype=float) * reward_scaling_factor
        totalTransactionCost += np.asarray(
            info.get("TotalTransactionCost", info.get("TransactionCost", 0.0)),
            dtype=float,
        )
        turnover += np.abs(nextState[:, 0].astype(float) - Hprev)

        state = nextState

    terminalTurnover = np.abs(state[:, 0].astype(float))
    turnover += terminalTurnover

    return {
        "totalReward": totalReward.astype(float),
        "terminalTurnover": terminalTurnover.astype(float),
        "turnover": turnover.astype(float),
        "totalTransactionCost": totalTransactionCost.astype(float),
    }


def evaluateBatchedPolicy(env, policyFunction, c=1.5, baseSeed=0):
    """
    Evaluates a batched policy run and computes summary performance statistics.

    Args:
        env: Batched hedging environment whose batchSize determines the number of episodes.
        policyFunction: Function mapping (env, state) to next hedge vector.
        c: Risk-aversion coefficient used in the Y = mean + c * std metric.
        baseSeed: Seed used for the batched episode.

    Returns:
        summary: Dictionary containing mean cost, standard deviation of cost,
            risk-adjusted metric, mean transaction cost, and mean turnover.
        df: DataFrame with one row per parallel episode.
    """
    out = runBatchedEpisode(env, policyFunction, seed=baseSeed)

    df = pd.DataFrame({
        "totalReward": out["totalReward"],
        "terminalTurnover": out["terminalTurnover"],
        "turnover": out["turnover"],
        "totalTransactionCost": out["totalTransactionCost"],
    })

    C = -df["totalReward"].to_numpy(dtype=float)
    meanC = float(np.mean(C))
    stdC = float(np.std(C, ddof=1)) if len(C) > 1 else 0.0
    risk_adj_Y = meanC + c * stdC

    summary = {
        "episodes": int(env.batchSize),
        "meanCostPct": meanC,
        "stdCostPct": stdC,
        "Y(mean+c*std)": risk_adj_Y,
        "meanTransactionCost": float(df["totalTransactionCost"].mean()),
        "meanTurnover": float(df["turnover"].mean()),
    }
    return summary, df