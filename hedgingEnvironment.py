import numpy as np
import pandas as pd
from typing import Literal, Optional, Tuple, Any
from datetime import datetime, timedelta
from helpers import blackScholesCallPriceDelta, stationaryDistribution, markovRegimeTransition, nextPriceGBM

KAPPASCALE = 1000

class hedgingEnvironment:
    def __init__(self, 
                 S0 : float             = 100.0, 
                 K : float              = 100.0, 
                 expiration : float     = 10/252, 
                 steps : float          = 10.0, 
                 r : float              = 0.0, 
                 q : float              = 0.0, 
                 mu : float             = 0.0, 
                 sigma : float          = 0.2, 
                 sigmaValuation : float = None, 
                 options : int          = 1,
                 Hmin : float           = None, 
                 Hmax : float           = None, 
                 trnsCostFunc : callable = None, 
                 deltaCutoff : float = 0.5,
                 deltaCostFunc : callable = None,
                 rng=None):

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

        #price variables
        self.S0, self.K, self.expiration, self.N = float(S0), float(K), float(expiration), int(steps)
        self.r, self.q, self.mu = float(r), float(q), float(mu)
        self.sigmaSimulation = float(sigma)
        self.sigmaValuation = float(sigma) if sigmaValuation == None else float(sigmaValuation)
        self.optionPrice0 = blackScholesCallPriceDelta(self.S0, self.K, self.r, self.q, self.sigmaValuation, self.expiration)[0]
        self.options = int(options)
        self.dt = self.expiration / self.N
        self.rng = rng if rng != None else np.random.default_rng()

        # position constraint variables
        if self.Hmin is None:
            self.Hmin = -1 * self.options
        if self.Hmax is None:
            self.Hmax = 1 * self.options
        else:
            self.Hmin, self.Hmax = float(Hmin), float(Hmax)

        #delta hedging variables
        self.defaultTransCost = 0.01
        self.transCostFunc = trnsCostFunc if trnsCostFunc is not None else lambda moneyTraded: self.defaultTransCost * moneyTraded
        
        # delta constraint variables
        self.deltaCutoff = deltaCutoff
        self.defaultDeltaPenalty = 0.05
        self.deltaPenaltyFunc = deltaCostFunc if deltaCostFunc is not None else lambda netDelta: self.defaultDeltaPenalty * netDelta

        #market impact variables - TO-DO

        # state variables
        self.i, self.S, self.H, self.V, self.L = 0, 0, 0, 0, 1

    def tau(self):
        """
            Returns remaining time-to-maturity tau_i = max(T - i*dt, 0).
        """
        return max(self.expiration - self.i * self.dt, 0)

    def stateVector(self):
        """
            Returns the current observable state as a NumPy array.
            [H, S, tau, kappa/regimeIndex]
        """
        base = [self.H, self.S, self.tau()]
        return np.array(base, dtype=np.float32)

    def reset(self):
        """
            Resets the episode to time i=0:
            - sets S = S0
            - picks the initial liquidity regime (stationary draw or fixed startingRegime)
            - applies optional regime burn-in transitions
            - sets kappa from the regime
            - sets initial hedge H = 0 (unhedged)
            - initializes option value V = -options * BSCallPrice(S, tau)

            Returns: (stateVector, initialReward)

            initialReward is always 0; setup costs are charged via applyInitialHedge()
        """
        # Reset counters and variables
        self.i = 0
        self.S = np.asarray(self.S0).item()

        # Set hedge and charge initial setup cost
        self.H = 0.0
        callPrice,_ = blackScholesCallPriceDelta(self.S, self.K, self.r, self.q, self.sigmaValuation, self.tau())[0]
        self.V = -self.options * callPrice
        initialReward = 0.0
        return self.stateVector(), float(initialReward)


    def step(self, Hnext):
        """
            Timing convention:
            1) The agent chooses H_{i+1} using the current state (H_i, S_i, tau_i, kappa_i).
            2) Transaction cost for the trade from H_i to H_{i+1} is charged using CURRENT kappa (kappa_i)
               and the next-step spot as execution proxy (S_{i+1}), matching Cao et al.:
               TC_{i+1} = kappa_i * | S_{i+1} * (H_{i+1} - H_i) |
            3) Price evolves to S_{i+1}; option is revalued to V_{i+1}.
            4) Reward (accounting P&L minus transaction cost):
                   R_{i+1} = (V_{i+1} - V_i) + H_i * (S_{i+1} - S_i) - TC_{i+1}
            5) Terminal liquidation at maturity (closing remaining hedge):
                   extra reward += - kappaUsed * | S_n * H_n |
                where kappaUsed is the kappa that applied to the final trade decision.
            6) Liquidity regime transitions AFTER the step (if not terminal) to define kappa_{i+1}:
                   L_{i+1} ~ P[L_i, :]
                   kappa_{i+1} = kappaLevels[L_{i+1}]

            Hnext: desired hedge position (shares of underlying) for the next period; clipped to [Hmin, Hmax]
        """
        # Clip action if parameters want to enforce it 
        # TO-DO: change this from Hnext to Htrade, for a more direct interpretation of the agent's action as 
        # "how much to trade" rather than "what position to move to"
        if self.Hmin is not None and self.Hmax is not None:
            Hnext = float(np.clip(Hnext, self.Hmin, self.Hmax))

        # 1) simulate price
        Snext = nextPriceGBM(self.S, self.mu, self.sigmaSimulation, self.dt, self.rng)
        Snext = float(np.asarray(Snext).item())
        spotExec = Snext
        tauNext = max(self.expiration - (self.i + 1) * self.dt, 0)

        # 2) allow 
        tradedH = Hnext - self.H
        moneyTraded = abs(spotExec * tradedH)
        transactionCost = self.transCostFunc(moneyTraded)
        
        # 3) Revalue option at next time step
        nextPrice, nextDelta = blackScholesCallPriceDelta(Snext, self.K, self.r, self.q, self.sigmaValuation, tauNext)[0]
        Vnext = -self.options * nextPrice

        # 4) Adding a penalty for net delta exposure.
        ##### TO-DO: parameterize this penalty
        netDelta = abs(self.options * nextDelta - Hnext)
        if netDelta > self.deltaCutoff * self.options:
            deltaCost = self.deltaPenaltyFunc(netDelta)
        else:
            deltaCost = 0.0

        # 5) reward = hedged P&L - transaction cost
        reward = (Vnext - self.V) + self.H * (Snext - self.S) - transactionCost - deltaCost

        self.i += 1
        self.S, self.H, self.V = float(Snext), float(Hnext), float(Vnext)

        done = (self.i >= self.N)
        terminalTC = 0

        if done:
            # liquidate remaining hedge without using transaction cost here,
            # since we want to maintain delta neutral position at expirey.
            # TO-DO: conisder the real world ramifications of this choice
            liquidation_amt = abs(spotExec * self.H)
            reward += -liquidation_amt

        info = {"Reward": float(reward), "TransactionCost": float(transactionCost), "TerminalTransactionCost": float(terminalTC),
                "TotalTransactionCost": float(transactionCost + terminalTC), "SpotExec": float(spotExec),
                "DeltaTraded": float(tradedH), "MoneyTraded": float(moneyTraded), "transactionCost": transactionCost }
        return self.stateVector(), float(reward), done, info


    def seed(self, seed=None):
        """
            Sets/overwrites the environment RNG in a single place.
            seed:
              - None: do nothing
              - int / np.integer: creates np.random.default_rng(seed)
              - np.random.Generator: uses it directly
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
        State can be [H, S, tau]
        Returns: (H, S, tau))
    """
    H, S, tau = float(state[0]), float(state[1]), float(state[2])
    return H, S, tau

def policyDeltaHedge(env, state):
    """
        Practitioner delta hedge: Htarget = options * DeltaCall(S, tau)
        Uses valuation sigma (env.sigmaValuation) for delta, consistent with Cao et al. practitioner delta approach.
    """
    _, S, tau, _ = unpackState(state)
    _, delta = blackScholesCallPriceDelta(S, env.K, env.r, env.q, env.sigmaValuation, tau)
    Htarget = env.options * delta
    return float(np.clip(Htarget, env.Hmin, env.Hmax))


## TO-DO: Review this policy, make a more realistic one that only trades when delta error exceeds a certain band
def policyDeltaHedgeWithBand(band=0.25):
    """
        Returns a policy(env, state) that implements delta hedging with an inaction band:
            if |Htarget - Hcurr| <= band: do nothing (no trade)
            else: move to Htarget (clipped)
    """
    def policy(env, state):
        Hcurr, S, tau, _ = unpackState(state)
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
        Runs one episode and returns bookkeeping for:
          - totalReward: sum of accounting P&L rewards (includes terminal liquidation transaction cost)
          - totalTransactionCost: includes setup transaction cost + per-step transaction cost + terminal liquidation transaction cost
          - turnover: includes setup turnover + per-step turnover + terminal liquidation turnover
          - turnoverPerRegime / transactionCostPerRegime: only per-step rehedge contributions (setup + terminal are reported separately)
    """
    env.rng = np.random.default_rng(seed)

    state, r0 = env.reset()
    
    # Use the environment's intrinsic initial option value. 
    # For paper environment, this is BS value. For grouped data, it's V_pathw[0].
    # (env.V is negative because we are short, so we take the absolute value)


    # episode loop #
    done = False
    while not done:
        Hprev = float(state[0])

        Hnext = float(policyFunction(env, state))
        nextState, reward, done, info = env.step(Hnext)
        
        totalReward += float(reward)

        tc = float(info.get("TotalTransactionCost", info.get("TransactionCost", 0.0)))
        totalTransactionCost += tc

        Hused = float(nextState[0])
        dH = abs(Hused - Hprev)

        turnover += dH

        state = nextState

    # terminal liquidation turnover #
    terminalTurnover = abs(float(state[0]))
    turnover += terminalTurnover


    totalCostProxy = -totalReward 

    return {"totalReward": float(totalReward), 
            "terminalTurnover": float(terminalTurnover),
            "turnover": float(turnover),
            "totalTransactionCost": float(totalTransactionCost)}

### POLICY EVALUATION ###
def evaluatePolicy(env, policyFunction, episodes=300, c=1.5, baseSeed=0):
    """
        Evaluate a policy over many episodes and return summary statistics.
    """
    rows = []

    for ep in range(episodes):
        env = env
        env.rng = np.random.default_rng(baseSeed + ep)
        out = runEpisode(env, policyFunction, seed=baseSeed + ep)
        rows.append(out)

    df = pd.DataFrame(rows)
    C = -df["totalReward"].values
    meanC = float(np.mean(C))
    stdC = float(np.std(C, ddof=1)) if len(C) > 1 else 0
    risk_adj_Y = meanC + c * stdC

    # average regime fractions across episodes
    frac0 = float(pd.Series([d[0] for d in df["timeFractionPerRegime"]]).mean())
    frac1 = float(pd.Series([d[1] for d in df["timeFractionPerRegime"]]).mean())
    frac2 = float(pd.Series([d[2] for d in df["timeFractionPerRegime"]]).mean())

    summary = {"episodes": int(episodes), "meanCostPct": meanC, "stdCostPct": stdC, "Y(mean+c*std)": risk_adj_Y,
               "meanTransactionCost": float(df["totalTransactionCost"].mean()), "meanTurnover": float(df["turnover"].mean()),
               "meanTimeFractionRegime0": frac0, "meanTimeFractionRegime1": frac1, "meanTimeFractionRegime2": frac2}
    return summary, df

def preprocessState(env, state, kappaScale=KAPPASCALE):
    """
        Normalize the raw state for neural network input.

        Raw state: [H, S, tau]
        Return normalized features with roughly comparable scales:
            H -> scaled by max(|Hmin|,|Hmax|)
            S -> x = log(S/K) (log-moneyness)
            tau -> scaled time-to-maturity: tau/T in [0, 1]

        env: environment object
        state: state vector
        kappaScale: scalar to scale kappa up
    """
    H, S, tau = float(state[0]), float(state[1]), float(state[2])
    hedgeScale = max(abs(env.Hmin), abs(env.Hmax), 1e-8)
    normalizedH = H / hedgeScale
    x = np.log(max(S, 1e-12) / max(env.K, 1e-12))
    normalizedTau = tau / max(env.T, 1e-12)
    return np.array([normalizedH, x, normalizedTau], dtype=np.float32)

def scaleActionToHedge(env, u):
    """
        Maps actor's scaled action u in [-1,1] -> hedge position H in [Hmin, Hmax] using an affine transformation

        env: environment object
        u: action (actor output)
    """
    u = float(np.clip(u, -1, 1))
    H = 0.5 * (u + 1) * (env.Hmax - env.Hmin) + env.Hmin
    return H

def scaleHedgeToAction(env, H):
    """
        Maps hedge position [Hmin, Hmax] -> actor's scaled acition u in [-1,1] using an affine transformation

        env: environment object
        H: hedge position
    """
    H = float(np.clip(H, env.Hmin, env.Hmax))
    u = 2 * (H - env.Hmin) / (env.Hmax - env.Hmin) - 1
    return u