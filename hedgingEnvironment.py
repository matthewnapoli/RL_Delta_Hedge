import numpy as np
import pandas as pd
from typing import Literal, Optional, Tuple, Any
from datetime import datetime, timedelta
from helpers import blackScholesCallPriceDelta, stationaryDistribution, markovRegimeTransition, nextPriceGBM

KAPPASCALE = 1000

### ENVIRONMENT CLASS ###
class hedgingEnvironment:
    """
        Discrete-time hedging environment with accounting P&L and proportional
        transaction costs.

        State (time i): x_i = [H_i, S_i, tau_i, z_i]
            - H_i: current hedge position (shares)
            - S_i: spot price
            - tau_i: time-to-maturity = T - i*dt
            - z_i: kappa_i (proportional transaction cost rate from current liquidity regime)

        Action (time i): choose the next hedge position H_{i+1} (clipped to [Hmin, Hmax]).

        Dynamics:
            - Underlying evolves according to the configured price process.
            - Liquidity regime follows a Markov chain: L_{i+1} ~ P[L_i, :]
              and transaction cost rate is kappa_i = kappaLevels[L_i].
            - For constant-kappa experiments, use P = I and kappaLevels = [k, k, k].

        Reward (accounting P&L):
            R_{i+1} = (V_{i+1} - V_i)
                      + H_i * (S_{i+1} - S_i)
                      - kappa_i * | S_{i+1} * (H_{i+1} - H_i) |

            where V_i is the mark-to-model option value (negative when short):
                V_i = -options * BSCallPrice(S_i, tau_i)

        Initial setup:
            reset() always starts unhedged (H=0) with R_0 = 0.
            Call applyInitialHedge(H0) to set the first hedge and charge
            setup cost: R_0 = -kappa_0 * |S_0 * H_0|

        Terminal liquidation (IMPORTANT):
            When the episode ends, we liquidate any remaining hedge H_n at S_n and charge:
                extra_reward = -kappa_used * | S_n * H_n |
    """

    def __init__(self, S0 : float =100, K : float =100, T : float = 10/252, 
                 steps : float=10, r : float=0, q : float=0, mu : float=0, 
                 sigma=0.2, sigmaValuation=None, options=1,
                 Hmin=0, Hmax=1.5, trnsCostFunc=None, rng=None):
        """
            Builds a discrete-time hedging environment for a short European call with proportional
            transaction costs. Transaction costs depend on a 3-state Markov liquidity regime.

            Inputs:
              - S0, K, T, steps: underlying start, strike, maturity, number of periods (N).
              - r, q: risk-free rate and dividend yield used for option mark-to-model pricing.
              - mu, sigma: drift/vol used to simulate the underlying under GBM (sigma = sigmaSimulation).
              - sigmaValuation: volatility used in Black-Scholes valuation for accounting P&L; if None, uses sigmaSimulation.
              - options: number of calls sold (positive int); portfolio holds -options * call.
              - Hmin, Hmax: bounds on hedge position (shares of underlying).
              
              - kappaLevels: 3-element list; proportional cost rates for regimes 0/1/2.
              - P: 3x3 transition matrix for regimes (rows sum to 1). Uses a default if None.
              - startingRegime: initial regime index if startFromStationary is False.

              - rng: NumPy Generator for randomness.
              - startFromStationary: if True, draw initial regime from stationary distribution pi.
              - burnin: number of regime transitions applied at reset before starting the episode.

            State at time i is [H_i, S_i, tau_i, kappa_i] (or regime index), where tau_i = T - i*dt.
        """
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



        self.S0, self.K, self.T, self.N = float(S0), float(K), float(T), int(steps)
        self.r, self.q, self.mu = float(r), float(q), float(mu)

        self.sigmaSimulation = float(sigma)
        self.sigmaValuation = float(sigma) if sigmaValuation == None else float(sigmaValuation)
        self.optionPrice0 = blackScholesCallPriceDelta(self.S0, self.K, self.r, self.q, self.sigmaValuation, self.T)[0]

        self.options = int(options)
        self.dt = self.T / self.N

        self.Hmin, self.Hmax = float(Hmin), float(Hmax)
        self.rng = rng if rng != None else np.random.default_rng()
        self.pi = stationaryDistribution(self.P)  # stationary dist of regimes


        # state variables
        self.i, self.S, self.H, self.V, self.L = 0, 0, 0, 0, 1
        self.kappat = self.kappaLevels[self.L]

    def tau(self):
        """
            Returns remaining time-to-maturity tau_i = max(T - i*dt, 0).
        """
        return max(self.T - self.i * self.dt, 0)

    def stateVector(self):
        """
            Returns the current observable state as a NumPy array.
            [H, S, tau, kappa/regimeIndex]
        """
        base = [self.H, self.S, self.tau(), self.kappat]
        return np.array(base, dtype=float)

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

        # choose initial regime #
        if self.startFromStationary:
            self.L = int(self.rng.choice(3, p=self.pi))
        else:
            self.L = self.startingRegime
        self.kappat = float(self.kappaLevels[self.L])

        # Apply burn‑in transitions
        for _ in range(self.burnin):
            self.L = markovRegimeTransition(self.L, self.P, self.rng)
        self.kappat = float(self.kappaLevels[self.L])

        # Set hedge and charge initial setup cost
        self.H = 0
        callPrice = blackScholesCallPriceDelta(self.S, self.K, self.r, self.q, self.sigmaValuation, self.tau())[0]
        self.V = -self.options * callPrice
        initialReward = 0
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
        # Clip action
        Hnext = float(np.clip(Hnext, self.Hmin, self.Hmax))

        # Regime/kappa used for this decision/trade
        regimeUsed = int(self.L)
        kappaUsed = float(self.kappat)

        # 1) simulate price
        Snext = nextPriceGBM(self.S, self.mu, self.sigmaSimulation, self.dt, self.rng)
        Snext = float(np.asarray(Snext).item())
        spotExec = Snext
        tauNext = max(self.T - (self.i + 1) * self.dt, 0)

        # 2) Cao accounting P&L: TC_{i+1} = kappa_i * | S_{i+1} * (H_{i+1} - H_i) |
        transactionCost = kappaUsed * abs(spotExec * (Hnext - self.H))

        # 3) revalue option at next time step
        callNext = blackScholesCallPriceDelta(Snext, self.K, self.r, self.q, self.sigmaValuation, tauNext)[0]
        Vnext = -self.options * callNext

        # 4) reward = hedged P&L - transaction cost
        reward = (Vnext - self.V) + self.H * (Snext - self.S) - transactionCost

        self.i += 1
        self.S, self.H, self.V = float(Snext), float(Hnext), float(Vnext)

        done = (self.i >= self.N)
        terminalTC = 0
        if done:
            # liquidate remaining hedge at Sn
            terminalTC = kappaUsed * abs(self.S * self.H)
            reward += -terminalTC
        else:
            # transition regime after step if not done
            self.L = markovRegimeTransition(self.L, self.P, self.rng)
            self.kappat = float(self.kappaLevels[self.L])

        info = {"TransactionCost": float(transactionCost), "TerminalTransactionCost": float(terminalTC),
                "TotalTransactionCost": float(transactionCost + terminalTC), "SpotExec": float(spotExec),
                "KappaUsed": float(kappaUsed), "RegimeUsed": int(regimeUsed), "KappaNext": float(self.kappat),
                "RegimeNext": int(self.L)
               }
        return self.stateVector(), float(reward), done, info

    def applyInitialHedge(self, H0):
        """
            Apply an initial hedge at time 0 without advancing time.

            This charges the proportional transaction cost for changing the hedge
            from the current holding H (typically 0 right after reset) to H0 using
            the current regime's kappa_0 and spot S_0:
                transactionCost0 = kappa_0 * | S_0 * (H0 - H_current) |

            Returns:
                (state, reward0, info) where reward0 = -transactionCost0 and info contains
                TransactionCost/TotalTransactionCost for bookkeeping.
        """
        if self.i != 0:
            raise RuntimeError("applyInitialHedge() must be called immediately after reset() (i==0).")

        H0 = float(np.clip(H0, self.Hmin, self.Hmax))
        deltaH0 = H0 - float(self.H)
        spotExec = float(self.S)
        kappaUsed = float(self.kappat)
        transactionCost0 = float(kappaUsed * abs(spotExec * deltaH0))
        self.H = H0

        info = {"TransactionCost": float(transactionCost0), "TerminalTransactionCost": 0, "TotalTransactionCost": float(transactionCost0),
                "SpotExec": float(spotExec), "KappaUsed": float(kappaUsed), "RegimeUsed": int(self.L), "IsInitial": True
               }
        return self.stateVector(), float(-transactionCost0), info

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
                  or [H, S, tau, kappa]
        Returns: (H, S, tau, kappa (or None))
    """
    H, S, tau = float(state[0]), float(state[1]), float(state[2])
    kappa = float(state[3]) if len(state) > 3 else None
    return H, S, tau, kappa

def policyDeltaHedge(env, state):
    """
        Practitioner delta hedge: Htarget = options * DeltaCall(S, tau)
        Uses valuation sigma (env.sigmaValuation) for delta, consistent with Cao et al. practitioner delta approach.
    """
    _, S, tau, _ = unpackState(state)
    _, delta = blackScholesCallPriceDelta(S, env.K, env.r, env.q, env.sigmaValuation, tau)
    Htarget = env.options * delta
    return float(np.clip(Htarget, env.Hmin, env.Hmax))

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
    # For paper environment, this is BS value. For grouped data, it's V_path[0].
    # (env.V is negative because we are short, so we take the absolute value)
    optionPrice0 = abs(float(env.V)) 
    
    totalReward = float(r0)
    turnover = 0.0
    totalTransactionCost = 0.0

    turnoverPerRegime = {0: 0.0, 1: 0.0, 2: 0.0}
    transactionCostPerRegime = {0: 0.0, 1: 0.0, 2: 0.0}
    regimeCounts = {0: 0, 1: 0, 2: 0}

    # initial hedge (setup) #
    H0 = float(policyFunction(env, state))
    state, reward0, info0 = env.applyInitialHedge(H0)
    totalReward += float(reward0)

    setupTransactionCost = float(info0.get("TotalTransactionCost", info0.get("TransactionCost", 0.0)))
    setupTurnover = abs(H0 - 0)
    totalTransactionCost += setupTransactionCost
    turnover += setupTurnover

    # episode loop #
    done = False
    while not done:
        Hprev = float(state[0])

        Hnext = float(policyFunction(env, state))
        nextState, reward, done, info = env.step(Hnext)
        
        totalReward += float(reward)

        tc = float(info.get("TotalTransactionCost", info.get("TransactionCost", 0.0)))
        totalTransactionCost += tc

        regimeUsed = int(info.get("RegimeUsed", 1))
        transactionCostPerRegime[regimeUsed] += tc

        Hused = float(nextState[0])
        dH = abs(Hused - Hprev)

        turnover += dH
        turnoverPerRegime[regimeUsed] += dH
        regimeCounts[regimeUsed] += 1

        state = nextState

    # terminal liquidation turnover #
    terminalTurnover = abs(float(state[0]))
    turnover += terminalTurnover

    steps = sum(regimeCounts.values())
    if steps > 0:
        timeFractionPerRegime = {k: regimeCounts[k] / steps for k in regimeCounts}
        avgTurnoverRatePerRegime = {k: turnoverPerRegime[k] / steps for k in turnoverPerRegime}
        avgTransactionCostRatePerRegime = {k: transactionCostPerRegime[k] / steps for k in transactionCostPerRegime}
    else:
        timeFractionPerRegime = {k: 0.0 for k in regimeCounts}
        avgTurnoverRatePerRegime = {k: 0.0 for k in turnoverPerRegime}
        avgTransactionCostRatePerRegime = {k: 0.0 for k in transactionCostPerRegime}

    totalCostProxy = -totalReward  
    totalCostPct = (totalCostProxy / max(optionPrice0, 1e-12)) * 100

    return {"totalReward": float(totalReward), 
            "totalCostProxy": float(totalCostProxy),
            "totalCostPct": float(totalCostPct),
            "setupTransactionCost": float(setupTransactionCost),
            "setupTurnover": float(setupTurnover),
            "terminalTurnover": float(terminalTurnover),
            "turnover": float(turnover),
            "totalTransactionCost": float(totalTransactionCost),
            "timeFractionPerRegime": timeFractionPerRegime,
            "turnoverPerRegime": turnoverPerRegime,
            "transactionCostPerRegime": transactionCostPerRegime,
            "avgTurnoverRatePerRegime": avgTurnoverRatePerRegime,
            "avgTransactionCostRatePerRegime": avgTransactionCostRatePerRegime,
            "regimeCounts": dict(regimeCounts)}

### REGIME NORMALIZATION (PER-STEP CONDITIONAL RATES) ###
def normalizeByRegime(episodeOutput):
    """
    Convert per-episode regime totals into PER-STEP conditional rates:
        turnoverRate[k] = turnoverPerRegime[k] / regimeCounts[k]
        tcRate[k] = transactionCostPerRegime[k] / regimeCounts[k]
    """
    K = [0, 1, 2]
    turnover = np.array([episodeOutput["turnoverPerRegime"][k] for k in K], dtype=float)
    transactionCost = np.array([episodeOutput["transactionCostPerRegime"][k] for k in K], dtype=float)
    counts = np.array([episodeOutput["regimeCounts"][k] for k in K], dtype=float)

    turnoverRate = np.divide(turnover, counts, out=np.zeros_like(turnover), where=counts > 0)
    transactionCostRate = np.divide(transactionCost, counts, out=np.zeros_like(transactionCost), where=counts > 0)

    return turnoverRate, transactionCostRate

### POLICY EVALUATION ###
def evaluatePolicy(environmentGenerator, policyFunction, episodes=300, c=1.5, baseSeed=0):
    """
        Evaluate a policy over many episodes and return summary statistics.
    """
    rows = []

    for ep in range(episodes):
        env = environmentGenerator()
        out = runEpisode(env, policyFunction, seed=baseSeed + ep)
        turnoverRate, transactionCostRate = normalizeByRegime(out)
        out["turnoverRateRegime0"] = turnoverRate[0]
        out["turnoverRateRegime1"] = turnoverRate[1]
        out["turnoverRateRegime2"] = turnoverRate[2]
        out["transactionCostRateRegime0"] = transactionCostRate[0]
        out["transactionCostRateRegime1"] = transactionCostRate[1]
        out["transactionCostRateRegime2"] = transactionCostRate[2]
        rows.append(out)

    df = pd.DataFrame(rows)
    C = df["totalCostPct"].values
    meanC = float(np.mean(C))
    stdC = float(np.std(C, ddof=1)) if len(C) > 1 else 0
    Y = meanC + c * stdC

    # average regime fractions across episodes
    frac0 = float(pd.Series([d[0] for d in df["timeFractionPerRegime"]]).mean())
    frac1 = float(pd.Series([d[1] for d in df["timeFractionPerRegime"]]).mean())
    frac2 = float(pd.Series([d[2] for d in df["timeFractionPerRegime"]]).mean())

    summary = {"episodes": int(episodes), "meanCostPct": meanC, "stdCostPct": stdC, "Y(mean+c*std)": Y,
               "meanTransactionCost": float(df["totalTransactionCost"].mean()), "meanTurnover": float(df["turnover"].mean()),
               "meanTimeFractionRegime0": frac0, "meanTimeFractionRegime1": frac1, "meanTimeFractionRegime2": frac2}
    return summary, df

def preprocessState(env, state, kappaScale=KAPPASCALE):
    """
        Normalize the raw state for neural network input.

        Raw state: [H, S, tau, kappa]
        Return normalized features with roughly comparable scales:
            H -> scaled by max(|Hmin|,|Hmax|)
            S -> x = log(S/K) (log-moneyness)
            tau -> scaled time-to-maturity: tau/T in [0, 1]
            kappa -> scaled up by 1,000 so it's on the order of 1-10

        env: environment object
        state: state vector
        kappaScale: scalar to scale kappa up
    """
    H, S, tau, kappa = float(state[0]), float(state[1]), float(state[2]), float(state[3])
    hedgeScale = max(abs(env.Hmin), abs(env.Hmax), 1e-8)
    normalizedH = H / hedgeScale
    x = np.log(max(S, 1e-12) / max(env.K, 1e-12))
    normalizedTau = tau / max(env.T, 1e-12)
    normalizedKappa = kappa * float(kappaScale)
    return np.array([normalizedH, x, normalizedTau, normalizedKappa], dtype=np.float32)

def scaleActionToHedge(env, u):
    """
        Maps actor's scaled acition u in [-1,1] -> hedge position H in [Hmin, Hmax] using an affine transformation

        env: environment object
        u: action (actor output)
    """
    u = float(np.clip(u, -1, 1))
    H = 0.5 * (u + 1) * (env.Hmax - env.Hmin) + env.Hmin # affine map [-1,1] -> [Hmin,Hmax]
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