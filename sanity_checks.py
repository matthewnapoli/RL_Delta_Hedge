import numpy as np
from envSpec import EnvSpec, makeEnvironment
from hedgingEnvironment import preprocessState, scaleActionToHedge, scaleHedgeToAction, KAPPASCALE
from replayBuffer import ReplayBuffer

def prechecks():
    # Define specs BEFORE they are used in any functions/lambdas
    paperSpec = EnvSpec(kappaMode="paper", maturity="1m", rebalancingFrequency="daily")
    stochSpec = EnvSpec(kappaMode="stochastic", maturity="1m", rebalancingFrequency="daily")

    # (Optional) Kept for completeness if you need them later in the script
    envFunc_paper = lambda: makeEnvironment(paperSpec, seed=0)
    envFunc_stoch = lambda: makeEnvironment(stochSpec, seed=0)

    def quickSanityChecks() -> dict:
        """
            Run quick sanity checks to validate environment and scaling
        """
        env = makeEnvironment(paperSpec)
        checks = {}

        # A) No trade: transaction cost should be 0
        s0, _ = env.reset()
        # Assuming s0[0] is the current hedge holding
        s1, r1, d1, info1 = env.step(float(s0[0])) 
        checks["A"] = (info1["TransactionCost"] == 0)

        # B) Big trade: transaction cost > 0
        s0, _ = env.reset()
        s1, r2, d2, info2 = env.step(env.Hmax)
        checks["B"] = (info2["TransactionCost"] > 0)

        # C) Preprocess range check: Validate values are finite
        try:
            statePreprocess = preprocessState(env, s0, kappaScale=KAPPASCALE)
            checks["C"] = np.isfinite(statePreprocess).all()
        except Exception:
            checks["C"] = False

        # D) Buffer sample shapes: Validate correct extraction
        try:
            buffer = ReplayBuffer(capacity=1000, stateDimension=4, actionDimension=1)
            for _ in range(300):
                s0, _ = env.reset()
                sp0 = preprocessState(env, s0, kappaScale=KAPPASCALE)
                u = np.random.uniform(-1, 1)
                H = scaleActionToHedge(env, u)
                s1, r, d, _ = env.step(H)
                sp1 = preprocessState(env, s1, kappaScale=KAPPASCALE)
                buffer.add(sp0, np.array([u], dtype=np.float32), r, sp1, d)

            batch = buffer.sample(32)
            checks["D"] = (
                batch["currentStates"].shape[0] == 32 and
                batch["actions"].shape[0] == 32 and
                batch["rewards"].shape[0] == 32 and
                batch["doneFlags"].shape[0] == 32
            )
        except Exception:
            checks["D"] = False

        return checks # FIX: Added the missing return statement

    def rewardFormulaSanityCheck() -> bool:
        """
            Manually compute one step of accounting P&L and compare to env.step()
        """
        env = makeEnvironment(paperSpec, seed=42)
        state, _ = env.reset()
        state, _, _ = env.applyInitialHedge(0.5)

        # snapshot pre-step values
        hCurr, sCurr, vCurr, kappaCurr = env.H, env.S, env.V, env.kappat

        # step with a known action
        hNext = 0.7
        state, reward, done, info = env.step(hNext)

        # post-step values
        sNext, vNext = env.S, env.V

        # manual calculation
        deltaV = vNext - vCurr
        hedgePnL = hCurr * (sNext - sCurr)
        transactionCost = kappaCurr * abs(sNext * (hNext - hCurr))
        manualReward = deltaV + hedgePnL - transactionCost
        
        if done:
            manualReward += -kappaCurr * abs(sNext * hNext)  # terminal liquidation

        diff = abs(reward - manualReward)
        
        # FIX: Return a simple boolean instead of a set
        return diff < 1e-10

    # Execute checks
    checks_quick = quickSanityChecks() 
    checks_reward = rewardFormulaSanityCheck()

    # FIX: Properly evaluate dictionary values and boolean separately
    quick_passed = all(checks_quick.values())

    if quick_passed and checks_reward:
        return "All checks passed."
    elif not checks_reward:
        return "Reward checks failed."
    else:
        # Note: If you want to know exactly which quick check failed, 
        # you could print or return the `checks_quick` dict here.
        return "Quick checks failed."+checks_quick