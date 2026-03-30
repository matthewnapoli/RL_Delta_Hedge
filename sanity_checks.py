import numpy as np
from hedgingEnvironment import (
    hedgingEnvironment,
    preprocessState,
    scaleActionToHedge,
)
from replayBuffer import ReplayBuffer


def prechecks():
    def make_env(seed: int = 0) -> hedgingEnvironment:
        env = hedgingEnvironment(
            S0=100.0,
            K=100.0,
            expiration=10 / 252,
            steps=10,
            r=0.0,
            q=0.0,
            mu=0.0,
            sigma=0.2,
            sigmaValuation=None,
            options=1,
            Hmin=None,
            Hmax=None,
            deltaCutoff=0.05,
            deltaPenaltyWeight=0.1,
            deltaPenaltyPower=2,
        )
        env.seed(seed)
        return env

    def quickSanityChecks() -> dict:
        """
        Run quick sanity checks to validate environment and scaling.
        """
        env = make_env(seed=0)
        checks = {}

        # A) No trade: transaction cost should be 0
        s0, _ = env.reset()
        s1, r1, d1, info1 = env.step(float(s0[0]))
        checks["A"] = info1["TransactionCost"] == 0

        # B) Big trade: transaction cost > 0
        s0, _ = env.reset()
        s1, r2, d2, info2 = env.step(env.Hmax)
        checks["B"] = info2["TransactionCost"] > 0

        # C) Preprocess range check: values should be finite
        try:
            statePreprocess = preprocessState(env, s0)
            checks["C"] = np.isfinite(statePreprocess).all()
        except Exception:
            checks["C"] = False

        # D) Buffer sample shapes
        try:
            buffer = ReplayBuffer(capacity=1000, stateDimension=3, actionDimension=1)

            for _ in range(300):
                s0, _ = env.reset()
                sp0 = preprocessState(env, s0)

                u = float(np.random.uniform(-1, 1))
                H = scaleActionToHedge(env, u)

                s1, r, d, _ = env.step(H)
                sp1 = preprocessState(env, s1)

                buffer.add(
                    sp0,
                    np.array([u], dtype=np.float32),
                    float(r),
                    sp1,
                    bool(d),
                )

            batch = buffer.sample(32)
            checks["D"] = (
                batch["currentStates"].shape[0] == 32
                and batch["actions"].shape[0] == 32
                and batch["rewards"].shape[0] == 32
                and batch["doneFlags"].shape[0] == 32
            )
        except Exception:
            checks["D"] = False

        return checks

    def rewardFormulaSanityCheck() -> bool:
        """
        Manually compute one step reward and compare to env.step().
        """
        env = make_env(seed=42)
        state, _ = env.reset()

        # Set an initial hedge directly
        env.H = 0.5

        # snapshot pre-step values
        hCurr = env.H
        sCurr = env.S
        vCurr = env.V

        # step with a known action
        hNext = 0.7
        state, reward, done, info = env.step(hNext)

        # post-step values
        sNext = env.S
        vNext = env.V

        # manual calculation
        deltaV = vNext - vCurr
        hedgePnL = hCurr * (sNext - sCurr)
        transactionCost = env.transCostFunc(abs(sNext * (hNext - hCurr)))

        # match new delta penalty logic
        _, nextDelta = __import__("helpers").blackScholesCallPriceDelta(
            sNext, env.K, env.r, env.q, env.sigmaValuation, env.tau()
        )
        netDelta = env.options * nextDelta - hNext
        excessDelta = max(abs(netDelta) - env.deltaCutoff * env.options, 0.0)
        deltaCost = env.deltaPenaltyFunc(excessDelta)

        manualReward = deltaV + hedgePnL - transactionCost - deltaCost

        if done:
            manualReward -= env.transCostFunc(abs(sNext * hNext))

        diff = abs(float(reward) - float(manualReward))
        return diff < 1e-10

    checks_quick = quickSanityChecks()
    checks_reward = rewardFormulaSanityCheck()

    quick_passed = all(checks_quick.values())

    if quick_passed and checks_reward:
        return "All checks passed."
    elif not checks_reward:
        return "Reward checks failed."
    else:
        return f"Quick checks failed: {checks_quick}"