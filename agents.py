import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from actor_critics import Actor, Critic, SecondMomentCritic
from dataclasses import dataclass 
from hedgingEnvironment import preprocessState, scaleActionToHedge, scaleHedgeToAction, KAPPASCALE
from replayBuffer import ReplayBuffer, DEVICE


@dataclass
class DDPGConfig:
    """
        Outputs a config object to pass to the agent
    """
    gamma: float = 1                    # finite horizon -> gamma (discount factor) = 1
    tau: float = 0.0001                 # target network update rate
    actorLearningRate: float = 0.0001   # actor learning rate (hyperparameter)
    criticLearningRate: float = 0.00004 # critic learning rate (hyperparameter)
    batchSize: int = 256                # number of transitions to sample from the replay buffer for one gradient update
    replaySize: int = 200000            # replay buffer capacity: maximum number of transitions kept in memory
    warmup: int = 32                    # steps before training begins
    trainEvery: int = 2                 # gradient steps per env step: how often to update network
    noiseStd: float = 0.005             # exploration noise in (scaled) action space
    noiseClip: float = 0.02             # clip noise
    hidden: int = 128                   # width of hidden layer
    kappaScale: float = KAPPASCALE      # for preprocessState


class DDPGAgent:
    def __init__(self, stateDimension, config: DDPGConfig):
        """
            Deep Deterministic Policy Gradient (risk‑neutral) agent

            Encapsulates the actor, critic, target networks, replay buffer and hyperparameters
            The agent interacts with the environment by selecting actions, storing transitions and performing gradient updates
        """
        if not isinstance(config, DDPGConfig):
            raise TypeError(f"DDPGAgent requires DDPGConfig, got {type(config).__name__}.")

        self.config = config
        self.actor = Actor(stateDimension, hidden=config.hidden).to(DEVICE)
        self.critic = Critic(stateDimension, hidden=config.hidden).to(DEVICE)

        self.actorTgt = Actor(stateDimension, hidden=config.hidden).to(DEVICE)
        self.criticTgt = Critic(stateDimension, hidden=config.hidden).to(DEVICE)

        self.actorTgt.load_state_dict(self.actor.state_dict())
        self.criticTgt.load_state_dict(self.critic.state_dict())

        self.actorOpt = optim.Adam(self.actor.parameters(), lr=config.actorLearningRate)
        self.criticOpt = optim.Adam(self.critic.parameters(), lr=config.criticLearningRate)

        self.buffer = ReplayBuffer(config.replaySize, stateDimension=stateDimension, actionDimension=1)

        self.totalSteps = 0

    @torch.no_grad()
    def selectAction(self, env, stateRaw, explore=True):
        """
            Preprocesses state -> actor outputs scaled action u in [-1,1] -> converts to actual hedge H
            If explore=True, adds Gaussian noise (clipped to [-1,1])

            env: for action scaling bounds
            stateRaw: raw env state [H,S,tau,kappa]
            explore: if True, add noise for exploration (vs exploitation)
        """
        S = preprocessState(env, stateRaw, kappaScale=self.config.kappaScale)
        St = torch.tensor(S, DEVICE=DEVICE).unsqueeze(0)
        u = float(self.actor(St).cpu().numpy()[0, 0])

        if explore:
            eps = np.random.normal(0, self.config.noiseStd)
            eps = np.clip(eps, -self.config.noiseClip, self.config.noiseClip)
            u = np.clip(u + eps, -1, 1)

        H = scaleActionToHedge(env, u)
        return float(H)

    def softUpdate(self, net, tgt):
        """
            Updates target net in place

            net: live network (actor or critic)
            tgt: target network
        """
        tau = self.config.tau
        for p, pTgt in zip(net.parameters(), tgt.parameters()):
            pTgt.data.mul_(1 - tau).add_(tau * p.data)

    def trainStep(self):
        """
            Perform a single gradient update on the critic and actor
            Returns None until the replay buffer contains at least max(warmup, batch_size) transitions

            Once buffer has enough data:
                1. samples batch
                2. computes target Q: y = r + gamma(1-d)*Qtgt(s',μtgt(s'))
                3. updates critic by MSE loss between Q(s,a) and y
                4. updates actor to maximize critic output: E[Q(s,μ(s))] (implemented as minimizing negative)
        """
        if self.buffer.size < max(self.config.warmup, self.config.batchSize):
            return None

        batch = self.buffer.sample(self.config.batchSize)
        state, action, reward, nextState, done = batch["currentStates"], batch["actions"], batch["rewards"], batch["nextStates"], batch["doneFlags"]

        # Compute target Q: y = r + gamma (1 - done) Qtgt(s', mu_tgt(s'))
        with torch.no_grad():
            nextAction = self.actorTgt(nextState)
            nextQ = self.criticTgt(nextState, nextAction)
            y = reward + self.config.gamma * (1.0 - done) * nextQ

        # Critic loss
        Q = self.critic(state, action)
        criticLoss = nn.MSELoss()(Q, y)
        self.criticOpt.zero_grad()
        criticLoss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.criticOpt.step()

        # Actor loss: maximize Q(s, actor(s)) <-> minimize -Q
        actionPred = self.actor(state)
        actorLoss = -self.critic(state, actionPred).mean()
        self.actorOpt.zero_grad()
        actorLoss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actorOpt.step()

        # Soft update targets
        self.softUpdate(self.actor, self.actorTgt)
        self.softUpdate(self.critic, self.criticTgt)

        output = {"criticLoss": float(criticLoss.item()), "actorLoss": float(actorLoss.item())}
        return output
    

@dataclass
class MeanStdDDPGConfig(DDPGConfig):
    """
        Configuration for the mean–standard deviation agent.
        Extends DDPGConfig with risk aversion and numerical stability parameters.

        riskAversion = c in: objective = E[G] - c * Std(G)
    """
    riskAversion: float = 1.5
    stdEps: float = 1e-8


class MeanStdDDPGAgent:
    """
        Risk‑averse (mean-std) agent optimizing E[G] − c*Std(G).

        Q1 target (critic1 for E[G]): y1 = r + gamma*(1-done)*Q1tgt(s',a')
        Q2 target (critic2 for E[G^2]): y2 = r^2 + 2*gamma*r*(1-done)*Q1tgt(s',a') + gamma^2*(1-done)*Q2tgt(s',a')
        Actor maximizes: Q1 - c*sqrt(max(Q2 - Q1^2, 0))
    """
    def __init__(self, stateDimension, config: MeanStdDDPGConfig):
        if not isinstance(config, MeanStdDDPGConfig):
            raise TypeError(f"MeanStdDDPGAgent requires MeanStdDDPGConfig, got {type(config).__name__}.")

        self.config = config

        # networks #
        self.actor = Actor(stateDimension, hidden=config.hidden).to(DEVICE)
        self.critic1 = Critic(stateDimension, hidden=config.hidden).to(DEVICE)
        self.critic2 = SecondMomentCritic(stateDimension, hidden=config.hidden).to(DEVICE)

        # targets #
        self.actorTgt = Actor(stateDimension, hidden=config.hidden).to(DEVICE)
        self.critic1Tgt = Critic(stateDimension, hidden=config.hidden).to(DEVICE)
        self.critic2Tgt = SecondMomentCritic(stateDimension, hidden=config.hidden).to(DEVICE)
        self.actorTgt.load_state_dict(self.actor.state_dict())
        self.critic1Tgt.load_state_dict(self.critic1.state_dict())
        self.critic2Tgt.load_state_dict(self.critic2.state_dict())

        # optimizers #
        self.actorOpt = optim.Adam(self.actor.parameters(), lr=config.actorLearningRate)
        self.critic1Opt = optim.Adam(self.critic1.parameters(), lr=config.criticLearningRate)
        self.critic2Opt = optim.Adam(self.critic2.parameters(), lr=config.criticLearningRate)

        # replay #
        self.buffer = ReplayBuffer(config.replaySize, stateDimension=stateDimension, actionDimension=1)
        self.totalSteps = 0

    @torch.no_grad()
    def selectAction(self, env, stateRaw, explore=True):
        s = preprocessState(env, stateRaw, kappaScale=self.config.kappaScale)
        st = torch.tensor(s, DEVICE=DEVICE).unsqueeze(0)
        u = float(self.actor(st).cpu().numpy()[0, 0])

        if explore:
            eps = np.random.normal(0.0, self.config.noiseStd)
            eps = float(np.clip(eps, -self.config.noiseClip, self.config.noiseClip))
            u = float(np.clip(u + eps, -1.0, 1.0))

        H = scaleActionToHedge(env, u)
        return float(H)

    def softUpdate(self, net, tgt):
        tau = float(self.config.tau)
        for p, pTgt in zip(net.parameters(), tgt.parameters()):
            pTgt.data.mul_(1.0 - tau).add_(tau * p.data)

    def trainStep(self):
        if self.buffer.size < max(self.config.warmup, self.config.batchSize):
            return None

        batch = self.buffer.sample(self.config.batchSize)
        state = batch["currentStates"]
        action = batch["actions"]
        reward = batch["rewards"]
        nextState = batch["nextStates"]
        done = batch["doneFlags"]
        notDone = (1.0 - done)

        gamma = float(self.config.gamma)
        c = float(self.config.riskAversion)

        # targets #
        with torch.no_grad():
            nextAction = self.actorTgt(nextState)

            q1Next = self.critic1Tgt(nextState, nextAction)
            q2Next = self.critic2Tgt(nextState, nextAction)

            # y1 = r + gamma*(1-d)*q1Next
            y1 = reward + gamma * notDone * q1Next
            # y2 = r^2 + 2*gamma*r*(1-d)*q1Next + gamma^2*(1-d)*q2Next
            y2 = reward.pow(2) + 2.0 * gamma * reward * notDone * q1Next + (gamma ** 2) * notDone * q2Next

        # critic 1 update #
        q1 = self.critic1(state, action)
        critic1Loss = nn.MSELoss()(q1, y1)
        self.critic1Opt.zero_grad()
        critic1Loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1Opt.step()

        # critic 2 update #
        q2 = self.critic2(state, action)
        critic2Loss = nn.MSELoss()(q2, y2)
        self.critic2Opt.zero_grad()
        critic2Loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2Opt.step()

        # actor update: maximize q1 - c*std  <=>  minimize: -(q1 - c*std)
        actionPred = self.actor(state)
        q1Pred = self.critic1(state, actionPred)
        q2Pred = self.critic2(state, actionPred)
        varPred = torch.relu(q2Pred - q1Pred.pow(2))
        stdPred = torch.sqrt(varPred + float(self.config.stdEps))
        actorLoss = -(q1Pred - c * stdPred).mean()
        self.actorOpt.zero_grad()
        actorLoss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actorOpt.step()

        # soft update targets #
        self.softUpdate(self.actor, self.actorTgt)
        self.softUpdate(self.critic1, self.critic1Tgt)
        self.softUpdate(self.critic2, self.critic2Tgt)

        criticLoss = (critic1Loss + critic2Loss) / 2
        return {"actorLoss": float(actorLoss.item()), "criticLoss": float(criticLoss.item()),
                "critic1Loss": float(critic1Loss.item()), "critic2Loss": float(critic2Loss.item())}
    

def trainDDPG(env, agent, episodes=2000, baseSeed=0, logEvery=50, debugFirstEpisode=True, debugSteps=5, reward_scalar=1):
    """
        Run reinforcement learning episodes to train an agent
        This training loop handles both risk‑neutral and risk‑averse agents

        env: a fresh environment instance of a hedging environment (not shared across episodes to ensure fresh RNG state)
        agent: DDPGAgent
        episodes: number of episodes to roll out
        baseSeed: for reproducibility
        logEvery: how often to print progress
        debugFirstEpisode: if True, prints detailed step diagnostics for the first episode
    """

    hist = {"episodeReward": [], "episodeTransactionCost": [], "actorLoss": [], "criticLoss": []}

    # for ep in tqdm(range(episodes)):
    for ep in range(episodes):
        env = env
        env.rng = np.random.default_rng(baseSeed + ep)
        state, r0 = env.reset()
        done = False
        episodeReward = float(r0)
        episodeTransactionCost = 0
        stepCount = 0

        sPreInitial = preprocessState(env, state, kappaScale=agent.config.kappaScale)
        H0 = agent.selectAction(env, state, explore=True)
        state, reward0, info0 = env.applyInitialHedge(H0)
        reward0 /= reward_scalar

        episodeReward += float(reward0)
        episodeTransactionCost += float(info0.get("TotalTransactionCost", info0.get("TransactionCost", 0)))

        sPostInitial = preprocessState(env, state, kappaScale=agent.config.kappaScale)
        u0 = scaleHedgeToAction(env, H0)
        agent.buffer.add(sPreInitial, np.array([u0], dtype=np.float32), np.array([reward0], dtype=np.float32), sPostInitial, False)
        agent.totalSteps += 1
        # Main episode loop
        while not done:
            # 1) choose hedge (in H-space) using actor + exploration noise
            Hnext = agent.selectAction(env, state, explore=True)

            # 2) step environment
            nextState, reward, done, info = env.step(Hnext)
            reward/=reward_scalar  # scale reward if needed (eg. for mean-std agent to keep magnitudes manageable)

            # 3) preprocess states for NN input
            s = preprocessState(env, state, kappaScale=agent.config.kappaScale)
            s2 = preprocessState(env, nextState, kappaScale=agent.config.kappaScale)

            # 4) store action in SCALED space u in [-1,1]
            u = scaleHedgeToAction(env, Hnext)
            agent.buffer.add(s, np.array([u], dtype=np.float32), np.array([reward], dtype=np.float32), s2, done)
            agent.totalSteps += 1

            # 5) train periodically
            if agent.totalSteps % agent.config.trainEvery == 0:
                losses = agent.trainStep()
                #print("trainStep output →", losses)
                if losses is not None:
                    hist["actorLoss"].append(losses["actorLoss"])
                    hist["criticLoss"].append(losses["criticLoss"])

            # 6) accumulate episode stats (every step)
            episodeReward += float(reward)
            episodeTransactionCost += float(info.get("TransactionCost", 0))

            # 7) debug print (first episode only)
            if debugFirstEpisode and ep == 0 and stepCount < debugSteps:
                tc = float(info.get("TransactionCost", 0))
                print(f"step={stepCount} reward={float(reward): .8f} cumReward={episodeReward: .8f} "f"tc={tc: .8f} S={float(env.S): .4f} H={float(env.H): .4f}")
                stepCount += 1
            state = nextState

        hist["episodeReward"].append(episodeReward)
        hist["episodeTransactionCost"].append(episodeTransactionCost)
        if (ep + 1) % logEvery == 0:
            avgReward = np.mean(hist["episodeReward"][-logEvery:])
            avgTransactionCost = np.mean(hist["episodeTransactionCost"][-logEvery:])
            print(f"Episode {ep+1}/{episodes} | avg reward: {avgReward:.8f} | avg TC: {avgTransactionCost:.8f} | buffer: {agent.buffer.size}")

    return hist


#### PLUGGING LEARNED POLICY INTO evaluatePolicy(...) ####
def makeReinforcementLearningPolicy(agent: "DDPGAgent | MeanStdDDPGAgent"):
    """
        Return a deterministic policy wrapper around an RL agent
        No exploration noise during evaluation
    """
    def policy(env, state):
        return agent.selectAction(env, state, explore=False)
    return policy