# https://github.com/tims457/rl_agents/blob/main/mbrl/ppo_torch.py
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import random
import time
import torch
import gym
import copy
import toml
import json
import wandb
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series
import torch.nn as nn
import torch.optim as optim
from matplotlib import animation, rc
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal

# from timer import timer
writer = SummaryWriter()
torch.random.manual_seed(0)
np.random.seed(0)
# plt.style.use(["science", "notebook", "grid", "no-latex"])  # pip install SciencePlots

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self, batch_size=32):
        self.states = []
        self.logits = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.advantages = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int32)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]
        actions = torch.stack(self.actions)
        actions = actions.to(device)
        return (
            torch.tensor(self.states, dtype=torch.float32, device=device),
            actions,
            torch.tensor(self.logits, dtype=torch.float32, device=device),
            torch.tensor(self.vals, dtype=torch.float32, device=device),
            torch.tensor(self.rewards, dtype=torch.float32, device=device),
            torch.tensor(self.dones),
            torch.tensor(self.advantages, dtype=torch.float32, device=device),
            batches,
        )

    def store_memory(self, state, action, logits, vals, reward, done, advantage):
        self.states.extend(state)
        self.actions.extend(action)
        self.logits.extend(logits)
        self.vals.extend(vals)
        self.rewards.extend(reward)
        self.dones.extend(done)
        self.advantages.extend(advantage)

    def clear_memory(self):
        self.states = []
        self.logits = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
        self.advantages = []


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
        torch.nn.init.constant_(m.bias, 0.0)

class Agent(nn.Module):
    def __init__(self, obs_space, action_space):
        super(Agent, self).__init__()
        self.action_space = action_space
        self.obs_space = obs_space
        self.actor = nn.Sequential(nn.Linear(obs_space, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, action_space))
        self.critic = nn.Sequential(nn.Linear(obs_space, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

        self.actor.apply(init_weights)
        self.critic.apply(init_weights)

    def forward(self, x):
        logits = self.actor(x)
        values = self.critic(x)
        if logits.shape[0] != 2:
            print("error")
        # dist = MultivariateNormal(logits, torch.diag(torch.exp(logits)))
        dist = Independent(Normal(logits, logits.exp()), 1)
        return dist, values, logits

class PPO:
    def __init__(self, env, agent, hyp, use_wandb=False, training=True):
        self.env = env
        self.reward_threshold = 150
        self.agent = agent
        self.training = training
        self.TEST_FREQ = hyp["test_freq"]
        self.REWARD_THRESHOLD = hyp["reward_threshold"]

        self.EPOCHS = hyp["epochs"]
        self.GAMMA = hyp["gamma"]
        self.GAE_LAMBDA = hyp["gae_lambda"]
        self.EPS = hyp["eps"]

        self.TIMESTEPS = hyp["timesteps"]
        self.BATCH_SIZE = hyp["batch_size"]
        self.N_EPOCHS = hyp["n_epochs"]

        self.C1 = hyp["c1"]  # 0.5  #critic loss coefficient
        self.C2 = hyp["c2"]  # entropy coefficient
        self.clip_value = hyp["clip_value"]
        self.V_EPS = hyp["v_eps"]

        # self.initial_learning_rate = hyp["lr0"]
        # self.opt = optim.Adam(agent.parameters(), lr=self.initial_learning_rate)
        self.opt = torch.optim.Adam(
            [
                {"params": self.agent.actor.parameters(), "lr": hyp["actor_lr"]},
                {"params": self.agent.critic.parameters(), "lr": hyp["critic_lr"]},
            ]
        )
        self.scheduler = optim.lr_scheduler.MultiplicativeLR(
            self.opt, lr_lambda=lambda x: hyp["lr_decay"]
        )
        # self.scheduler = optim.lr_scheduler.MultiStepLR(
        #     self.opt, milestones=[100, 200, 400, 9e9], gamma=hyp["lr_decay"]
        # )
        self.decay_lr = True
        self.step = 0

        self.print_frequency = 1
        self.render = False
        self.use_wandb = use_wandb
        self.loss_hist = []
        self.reward_hist = []
        self.actor_loss_hist = []
        self.critic_loss_hist = []
        self.entropy_hist = []
        self.grad_norm_hist = []
        self.episode_reward_hist= []

    def select_action(self, state):
        dist, value, policy_logits = self.agent(state)
        if self.training:
            action = dist.sample()
        else:
            action = dist.mean
        return action, value, policy_logits

    def compute_advantages(self, values, next_values, rewards, dones):

        deltas = np.zeros((len(rewards)))
        for t, (r, v, nv, d) in enumerate(zip(rewards, values, next_values, dones)):
            deltas[t] = r + self.GAMMA * (1 - d) * nv - v

        advantages = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            advantages[t] = (
                advantages[t]
                + (1 - dones[t]) * self.GAMMA * self.GAE_LAMBDA * advantages[t + 1]
            )

        targets = advantages + values

        advantages -= advantages.mean()
        advantages /= advantages.std() + 1e-8

        return advantages, targets

    def learn(self, mem, targets):

        for epoch in range(self.N_EPOCHS):
            total_loss = 0
            total_entropy_loss = 0
            total_actor_loss = 0
            total_critic_loss = 0

            (
                states,
                actions,
                logits_old,
                values,
                rewards,
                dones,
                advantages,
                batches,
            ) = mem.generate_batches()

            # action_idx = torch.stack(
            #     (torch.arange(0, len(rewards), device=device), actions), 1
            # )
            action_idx = torch.arange(0, len(rewards), device=device)

            old_probs = nn.functional.softmax(logits_old, dim=1)
            old_probs = old_probs[list(action_idx.T)]
            old_probs.to(device)
            self.agent.to(device)

            for batch in batches:
                dist, critic_value, logits = self.agent(states[batch])
                probs = torch.nn.functional.softmax(logits, dim=1)

                # action_idx_batch = torch.stack(
                #     (torch.arange(0, len(probs), device=device), actions[batch]), 1
                # )
                action_idx_batch = torch.arange(0, len(probs), device=device)

                probs = probs[list(action_idx_batch.T)]

                critic_value = torch.squeeze(critic_value)

                entropy = torch.mean(dist.entropy())

                # distribution ratio
                r_theta = torch.exp(probs - torch.squeeze(old_probs[batch]))

                advs = advantages[batch]
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)

                # policy clipping
                policy_obj = -r_theta * advs
                clipped_r_theta = -(
                    torch.clip(r_theta, 1 - self.EPS, 1 + self.EPS) * advs
                )
                # compute losses
                actor_loss = torch.mean(torch.maximum(policy_obj, clipped_r_theta))

                # clip value function
                if self.clip_value:
                    critic_loss_unclipped = torch.square(targets[batch] - critic_value)
                    v_loss_clipped = targets[batch] + torch.clamp(
                        critic_value - targets[batch], -self.V_EPS, self.V_EPS
                    )
                    critic_loss_clipped = torch.square(v_loss_clipped - targets[batch])
                    v_loss_max = torch.max(critic_loss_unclipped, critic_loss_clipped)
                    critic_loss = torch.mean(v_loss_max)
                else:
                    critic_loss = torch.mean(
                        torch.square(targets[batch] - critic_value)
                    )

                loss = actor_loss + self.C1 * critic_loss + self.C2 * entropy

                total_loss += loss
                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
                total_entropy_loss += entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(self.agent.parameters(), 0.5)
                self.opt.step()

                self.step += 1

            self.scheduler.step()

            self.loss_hist.append(total_loss.cpu().detach().numpy())
            self.actor_loss_hist.append(total_actor_loss.cpu().detach().numpy())
            self.critic_loss_hist.append(total_critic_loss.cpu().detach().numpy())
            self.entropy_hist.append(total_entropy_loss.cpu().detach().numpy())
            writer.add_scalar("training/loss", total_loss, self.step)
            writer.add_scalar("training/actor_loss", total_actor_loss, self.step)
            writer.add_scalar("training/critic_loss", total_critic_loss, self.step)
            writer.add_scalar("training/entropy_loss", total_entropy_loss, self.step)
            writer.add_scalar(
                "training/learning_rate", self.scheduler.get_last_lr()[0], self.step
            )
            if self.use_wandb:
                wandb.log(
                    {
                        "training/total_loss": total_loss,
                        "training/actor_loss": total_actor_loss,
                        "training/entropy": total_entropy_loss,
                        "training/critic_loss": total_critic_loss,
                        "training/learning_rate": self.scheduler.get_last_lr()[0],
                        "time/epoch": self.epoch,
                        "time/env_step": self.env_step,
                    }
                )

    def train(self):

        self.episode_reward_hist = []
        self.loss_hist = []
        self.entropy_hist = []
        self.actor_loss_hist = []
        self.critic_loss_hist = []
        mem = Memory(self.BATCH_SIZE)

        step = 0
        done = False
        episode_reward = 0
        episode = 1
        self.env_step = 0

        for epoch in range(self.EPOCHS):
            rewards = []
            actions = []
            values = []
            states = []
            next_states = []
            dones = []
            policy_logits = []
            self.epoch = epoch
            self.training = True
            state = self.env.reset()

            for t in range(self.TIMESTEPS):

                if self.render:
                    self.env.render()

                action, value, pls = self.select_action(
                    torch.tensor(state, dtype=torch.float32)
                )
                next_state, reward, done, _ = self.env.step(action.detach().numpy())
                episode_reward += reward

                actions.append(action)
                values.append(value.detach().numpy())
                rewards.append(reward)
                states.append(state)
                next_states.append(next_state)
                dones.append(done)
                policy_logits.append(pls.detach().numpy())

                if done:
                    state = self.env.reset()
                    step = 0
                    self.episode_reward_hist.append(episode_reward)
                    mean_episode_reward = np.mean(self.episode_reward_hist[-20:])
                    writer.add_scalar("training/reward", episode_reward, self.step)
                    writer.add_scalar(
                        "training/mean_reward", mean_episode_reward, self.step,
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "training/reward": episode_reward,
                                "training/mean_reward": mean_episode_reward,
                                "time/env_step": self.env_step,
                                "time/epoch": self.epoch,
                            }
                        )
                    episode_reward = 0
                    episode += 1

                else:
                    step += 1
                    state = next_state
                self.env_step += 1

            _, next_values, _ = self.agent(
                torch.tensor(next_states, dtype=torch.float32)
            )

            advantages, targets = self.compute_advantages(
                values, next_values.detach().numpy(), rewards, dones
            )

            mem.store_memory(
                states, actions, policy_logits, values, rewards, dones, advantages
            )
            writer.add_histogram("training/values", np.array(values), self.step)
            writer.add_histogram("training/rewards", np.array(rewards), self.step)
            writer.add_histogram("training/next_values", next_values, self.step)
            writer.add_histogram("training/advantages", advantages, self.step)
            writer.add_histogram("training/targets", targets, self.step)
            self.learn(mem, torch.tensor(targets, device=device))

            mem.clear_memory()
            self.agent.to("cpu")

            if epoch % self.print_frequency == 0:
                mean_reward = np.mean(self.episode_reward_hist[-50:])
                print(
                    f"Epoch: {epoch}\tLoss: {self.loss_hist[-1]:.2f}\tReward: {self.episode_reward_hist[-1]:.2f}\tMean reward: {mean_reward:.2f}\tLR: {self.opt.param_groups[0]['lr']:.2e}\tStep: {self.env_step:.1e}"
                )

            if (epoch + 1) % self.TEST_FREQ == 0:
                test_reward = self.test(30)

                if test_reward > self.REWARD_THRESHOLD:
                    print(f"Solved in {epoch} epochs")
                    break
                pass

    def plot_training(self):

        window = 10
        plot_alpha = 0.7
        blue = "#4184f3"
        style = ":"

        # loss and reward
        plt.figure(figsize=(10, 7))
        plt.subplot(2, 1, 1)
        plt.plot(self.loss_hist, blue, linestyle=style, label="Loss", alpha=plot_alpha)
        plt.plot(
            Series(self.loss_hist).rolling(window).mean(),
            blue,
            label=f"Rolling mean, {window}",
        )

        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.subplot(2, 1, 2)
        plt.plot(
            self.episode_reward_hist,
            blue,
            linestyle=style,
            label="Reward",
            alpha=plot_alpha,
        )
        plt.plot(
            Series(self.episode_reward_hist).rolling(window).mean(),
            blue,
            label=f"Rolling mean, {window}",
        )
        plt.ylabel("Episode reward")
        plt.xlabel("Training episode")

        plt.savefig("training_loss_reward.png")

        # diagnostics
        plt.figure(figsize=(10, 12))
        plt.subplot(3, 1, 1)
        plt.plot(
            self.actor_loss_hist, blue, linestyle=style, label="Loss", alpha=plot_alpha
        )
        plt.plot(
            Series(self.actor_loss_hist).rolling(window).mean(),
            blue,
            label=f"Rolling mean, {window}",
        )
        plt.ylabel("Actor loss")
        plt.xlabel("Epochs")

        plt.subplot(3, 1, 2)
        plt.plot(
            self.critic_loss_hist, blue, linestyle=style, label="Loss", alpha=plot_alpha
        )
        plt.plot(
            Series(self.critic_loss_hist).rolling(window).mean(),
            blue,
            label=f"Rolling mean, {window}",
        )
        plt.ylabel("Critic loss")
        plt.xlabel("Epochs")

        plt.subplot(3, 1, 3)
        plt.plot(
            self.entropy_hist, blue, linestyle=style, label="Loss", alpha=plot_alpha
        )
        plt.plot(
            Series(self.entropy_hist).rolling(window).mean(),
            blue,
            label=f"Rolling mean, {window}",
        )
        plt.ylabel("Entropy loss")
        plt.xlabel("Epochs")

        plt.savefig("training_loss_diag.png")

        plt.show()

    # def save_ppo_toml(self, path):
    #     with open(f"{path}/ppo.text", "w") as f:
    #         f.write(str(ppo.__dict__))

    def save(self):
        dt_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        path = "saved_models/saved_ppo_model_torch_{0}".format(dt_str)
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(self.agent.state_dict(), f"{path}/agent")
        # self.save_ppo_toml(path)
        print("Model saved")
        return path

    def load(self, PATH):
        self.agent = torch.load(PATH)
        print(f"Model {PATH} loaded")

    #######TESTING########
    def test(self, episodes, render=False):

        print("Testing agent")

        self.training = False
        frames = []
        test_episode_reward_hist = []

        if render:

            fig = plt.figure()

        for episode in range(episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            step = 0

            while not done:
                action, value, logits = self.select_action(
                    torch.tensor(state, dtype=torch.float32)
                )

                new_state, reward, done, info = self.env.step(action.detach().numpy())

                if render:
                    img = plt.imshow(self.env.render("rgb_array"))
                    plt.grid()
                    frames.append([img])

                state = new_state
                episode_reward += reward
                step += 1

            print(f"Episode {episode} lasted {step} steps. Reward: {episode_reward}")
            test_episode_reward_hist.append(episode_reward)
            writer.add_scalar("test/test_reward", episode_reward, self.step)
            if self.use_wandb:
                wandb.log(
                    {
                        "test/test_reward": episode_reward,
                        # "time/env_step": self.env_step,
                        "time/epoch": self.epoch,
                    }
                )
        writer.add_scalar(
            "test/mean_test_reward", np.mean(test_episode_reward_hist), self.step
        )
        if self.use_wandb:
            wandb.log(
                {
                    "test/mean_test_reward": np.mean(test_episode_reward_hist),
                    "time/env_step": self.env_step,
                    "time/epoch": self.epoch,
                }
            )
        if render:
            an = animation.ArtistAnimation(
                fig, frames, interval=100, repeat_delay=1000, blit=True
            )
            writergif = animation.PillowWriter(fps=30)
            an.save("animation.gif", writer=writergif)
        return np.mean(test_episode_reward_hist)

if __name__ == "__main__":
    env = gym.make('LunarLanderContinuous-v2')

    hyperparameter_defaults = dict(
        gamma=0.99,
        gae_lambda=0.95,
        eps=0.2,
        batch_size=128,
        timesteps=5000,
        n_epochs=40,
        c1=0.5,
        c2=0.001,
        clip_value=True,
        v_eps=0.5,
        actor_lr=3e-4,
        critic_lr=1e-3,
        decay_lr=True,
        lr_decay=1,
        epochs=150,
        test_freq=20,
        reward_threshold=150,
    )
    agent = Agent(env.observation_space.shape[0], env.action_space.shape[0])
    ppo = PPO(env, agent, hyperparameter_defaults, use_wandb=False)
    ppo.train()
    # ppo.plot_training()
    path = ppo.save()
    ppo.test(10, render=True)