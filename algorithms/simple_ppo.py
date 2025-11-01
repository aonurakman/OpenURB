import random
from typing import Sequence

import torch
import torch.nn as nn
import torch.optim as optim

from baseline_models import BaseLearningModel

__all__ = ["Network", "PPO"]

class Network(nn.Module):
    def __init__(self, in_size: int, out_size: int, num_hidden: int, widths: Sequence[int]):
        super().__init__()
        assert len(widths) == (num_hidden + 1), "DQN widths and number of layers mismatch!"
        self.input_layer = nn.Linear(in_size, widths[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(widths[idx], widths[idx + 1]) for idx in range(num_hidden)
        )
        self.out_layer = nn.Linear(widths[-1], out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        return self.out_layer(x)
    

### A simplified single-step actor-only PPO implementation for single-step decisions.
class PPO(BaseLearningModel):
    def __init__(self, state_size, action_space_size,
                 device="cpu", batch_size=16, lr=0.003, num_epochs=4, 
                 num_hidden=2, widths=[32, 64, 32], clip_eps=0.2, 
                 normalize_advantage=True, entropy_coef=0.3):
        super().__init__()
        self.device = device
        self.action_space_size = action_space_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.clip_eps = clip_eps
        self.normalize_advantage = normalize_advantage
        self.entropy_coef = entropy_coef

        self.policy_net = Network(state_size, action_space_size, num_hidden, widths).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.softmax = nn.Softmax(dim=-1)
        
        self.loss = list()
        self.memory = list()
        self.deterministic = False

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.policy_net(state_tensor)
            #logits = torch.clamp(logits, -10, 10)
            #logits = (logits - logits.min()) / (logits.max() - logits.min())
            #self.logits = logits
            probs = self.softmax(logits)
        dist = torch.distributions.Categorical(probs)
        if not self.deterministic: action = dist.sample().item()
        else: action = torch.argmax(probs).item()
        self.last_state = state
        self.last_action = action
        self.last_log_prob = dist.log_prob(torch.tensor(action)).item()
        return action

    def push(self, reward):
        self.memory.append((self.last_state, self.last_action, self.last_log_prob, reward))
        del self.last_state, self.last_action, self.last_log_prob

    def learn(self):
        if len(self.memory) < self.batch_size: return
        step_loss = list()

        for _ in range(self.num_epochs):
            batch = random.sample(self.memory, self.batch_size)
            states, actions, old_log_probs, rewards = zip(*batch)
            states_tensor = torch.FloatTensor(states).to(self.device)
            actions_tensor = torch.LongTensor(actions).to(self.device)
            old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
            rewards_tensor = torch.FloatTensor(rewards).to(self.device)
            # print(f"""
            # States: {states_tensor}, Actions: {actions_tensor},
            # Old Log Probs: {old_log_probs_tensor}, Rewards: {rewards_tensor}
            #       """)

            logits = self.policy_net(states_tensor)
            probs = self.softmax(logits)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions_tensor)

            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            #advantage = rewards_tensor
            if self.normalize_advantage: advantage = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
            else: advantage = rewards_tensor

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
            #loss = -torch.min(surr1, surr2).mean()
            entropy = dist.entropy().mean()
            loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()
            step_loss.append(loss.item())

        self.loss.append(sum(step_loss) / len(step_loss))
        self.memory.clear()
    