"""Double DQN used to choose the active-layer budget."""

from collections import deque
import random
from typing import Deque, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch import optim


Transition = Tuple[Sequence[float], int, float, Sequence[float]]


class QNetwork(nn.Module):
    """Three-layer MLP used by the online and target Q networks."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = torch.relu(self.fc1(state))
        state = torch.relu(self.fc2(state))
        return self.fc3(state)


class DoubleDQNAgent:
    """Cold-start Double DQN matching the RLCD-FedLoRA controller."""

    def __init__(
        self,
        state_size: int = 3,
        action_size: int = 12,
        hidden_size: int = 64,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        memory_size: int = 5000,
        tau: float = 0.05,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.hidden_size = int(hidden_size)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.batch_size = int(batch_size)
        self.tau = float(tau)
        self.device = torch.device(device)
        self.memory = deque(maxlen=int(memory_size))  # type: Deque[Transition]
        self._python_rng = random.Random(int(seed))
        self._numpy_rng = np.random.default_rng(int(seed))

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.q_network = QNetwork(
                self.state_size, self.action_size, self.hidden_size
            ).to(self.device)
            self.target_network = QNetwork(
                self.state_size, self.action_size, self.hidden_size
            ).to(self.device)
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=float(learning_rate)
        )
        self.update_target_network()

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    def soft_update_target_network(self) -> None:
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_network.parameters(), self.q_network.parameters()
            ):
                target_param.copy_(
                    self.tau * online_param + (1.0 - self.tau) * target_param
                )

    def remember(
        self,
        state: Sequence[float],
        action: int,
        reward: float,
        next_state: Sequence[float],
    ) -> None:
        if not 0 <= int(action) < self.action_size:
            raise ValueError("Action is outside the DDQN action space")
        self.memory.append(
            (tuple(state), int(action), float(reward), tuple(next_state))
        )

    def act(self, state: Sequence[float]) -> int:
        """Return action index; RLCD-FedLoRA maps it to budget `action + 1`."""
        if self._numpy_rng.random() <= self.epsilon or len(self.memory) == 0:
            return self._python_rng.randint(0, self.action_size - 1)

        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def replay(self) -> Optional[float]:
        if len(self.memory) == 0:
            return None

        sample_size = min(len(self.memory), self.batch_size)
        batch = self._python_rng.sample(list(self.memory), sample_size)
        states, actions, rewards, next_states = zip(*batch)

        states_tensor = torch.as_tensor(
            np.asarray(states), dtype=torch.float32, device=self.device
        )
        actions_tensor = torch.as_tensor(
            actions, dtype=torch.long, device=self.device
        )
        rewards_tensor = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        )
        next_states_tensor = torch.as_tensor(
            np.asarray(next_states), dtype=torch.float32, device=self.device
        )

        predicted = self.q_network(states_tensor).gather(
            1, actions_tensor.unsqueeze(1)
        ).squeeze(1)
        with torch.no_grad():
            next_actions = self.q_network(next_states_tensor).argmax(dim=1)
            next_values = self.target_network(next_states_tensor).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            targets = rewards_tensor + self.gamma * next_values

        loss = nn.functional.mse_loss(predicted, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update_target_network()

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(
                self.epsilon_min, self.epsilon * self.epsilon_decay
            )
        return float(loss.detach().cpu().item())

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.q_network.parameters()) * 2

    def state_dict(self) -> Dict[str, object]:
        return {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "memory": list(self.memory),
            "python_rng_state": self._python_rng.getstate(),
            "numpy_rng_state": self._numpy_rng.bit_generator.state,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.q_network.load_state_dict(state["q_network"])
        self.target_network.load_state_dict(state["target_network"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.memory.clear()
        self.memory.extend(state.get("memory", []))
        if "python_rng_state" in state:
            self._python_rng.setstate(state["python_rng_state"])
        if "numpy_rng_state" in state:
            self._numpy_rng.bit_generator.state = state["numpy_rng_state"]
