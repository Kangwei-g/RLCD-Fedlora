"""DDQN layer-count controller and round-aligned training feedback."""
import json
import math
import random
from collections import deque
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from jl_controller_variants import get_spec

def encode_state(raw, rounds, normalize):
    t, delta, loss = map(float, raw)
    if rounds < 1 or not all((math.isfinite(x) for x in (t, delta, loss))) or min(t, delta, loss) < 0:
        raise ValueError('Invalid controller state')
    if normalize:
        return [min(t / rounds, 1.0), delta / (1.0 + delta), loss / (1.0 + loss)]
    return [t, delta, loss]

def reward_potential(metrics, mode):
    if mode in ('adaptive_f1_gain', 'fixed_f1_gain'):
        value = float(metrics['macro_f1'])
    elif mode == 'ce_gain':
        value = math.exp(-float(metrics['balanced_ce']))
    elif mode == 'brier_gain':
        value = 1.0 - float(metrics['balanced_brier'])
    elif mode == 'mixed_gain':
        value = 0.5 * (float(metrics['macro_f1']) + 1.0 - float(metrics['balanced_brier']))
    else:
        raise ValueError(mode)
    if not math.isfinite(value) or not -1e-08 <= value <= 1.0 + 1e-08:
        raise ValueError('Reward potential must be finite and in [0,1]')
    return min(1.0, max(0.0, value))

def epsilon_at(round_number, fast):
    if fast:
        return max(0.05, 0.5 - 0.45 * max(0, round_number - 2) / 78.0)
    return max(0.01, 0.995 ** max(0, round_number - 1))

class QNetwork(nn.Module):

    def __init__(self, actions, bounded):
        super().__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, actions)
        self.bounded = bounded

    def forward(self, x):
        x = self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))
        return torch.tanh(x) if self.bounded else x

class StudyDDQN:

    def __init__(self, spec, seed, actions):
        self.spec, self.actions = (spec, actions)
        self.rng = random.Random(seed + 17011)
        self.replay_rng = random.Random(seed + 29009)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 17011)
            self.q = QNetwork(actions, spec['bounded_q'])
            self.target = QNetwork(actions, spec['bounded_q'])
        self.target.load_state_dict(self.q.state_dict())
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=spec['learning_rate'])
        self.memory = deque(maxlen=5000)
        self.update_count = 0

    def choose(self, context, round_number):
        with torch.no_grad():
            values = self.q(torch.tensor([context], dtype=torch.float32))[0]
        if not torch.isfinite(values).all():
            raise RuntimeError('Nonfinite DDQN action values')
        epsilon = epsilon_at(round_number, self.spec['fast_epsilon'])
        explored = self.rng.random() < epsilon
        action = self.rng.randrange(self.actions) if explored else int(values.argmax())
        return (action, dict(epsilon=epsilon, explored=explored, values=values.tolist()))

    def observe(self, context, action, reward, next_context, done):
        self.memory.append((list(context), int(action), float(reward), list(next_context), bool(done)))
        losses = []
        if len(self.memory) < self.spec['replay_start']:
            return dict(losses=losses, replay_size=len(self.memory), update_count=self.update_count)
        for _ in range(self.spec['updates']):
            batch = self.replay_rng.sample(list(self.memory), min(64, len(self.memory)))
            states, actions, rewards, next_states, terminals = zip(*batch)
            states = torch.tensor(states, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            dones = torch.tensor(terminals, dtype=torch.float32) if self.spec['terminal'] else torch.zeros(len(batch))
            predicted = self.q(states).gather(1, actions[:, None]).squeeze(1)
            with torch.no_grad():
                next_actions = self.q(next_states).argmax(dim=1)
                next_values = self.target(next_states).gather(1, next_actions[:, None]).squeeze(1)
                targets = rewards + self.spec['gamma'] * (1.0 - dones) * next_values
                if self.spec['bounded_q']:
                    targets = targets.clamp(-1.0, 1.0)
            loss = F.smooth_l1_loss(predicted, targets) if self.spec['huber'] else F.mse_loss(predicted, targets)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite DDQN loss')
            self.optimizer.zero_grad()
            loss.backward()
            if self.spec['grad_clip'] is not None:
                nn.utils.clip_grad_norm_(self.q.parameters(), self.spec['grad_clip'], error_if_nonfinite=True)
            self.optimizer.step()
            with torch.no_grad():
                for target, online in zip(self.target.parameters(), self.q.parameters()):
                    target.lerp_(online, self.spec['tau'])
            self.update_count += 1
            losses.append(float(loss.detach()))
        return dict(losses=losses, replay_size=len(self.memory), update_count=self.update_count, target_min=float(targets.min()), target_max=float(targets.max()))

    def snapshot(self):
        return dict(q=self.q.state_dict(), target=self.target.state_dict(), optimizer=self.optimizer.state_dict(), memory=list(self.memory), action_rng=self.rng.getstate(), replay_rng=self.replay_rng.getstate(), update_count=self.update_count)

class ControllerStudy:

    def __init__(self, name, seed, rounds, output=None, actions=12, budgets=None):
        self.name, self.spec = (name, get_spec(name))
        self.seed, self.rounds, self.actions = (int(seed), int(rounds), int(actions))
        self.budgets = tuple(range(1, actions + 1)) if budgets is None else tuple(budgets)
        if len(self.budgets) != actions or len(set(self.budgets)) != actions or any((not isinstance(g, int) or g < 1 for g in self.budgets)):
            raise ValueError('Controller budgets must be distinct positive integers, one per action')
        self.output = Path(output) if output is not None else None
        self.agent = StudyDDQN(self.spec, seed, actions)
        self.pending, self.previous_potential, self.last_round = (None, None, 0)

    @property
    def epsilon(self):
        return epsilon_at(self.last_round, self.spec['fast_epsilon']) if self.spec['kind'] == 'ddqn' else None

    def finish_round(self, round_number, raw_state, metrics, executed_g):
        if round_number != self.last_round + 1 or executed_g not in self.budgets:
            raise ValueError('Controller observations must be ordered and have a valid executed G')
        if int(raw_state[0]) != round_number or round_number > self.rounds:
            raise ValueError('State round must match the observation')
        if self.spec['reward_threshold'] is not None and abs(float(metrics['threshold']) - self.spec['reward_threshold']) > 1e-09:
            raise ValueError('Reward threshold differs from the prespecified fixed value')
        context = encode_state(raw_state, self.rounds, self.spec['normalize'])
        potential = reward_potential(metrics, self.spec['reward'])
        reward = potential - self.previous_potential if self.previous_potential is not None else None
        terminal = round_number == self.rounds
        learned = None
        if self.pending is not None:
            if executed_g != self.budgets[self.pending['action']]:
                raise RuntimeError('Executed G does not match the action to receive credit')
            learned = self.agent.observe(self.pending['context'], self.pending['action'], reward, context, terminal)
        elif round_number > 2:
            raise RuntimeError('Missing controller decision after warm-up')
        event = dict(round=round_number, variant=self.name, raw_state=list(map(float, raw_state)), context=context, executed_G=executed_g, warmup=round_number <= 2, potential=potential, reward=reward, reward_mode=self.spec['reward'], reward_threshold=float(metrics['threshold']), server_macro_f1=float(metrics['macro_f1']), server_balanced_ce=float(metrics['balanced_ce']), server_balanced_brier=float(metrics['balanced_brier']), credited_decision=self.pending, learning=learned, terminal=terminal)
        self.pending = None
        next_g = None
        if not terminal and round_number >= 2:
            action, diagnostics = self.agent.choose(context, round_number)
            self.pending = dict(context=context, action=int(action), G=self.budgets[action], planned_for_round=round_number + 1, **diagnostics)
            next_g = self.budgets[action]
        event['next_decision'] = self.pending
        self.previous_potential, self.last_round = (potential, round_number)
        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=True)
            with (self.output / 'jl_controller_trajectory.jsonl').open('a') as stream:
                stream.write(json.dumps(event, allow_nan=False) + '\n')
            self.save()
        return (next_g, event)

    def save(self):
        state = dict(variant=self.name, spec=self.spec, seed=self.seed, rounds=self.rounds, budgets=list(self.budgets), last_round=self.last_round, previous_potential=self.previous_potential, pending=self.pending, controller=self.agent.snapshot(), note='Controller state only; not an exact federated-training resume checkpoint')
        path = self.output / 'jl_study_controller_last.pt'
        temp = path.with_suffix('.tmp')
        torch.save(state, temp)
        temp.replace(path)
