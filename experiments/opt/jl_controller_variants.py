"""DDQN reference used in the paper experiments."""
from copy import deepcopy
BASE = dict(kind='ddqn', normalize=False, fast_epsilon=False, coordinate_exploration=0.2, reward='adaptive_f1_gain', reward_threshold=None, bounded_q=False, terminal=False, replay_start=1, updates=1, huber=False, grad_clip=None, tau=0.05, learning_rate=0.001, gamma=0.99)
VARIANTS = {'ddqn_s01': dict(BASE, label='RLCD-FedLoRA DDQN controller')}
ORDER = tuple(VARIANTS)

def get_spec(name):
    return deepcopy(VARIANTS[name])

def client_shuffle_seed(seed, round_index, client_id):
    if min(seed, round_index, client_id) < 0:
        raise ValueError('Seed, round and client indices must be nonnegative')
    return (int(seed) * 1000000007 + int(round_index) * 1000003 + int(client_id) * 1009 + 503) % (2 ** 63 - 1)
