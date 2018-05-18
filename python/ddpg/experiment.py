import numpy
import time
import pprint
import tempfile
import io
import json

import ddpg

class Config(ddpg.ddpg.Config):
    def __init__(self):
        super(Config, self).__init__()

        self.memory_capacity = 10000
        self.batch_size = 32
        self.max_episodes = 200
        self.max_ep_steps = 200
        self.var_decay = 0.9995
        self.uniform_decay = 0.9995
        self.var = 3
        self.uniform_ratio = 1

class Experiment:
    def __init__(self, config=Config()):
        self._log_file = tempfile.mktemp(prefix='ddpg-', suffix='.log.txt')

        pprint.pprint({'log': self._log_file})

        self._config = config

        self._env = ddpg.toribash_env.ToribashEnvironment()

        self._s_dim = ddpg.toribash_env.toribash_state_t.DIM
        self._a_dim = ddpg.toribash_env.toribash_action_t.DIM // 2
        self._a_bound = ddpg.toribash_env.toribash_action_t.BOUNDS[:self._a_dim, :].T

        self._ddpg = ddpg.ddpg.DDPG(
            self._a_dim,
            self._s_dim,
            self._a_bound,
            config=self._config)

        self._var = self._config.var
        self._uniform_ratio = self._config.uniform_ratio

        self._t1 = time.time()

    @staticmethod
    def _generate_bounded_uniform(_bound):
        bound = _bound.reshape(2, -1)

        res = numpy.empty(bound.shape[1:], dtype=bound.dtype)

        for k in range(bound.shape[1]):
            res[k] = numpy.random.randint(
                bound[0, k],
                bound[1, k] + 1,
                dtype=res.dtype)

        return res.reshape(_bound.shape[1:])

    def _log(self, data):
        with io.open(self._log_file, 'a+') as f:
            f.write(u'' + json.dumps(data)+'\n')

        pprint.pprint(data)

    def train(self):
        self._env.lua_dostring(b'_toggle_ui()')
        self._env.lua_dostring(b'')
        initial_state = self._env.read_state()
        s = initial_state

        for i in range(self._config.max_episodes):
            ep_reward = 0
            for j in range(self._config.max_ep_steps):
                #if RENDER:
                #    env.render()

                # Add exploration noise
                _a_opponent = numpy.array([4,] * 20 + [0, 0], dtype=numpy.int32)

                _a = self._ddpg.choose_action(s.to_tensor())

                if numpy.random.uniform(0, 1) < self._uniform_ratio:
                    random_mixture = self._generate_bounded_uniform(self._a_bound)
                else:
                    random_mixture = numpy.random.normal(_a, self._var)

                a = numpy.round(numpy.clip(
                        numpy.float32(random_mixture),
                        self._a_bound[0, :],
                        self._a_bound[1, :]))

                self._env.make_action(numpy.concatenate([numpy.int32(a), _a_opponent]))

                _q = self._ddpg.get_q(s.to_tensor(), _a)

                self._env.lua_dostring(b'')
                s_ = self._env.read_state()
                r = numpy.double(s_.players[0].score) - \
                    numpy.double(s.players[0].score) + \
                    numpy.double(s_.players[0].injury) - \
                    numpy.double(s.players[0].injury) + \
                    -(numpy.double(s_.players[1].score) - \
                    numpy.double(s.players[1].score)) + \
                    -(numpy.double(s_.players[0].injury) - \
                    numpy.double(s.players[0].injury))

                pprint.pprint({
                    'a': a,
                    '_a': numpy.int32(_a),
                    '_q': _q,
                    'r': r,
                    'ddpg.pointer': self._ddpg.pointer})

                self._ddpg.store_transition(s.to_tensor(), a, r, s_.to_tensor())

                if self._ddpg.pointer >= self._config.memory_capacity:
                    self._var *= self._config.var_decay
                    self._uniform_ratio *= self._config.uniform_decay
                    self._ddpg.learn()

                if s_.world_state.match_frame == 0:
                    self._log({
                        'episode': i,
                        'reward': ep_reward,
                        'var': self._var,
                        'uniform_ratio': self._uniform_ratio,
                        'ddpg.pointer': self._ddpg.pointer})
                    break
                else:
                    ep_reward += r

                s = s_
                #if j == self._config.max_ep_steps - 1:

        print('Running time: ', time.time() - self._t1)

class SimpleExperiment:
    def __init__(self, config=Config()):
        self._log_file = tempfile.mktemp(prefix='ddpg-', suffix='.log.txt')

        pprint.pprint({'log': self._log_file})

        self._config = config
        self._env = ddpg.toribash_env.SimpleEnvironment()

        self._s_dim, self._a_dim, self._a_bound = self._env.get_parameters()

        self._ddpg = ddpg.ddpg.DDPG(
            self._s_dim,
            self._a_dim,
            self._a_bound,
            config = self._config)

        self._var = self._config.var

        self._t1 = time.time()

    def _log(self, data):
        with io.open(self._log_file, 'a+') as f:
            f.write(u'' + json.dumps(data)+'\n')

        pprint.pprint(data)

    def train(self):
        s = None

        for i in range(self._config.max_episodes):
            self._env.reset()

            ep_reward = 0

            s = self._env.read_state()

            for j in range(self._config.max_ep_steps):
                a = self._ddpg.choose_action(s)

                a_tilde = numpy.clip(
                    numpy.random.normal(a, self._var),
                    self._a_bound[0, :],
                    self._a_bound[1, :])

                try:
                    s_, r = self._env.make_action(a_tilde)
                except:
                    break

                self._ddpg.store_transition(s, a, r, s_)

                if self._ddpg.pointer >= self._config.memory_capacity:
                    self._var *= self._config.var_decay
                    self._ddpg.learn()

                ep_reward += r

                s = s_

            self._log({
                'episode': '%d' % i,
                'reward': '%d' % ep_reward,
                'var': '%.2lf' % self._var,
                'ddpg.pointer': '%d' % self._ddpg.pointer})

        print('Running time: ', time.time() - self._t1)
