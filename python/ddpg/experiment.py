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

        self._var = 3
        self._uniform_ratio = 1

        self._t1 = time.time()

    def _generate_bounded_uniform(self, _bound):
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
                self._env.lua_dostring(b'')
                s_ = self._env.read_state()
                r = numpy.abs(
                    numpy.double(s_.players[0].score) - \
                    numpy.double(s.players[0].score)) + \
                    numpy.abs(
                    numpy.double(s_.players[1].score) - \
                    numpy.double(s.players[1].score))

                self._ddpg.store_transition(s.to_tensor(), a, r / 10, s_.to_tensor())

                if self._ddpg.pointer % self._config.memory_capacity == 0:
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
