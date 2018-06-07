import numpy
import time
import pprint
import tempfile
import io
import json
import matplotlib.pyplot
import pandas

import ddpg

class Config(ddpg.ddpg.Config):
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)

        self.memory_capacity = 10000
        self.c_batch_size = 32
        self.a_batch_size = 32
        self.max_episodes = 200
        self.max_ep_steps = 200
        self.var_decay = 0.9995
        self.var_min = 0.1
        self.epochs = 3
        self.uniform_decay = 0.9995
        self.var = 3
        self.uniform_ratio = 0
        self.dump_transition = False
        self.filter_validate_only = True
        self.timesteps = 3

class Experiment:
    def __init__(self, config=Config()):
        self._log_file = tempfile.mktemp(prefix='ddpg-', suffix='.log.txt')

        pprint.pprint({'log': self._log_file})

        self._config = config

        self._env = ddpg.toribash_env.ToribashEnvironment()

        self._ddpg = ddpg.ddpg.DDPG(
            self._env.action_dim,
            self._env.state_dim,
            self._env.action_bound,
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

        if self._config.filter_validate_only and data['validate'] or \
            not self._config.filter_validate_only:
            pprint.pprint(data)

    def show_stat(self):
        d = None
        with io.open(self._log_file, 'r') as f:
            d = pandas.DataFrame([json.loads(o) for o in f.read().split('\n') if len(o) > 0])

        matplotlib.pyplot.plot(d['episode'], d['reward'])
        matplotlib.pyplot.show()

    def train(self):
        s = None
        validate = None

        for i in range(self._config.max_episodes):
            s = self._env.reset()
            ep_reward = 0

            rb = []

            pprint.pprint({'ddpg.pointer': self._ddpg.pointer})

            if i % (self._config.max_episodes // self._config.epochs) == 0 or \
                i == self._config.max_episodes - 1:
                validate = True
            else:
                validate = False

            for j in range(self._config.max_ep_steps):
                if j == 0:
                    multiple_steps_ = [s] * self._config.timesteps
                elif j < self._config.timesteps:
                    multiple_steps_ = [rb[0][0]] * (self._config.timesteps - j) + \
                        [rb[i2][0] for i2 in range(j)]
                else:
                    multiple_steps_ = [rb[i2][0] for i2 in \
                        range(j - self._config.timesteps, j)]

                multiple_steps = numpy.stack(multiple_steps_)

                _a = self._ddpg.choose_action(multiple_steps)

                if validate:
                    random_mixture = _a
                else:
                    if numpy.random.uniform(0, 1) < self._uniform_ratio:
                        random_mixture = self._generate_bounded_uniform(self._a_bound)
                    else:
                        random_mixture = numpy.random.normal(_a, self._var)

                a = numpy.round(numpy.clip(
                        numpy.float32(random_mixture),
                        self._env.action_bound[0, :],
                        self._env.action_bound[1, :]))

                s_, r, done = self._env.step(a)

                _q = self._ddpg.get_q(multiple_steps, _a)

                if self._config.dump_transition:
                    pprint.pprint({
                        #'a': a,
                        '_a': numpy.round(_a),
                        '_q': _q,
                        'r': r,
                        #'s': s,
                        #'s_': s_,
                        #'j': j,
                        #'ddpg.pointer': self._ddpg.pointer})
                        })

                rb.append([s, a, r, s_])

                if self._ddpg.pointer >= self._config.memory_capacity and \
                    not validate:
                    self._var = max(self._var * self._config.var_decay, self._config.var_min)
                    self._uniform_ratio *= self._config.uniform_decay
                    self._ddpg.learn()

                if done or j == self._config.max_ep_steps - 1:
                    self._log({
                        'episode': i,
                        'reward': ep_reward,
                        'var': self._var,
                        'uniform_ratio': self._uniform_ratio,
                        'ddpg.pointer': self._ddpg.pointer,
                        'validate': validate
                        })
                    break
                else:
                    ep_reward += r

                s = s_

            if not validate:
                for tr in rb:
                    r = (ep_reward / len(rb) + tr[2])
                    if numpy.abs(r) > 1e-6:
                        r = r / (2 * numpy.abs(ep_reward))
                    self._ddpg.store_transition(tr[0], tr[1], r, tr[3])
                

        print('Running time: ', time.time() - self._t1)


class SimpleExperiment:
    def __init__(self, config=Config(), env=None):
        self._log_file = tempfile.mktemp(prefix='ddpg-', suffix='.log.txt')

        pprint.pprint({'log': self._log_file})

        self._config = config
        if env is None:
            self._env = ddpg.toribash_env.SimpleEnvironment()
        else:
            self._env = env

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

        if self._config.filter_validate_only and data['validate'] or \
            not self._config.filter_validate_only:
            pprint.pprint(data)

    def train(self):
        s = None
        validate = None

        for i in range(self._config.max_episodes):
            self._env.reset()

            ep_reward = 0

            s = self._env.read_state()

            if i % (self._config.max_episodes // self._config.epochs) == 0 or \
                i == self._config.max_episodes - 1:
                validate = True
            else:
                validate = False

            for j in range(self._config.max_ep_steps):
                a = self._ddpg.choose_action(s)

                if not validate:
                    a_tilde = numpy.clip(
                        numpy.random.normal(a, self._var),
                        self._a_bound[0, :],
                        self._a_bound[1, :])
                else:
                    a_tilde = a

                try:
                    s_, r = self._env.make_action(numpy.round(a_tilde))
                except ValueError:
                    break

                if self._config.dump_transition:
                    pprint.pprint({
                        's': s,
                        'a': a,
                        'a_tilde': a_tilde,
                        's_': s_,
                        'r': r})

                if not validate:
                    self._ddpg.store_transition(s, a_tilde, r, s_)

                    if self._ddpg.pointer >= self._config.memory_capacity:
                        self._var = max(self._var * self._config.var_decay, self._config.var_min)
                        self._ddpg.learn()

                ep_reward += r

                s = s_

            if validate:
                q_values = self._ddpg.sess.run(self._ddpg._q, {
                    self._ddpg.S: (numpy.arange(24) % 12)[:, numpy.newaxis],
                    self._ddpg.a: (numpy.array([0] * 12 + [1] * 12))[:, numpy.newaxis]}) \
                    .reshape(2,12).T
                actions = self._ddpg.sess.run(self._ddpg.a, {
                    self._ddpg.S: numpy.arange(12)[:, numpy.newaxis]})
            else:
                q_values = numpy.array([])
                actions = numpy.array([])

            self._log({
                'episode': '%d' % i,
                'reward': '%d' % ep_reward,
                'var': '%.2lf' % self._var,
                'ddpg.pointer': '%d' % self._ddpg.pointer,
                'validate': validate,
                'q_values': q_values.tolist(),
                'actions': [float(v) for v in actions],
            })

        print('Running time: ', time.time() - self._t1)
