import tensorflow
import numpy
import pprint


class Config:
    def __init__(self):
        self.lr_a = 0.001    # learning rate for actor
        self.lr_c = 0.002    # learning rate for critic
        self.gamma = 0.9     # reward discount
        self.tau = 0.01      # soft replacement
        self.model_capacity = 1


class DDPG(object):
    def __init__(
        self,
        a_dim,
        s_dim,
        a_bound,
        config):

        self._config = config

        self.memory = numpy.zeros((self._config.memory_capacity, s_dim * 2 + a_dim + 1), dtype=numpy.float32)
        self.pointer = 0
        self.sess = tensorflow.Session()

        self.a_dim = a_dim
        self.s_dim = s_dim
        self.a_bound = tensorflow.constant(a_bound, dtype=tensorflow.float32)

        self.S = tensorflow.placeholder(tensorflow.float32, [None, s_dim], 's')
        self.S_ = tensorflow.placeholder(tensorflow.float32, [None, s_dim], 's_')
        self.R = tensorflow.placeholder(tensorflow.float32, [None, 1], 'r')

        self.a = self._build_a(self.S,)
        self._q = self._build_c(self.S, self.a, )
        a_params = tensorflow.get_collection(tensorflow.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tensorflow.get_collection(tensorflow.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tensorflow.train.ExponentialMovingAverage(decay=1 - self._config.tau)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tensorflow.reduce_mean(self._q)  # maximize the q
        self.atrain = tensorflow.train.AdamOptimizer(self._config.lr_a).minimize(a_loss, var_list=a_params)

        with tensorflow.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + self._config.gamma * q_
            self._td_error = tensorflow.losses.mean_squared_error(labels=q_target, predictions=self._q)

            adam_optimizer = tensorflow.train.AdamOptimizer(self._config.lr_c)
            self.ctrain_grads = adam_optimizer.compute_gradients(self._td_error, var_list=c_params)
            self.ctrain = adam_optimizer.minimize(self._td_error, var_list=c_params)

        self.sess.run(tensorflow.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[numpy.newaxis, :]})[0]

    def get_q(self, s, a):
        return self.sess.run(
            self._q,
            {
                self.S: s[numpy.newaxis, :],
                self.a: a[numpy.newaxis, :]
            })[0]

    def learn(self):
        indices = numpy.random.choice(self._config.memory_capacity, size=self._config.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})

        self.ctrain_grads_tensor = \
            self.sess.run(self.ctrain_grads, {self.S: bs, self.a: ba, self.R: br, self.S_: bs})

        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = numpy.hstack((s, a, [r], s_))
        index = self.pointer % self._config.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tensorflow.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tensorflow.layers.dense(s, self._config.model_capacity, activation=tensorflow.nn.relu, name='l1', trainable=trainable)
            a = tensorflow.layers.dense(net, self.a_dim, activation=tensorflow.nn.tanh, name='a', trainable=trainable)
            return tensorflow.clip_by_value(a, self.a_bound[0, :], self.a_bound[1, :], name='clipped_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tensorflow.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = self._config.model_capacity
            w1_s = tensorflow.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tensorflow.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tensorflow.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tensorflow.nn.relu(tensorflow.matmul(s, w1_s) + tensorflow.matmul(a, w1_a) + b1)
            return tensorflow.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
