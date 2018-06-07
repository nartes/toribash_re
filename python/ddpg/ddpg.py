import tensorflow
import numpy
import pprint


class Config:
    def __init__(self, model_capacity=1):
        self.lr_a = 0.001    # learning rate for actor
        self.lr_c = 0.002    # learning rate for critic
        self.gamma = 0.9     # reward discount
        self.tau = 0.01      # soft replacement
        self.model_capacity = model_capacity
        self.layers_amount = 3
        self.a_dense_sizes = [(32 << self.model_capacity), ] * self.layers_amount
        self.c_dense_sizes = [(32 << self.model_capacity), ] * self.layers_amount
        self.learn_a = True
        self.learn_c = True
        self.timesteps = 3


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

        self.S = tensorflow.placeholder(tensorflow.float32, [None, self._config.timesteps, s_dim], 's')
        self.S_ = tensorflow.placeholder(tensorflow.float32, [None, self._config.timesteps, s_dim], 's_')
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

        self.a_loss = - tensorflow.reduce_mean(self._q)  # maximize the q
        self.atrain = tensorflow.train.RMSPropOptimizer(self._config.lr_a).minimize(self.a_loss, var_list=a_params)

        with tensorflow.control_dependencies(target_update):    # soft replacement happened at here
            self._q_target = self.R + self._config.gamma * q_
            self._td_error = tensorflow.squared_difference(self._q_target, self._q)

            self.c_optimizer = tensorflow.train.RMSPropOptimizer(self._config.lr_c)
            self.ctrain_grads = self.c_optimizer.compute_gradients(self._td_error, var_list=c_params)
            self.ctrain = self.c_optimizer.minimize(self._td_error, var_list=c_params)

        self.sess.run(tensorflow.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[numpy.newaxis, :, :]})[0]

    def get_q(self, s, a):
        return self.sess.run(
            self._q,
            {
                self.S: s[numpy.newaxis, :, :],
                self.a: a[numpy.newaxis, :]
            })[0]

    def learn(self):
        for k in range(2):
            if k == 0 and not self._config.learn_a or \
                k == 1 and not self._config.learn_c:
                continue

            if k == 0:
                batch_size = self._config.a_batch_size
            else:
                batch_size = self._config.c_batch_size

            indices_ = numpy.random.choice(self._config.memory_capacity - self._config.timesteps, batch_size)
            indices = (numpy.arange(self._config.timesteps * batch_size, dtype=numpy.int32) % \
                self._config.timesteps).reshape(-1, self._config.timesteps) + \
                indices_.reshape(batch_size, 1)

            bt = self.memory[indices.reshape(-1), :]
            bs = bt[:, :self.s_dim].reshape(batch_size, self._config.timesteps, -1)
            ba = self.memory[indices[:, -1].reshape(-1), :][:, self.s_dim: self.s_dim + self.a_dim]
            br = self.memory[indices[:, -1].reshape(-1), :][:, -self.s_dim - 1: -self.s_dim]
            bs_ = bt[:, -self.s_dim:].reshape(batch_size, self._config.timesteps, -1)

            if k == 0:
                self.sess.run(self.atrain, {self.S: bs})
                if self._config.dump_transition:
                    pprint.pprint({'a_loss:': self.sess.run(self.a_loss, {self.S: bs})})
            else:
                self.ctrain_grads_tensor = \
                    self.sess.run(self.ctrain_grads, {self.S: bs, self.a: ba, self.R: br, self.S_: bs})

                self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
                if self._config.dump_transition:
                    pprint.pprint({'td_error': \
                        self.sess.run(
                            self._td_error,
                            {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})})

    def store_transition(self, s, a, r, s_):
        transition = numpy.hstack((s, a, [r], s_))
        index = self.pointer % self._config.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tensorflow.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            s_ = tensorflow.reshape(s, shape=(-1, self._config.timesteps * self.s_dim))

            init_w = tensorflow.contrib.layers.xavier_initializer()
            init_b = tensorflow.constant_initializer(0.001)

            net = tensorflow.layers.dense(
                s_,
                self._config.a_dense_sizes[0],
                activation=tensorflow.nn.relu,
                name='l1',
                kernel_initializer=init_w,
                bias_initializer=init_b,
                trainable=trainable)

            i = 0
            for ds in self._config.a_dense_sizes[1:]:
                net = tensorflow.layers.dense(
                    net,
                    ds,
                    activation= \
                        tensorflow.nn.relu if i == len(self._config.a_dense_sizes) - 2 \
                        else tensorflow.nn.relu6,
                    name='l%d' % (i + 2),
                    kernel_initializer=init_w,
                    bias_initializer=init_b,
                    trainable=trainable)
                i += 1

            a = tensorflow.layers.dense(
                net, self.a_dim, activation=tensorflow.nn.tanh, name='a', trainable=trainable)

            return tensorflow.clip_by_value(
                a * (self.a_bound[1, :] - self.a_bound[0, :]) / 2.0 + \
                self.a_bound[0, :] + (self.a_bound[1, :] - self.a_bound[0, :]) / 2.0,
                self.a_bound[0, :], self.a_bound[1, :], name='clipped_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tensorflow.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            init_w = tensorflow.contrib.layers.xavier_initializer()
            init_b = tensorflow.constant_initializer(0.001)

            s_ = tensorflow.reshape(s, shape=(-1, self._config.timesteps * self.s_dim))

            n_l1 = self._config.c_dense_sizes[0]
            w1_s = tensorflow.get_variable('w1_s', [s_.shape[-1], n_l1], trainable=trainable,
                    initializer=init_w)
            w1_a = tensorflow.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable,
                    initializer=init_w)
            b1 = tensorflow.get_variable('b1', [1, n_l1], trainable=trainable, initializer=init_b)
            net = tensorflow.nn.relu(tensorflow.matmul(s_, w1_s) + tensorflow.matmul(a, w1_a) + b1)

            i = 0
            for ds in self._config.c_dense_sizes[1:]:
                net = tensorflow.layers.dense(
                    net,
                    ds,
                    activation= \
                        tensorflow.nn.relu if i == len(self._config.c_dense_sizes) - 2 \
                        else tensorflow.nn.relu6,
                    name='l%d' % (i + 1),
                    kernel_initializer=init_w,
                    bias_initializer=init_b,
                    trainable=trainable)
                i += 1

            return tensorflow.layers.dense(
                net, 1, trainable=trainable, activation=tensorflow.nn.tanh) * 2  # Q(s,a)
