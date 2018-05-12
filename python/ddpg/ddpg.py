"""
Note: This is a updated version from my previous code,
for the target network, I use moving average to soft replace target parameters instead using assign function.
By doing this, it has 20% speed up on my machine (CPU).

Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
"""

import tensorflow
import numpy
import time
import ctypes
import io
import enum
import time
import os



#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
ENV_NAME = 'Pendulum-v0'


###############################  DDPG  ####################################


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = numpy.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=numpy.float32)
        self.pointer = 0
        self.sess = tensorflow.Session()

        self.a_dim = a_dim
        self.s_dim = s_dim
        self.a_bound = tensorflow.constant(a_bound, dtype=tensorflow.float32)

        self.S = tensorflow.placeholder(tensorflow.float32, [None, s_dim], 's')
        self.S_ = tensorflow.placeholder(tensorflow.float32, [None, s_dim], 's_')
        self.R = tensorflow.placeholder(tensorflow.float32, [None, 1], 'r')

        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )
        a_params = tensorflow.get_collection(tensorflow.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tensorflow.get_collection(tensorflow.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tensorflow.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tensorflow.reduce_mean(q)  # maximize the q
        self.atrain = tensorflow.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tensorflow.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tensorflow.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tensorflow.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tensorflow.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[numpy.newaxis, :]})[0]

    def learn(self):
        indices = numpy.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = numpy.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tensorflow.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tensorflow.layers.dense(s, 30, activation=tensorflow.nn.relu, name='l1', trainable=trainable)
            a = tensorflow.layers.dense(net, self.a_dim, activation=tensorflow.nn.tanh, name='a', trainable=trainable)
            return tensorflow.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tensorflow.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 30
            w1_s = tensorflow.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tensorflow.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tensorflow.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tensorflow.nn.relu(tensorflow.matmul(s, w1_s) + tensorflow.matmul(a, w1_a) + b1)
            return tensorflow.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


###############################  training  ####################################

class message_e(enum.Enum):
    TORIBASH_STATE      = 0
    TORIBASH_ACTION     = 1

class toribash_state(ctypes.Structure):
    class player(ctypes.Structure):
        _fields_ = [
            ('joints', ctypes.c_int32 * 20),
            ('grips', ctypes.c_int32 * 2)]

    _fields_ = [('players', player * 2)]

    def to_tensor(self):
        return numpy.concatenate([
            self.players[0].joints,
            self.players[0].grips,
            self.players[1].joints,
            self.players[1].grips])

    DIM = (20 + 2) * 2

class toribash_action(ctypes.Structure):
    class player(ctypes.Structure):
        _fields_ = [
            ('joints', ctypes.c_int32 * 20),
            ('grips', ctypes.c_int32 * 2)]
    _fields_ = [('players', player * 2)]

    DIM = (20 + 2) * 2
    BOUNDS = numpy.array(([[1,4],] * 20 + [[1,2],] * 2) * 2, dtype=numpy.int32)

    def to_tensor(self):
        return numpy.concatenate([
            self.players[0].joints,
            self.players[0].grips,
            self.players[1].joints,
            self.players[1].grips])

    @classmethod
    def from_tensor(cls, act_tensor):
        act = cls()

        for p in [0, 1]:
            for j in range(0, 20):
                act.players[p].joints[j] = act_tensor[p * 22 + j]
            for g in range(0, 2):
                act.players[p].grips[g] = act_tensor[p * 22 + g + 20]

        return act

def send_bytes(data):
    ddpg_socket_out = io.open("/tmp/patch_toribash_environment_ddpg_socket_in", "wb")
    ddpg_socket_out.write(data)
    ddpg_socket_out.close()

def read_bytes():
    fname_in = "/tmp/patch_toribash_environment_ddpg_socket_out"
    while not os.path.exists(fname_in):
       time.sleep(0.001)

    ddpg_socket_in = io.open(fname_in, "rb")

    res =  ddpg_socket_in.read()
    ddpg_socket_in.close()

    return res

def read_state():
    z = read_bytes()

    assert ctypes.c_int32.from_buffer_copy(z[:4]).value == message_e.TORIBASH_STATE.value
    assert ctypes.sizeof(toribash_state) == ctypes.c_int32.from_buffer_copy(z[4:][:4]).value
    st = toribash_state.from_buffer_copy(z[8:])

    return st

def make_action(a_tensor):
    act = toribash_action.from_tensor(a_tensor)

    send_bytes(
        bytes(ctypes.c_int32(message_e.TORIBASH_ACTION.value)) +
        bytes(ctypes.c_int32(ctypes.sizeof(toribash_action))) +
        bytes(act))

class Experiment:
    def __init__(self):
        self._s_dim = toribash_state.DIM
        self._a_dim = toribash_action.DIM
        self._a_bound = toribash_action.BOUNDS.T

        self._ddpg = DDPG(a_dim, s_dim, a_bound)

        self._var = 3  # control exploration

        self._t1 = time.time()

    def train(self):
        for i in range(MAX_EPISODES):
            s = read_state().to_tensor()
            ep_reward = 0
            for j in range(MAX_EP_STEPS):
                #if RENDER:
                #    env.render()

                # Add exploration noise
                a = self._ddpg.choose_action(s)
                a = numpy.clip(numpy.random.normal(a, self._var), -2, 2)    # add randomness to action selection for exploration
                make_action(numpy.int32(a))
                r = 1
                s_ = read_state().to_tensor()

                self._ddpg.store_transition(s, a, r / 10, s_)

                if self._ddpg.pointer > MEMORY_CAPACITY:
                    var *= .9995    # decay the action randomness
                    self._ddpg.learn()

                s = s_
                ep_reward += r
                if j == MAX_EP_STEPS-1:
                    print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                    # if ep_reward > -300:RENDER = True
                    break

        print('Running time: ', time.time() - self._t1)
