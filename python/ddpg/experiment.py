import numpy
import time

import ddpg

class Experiment:
    def __init__(self):
        self._env = ddpg.toribash_env.ToribashEnvironment()

        self._s_dim = ddpg.toribash_env.toribash_state_t.DIM
        self._a_dim = ddpg.toribash_env.toribash_action_t.DIM
        self._a_bound = ddpg.toribash_env.toribash_action_t.BOUNDS.T

        self._ddpg = ddpg.ddpg.DDPG(self._a_dim, self._s_dim, self._a_bound)

        self._var = 3  # control exploration

        self._t1 = time.time()


    def train(self):
        self._env.lua_dostring(b'_toggle_ui()')
        self._env.lua_dostring(b'')
        s = self._env.read_state()

        for i in range(ddpg.ddpg.MAX_EPISODES):
            ep_reward = 0
            for j in range(ddpg.ddpg.MAX_EP_STEPS):
                #if RENDER:
                #    env.render()

                # Add exploration noise
                a = self._ddpg.choose_action(s.to_tensor())
                a = numpy.clip(
                    numpy.random.normal(a, self._var),
                    self._a_bound[0, :],
                    self._a_bound[1, :])    # add randomness to action selection for exploration
                self._env.make_action(numpy.int32(a))
                self._env.lua_dostring(b'')
                s_ = self._env.read_state()
                r = numpy.double(s_.players[0].score) - numpy.double(s.players[0].score)

                self._ddpg.store_transition(s.to_tensor(), a, r / 10, s_.to_tensor())

                if self._ddpg.pointer > ddpg.ddpg.MEMORY_CAPACITY:
                    self._var *= .9995    # decay the action randomness
                    self._ddpg.learn()

                s = s_
                ep_reward += r
                if j == ddpg.ddpg.MAX_EP_STEPS-1:
                    print('Episode: %i, Reward: %i, Explore: %.2f' % (i, int(ep_reward), self._var))
                    # if ep_reward > -300:RENDER = True
                    break

        print('Running time: ', time.time() - self._t1)
