import numpy
import scipy
import io
import pickle
import h5py

import keras.models
import keras.layers
import keras.optimizers

import rl.processors
import rl.agents
import rl.memory
import rl.random

import ddpg


def add_uniform(_range):
    params = [
        numpy.arange(len(_range)),
        numpy.ones((len(_range))) / len(_range)]

    gen = scipy.stats.rv_discrete(name='custom', values=params)

    return lambda k=1: [_range[v] for v in gen.rvs(size=k)]

#class LocalProcessor(rl.processors.WhiteningNormalizerProcessor):
class LocalProcessor(rl.core.Processor):
    def __init__(
        self,
        env,
        *args,
        action_filter=None,
        **kwargs):
        super(LocalProcessor, self).__init__(*args, **kwargs)

        self.env = env

        if action_filter is None:
            action_filter = numpy.arange(
                self.env.action_bound.shape[1] // 2,
                dtype=numpy.int32)


        self.action_filter = action_filter

        tmp_ab = self.env.action_bound.copy()
        tmp_ab[1, :20] = 2
        tmp_ab[1, 22:42] = 2

        self.action_bound = tmp_ab[:, action_filter]
        self.action_dim = self.action_bound.shape[1]

        self.state_dim = self.env.state_dim + 3

        self.opponent_type = None

        self.cur_opponent = None
        self.opponent = add_uniform(['all_fixed', 'all_relaxed', 'ddpg'])
        #self.run_away = add_uniform([True, False])
        self.run_away = add_uniform([True])
        self.cur_opponent = None
        self.is_run_away = None
        self.goal_pos_3d = None

    def process_step(self, observation, reward, done, info):
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)

        joints_pos_3d = [numpy.array(self.env._cur_state.players[k].joints_pos_3d) for k in range(2)]

        if self.env._prev_state is not None:
            joints_pos_3d_prev = \
                [numpy.array(self.env._prev_state.players[k].joints_pos_3d) for k in range(2)]
        else:
            joints_pos_3d_prev = joints_pos_3d.copy()

        if self.env._prev_state is not None:
            injured = self.env._cur_state.players[0].injury - \
                self.env._prev_state.players[0].injury
        else:
            injured = self.env._cur_state.players[0].injury

        if self.is_run_away:
            dist_diff = ( \
                numpy.sqrt(numpy.sum(numpy.square(
                    self.goal_pos_3d - \
                    numpy.mean(joints_pos_3d[0], axis=1)))) - \
                numpy.sqrt(numpy.sum(numpy.square(
                    self.goal_pos_3d - \
                    numpy.mean(joints_pos_3d_prev[0], axis=1)))))

            reward = (1 if dist_diff > 0 else -1) + \
                (-1 if injured > 0 else 1)
        else:
            reward -= 10 * numpy.sqrt(numpy.sum(numpy.square(
                numpy.mean(joints_pos_3d[0], axis=1), - \
                numpy.mean(joints_pos_3d[1], axis=1)))) - \
                numpy.sqrt(numpy.sum(numpy.square(
                    numpy.mean(joints_pos_3d_prev[0], axis=1), - \
                    numpy.mean(joints_pos_3d_prev[1], axis=1))))

        return observation, reward, done, info

    def process_observation(self, observation):
        if self.env._cur_state.world_state.match_frame == 0:
            self.cur_opponent = self.opponent()[0]
            self.is_run_away = self.run_away()[0]

            self.goal_pos_3d = numpy.zeros(3)

            if self.is_run_away:
                self.run_dimension = numpy.random.randint(0, 3, 1)[0]
                self.run_direction = numpy.random.randint(0, 2, 1)[0]
                self.goal_pos_3d[self.run_dimension] = 200 * (2 * self.run_direction - 1)

        observation = numpy.concatenate([observation, self.goal_pos_3d])

        return observation

    def clip_action(self, action):
        delta = self.action_bound[1, :] - self.action_bound[0, :]
        mid = self.action_bound[0, :] + delta / 2

        return numpy.round(numpy.clip(
            numpy.float32(action * (delta / 2) + mid),
            self.action_bound[0, :],
            self.action_bound[1, :]))

    def set_actor(self, actor_train, actor_opponent):
        self.actor_train = actor_train
        self.actor_opponent = actor_opponent

    def set_memory(self, memory):
        self.memory = memory

    def set_random_process(self, random_process):
        self.random_process = random_process

    def process_action(self, action):
        action = self.clip_action(action)

        a = numpy.concatenate([self.env.action_bound[0, :]])

        if self.cur_opponent == 'all_fixed':
            _a_opponent = numpy.array([3,] * 20 + [0, 0], dtype=numpy.int32)
            a[22:] = _a_opponent
        elif self.cur_opponent == 'all_relaxed':
            _a_opponent = numpy.array([4,] * 20 + [0, 0], dtype=numpy.int32)
            a[22:] = _a_opponent
        elif self.cur_opponent == 'ddpg':
            state = numpy.stack(
                self.memory.get_recent_state(
                    numpy.concatenate(
                        [self.env._cur_state.to_tensor().flatten(), self.goal_pos_3d])))
            state = numpy.hstack([ \
                state[:, :1],
                state[:, 1:][:, :-3][:, (self.env.state_dim - 1) // 2:],
                state[:, 1:][:, :-3][:, :(self.env.state_dim - 1) // 2],
                state[:, -3:]])
            self.actor_opponent.set_weights(self.actor_train.get_weights())
            opponent_action = self.actor_opponent.predict(state[numpy.newaxis, :, :])[0, :]
            opponent_action += self.random_process.sample()
            _a_opponent = numpy.int32(self.clip_action(opponent_action))

            a[22:][self.action_filter] = _a_opponent
        else:
            raise ValueError('Wrong opponent type')

        a[self.action_filter] = action

        return a


class LocalMemory(rl.memory.SequentialMemory):
    def __init__(
        self,
        min_train_window_length=1,
        train_window_length=32,
        annealing_steps=1000,
        *args,
        **kwargs):

        super(LocalMemory, self).__init__(*args, **kwargs)

        self._max_train_window_length = train_window_length
        self._min_train_window_length = 1
        self._step = 0
        self._annealing_steps = annealing_steps
        self._train_window_length = self._min_train_window_length
        self._samples = None
        self._iteration = None
        self._batch_size = None

        self.agent = None

    def set_agent(self, agent):
        self.agent = agent

    def save(self, file_path):
        with h5py.File(file_path, 'w') as f:

            attrs = {
                'config': {
                    'min_train_window_length': self._min_train_window_length,
                    'max_train_window_length': self._max_train_window_length,
                    'train_window_length': self._train_window_length,
                    'annealing_steps': self._annealing_steps,
                    'step': self._step,
                    'samples': self._samples,
                    'iteration': self._iteration,
                    'batch_size': self._batch_size
                },
                'native_config': {
                    'window_length': self.window_length,
                    'ignore_episode_boundaries': self.ignore_episode_boundaries,
                    'limit': self.limit
                }
            }

            for k, v in attrs.items():
                g = f.create_group(k)

                for k, v in v.items():
                    if v is not None:
                        g.attrs[k] = v

            g = f.create_group('ring_buffers')

            for k, v in ({
                    'observations': self.observations,
                    'terminals': self.terminals,
                    'actions': self.actions,
                    'rewards': self.rewards
                    }).items():
                g[k] = numpy.stack(v.data)

    def load(self, file_path):
        self._samples = None
        self._iteration = None
        self._batch_size = None

        with h5py.File(file_path, 'r') as f:
            for k, v in f['config'].attrs.items():
                if k in [
                    'min_train_window_length',
                    'max_train_window_length',
                    'train_window_length',
                    'annealing_steps',
                    'step',
                    'samples',
                    'iteration',
                    'batch_size']:
                    setattr(self, '_' + k, v)

            for k, v in f['native_config'].attrs.items():
                if k in ['window_length', 'ignore_episode_boundaries', 'limit']:
                    setattr(self, k, v)

            for k, v in f['ring_buffers'].items():
                if k in ['observations', 'terminals', 'actions', 'rewards']:
                    rb = getattr(self, k)
                    rb.length = v.shape[0]
                    rb.maxlen = self.limit
                    rb.data = [e for e in numpy.array(v)]

    def sample(self, batch_size, validation_split=0.3, is_test=False):
        if self._samples is None or \
            self._iteration == self._train_window_length or \
            self._batch_size != batch_size:

            self._step += 1

            if self._step % self._annealing_steps == 0:
                self._train_window_length = max(
                    self._max_train_window_length,
                    self._train_window_length + 1)

            if self.agent is not None:
                self.agent.reset_train_states()

            self._batch_size = batch_size

            split_index = int(self.nb_entries * (1.0 - validation_split))

            if not is_test:
                indexes = rl.memory.sample_batch_indexes(
                    self.window_length + self._train_window_length - 1 + \
                    split_index,
                    self.nb_entries - 1,
                    size=self._batch_size)
            else:
                indexes = rl.memory.sample_batch_indexes(
                    self.window_length + self._train_window_length - 1,
                    self.nb_entries - 1 - split_index,
                    size=self._batch_size)

            batch_idxs = []
            for i in indexes:
                for k in range(-self._train_window_length + 1, 1):
                    batch_idxs.append(i + k)

            self._samples = super(LocalMemory, self).sample(
                self._batch_size * self._train_window_length,
                batch_idxs)
            self._iteration = 0

        ret = self._samples[self._iteration :: self._train_window_length]

        self._iteration += 1

        return ret

    def sample_to_critic_batch(self, sample):
        return [ \
            numpy.stack([o.action for o in sample]), \
            numpy.stack([numpy.stack(o.state0) for o in sample]), \
        ], \
            keras.utils.to_categorical(
                numpy.stack([o.reward for o in sample]), 5)


class KerasSequence(keras.utils.Sequence):
    def __init__(self, m, batch_size, validation_split, length):
        self._m = m
        self._batch_size = batch_size
        self._validation_split = validation_split
        self._length = length

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if idx < self._length * self._validation_split:
            return self._m.sample(
                batch_size=self._batch_size,
                validation_split=self._validation_split,
                is_test=False)
        else:
            return self._m.sample(
                batch_size=self._batch_size,
                validation_split=self._validation_split,
                is_test=True)


class LocalAgent(rl.agents.DDPGAgent):
    def __init__(self, actor_train, actor_validate, *args, **kwargs):
        super(LocalAgent, self).__init__(*args, actor=actor_train, **kwargs)

        assert type(self.memory) is LocalMemory

        self.actor_train = actor_train
        self.actor_validate = actor_validate

    def reset_train_states(self):
        if self.compiled:
            self.actor_train.reset_states()
            self.critic.reset_states()
            self.target_actor.reset_states()
            self.target_critic.reset_states()

    def reset_states(self):
        if self.random_process is not None:
            self.random_process.reset_states()
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.actor_validate.reset_states()

    def select_action(self, state):
        batch = self.process_state_batch([state])
        self.actor_validate.set_weights(self.actor_train.get_weights())
        action = self.actor_validate.predict_on_batch(batch).flatten()
        assert action.shape == (self.nb_actions,)

        # Apply noise, if a random process is set.
        if self.training and self.random_process is not None:
            noise = self.random_process.sample()
            assert noise.shape == action.shape
            action += noise

        return action


class KerasRlExperiment:
    def __init__(
        self,
        capacity=2,
        batch_size=128,
        warm_up=10000,
        env=None):

        if env is None:
            env = ddpg.toribash_env.ToribashEnvironment()

        self.env = env
        self.window_length = 3
        self.min_train_window_length = 10
        self.train_window_length = 20
        self.batch_size = batch_size
        self.warm_up = warm_up
        self.capacity = capacity

        self.processor = LocalProcessor(
            env=self.env,
            action_filter=numpy.arange(22, dtype=numpy.int32))

        numpy.random.seed(123)
        self.nb_actions = self.processor.action_dim

        self.generate_models()
        self.init_agent()

    def generate_models(self):
        self.action_input = keras.layers.Input(
            batch_shape=(self.batch_size, self.nb_actions,),
            name='action_input')

        self.observation_input = keras.layers.Input(
            batch_shape=(self.batch_size, self.window_length) + (self.processor.state_dim,),
            name='observation_input')

        self.actors = []

        for i in range(3):
            if i == 0:
                actor_input = self.observation_input
            else:
                actor_input = keras.layers.Input(shape=(self.window_length, self.processor.state_dim))

            #x = keras.layers.TimeDistributed(keras.layers.Dense(32 << self.capacity))(actor_input)
            #x = keras.layers.Dropout(0.3)(x)

            #x = keras.layers.RNN(
            #    [keras.layers.LSTMCell(32 << self.capacity, dropout=0.3) \
            #    for k in range(3)],
            #    stateful=True if i == 0 else False) \
            #    (x)

            #x = keras.layers.Dense(self.nb_actions)(x)
            #x = keras.layers.Activation('tanh')(x)

            x = keras.layers.Flatten()(actor_input)
            x = keras.layers.Dropout(0.3)(x)
            x = keras.layers.Dense(128, activation='relu')(x)
            x = keras.layers.Dropout(0.3)(x)
            x = keras.layers.Dense(self.nb_actions, activation='tanh')(x)

            actor = keras.models.Model(inputs=actor_input, outputs=x)

            print(actor.summary())

            self.actors.append(actor)

        #x = keras.layers.TimeDistributed(
        #    keras.layers.Dense(32 << self.capacity))(self.observation_input)
        #x = keras.layers.Dropout(0.3)(x)
        #x2 = keras.layers.Dense(32 << self.capacity)(self.action_input)
        #x2 = keras.layers.Dropout(0.3)(x2)
        #x2 = keras.layers.RepeatVector(self.window_length)(x2)
        #x = keras.layers.Concatenate()([x, x2])
        #x = keras.layers.RNN(
        #    [keras.layers.LSTMCell(32 << self.capacity, dropout=0.3) for k in range(3)],
        #    stateful=True)(x)
        #x = keras.layers.Dense(1)(x)
        #x = keras.layers.Activation('linear')(x)

        x = keras.layers.Concatenate()([ \
            keras.layers.Flatten()(self.observation_input),
            self.action_input])
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(5, activation='softmax')(x)

        critic = keras.models.Model(inputs=[self.action_input, self.observation_input], outputs=x)
        print(critic.summary())

        self.critic = critic

    def init_agent(self):

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        self.memory = LocalMemory(
            limit=self.warm_up,
            window_length=self.window_length,
            min_train_window_length=self.min_train_window_length,
            train_window_length=self.train_window_length)

        self.random_process = \
            rl.random.GaussianWhiteNoiseProcess(
                size=self.nb_actions,
                mu=0.,
                sigma=9.0,
                sigma_min=0.0,
                n_steps_annealing=self.warm_up * 4)

        self.processor.set_actor(
            actor_train=self.actors[0],
            actor_opponent=self.actors[2])
        self.processor.set_memory(self.memory)
        self.processor.set_random_process(
            rl.random.GaussianWhiteNoiseProcess(
                size=self.nb_actions,
                mu=0.,
                sigma=9.0,
                sigma_min=4.0,
                n_steps_annealing=self.warm_up * 4))

        self.agent = LocalAgent(
            batch_size=self.batch_size,
            nb_actions=self.nb_actions,
            actor_train=self.actors[0],
            actor_validate=self.actors[1],
            critic=self.critic,
            critic_action_input=self.action_input,
            memory=self.memory,
            nb_steps_warmup_critic=self.warm_up,
            nb_steps_warmup_actor=self.warm_up,
            random_process=self.random_process,
            gamma=.99,
            target_model_update=1e-3,
            processor=self.processor)

        self.memory.set_agent(self.agent)

        self.agent.compile([ \
            keras.optimizers.RMSprop(lr=1e-3),
            keras.optimizers.RMSprop(lr=1e-3)],
            metrics=['mse', 'mae'])

    def train(self, epochs=20, nb_steps=None):
        self.log_interval = self.warm_up // 2

        if nb_steps is None:
            nb_steps = self.warm_up

        # Okay, now it's time to learn something! We visualize the training here for show, but this
        # slows down training quite a lot. You can always safely abort the training prematurely using
        # Ctrl + C.
        for k in range(epochs):
            if k > 0:
                self.agent.nb_steps_warmup_critic = 0
                self.agent.nb_steps_warmup_actor = 0

            self.agent.fit(
                self.env,
                nb_steps=nb_steps,
                visualize=False,
                verbose=1,
                log_interval=self.log_interval)

            # After training is done, we save the final weights.
            self.agent.save_weights('ddpg_{}_weights_{}.h5f'.format('mujoco', k), overwrite=True)

            # Finally, evaluate our algorithm for 1 episodes.
            self.agent.test(self.env, nb_episodes=10, visualize=False, nb_max_episode_steps=200)
