import numpy
import keras.models
import keras.layers
import keras.optimizers

import rl.processors
import rl.agents
import rl.memory
import rl.random

import ddpg

class MujocoProcessor(rl.processors.WhiteningNormalizerProcessor):
    def __init__(self, env, *args, **kwargs):
        super(MujocoProcessor, self).__init__(*args, **kwargs)
        self._env = env

    def process_action(self, action):
        delta = self._env.action_bound[1, :] - self._env.action_bound[0, :]
        mid = self._env.action_bound[0, :] + delta / 2

        return numpy.round(numpy.clip(
            numpy.float32(action * (delta / 2) + mid),
            self._env.action_bound[0, :],
            self._env.action_bound[1, :]))


class KerasRlExperiment:
    def __init__(self):
        # Get the environment and extract the number of actions.
        self.env = ddpg.toribash_env.ToribashEnvironment(action_filter=numpy.arange(22, dtype=numpy.int32))

        numpy.random.seed(123)
        self.nb_actions = self.env.action_dim

        self.generate_models()
        self.init_agent()

    def generate_models(self):
        # Next, we build a very simple model.
        actor = keras.models.Sequential()
        actor.add(keras.layers.Flatten(input_shape=(1,) + (self.env.state_dim,)))
        actor.add(keras.layers.Dense(400))
        actor.add(keras.layers.Dropout(0.2))
        actor.add(keras.layers.Activation('relu'))
        actor.add(keras.layers.Dense(400))
        actor.add(keras.layers.Dropout(0.2))
        actor.add(keras.layers.Activation('relu'))
        actor.add(keras.layers.Dense(400))
        actor.add(keras.layers.Dropout(0.1))
        actor.add(keras.layers.Activation('relu'))
        actor.add(keras.layers.Dense(300))
        actor.add(keras.layers.Dropout(0.1))
        actor.add(keras.layers.Activation('relu'))
        actor.add(keras.layers.Dense(self.nb_actions))
        actor.add(keras.layers.Activation('tanh'))
        print(actor.summary())

        self.actor = actor

        self.action_input = keras.layers.Input(shape=(self.nb_actions,), name='action_input')
        self.observation_input = keras.layers.Input(shape=(1,) + (self.env.state_dim,), name='observation_input')
        flattened_observation = keras.layers.Flatten()(self.observation_input)
        x = keras.layers.Dense(400)(flattened_observation)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dense(400)(flattened_observation)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Concatenate()([x, self.action_input])
        x = keras.layers.Dense(400)(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dense(300)(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dense(300)(x)
        x = keras.layers.Dropout(0.1)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dense(1)(x)
        x = keras.layers.Activation('linear')(x)
        critic = keras.models.Model(inputs=[self.action_input, self.observation_input], outputs=x)
        print(critic.summary())

        self.critic = critic

    def init_agent(self, warm_up=200):
        self.warm_up = warm_up

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        self.memory = rl.memory.SequentialMemory(limit=40000, window_length=1)
        self.random_process = \
            rl.random.GaussianWhiteNoiseProcess(
                size=self.nb_actions, mu=0., sigma=9.0, sigma_min=0.1, n_steps_annealing=16000)
        self.agent = rl.agents.DDPGAgent(
            nb_actions=self.nb_actions,
            actor=self.actor,
            critic=self.critic,
            critic_action_input=self.action_input,
            memory=self.memory,
            nb_steps_warmup_critic=warm_up,
            nb_steps_warmup_actor=warm_up,
            random_process=self.random_process,
            gamma=.99,
            target_model_update=1e-3,
            processor=MujocoProcessor(self.env))

        self.agent.compile([keras.optimizers.Adam(lr=1e-4), keras.optimizers.Adam(lr=1e-3)], metrics=['mae'])

    def train(self, epochs=10, log_interval=1000, nb_steps=None):
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
                log_interval=log_interval)

            # After training is done, we save the final weights.
            self.agent.save_weights('ddpg_{}_weights_{}.h5f'.format('mujoco', k), overwrite=True)

            # Finally, evaluate our algorithm for 1 episodes.
            self.agent.test(self.env, nb_episodes=1, visualize=False, nb_max_episode_steps=200)
