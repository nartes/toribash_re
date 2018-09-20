import numpy
import scipy
import io
import copy
import os
import tempfile
import pprint
import pickle
import h5py
import pandas
import functools
import multiprocessing
import multiprocessing.sharedctypes
import ctypes
import matplotlib.pyplot
import sklearn
import sklearn.ensemble
import sklearn.cluster
import sklearn.neural_network
import collections

import keras.models
import keras.layers
import keras.optimizers

import tensorflow

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
        limit,
        min_train_window_length=1,
        train_window_length=32,
        annealing_steps=1000,
        *args,
        **kwargs):

        super(LocalMemory, self).__init__(limit, *args, **kwargs)

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


class Datasets:
    @classmethod
    def get_entries_count(cls, raw_states_path):
        return os.stat(raw_states_path).st_size \
            // ctypes.sizeof(ddpg.toribash_env.toribash_state_t)

    @classmethod
    def get_dataset(
        cls,
        raw_states_path,
        start_entry=None,
        end_entry=None,
        rs_ctypes=None):
        if start_entry is None:
            start_entry = 0

        entries_count = cls.get_entries_count(raw_states_path)

        if end_entry is None:
            end_entry = entries_count

        assert start_entry >= 0 and start_entry < end_entry and end_entry <= entries_count

        if rs_ctypes is None:
            rs_ctypes = (ddpg.toribash_env.toribash_state_t * (end_entry - start_entry + 1))()

        with io.open(raw_states_path, 'rb') as f:
            f.seek(start_entry * ctypes.sizeof(ddpg.toribash_env.toribash_state_t))
            assert f.readinto(numpy.frombuffer(rs_ctypes, dtype=numpy.uint8)) == \
                ctypes.sizeof(rs_ctypes)

        return rs_ctypes

    @classmethod
    def get_dataset_random_part(
        cls,
        rs_dat_paths,
        dataset_split=1.0,
        granularity=10,
        validation_split=0.3,
        is_test=False,
        rs_ctypes=None):

        sizes = [cls.get_entries_count(p) for p in rs_dat_paths]

        z = numpy.sum(sizes)

        chunk_size = int(z * dataset_split / granularity)

        a = (z + chunk_size - 1) // chunk_size

        c = int(a * (1 - validation_split))
        d = a - c

        f = int(c * dataset_split)
        g = int(d * dataset_split)

        chunk_ids = numpy.sort(
            numpy.random.choice(d if is_test else c, g if is_test else f, replace=False)) \
            + (c if is_test else 0)

        if not is_test:
            chunks_boundaries = [0, int(z * (1 - validation_split))]
        else:
            chunks_boundaries = [int(z * (1 - validation_split)), z]

        subset_entries = chunk_ids.size * chunk_size

        if rs_ctypes is None:
            rs_ctypes = (ddpg.toribash_env.toribash_state_t * subset_entries)()

        assert len(rs_ctypes) >= subset_entries

        cum_sizes = numpy.cumsum(sizes)

        rs_pos = 0
        k = 0
        cur_pos = 0

        for i in chunk_ids:
            start_pos = max(chunk_size * i, chunks_boundaries[0])
            end_pos = min(chunk_size * (i + 1), chunks_boundaries[1])

            while cum_sizes[k] - sizes[k] + cur_pos < start_pos:
                if cum_sizes[k] < start_pos:
                    k += 1
                    cur_pos = 0
                else:
                    cur_pos = start_pos - cum_sizes[k] + sizes[k]

            assert cum_sizes[k] - sizes[k] + cur_pos < end_pos

            cur_start_pos = cum_sizes[k] - sizes[k] + cur_pos
            cur_end_pos = min(cum_sizes[k], end_pos)

            cls.get_dataset(
                rs_dat_paths[k],
                start_entry=cur_start_pos - cum_sizes[k] + sizes[k],
                end_entry=cur_end_pos - cum_sizes[k] + sizes[k],
                rs_ctypes = \
                    (rs_ctypes._type_ * (cur_end_pos - cur_start_pos)).from_buffer( \
                        numpy.frombuffer(rs_ctypes, numpy.uint8) \
                        .reshape(-1, ctypes.sizeof(rs_ctypes._type_)) \
                        [rs_pos:][:cur_end_pos - cur_start_pos]))

            rs_pos += cur_end_pos - cur_start_pos

            cur_pos += cur_end_pos - cur_start_pos

        rs_numpy = numpy.frombuffer(rs_ctypes, numpy.uint8) \
                .reshape(-1, ctypes.sizeof(rs_ctypes._type_))

        rs_numpy[rs_pos:] = rs_numpy[:rs_numpy.shape[0] - rs_pos]

        return rs_ctypes

    @classmethod
    def get_some_features(cls, raw_states, player=0):
        assert player in [0, 1]

        if player==0:
            injury_lambda = lambda rs: rs.players[1].injury - 0.95 * rs.players[0].injury
        else:
            injury_lambda = lambda rs: rs.players[0].injury - 0.95 * rs.players[1].injury

        d = pandas.DataFrame({
            'injuries': numpy.array([injury_lambda(rs) for rs in raw_states])})
        d['diff_injuries'] = d['injuries']
        d['diff_injuries'].iloc[1:] = d['injuries'].diff()
        d['match_frame'] = numpy.array([rs.world_state.match_frame for rs in raw_states])
        d['id'] = d.index

        return d

    @classmethod
    def scale_discretize_and_clip_value(
        cls,
        value,
        scale,
        clip_value):

        assert scale >= 0.0

        cur = value.copy()
        i1 = numpy.logical_and(numpy.abs(cur) < 10 ** scale, numpy.abs(cur) > 1e-6)
        cur[i1] = numpy.sign(cur[i1]) * (10 ** (scale + 1))
        cur /= 10 ** scale
        cur[numpy.abs(cur) < 1] = 1.0
        cur = numpy.round(numpy.sign(cur) * numpy.log10(numpy.abs(cur)))
        cur = numpy.clip(cur, -clip_value, clip_value)

        return cur


    @classmethod
    def get_distribution_features(
        cls,
        features,
        window_length,
        train_window_length,
        scale_diff_injury=2,
        maximum_log_diff_injury=2):

        assert maximum_log_diff_injury > 0
        assert scale_diff_injury > 0.0

        total_sequence_length = train_window_length + 1
        window_size = total_sequence_length - window_length

        raw = features

        z1 = raw['id'].values
        z2 = (numpy.arange(
            (z1.size - total_sequence_length + 1) * \
            total_sequence_length) % total_sequence_length) + \
            numpy.arange(
                z1.size - total_sequence_length + 1) \
                .repeat(total_sequence_length)
        z3 = z2.reshape(-1, total_sequence_length)

        mf_z3 = raw['match_frame'].values[z3]
        di_z3 = raw['diff_injuries'].values[z3]

        i1 = (numpy.sum(mf_z3[:, 1:] == 0, axis=1) == 0)

        assert window_size == 1

        i2 = cls.scale_discretize_and_clip_value(
            di_z3[:, -window_size:],
            scale_diff_injury,
            maximum_log_diff_injury).flatten()

        #i2 = numpy.sum(di_z3[:, -window_size:] > 0, axis=1)

        i3 = z3[:, -1][i1]

        s_i3 = pandas.DataFrame({'di_sum': i2[i1], 'id': i3}).sort_values(by=['di_sum'])

        u_i3 = numpy.unique(s_i3['di_sum'], return_counts=True)

        ps_u_i3 = numpy.concatenate([[0], numpy.cumsum(u_i3[1])])

        eta_k = u_i3[0].size - 1
        eta_l = numpy.arange(eta_k + 1)

        #eta_pi = scipy.misc.factorial(eta_k) / \
        #    scipy.misc.factorial(eta_l) / \
        #    scipy.misc.factorial(eta_k - eta_l) / \
        #    (2.0 ** eta_k)
        eta_pi = numpy.ones(eta_l.size) / eta_l.size

        eta_g = scipy.stats.rv_discrete(values=(eta_l, eta_pi))

        return {
            'distribution_features': {
                'z1': z1,
                'z2': z2,
                'z3': z3,
                'mf_z3': mf_z3,
                'di_z3': di_z3,
                'i1': i1,
                'i2': i2,
                'i3': i3,
                's_i3': s_i3,
                'u_i3': u_i3,
                'ps_u_i3': ps_u_i3,
                'eta_k': eta_k,
                'eta_l': eta_l,
                'eta_pi': eta_pi,
                'eta_g': eta_g
            },
            'attrs': {
                'window_length': window_length,
                'train_window_length': train_window_length,
                'window_size': window_size,
                'total_sequence_length': total_sequence_length
            }
        }

    @classmethod
    def get_balanced_raw_states_with_features(cls, raw_states, features, distribution_features):
        df = distribution_features['distribution_features']
        attrs = distribution_features['attrs']

        min_group_id = numpy.argmin(df['u_i3'][1])
        min_size = df['u_i3'][1][min_group_id]
        min_coef = df['eta_pi'] / df['eta_pi'][min_group_id]
        group_norm_coef = numpy.max(min_coef)
        group_sizes = numpy.int64(numpy.minimum(min_size * min_coef, df['u_i3'][1]))
        total_size = numpy.sum(group_sizes) * attrs['total_sequence_length']

        b_raw_states = (raw_states._type_ * total_size)()

        eta = numpy.arange(df['u_i3'][1].size).repeat(group_sizes)

        hamma = numpy.concatenate([
            numpy.random.permutation(numpy.arange(df['u_i3'][1][k]) + \
                df['ps_u_i3'][k])[:group_sizes[k]] \
            for k in range(len(group_sizes))])

        permute_eta_hamma_ids = numpy.random.permutation(numpy.arange(eta.size))

        eta_perm = eta[permute_eta_hamma_ids]
        hamma_perm = hamma[permute_eta_hamma_ids]

        fetch_ids = df['z1'][df['s_i3']['id'].values[hamma_perm]]

        fetch_ids_range = fetch_ids.repeat(attrs['total_sequence_length']) + \
            numpy.arange(
                attrs['total_sequence_length'] * fetch_ids.size) \
                % attrs['total_sequence_length'] - attrs['total_sequence_length'] + 1

        for k in range(len(b_raw_states)):
            b_raw_states[k] = raw_states[fetch_ids_range[k]]

        attrs['sequences_count'] = fetch_ids.size

        return {
            'features': features.iloc[fetch_ids_range],
            'raw_states': b_raw_states,
            'distribution_features': {
                'eta_perm': eta_perm,
                'hamma_perm': hamma_perm,
                'fetch_ids': fetch_ids
            },
            'attrs': attrs
        }

    @classmethod
    def generate_balanced_idxs(cls, distribution_features, batch_size):
        df = distribution_features['distribution_features']
        attrs = copy.deepcopy(distribution_features['attrs'])
        attrs['batch_size'] = batch_size

        while True:
            eta = df['eta_g'].rvs(
                size=attrs['batch_size'])

            xhi = numpy.random.randint(
                0,
                numpy.iinfo(numpy.int64).max,
                attrs['batch_size'])

            hamma = df['ps_u_i3'][eta] + xhi % df['u_i3'][1][eta]

            yield df['z1'][df['s_i3']['id'].values[hamma]]

    @classmethod
    def sample_brswf(cls, brswf, batch_size):
        return cls.generate_normalized_dataset(
            brswf['raw_states'],
            brswf['features'],
            brswf,
            cls.generate_balanced_idxs(
                brswf, batch_size=batch_size))

    @classmethod
    def generate_normalized_dataset(
        cls,
        raw_states,
        features,
        distribution_features,
        balanced_idxs,
        has_noise_injection=True):

        attrs = distribution_features['attrs']

        while True:
            indexes = next(balanced_idxs)

            batch_idxs = []

            X = ([], [])
            Y = []

            for i in indexes:
                for k in range(-attrs['window_size'] + 1, 1):
                        batch_idxs.append(i + k)

            for i in batch_idxs:
                assert i >= attrs['window_length'] and i < len(raw_states)

                x = numpy.stack(sum([
                    [numpy.array(raw_states[k].players[p].joints_pos_3d) \
                        for p in range(2)] for k in range(i - attrs['window_length'], i)], []))

                if has_noise_injection:
                    X[0].append(numpy.random.normal(x, 1e-5))
                else:
                    X[0].append(x)

                actions_list = sum([
                    [numpy.concatenate([numpy.array(raw_states[k].players[p].joints),
                     numpy.array(raw_states[k].players[p].grips)]) for p in range(2)] \
                     for k in range(i - attrs['window_length'] + 1, i + 1)], [])
                actions = numpy.stack(actions_list)

                if has_noise_injection:
                    X[1].append(numpy.random.normal(actions, 0.1))
                else:
                    X[1].append(actions)

            Y = features['diff_injuries'].values[batch_idxs]

            samples = [numpy.stack(x) for x in X], Y

            for iteration in range(attrs['window_size']):
                ret = [x[iteration :: attrs['window_size']] for x in samples[0]], \
                    samples[1][iteration :: attrs['window_size']]
                yield ret


class RawStatesMemory:
    def __init__(
        self,
        hdf_paths,
        dataset_split=1.0,
        validation_split=0.3,
        window_length=3,
        train_window_length=3,
        batch_size=64,
        player=0,
        scale_diff_injury=2,
        maximum_log_diff_injury=2):

        assert maximum_log_diff_injury > 0
        assert scale_diff_injury > 0.0

        self.hdf_paths = hdf_paths
        self.dataset_split = dataset_split

        self.batch_size = batch_size
        self.maximum_log_diff_injury = maximum_log_diff_injury
        self.scale_diff_injury = scale_diff_injury

        self.brswf = {}

        for is_test in [False, True]:
            rs = Datasets.get_dataset_random_part(
                self.hdf_paths,
                is_test=is_test,
                dataset_split=dataset_split,
                validation_split=validation_split)

            fs = Datasets.get_some_features(rs, player=player)

            df = Datasets.get_distribution_features(
                fs,
                window_length=window_length,
                train_window_length=train_window_length,
                scale_diff_injury=scale_diff_injury,
                maximum_log_diff_injury=self.maximum_log_diff_injury)
            pprint.pprint({
                'is_test': is_test,
                'u_i3_counts': df['distribution_features']['u_i3'][1]
            })

            df['raw_states'] = rs
            df['features'] = fs
            self.brswf[is_test] = df

            #Datasets.get_balanced_raw_states_with_features(rs, fs, df)

        self.sample_generator = [ \
            functools.partial(
                Datasets.sample_brswf,
                self.brswf[is_test],
                batch_size=batch_size)
            for is_test in [False, True]]

    def attrs(self, is_test=False):
        attrs = copy.deepcopy(self.brswf[is_test]['attrs'])
        attrs['batch_size'] = self.batch_size

        return attrs


    def nb_entries(self, is_test=False):
        return len(self.brswf[is_test]['raw_states'])

    def prioritized_sample(self, is_test=False):
        while True:
            yield from self.sample_generator[is_test]()

    def prioritized_sample_classification(self, **kwargs):
        g = self.prioritized_sample(**kwargs)
        while True:
            b = next(g)

            y_test = Datasets.scale_discretize_and_clip_value(
                b[1],
                self.scale_diff_injury,
                self.maximum_log_diff_injury) + self.maximum_log_diff_injury

            yield b[0], keras.utils.to_categorical(y_test, self.maximum_log_diff_injury * 2 + 1)

    def prioritized_sample_regression(self, **kwargs):
        g = self.prioritized_sample(**kwargs)
        while True:
            b = next(g)

            yield b

    def sample_for_ac_train(self, player=0, **kwargs):
        assert player in [0, 1]

        g = self.prioritized_sample_classification(**kwargs)

        while True:
            b = next(g)

            if player == 0:
                opponent_action_index = -1
            else:
                opponent_action_index = -2


            i1 = b[0][1][:, opponent_action_index, :]

            yield ([\
                b[0][0],
                b[0][1][:, :-2, :],
                i1.reshape((i1.shape[0], 1) + i1.shape[1:])], b[1])


class PatchedInputLayer(keras.layers.InputLayer):
    def __init__(self, numpy_input_tensor=None, *args, **kwargs):
        super(PatchedInputLayer, self).__init__(
            *args,
            **kwargs,
            input_tensor=keras.backend.tf.convert_to_tensor(numpy_input_tensor))

        self.numpy_input_tensor = numpy_input_tensor

    def get_config(self):
        config = {'numpy_input_tensor': self.numpy_input_tensor}

        base_config = super(PatchedInputLayer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class Models:
    def __init__(
        self,
        batch_size=32,
        window_length=1,
        maximum_log_diff_injury=2):

        assert maximum_log_diff_injury > 0

        assert window_length >= 1

        self.maximum_log_diff_injury = maximum_log_diff_injury

        self.input_shapes = [ \
            (2 * window_length, 3, 20),
            (2 * window_length, 22),
            (2 * (window_length - 1), 22),
            ]

        self.output_shapes = [
            (1,),
            (maximum_log_diff_injury * 2 + 1,),
            (22,)
            ]

        te = ddpg.toribash_env.ToribashEnvironment(None)

        self.output_bounds = [
            None,
            None,
            (te.action_bound[0, :22], te.action_bound[1, :22])
            ]

        self.batch_size = batch_size
        self.window_length = window_length

    def common_model(
        self,
        capacity=1,
        depth=[1, 1],
        inputs_idxs=numpy.s_[:],
        activations=('relu',),
        filters=(16, 16),
        dropout=0.3,
        is_regression=False,
        is_critic=True,
        is_batch_normalized=False,
        use_lstm=False):

        if type(depth) is not list:
            depth = [depth, depth]

        assert len(depth) == 2

        all_inputs = [keras.layers.Input(
            batch_shape=(self.batch_size,) + input_shape) for input_shape in self.input_shapes]

        inputs = []

        net = []

        for index, input_layer in zip(inputs_idxs, numpy.array(all_inputs)[inputs_idxs]):
            x = input_layer

            if index in [0]:
                x = keras.layers.Conv2D(
                    data_format='channels_first',
                    filters=filters[0],
                    kernel_size=(1, 1),
                    padding='same',
                    activation=activations[0],
                    kernel_initializer='glorot_normal',
                    bias_initializer='glorot_normal',
                    strides=(1, 1),
                    )(x)
                x = keras.layers.Dropout(dropout)(x)
                if is_batch_normalized:
                    x = keras.layers.BatchNormalization()(x)
                x = keras.layers.Conv2D(
                    data_format='channels_first',
                    filters=filters[0],
                    kernel_size=(3, 4),
                    padding='same',
                    activation=activations[0],
                    kernel_initializer='glorot_normal',
                    bias_initializer='glorot_normal',
                    strides=(1, 1),
                    )(x)
                x = keras.layers.Dropout(dropout)(x)
                if is_batch_normalized:
                    x = keras.layers.BatchNormalization()(x)
                if True:
                    x = keras.layers.Conv2D(
                        data_format='channels_first',
                        filters=filters[0],
                        kernel_size=(3, 4),
                        padding='valid',
                        activation=activations[0],
                        kernel_initializer='glorot_normal',
                        bias_initializer='glorot_normal',
                        strides=(3, 4),
                        )(x)
                    x = keras.layers.Dropout(dropout)(x)
                    if is_batch_normalized:
                        x = keras.layers.BatchNormalization()(x)
                if True:
                    x = keras.layers.Conv2D(
                        data_format='channels_first',
                        filters=filters[0],
                        kernel_size=(1, 4),
                        padding='valid',
                        activation=activations[0],
                        kernel_initializer='glorot_normal',
                        bias_initializer='glorot_normal',
                        strides=(1, 4),
                        )(x)
                    x = keras.layers.Dropout(dropout)(x)
                    if is_batch_normalized:
                        x = keras.layers.BatchNormalization()(x)
                if use_lstm:
                    x = keras.layers.Reshape(
                        target_shape=tuple([1, numpy.prod(x.shape.as_list()[1:])]))(x)
                    x = keras.layers.LSTM(
                        filters[1],
                        kernel_initializer='glorot_normal',
                        recurrent_initializer='glorot_normal',
                        bias_initializer='glorot_normal',
                        activation=activations[0],
                        dropout=dropout,
                        recurrent_dropout=dropout,
                        return_sequences=False)(x)
            elif index in [1, 2]:
                x = keras.layers.Conv1D(
                    data_format='channels_first',
                    filters=filters[0],
                    kernel_size=(1,),
                    padding='same',
                    activation=activations[0],
                    kernel_initializer='glorot_normal',
                    bias_initializer='glorot_normal',
                    strides=(1,),
                    )(x)
                x = keras.layers.Dropout(dropout)(x)
                if is_batch_normalized:
                    x = keras.layers.BatchNormalization()(x)
                x = keras.layers.Conv1D(
                    data_format='channels_first',
                    filters=filters[0],
                    kernel_size=(3,),
                    padding='same',
                    activation=activations[0],
                    kernel_initializer='glorot_normal',
                    bias_initializer='glorot_normal',
                    strides=(1,),
                    )(x)
                x = keras.layers.Dropout(dropout)(x)
                if is_batch_normalized:
                    x = keras.layers.BatchNormalization()(x)
                x = keras.layers.Conv1D(
                    data_format='channels_first',
                    filters=filters[0],
                    kernel_size=(3,),
                    padding='valid',
                    activation=activations[0],
                    kernel_initializer='glorot_normal',
                    bias_initializer='glorot_normal',
                    strides=(3,),
                    )(x)
                x = keras.layers.Dropout(dropout)(x)
                if is_batch_normalized:
                    x = keras.layers.BatchNormalization()(x)
                x = keras.layers.Conv1D(
                    data_format='channels_first',
                    filters=filters[0],
                    kernel_size=(3,),
                    padding='valid',
                    activation=activations[0],
                    kernel_initializer='glorot_normal',
                    bias_initializer='glorot_normal',
                    strides=(3,),
                    )(x)
                x = keras.layers.Dropout(dropout)(x)
                if is_batch_normalized:
                    x = keras.layers.BatchNormalization()(x)
                x = keras.layers.Conv1D(
                    data_format='channels_first',
                    filters=filters[0],
                    kernel_size=(2,),
                    padding='valid',
                    activation=activations[0],
                    kernel_initializer='glorot_normal',
                    bias_initializer='glorot_normal',
                    strides=(2,),
                    )(x)
                x = keras.layers.Dropout(dropout)(x)
                if is_batch_normalized:
                    x = keras.layers.BatchNormalization()(x)
                if use_lstm:
                    x = keras.layers.Reshape(
                        target_shape=tuple([1, numpy.prod(x.shape.as_list()[1:])]))(x)
                    x = keras.layers.LSTM(
                        filters[1],
                        kernel_initializer='glorot_normal',
                        recurrent_initializer='glorot_normal',
                        bias_initializer='glorot_normal',
                        activation=activations[0],
                        dropout=dropout,
                        recurrent_dropout=dropout,
                        return_sequences=False)(x)

            if len(x.shape) > 2:
                x = keras.layers.Flatten()(x)

            for d in range(depth[0]):
                x = keras.layers.Dense(32 << capacity, activation=activations[0])(x)
                x = keras.layers.Dropout(0.1)(x)

            net.append(x)

            inputs.append(input_layer)

        if len(net) > 1:
            x = keras.layers.Concatenate()(net)
        else:
            x = net[0]

        y = x

        for d in range(depth[1]):
            y = keras.layers.Dense(32 << capacity, activation=activations[0])(y)
            y = keras.layers.Dropout(dropout)(y)

        if is_critic:
            if is_regression:
                y = keras.layers.Dense(*self.output_shapes[0], activation='elu')(y)
            else:
                y = keras.layers.Dense(*self.output_shapes[1], activation='softmax')(y)
        else:
            y = keras.layers.Dense(*self.output_shapes[2], activation='elu')(y)

            bounds_layers =  \
                [PatchedInputLayer(numpy_input_tensor=numpy.float32(b)) \
                for b in self.output_bounds[2]]

            bounds_tensors = [il._inbound_nodes[0].output_tensors[0] for il in bounds_layers]

            inputs.extend(bounds_tensors)

            def clip(args):
                import keras
                return keras.backend.tf.clip_by_value(*args)

            y = keras.layers.Lambda(
                clip,
                output_shape=tuple(y.shape.as_list()[1:]))([y, *bounds_tensors])

        model = keras.models.Model(inputs=inputs, outputs=y)

        model.summary()

        return model

    def actor_critic(self, actor, critic, player=0):
        assert not critic.trainable

        assert player in [0, 1]

        i1_opponent = keras.layers.Input(batch_shape=(self.batch_size, 1, self.input_shapes[1][-1]))

        i1_actor_reshape = keras.layers.Reshape(
            target_shape=(i1_opponent.shape.as_list()[1:]))(actor.outputs[0])

        actions = [actor.inputs[1]]

        if player == 0:
            actions += [i1_actor_reshape, i1_opponent]
        else:
            actions += [i1_opponent, i1_actor_reshape]

        i1_all = keras.layers.Concatenate(axis=1)(actions)

        return keras.models.Model(
            inputs=[actor.inputs[0], actor.inputs[1], i1_opponent] + actor.inputs[2:],
            outputs=critic([actor.inputs[0], i1_all]))

    def ac_compile(self, ac):
        ac.compile(
            optimizer=keras.optimizers.Adam(lr=1e-4),
            loss=lambda yt, yp: \
                -keras.backend.relu((yp[:, 1] - 0.5) * 2))


class Helpers:
    def __init__(self, rsm):
        self.rsm = rsm

    def train_classification(
        self,
        model,
        train_steps_multiplier=1.0,
        test_steps_multiplier=1.0,
        epochs=10,
        verbose=1):

        model.fit_generator(
            self.rsm.prioritized_sample_classification(),
            validation_data=self.rsm.prioritized_sample_classification(is_test=True),
            steps_per_epoch=int(self.rsm.nb_entries() // \
                self.rsm.attrs()['batch_size'] * train_steps_multiplier),
            validation_steps=int(self.rsm.nb_entries(is_test=True) // \
                self.rsm.attrs()['batch_size'] * test_steps_multiplier),
            epochs=epochs,
            verbose=verbose)

    def train_regression(
        self,
        model,
        train_steps_multiplier=1.0,
        test_steps_multiplier=1.0,
        epochs=10,
        verbose=1):

        model.fit_generator(
            self.rsm.prioritized_sample_regression(),
            validation_data=self.rsm.prioritized_sample_regression(is_test=True),
            steps_per_epoch=int(self.rsm.nb_entries() // \
                self.rsm.attrs()['batch_size'] * train_steps_multiplier),
            validation_steps=int(self.rsm.nb_entries(is_test=True) // \
                self.rsm.attrs()['batch_size'] * test_steps_multiplier),
            epochs=epochs,
            verbose=verbose)

    def ac_train(
        self,
        model,
        train_steps_multiplier=1.0,
        test_steps_multiplier=1.0,
        epochs=10,
        verbose=1,
        player=0):

        model.fit_generator(
            self.rsm.sample_for_ac_train(player=player),
            validation_data=self.rsm.sample_for_ac_train(is_test=True, player=player),
            steps_per_epoch=int(self.rsm.nb_entries() // \
                self.rsm.attrs()['batch_size'] * train_steps_multiplier),
            validation_steps=int(self.rsm.nb_entries(is_test=True) // \
                self.rsm.attrs()['batch_size'] * test_steps_multiplier),
            epochs=epochs,
            verbose=verbose)

    @classmethod
    def model_input_from_raw_states(cls, raw_states, actor, shift=False, shrink=False):
        attrs = {
            'window_length': actor.input_shape[0][1] // 2,
            'window_size': 1,
            'batch_size': actor.input_shape[0][0]
        }

        assert shrink in ['begin', 'end', None, False]

        if len(raw_states) < attrs['window_length'] + 1:
            extended_raw_states = \
                raw_states[:1] * (attrs['window_length'] + 1 - len(raw_states)) + \
                raw_states
        else:
            extended_raw_states = raw_states[-(attrs['window_length'] + 1):]

        if shift:
            extended_raw_states.append(extended_raw_states[-1])

        features = pandas.DataFrame({
            'diff_injuries': numpy.zeros(len(extended_raw_states))
        })


        balanced_idxs = (i for i in [[len(extended_raw_states) - 1] * attrs['batch_size']])

        b = next(Datasets.generate_normalized_dataset(
            extended_raw_states,
            features,
            {'attrs': attrs},
            balanced_idxs,
            has_noise_injection=False))

        if shrink == 'begin':
            actions = b[0][1][:, 2:, :]
        elif shrink == 'end':
            actions = b[0][1][:, :-2, :]
        else:
            actions = b[0][1]

        return [b[0][0], actions]

    @classmethod
    def play_with_actor(cls, actor, client_key, has_workaround=False, player=0):
        assert player in [0, 1]

        try:
            raw_states = []

            te = ddpg.toribash_env.ToribashEnvironment(toribash_msg_queue_key=client_key)

            while True:
                st = te.reset()
                while True:
                    raw_states.append(te._cur_state)
                    act_player = actor.predict(
                        cls.model_input_from_raw_states(
                            raw_states,
                            actor,
                            shift=True,
                            shrink='end'))[0, ...].flatten()
                    if has_workaround:
                        act_player = numpy.clip(
                            act_player,
                            te.action_bound[0, :22],
                            te.action_bound[1, :22])
                    act_opponent = numpy.array([3,] * 20 + [1,] * 2, dtype=numpy.float32)

                    if player == 0:
                        act_total = [act_player, act_opponent]
                    else:
                        act_total = [act_opponent, act_player]

                    o, r, d, i = te.step(numpy.concatenate(act_total))
                    if d:
                        break
                if len(raw_states) > 100:
                    raw_states = raw_states[-100:]
        except KeyboardInterrupt:
            pass
        finally:
            te.close()

    @classmethod
    def mine_dataset(cls, client_key, steps_limit, save_prefix):
        assert steps_limit > 100

        iteration = 0
        while True:
            raw_states = []

            te = ddpg.toribash_env.ToribashEnvironment(toribash_msg_queue_key=client_key)
            while len(raw_states) < steps_limit:
                print("\rraw_states len is %d" % len(raw_states), end='')
                st = te.reset()
                while True:
                    raw_states.append(te._cur_state)
                    act = ddpg.experiment.Experiment._generate_bounded_uniform(te.action_bound)
                    o, r, d, i = te.step(act)
                    if d:
                        break
            te.close()

            b = (raw_states[0].__class__ * len(raw_states))()
            for k, v in zip(range(len(raw_states)), raw_states):
                if k % (len(raw_states) // 100) == 0:
                    print("\r%%=%f" % (k / len(raw_states) * 100))
                b[k] = v

            b2 = numpy.frombuffer(b, dtype=numpy.uint8)

            outf = tempfile.mktemp(
                dir='tmp',
                prefix='mine-dataset-%s-' % save_prefix,
                suffix='.dat')

            b2.tofile(outf)

            print(iteration)

            iteration += 1


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
        window_length=3,
        env=None):

        if env is None:
            env = ddpg.toribash_env.ToribashEnvironment()

        self.env = env
        self.window_length = window_length
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
            self.observation_input,
            keras.layers.Reshape(
                (self.window_length, self.nb_actions))(self.action_input)])
        x = keras.layers.RNN(
            [keras.layers.LSTMCell(256, dropout=0.3)] * 1,
            stateful=True)(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(256, activation='relu')(x)
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

    def train(self, epochs=20, nb_steps=None, verbose=1):
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
                verbose=verbose,
                log_interval=self.log_interval)

            # After training is done, we save the final weights.
            self.agent.save_weights('ddpg_{}_weights_{}.h5f'.format('mujoco', k), overwrite=True)

            # Finally, evaluate our algorithm for 1 episodes.
            self.agent.test(self.env, nb_episodes=10, visualize=False, nb_max_episode_steps=200)
