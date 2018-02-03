import io
import json
import re
import unittest
import glob
import os
import numpy
import pprint
import sys
import functools
import matplotlib.pyplot
import mpl_toolkits.mplot3d
import pickle
import copy
import tempfile
import pandas


class Statistics:
    def __init__(self):
        self._setup()

    def _setup(self):
        self._env = {}

        self._env['project_root'] = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), '..')
        self._env['replays'] = os.path.join(
            self._env['project_root'], 'build', 'toribash', '**', '*.rpl'
        )

        self._replays = glob.glob(self._env['replays'], recursive=True)

        self._env['verbose'] = True

    def parse_replay(self, fpath):
        replay_string = None

        with io.open(fpath, 'r') as f:
            replay_string = f.read()

        res = {'entries': []}

        lines = re.compile(r'[\n\r]+').split(replay_string)
        k = 0

        while k < len(lines):
            _not_white_space_regex = re.compile(r'[^\s]+')

            l = lines[k]

            _mg = None

            _mg = re.compile(r'\#?SCORE(.*)').match(l)
            if _mg:
                res['score'] = [int(x[0])
                                for x in _not_white_space_regex.finditer(_mg[1])]
                k += 1
                continue

            if re.compile(r'WORLD_SHADER(.*)').match(l) or \
                    re.compile(r'^#(.*)').match(l) or l == '':

                k += 1
                continue

            _mg = re.compile(r'NEWGAME(.*)').match(l)
            if _mg:
                res['newgame'] = _mg[1]
                k += 1
                continue

            if res.get('newgame') is None:
                k += 1
                continue

            _mg = re.compile(r'FRAME\s+(.*);(.*)').match(l)
            if _mg:
                entry = {
                    'frame': {},
                    'players': {},
                    'elements': {}
                }

                _frame_ints = \
                    [int(_mg[1])] + \
                    [int(x[0])
                     for x in _not_white_space_regex.finditer(_mg[2])]

                entry['frame'] = {
                    'timestamp': _frame_ints[0],
                    'score': _frame_ints[-2:],
                    'raw': _frame_ints
                }

                res['entries'].append(entry)
                k += 1
                continue

            for key in [
                'joint', 'pos', 'qat', 'linvel', 'angvel', 'grip', 'pos',
                'epos', 'eqat', 'elinvel', 'eangvel',
                    'emote', 'crush', 'cam', None]:
                _regex = re.compile(
                    r'({KEY})\s+([^;]+);(.*)'.replace('{KEY}', key.upper()))

                assert key != None

                _mg = _regex.match(l)

                if _mg is None:
                    continue

                _id = int(_mg[2])

                _raw = _mg.groups()[-1]

                _entry = {
                    'raw': _raw
                }

                if key == 'joint':
                    _joint_ints = [int(x[0])
                                   for x in _not_white_space_regex.finditer(_raw)]
                    _entry['ids'] = _joint_ints[0::2]
                    _entry['states'] = _joint_ints[1::2]
                elif key in ['emote', 'crash', 'cam']:
                    pass
                else:
                    _entry['floats'] = \
                        [float(x[0])
                         for x in _not_white_space_regex.finditer(_raw)]

                if key in ['epos', 'eqat', 'elinvel', 'eangvel']:
                    _key = key[1:]
                    _items = res['entries'][-1]['elements']
                else:
                    _key = key
                    _items = res['entries'][-1]['players']

                _item = _items.get(_id, {})

                _item[_key] = _entry

                _items[_id] = _item

                if _mg is not None:
                    break

            k += 1

        return res

    def cache_replays(self):
        res = []

        p = 0
        i = 0

        for f in self._replays:
            if self._env['verbose']:
                print("[%.2f] %s" % (i / len(self._replays) * 100, f))

            res.append({
                'fname': f,
                'replay': self.parse_replay(f)
            })

            dump_threshold = 10
            i += 1

            if i == len(self._replays) - 1 or \
                    1.0 * (i - p) / len(self._replays) * 100 > dump_threshold:
                out_name = os.path.join(
                    self._env['project_root'],
                    'build', 'replays_%d_%d.dat' % (i, dump_threshold))

                if self._env['verbose']:
                    print('[] [dump] %s' % out_name)

                with io.open(out_name, 'wb') as outf:
                    pickle.dump(res, outf)
                p = i

                del res
                res = []

    def helper(self, replays):
        res = []

        for r in replays:
            res.append(
                numpy.array(
                    [e['frame']['raw'][1:] for e in r['replay']['entries']],
                    dtype=numpy.int64
                )
            )

        return res

    def helper_2(self, scores):
        return numpy.hstack([
            numpy.max(scores[:, [0, 3]], axis=1).reshape(-1, 1),
            numpy.max(scores[:, [1, 2]], axis=1).reshape(-1, 1)
        ])

    def helper_3(self, count=-1):
        _files = []
        b = []

        __data = glob.glob('build/replays_*_*.dat')

        if count == -1:
            _data = __data
        else:
            _data = __data[:count]

        for _f in _data:
            a = pickle.load(io.open(_f, 'rb'))

            _b = self.helper(a)

            b.extend(copy.deepcopy(_b))

            del _b

            _files.extend([copy.deepcopy(x['fname']) for x in a])

            del a

        res = {
            '_files': _files,
            'b': b
        }

        _out_name = tempfile.mktemp(
            dir='build', prefix='helper_3_', suffix='.dat')

        if self._env['verbose']:
            print('[Statistics] [helper_3] dump to %s' % _out_name)

        with io.open(_out_name, 'wb') as outf:
            pickle.dump(res, outf)

        return res

    def helper_4(self, _b, _files, _threshold=10 ** 4):
        _i1 = [x.size > 0 and len(x.shape) ==
               2 and x.shape[1] == 4 for x in _b]

        b = [_b[k] for k in numpy.where(_i1)[0]]

        #e = numpy.concatenate([numpy.array([0]), numpy.cumsum([x.shape[0] for x in b])])

        #c = numpy.concatenate(b)
        #h = numpy.max(c[:, :2], axis=1)

        #i = numpy.where(h > _threshold)[0]
        #j = numpy.unique([numpy.max(numpy.where(z >= e)[0]) for z in i])

        s = numpy.array([numpy.max(x[-1, :]) for x in b])
        j = numpy.where(s > _threshold)[0]

        return {
            'b': b,
            '_i1': _i1,
            's': s,
            #'c': c,
            #'h': h,
            #'e': e,
            #'i': i,
            'j': j,
            'f': _files,
            'replays': '\n'.join([r'"' + os.path.split(_files[x])[1] + r'"' for x in j])
        }

    def helper_5(self, a, p=0):
        res = []
        for r in a:
            _res = {'joint': [], 'score': []}
            e_id = 0
            for e in r['replay']['entries']:
                if e['players'].get(p) is not None and e['players'][p].get('joint') is not None:
                    _res['joint'].append(
                        {'e_id': e_id, 'data': e['players'][p]['joint']})
                if e.get('frame') is not None:
                    _res['score'].append({'e_id': e_id, 'data': e['frame']})
                e_id += 1

            res.append(_res)

        return res

    def helper_6(self, b):
        res_1 = {'e_id': [], 'score': [], 'r_id': []}
        res_2 = {'e_id': [], 'joint': [], 'r_id': []}

        r_id = 0
        for y in b:
            for x in y['score']:
                res_1['e_id'].append(x['e_id'])
                res_1['r_id'].append(r_id)
                res_1['score'].append(x['data']['raw'][1:3])

            for x in y['joint']:
                res_2['e_id'].append(x['e_id'])
                res_2['r_id'].append(r_id)
                res_2['joint'].append(x['data']['raw'])

            r_id += 1

        res = {
            'scores': pandas.DataFrame(res_1),
            'joints': pandas.DataFrame(res_2),
        }

        res['merged'] = pandas.merge(res['scores'], res['joints'], left_on=['r_id', 'e_id'],
                                     right_on=['r_id', 'e_id'], how='outer')

        res['merged']['score0'] = numpy.array(
            list(res['merged']['score'].to_dict().values()))[:, 0]
        res['merged']['score1'] = numpy.array(
            list(res['merged']['score'].to_dict().values()))[:, 1]

        return res

    def helper_7(self, d, plot=True):
        z = numpy.diff(numpy.vstack([
            d['merged']['score0'],
            d['merged']['score1']]), axis=1).T

        _i1 = numpy.where(d['merged']['e_id'][1:] != 0)
        zzz = z[_i1[0]]
        _zz = zzz - numpy.mean(zzz)
        zz = _zz / numpy.sqrt(numpy.var(_zz))

        if plot:
            matplotlib.pyplot.figure()
            matplotlib.pyplot.plot(zzz)
            matplotlib.pyplot.figure()
            matplotlib.pyplot.plot(zz)
            matplotlib.pyplot.show()

        return {
            'z': z,
            'zzz': zzz,
            'zz': zz
        }

    def helper_8(self, b, method='scatter', *args, **kwargs):
        if not isinstance(b, list):
            b = [b]

        fig = matplotlib.pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')

        for line in b:
            getattr(ax, method)(
                xs=line[:, 0], ys=line[:, 1], zs=line[:, 2], marker='x', *args, **kwargs)

        fig.show()

    def helper_9(self, c):
        figs = []
        for k in range(c.shape[0]):
            b = c[k, :].squeeze()
            fig = matplotlib.pyplot.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xs=b[:, 0], ys=b[:, 1], zs=b[:, 2], marker='x')
            figs.append(fig)

        for fig in figs:
            fig.show()

    def helper_10(self, a, base_r_id=0):
        res = dict([(x, []) for x in
                    ['r_id', 'e_id', 'p_id', 'pos',
                     'action_states', 'action_ids', 'timestamp', 'score']])

        r_id = base_r_id
        for r in a:
            e_id = 0
            for e in r['replay']['entries']:
                for p_id in e['players'].keys():
                    p = e['players'][p_id]
                    res['r_id'].append(r_id)
                    res['e_id'].append(e_id)
                    res['p_id'].append(p_id)

                    if p.get('pos') is not None:
                        res['pos'].append(numpy.array(
                            p['pos']['floats']).reshape(-1, 3))
                    else:
                        res['pos'].append([])
                    if p.get('joint'):
                        res['action_ids'].append(p['joint']['ids'])
                        res['action_states'].append(p['joint']['states'])
                    else:
                        res['action_ids'].append([])
                        res['action_states'].append([])

                    res['timestamp'].append(e['frame']['timestamp'])
                    res['score'].append(e['frame']['raw'][1:])

                e_id += 1

            r_id += 1

        return pandas.DataFrame.from_dict(res)

    def helper_11(self, b, p_id=0, r_id=0):
        pos = b[b['p_id'] == p_id][b['r_id'] == r_id]['pos'] \
            .apply(functools.partial(numpy.mean, axis=0))

        return numpy.stack(pos)

    def helper_12(self, b, r_id=0):
        self.helper_8([
            self.helper_11(b, p_id=1, r_id=r_id),
            self.helper_11(b, p_id=0, r_id=r_id)], method='plot')

    def minus_list(self, a, b):
        return [x for x in a if not x in b]

    def _helper_13(self, b):
        c = b
        _d = c[c['pos'].apply(len) != 0]
        d = _d[['pos', 'timestamp', 'e_id', 'r_id', 'p_id']]
        e = d.assign(mean_pos=lambda x: x['pos'].apply(
            functools.partial(numpy.mean, axis=0)))
        z = {'r_id': [], 'p_id': [], 'e_id': [],
             'mean_pos': [], 'timestamp': []}
        for r_id in e['r_id'].unique():
            f = e[e['r_id'] == r_id]
            g = numpy.unique(f['timestamp'], return_index=True)[1]
            for p_id in f['p_id'].unique():
                i = f[f['p_id'] == p_id]
                e_ids = numpy.sort(f.iloc[g]['e_id'])
                _mean_pos = i['mean_pos']
                _e_ids = i['e_id']
                h = {
                    'r_id': [r_id, ] * e_ids.size,
                    'p_id': [p_id, ] * e_ids.size,
                    'e_id': numpy.array(e_ids).tolist(),
                    'mean_pos': numpy.apply_along_axis(lambda x: numpy.interp(e_ids, _e_ids, x),
                                                       axis=0, arr=numpy.stack(_mean_pos)).tolist(),
                    'timestamp': numpy.sort(numpy.array(f['timestamp'].unique())).tolist()
                }
                for k in h.keys():
                    z[k].extend(h[k])
                del h
                del i
                del e_ids
                del _e_ids
            del f
            del g

        y = pandas.DataFrame.from_dict(z)
        w = pandas.merge(_d[self.minus_list(list(_d.columns), ['timestamp'])], y,
                         on=self.minus_list(list(y.columns), ['mean_pos', 'timestamp']), how='outer')
        u = w.sort_values(by=['r_id', 'e_id', 'p_id'])

        return u

    def helper_13_merge_mean_pos(self, a):
        b = a.sort_values(by=['r_id', 'e_id'])

        b['mean_pos'] = list(numpy.mean(numpy.stack(b['pos'].values), axis=1))

        c = a.groupby(['r_id', 'e_id'])[['r_id', 'e_id']].first().values

        timestamp = a.groupby(['r_id', 'e_id'])['timestamp'].first().values

        d = {'p_id': [], 'common_mean_pos': []}

        assert b['p_id'].unique().size == 2

        for p_id in b['p_id'].unique():
            e = b[b['p_id'] == p_id][['mean_pos', 'timestamp']]
            cur_mean_pos = numpy.stack(e['mean_pos'].values)
            f = numpy.array([numpy.interp(timestamp, e['timestamp'].values, cur_mean_pos[:, k]) for k in
                             range(cur_mean_pos.shape[-1])]).T
            d['p_id'].append(numpy.array(
                [p_id], dtype=numpy.uint8).repeat(f.shape[0]))
            d['common_mean_pos'].extend(list(f))

        g = pandas.DataFrame({
            'r_id': c[:, 0].repeat(2).reshape(-1, 2).T.reshape(-1),
            'e_id': c[:, 1].repeat(2).reshape(-1, 2).T.reshape(-1),
            'p_id': numpy.concatenate(d['p_id']),
            'mean_pos': d['common_mean_pos'],
            'timestamp': timestamp.repeat(2).reshape(-1, 2).T.reshape(-1)
        })

        return pandas.merge(a[self.minus_list(a.columns, ['timestamp'])],
                            g, on=['r_id', 'e_id', 'p_id'], how='right')

    def helper_13(self, a):
        a = a[a['p_id'] < 2]
        a = a[a['pos'].apply(lambda x: isinstance(x, numpy.ndarray))]
        a = a[a['score'].apply(lambda x: len(x)) == 4]

        a = self.helper_13_merge_mean_pos(a)

        return a

    def helper_14(self, b, p_id=0, r_id=0):
        pos = b[(b['r_id'] == r_id) & (b['p_id'] == p_id)]['mean_pos']

        return numpy.stack(pos)

    def helper_15(self, b, r_id=0):
        rest = b[b['r_id'] == r_id][['timestamp', 'score']]
        _i1 = numpy.where([not x is numpy.NaN for x in rest['score']])[0]

        self.helper_16(numpy.stack([
            self.helper_14(b, p_id=1, r_id=r_id),
            self.helper_14(b, p_id=0, r_id=r_id)]), method='plot',
            rest=rest.iloc[_i1][['timestamp', 'score']])

    def helper_17(self, b):
        assert len(b.shape) == 3 and b.shape[0] == 2

        return numpy.sqrt(numpy.sum(numpy.square(b[0, :] - b[1, :]), axis=-1))

    def helper_16(self, b, rest, method='scatter', *args, **kwargs):
        figs = []

        fig = matplotlib.pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')

        for k in range(b.shape[0]):
            line = b[k, :].reshape(-1, 3)
            getattr(ax, method)(
                xs=line[:, 0], ys=line[:, 1], zs=line[:, 2], marker='x', *args, **kwargs)

        figs.append(fig)

        fig = matplotlib.pyplot.figure()
        ax2 = fig.add_subplot(211)
        ax2.plot(self.helper_17(b))

        ax3 = fig.add_subplot(212)
        ax3.plot(rest['timestamp'], numpy.stack(rest['score'])[:, :2])

        figs.append(fig)

        for f in figs:
            f.show()

    def helper_18(self, c):
        for r_id in c[(c['score'].apply(numpy.max) > 10 ** 5)]['r_id'].unique():
            self.helper_15(c, r_id=r_id)

    def _helper_19(self, c):
        res = {'r_id': [], 'distance': [], 'score': []}
        for r_id in c['r_id'].unique():
            res['r_id'].append(r_id)
            res['distance'].append(
                self.helper_17(numpy.stack([
                    self.helper_14(c, p_id=p_id, r_id=r_id) for p_id in [0, 1]
                ])).mean()
            )
            res['score'].append(
                c[(c['r_id'] == r_id) & c['score'].notna()]['score']
                .apply(lambda x: numpy.max(x[:2])).max()
            )

        return pandas.DataFrame(res)

    def helper_19(self, a, **kwargs):
        a = a.sort_values(by=['r_id', 'e_id', 'p_id'])

        b = a[a['score'].notna()]
        b['score'] = b['score'].apply(numpy.array)
        c = b.groupby(['r_id']).agg({
            'score': lambda x: numpy.max(numpy.stack(x)[:, :2].reshape(-1)),
            'r_id': lambda x: x.values[0]
        })

        d = self.helper_19_distance(a, **kwargs)

        return pandas.DataFrame({
            'r_id': c['r_id'].values,
            'score': c['score'].values,
            'distance': d.values
        })

    def helper_19_distance(self, a, f=lambda x: x.mean()):
        e = a.sort_values(by=['r_id', 'e_id', 'p_id']) \
             .groupby(['r_id'])['mean_pos'] \
             .apply(lambda x:
                    f(numpy.sqrt(numpy.sum(numpy.square(numpy.diff(
                        numpy.stack(x).reshape(-1, 2, 3), axis=1)), axis=-1)).squeeze()))

        return e

    def helper_20(self):
        a = pickle.load(io.open("build/replays_1332_10.dat", 'rb'))
        b = self.helper_10(a)
        c = self.helper_13(b)
        e = self.helper_19(c)
        matplotlib.pyplot.scatter(e['distance'], e['score'])
        matplotlib.pyplot.show()

    def helper_21(self, count=-1):
        res = []

        _data = sorted(glob.glob('build/replays_*_*.dat'))[:count]

        base_r_id = 0

        count = 0

        def progress(_count, progr, state='start'):
            print("[helper_21] %s #%d of %d, progress %.2f%%"
                  % (state, _count, len(_data), 100.0 * (progr) / len(_data)))

        for _f in _data:
            progress(count + 1, count)
            a = None

            with io.open(_f, 'rb') as _inf:
                a = pickle.load(_inf)

            b = self.helper_10(a, base_r_id=base_r_id)

            res.append(b)

            base_r_id += b.index.size

            del a

            progress(count + 1, count + 1, state='end')

            count += 1

        return res

    def helper_22(self, a):
        for k in range(len(a)):
            pickle.dump(a[k], io.open('build/p%d.dat' % k, 'wb'))

    def helper_23(self):
        a = []

        for k in sorted([
                int(re.compile(r'\d').search(x)[0]) for x in
                glob.glob('build/p*.dat')]):
            a.append(pickle.load(io.open('build/p%d.dat' % k, 'rb')))

        return pandas.concat(a)

    def helper_24(self, c):
        figs = []

        for _prefix, _func in [('max_minus_min', lambda x: x.max() - x.min()),
                               ('mean', lambda x: x.mean()),
                               ('min', lambda x: x.min()),
                               ('max', lambda x: x.max()),
                               ('max_diff', lambda x: numpy.max(numpy.abs(numpy.diff(x))))]:
            d = self.helper_19(c, f=_func)
            fig = matplotlib.pyplot.figure()
            ax = fig.add_subplot(111)
            ax.scatter(d['distance'], d['score'])
            figs.append((_prefix, fig))

        for _prefix, f in figs:
            f.savefig(tempfile.mktemp(
                dir='build', prefix='helper_24_' + _prefix + '_', suffix='.svgz'))
            f.show()

    def helper_25(self):
        b = self.helper_23()
        c = self.helper_13(b)
        self.helper_24(c)

    def helper_26(self, n_samples):
        samples = []

        one = numpy.random.randint(0, 100, (n_samples, 2)) / 100.0
        two = numpy.random.randint(0, 100, (n_samples, 2)) / 100.0

        samples.append([one, two])

        one = numpy.random.randint(0, 50, (n_samples, 2)) / 100.0
        two = numpy.random.randint(50, 100, (n_samples, 2)) / 100.0

        samples.append([one, two])

        return samples

    def helper_27_plot_all(self, one, two, sep, b, z, dots, crit, score, norm):
        fig1 = matplotlib.pyplot.figure()

        ax1 = fig1.add_subplot(221)
        ax1.plot(one[:, 0], one[:, 1], 'bo', two[:, 0], two[:, 1], 'rx')

        ax2 = fig1.add_subplot(222)
        ax2.plot(norm, 'x')

        ax3 = fig1.add_subplot(223)
        ax3.plot(numpy.int8(score) * 2 - 1, 'rx')

        ax4 = fig1.add_subplot(224)
        ax4.plot(b, 'o')

        fig1.show()

    def helper_27_construct_sv(self, one, two, sep):
        _one = sep.dot(one.T)
        _two = sep.dot(two.T)

        _best_one_i = [
            numpy.where(_one[k, :] < numpy.min(_one[k, :]) + 1e-6)[0]
            for k in range(_one.shape[0])
        ]

        _best_two_i = [
            numpy.where(_two[k, :] > numpy.max(_two[k, :]) - 1e-6)[0]
            for k in range(_two.shape[0])
        ]

        _best_one = one[[x[0] for x in _best_one_i]]
        _best_two = two[[x[0] for x in _best_two_i]]

        sep_norm = numpy.sqrt(numpy.sum(numpy.square(sep), axis=1))

        _dist = (
            numpy.array([a[x[0]] for a, x in zip(list(_one), _best_one_i)]) -
            numpy.array([a[x[0]] for a, x in zip(list(_two), _best_two_i)])
        ) / sep_norm

        _sep = sep / sep_norm.reshape(-1, 1) * _dist.reshape(-1, 1)

        z = _best_two + _sep / 2.0
        w = 2.0 * sep / sep_norm.reshape(-1, 1) / _dist.reshape(-1, 1)
        b = numpy.diag(-w.dot(z.T)).reshape(-1, 1)

        return pandas.DataFrame({
            'sep': list(w),
            'b': list(b),
            'z': list(z),
            '_best_one_i': _best_one_i,
            '_best_two_i': _best_two_i
        })

    def helper_27_decision_function(self, sep, z, b, samples):
        return sep.dot(samples.T) + b

    def helper_27_visualize(self, sep, z, b, samples):
        xx, yy = numpy.meshgrid(
            numpy.linspace(
                numpy.min(samples[:, 0]) - 0.1, numpy.max(samples[:, 0]) + 0.1, 100),
            numpy.linspace(
                numpy.min(samples[:, 1]) - 0.1, numpy.max(samples[:, 1]) + 0.1, 100),
        )
        Z = self.helper_27_decision_function(sep, z, b, numpy.c_[xx.ravel(), yy.ravel()]) \
            .reshape(xx.shape)

        matplotlib.pyplot.imshow(
            Z, interpolation='bilinear', origin='lower',
            extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap='gray')

        matplotlib.pyplot.contour(
            xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k',
            alpha=0.5)

        matplotlib.pyplot.scatter(
            samples[:, 0], samples[:, 1],
            c=self.helper_27_decision_function(sep, z, b, samples) > 0, edgecolors='k')

        matplotlib.pyplot.show()

    def helper_27_search_dumb(self, one, two, n_tries=10):
        _one_i = numpy.random.permutation(
            numpy.arange(max(one.shape[0], n_tries)) % one.shape[0])[:n_tries]
        _two_i = numpy.random.permutation(
            numpy.arange(max(two.shape[0], n_tries)) % two.shape[0])[:n_tries]

        _sep = one[_one_i] - two[_two_i]
        return self.helper_27_construct_sv(one, two, _sep)[['sep', 'b', 'z']]

    def helper_27_search_bs(self, one, two, n_tries):
        sep = numpy.random.randint(-10 ** 5, 10 ** 5, (n_tries, 2)) / 100.0
        #b = numpy.random.randint(-50, 50, (n_tries, 1)) / 100.0
        b = numpy.array([1]).repeat(n_tries).reshape(-1, 1)
        z = numpy.stack(
            [numpy.ones(n_tries), (-b.squeeze() - sep[:, 0]) / sep[:, 1]]).T

        return pandas.DataFrame({
            'sep': sep.tolist(),
            'z': z.tolist(),
            'b': b.tolist()
        })

    def helper_27_check(self, one, two, separators, plot_all=False):
        sep = numpy.stack(separators['sep'].values)
        b = numpy.stack(separators['b'].values)
        z = numpy.stack(separators['z'].values)

        dots = [sep.dot(one.T) + b - 1, sep.dot(two.T) + b + 1]

        crit = [
            numpy.sum(~(dots[0] >= -1e-3), axis=1) == 0,
            numpy.sum(~(dots[1] <= 1e-3), axis=1) == 0
        ]

        score = numpy.logical_and(crit[0], crit[1])

        norm = sep.dot(sep.T).diagonal()

        if plot_all:
            self.helper_27_plot_all(
                one, two, sep, b, z, dots, crit, score, norm)

        t = pandas.DataFrame({
            'sep': list(sep),
            'b': list(b),
            'z': list(z),
            'norm': norm,
            'score': score
        })

        return t[t['score'] == 1].sort_values(by=['norm']).head(5)

    def helper_27_search(self, one, two, n_tries, hints, plot_all=False):
        separators = pandas.concat([
            self.helper_27_search_dumb(one, two),
            self.helper_27_search_bs(one, two, n_tries)
        ])

        sep = numpy.stack(separators['sep'].values)
        b = numpy.stack(separators['b'].values)
        z = numpy.stack(separators['z'].values)

        if len(hints) > 0:
            sep = numpy.concatenate([sep, hints[0]])
            b = numpy.concatenate([b, hints[1]])
            z = numpy.concatenate([z, hints[2]])

        return self.helper_27_check(
            one,
            two,
            pandas.DataFrame({
                'sep': list(sep),
                'z': list(z),
                'b': list(b)
            }),
            plot_all=plot_all
        )

    def helper_27(self, samples, n_tries=10 ** 5, hints=[], **kwargs):
        one, two = samples

        batch_size = max(n_tries // 1000, 100)

        best = None

        for k in range((n_tries + batch_size - 1) // batch_size):
            r = self.helper_27_search(
                one,
                two,
                min(batch_size, n_tries - batch_size * k),
                hints=hints,
                **kwargs
            )

            if best is None:
                best = r
            else:
                best.append(r)

        if best.index.size > 0:
            self.helper_27_visualize(
                best['sep'].values[0],
                best['z'].values[0],
                best['b'].values[0],
                numpy.concatenate([one, two])
            )

    def helper_28(self, n_tries=10 ** 2, n_samples=10 ** 1):
        for s in self.helper_26(n_samples):
            self.helper_27(s, n_tries=n_tries)

    def helper_29_sample_for_perceptron(
        self,
        dist=None,
    ):
        if dist is None:
            dist = (numpy.random.rand(1)[0] * 0.8 + 0.1) / 2

        phi = numpy.random.rand(2) * 2 - 1
        phi /= numpy.sqrt(numpy.sum(numpy.square(phi)))

        phi /= dist

        b = -phi.dot(numpy.array([0.5, 0.5]).T)

        X = None
        Y = None

        while X is None or X.shape[0] < 100:
            _X = numpy.random.rand(1000, 2) * 3 - 1

            _Y = (phi.dot(_X.T) + b).reshape(-1)

            _i = numpy.abs(_Y) >= 1.0

            _Y = _Y > 0

            if numpy.sum(_i) > 0:
                if X is None:
                    X = _X[_i]
                else:
                    X = numpy.vstack(X, _X[_i])

                if Y is None:
                    Y = _Y[_i]
                else:
                    Y = numpy.vstack(Y, _Y[_i])

        return X[:100], Y[:100], dist, phi, b

    def helper_29_perceptron(
            self,
            X,
            Y,
            iter_tol=10 ** 6,
            plambda=1,
            stopping_rule=lambda p: False):
        n = X.shape[0]

        _j = numpy.random.permutation(
            numpy.arange(max(n, iter_tol))[:iter_tol] % n)

        w = numpy.zeros(X.shape[1] + 1)

        l = 0
        m = 0
        for k in range(iter_tol):
            xhi = numpy.append(X[_j[k] % n, :], plambda)
            y = numpy.double(Y[_j[k] % n]) * 2.0 - 1

            d = y * w.dot(xhi.T)
            _i = d <= +1e-6

            _w = w
            if numpy.sum(_i) > 0:
                _w[_i] = (w + y * xhi.T)[_i]
                m += 1

            if stopping_rule({
                'k': k,
                'j': _j[k],
                'l': l,
                'n': n,
                'a': numpy.sum(_i) == 0,
                'X': X,
                'Y': Y,
                'm': m,
                'w': w
            }):
                break

            w = _w
            l += 1

        return w[:2], w[2], l, m

    def helper_29(self, iter_tol=10 ** 5, plambda=1):
        X, Y = self.helper_29_sample_for_perceptron()[:2]
        w, b, l = self.helper_29_perceptron(
            X, Y, iter_tol=iter_tol, plambda=plambda)[:3]
        pprint.pprint({'w': w, 'b': b, 'l': l})
        self.helper_29_visualize(w, b, X, Y)

    def helper_29_visualize(self, w, b, X, Y):
        xx, yy = numpy.meshgrid(
            numpy.linspace(
                -0.1 + numpy.min(X[:, 0]), 0.1 + numpy.max(X[:, 0]), 1000),
            numpy.linspace(
                -0.1 + numpy.min(X[:, 1]), 0.1 + numpy.max(X[:, 1]), 1000)
        )
        Z = (w.dot(numpy.c_[xx.ravel(), yy.ravel()].T) + b).reshape(xx.shape)

        matplotlib.pyplot.imshow(
            Z, interpolation='bilinear', origin='lower',
            extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap='gray')

        matplotlib.pyplot.contour(
            xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k',
            alpha=0.5)

        matplotlib.pyplot.scatter(
            X[:, 0], X[:, 1],
            c=Y, edgecolors='k')

    def helper_30(self, iter_tol=10 ** 5):
        plambda = None
        X, Y, dist, phi = [None, ] * 4
        n = None
        _X = None
        D = None
        rho = None
        M = None
        m_l = None
        m_m = None
        m_w = None
        m_b = None
        w, b, l, m = [None, ] * 4
        zz = None

        while True:
            plambda = 1

            X, Y, dist, phi, b = self.helper_29_sample_for_perceptron()

            n = X.shape[0]

            _X = numpy.hstack([X, plambda * numpy.ones((X.shape[0], 1))])
            D = numpy.sqrt(numpy.max(numpy.sum(numpy.square(_X), axis=1)))
            rho = dist

            M = numpy.int(numpy.square(D / rho))

            m_l = None
            m_m = None
            m_w = None
            m_b = None

            def stopping_rule(p):
                nonlocal m_m
                nonlocal m_l
                nonlocal m_w
                nonlocal m_b

                if p['m'] >= M and m_m is None:
                    print(M, p['m'])
                    m_l = p['l']
                    m_m = p['m']
                    m_w = p['w'][:2].copy()
                    m_b = p['w'][2].copy()

                return False

            w, b, l, m = self.helper_29_perceptron(
                X,
                Y,
                iter_tol=iter_tol,
                stopping_rule=stopping_rule,
                plambda=plambda)

            zz = (numpy.double(Y) * 2 - 1) * \
                (numpy.double(w).dot(numpy.double(X).T) + numpy.double(b))
            pprint.pprint(zz)

            if numpy.sum(~(Y * (w.dot(X.T) + b) > -1e+6)) > 0:
                break

            break

        pprint.pprint({'M': M, 'D': D, 'rho': rho})
        pprint.pprint({
            'w': w, 'b': b, 'l': l, 'm': m, 'n': n, 'g': l < n})

        matplotlib.pyplot.subplot(121)

        if m_m is not None:
            pprint.pprint({
                'm_m': m_m, 'm_l': m_l, 'm_w': m_w, 'm_b': m_b})
            self.helper_29_visualize(m_w, m_b, X, Y)

        matplotlib.pyplot.subplot(122)
        self.helper_29_visualize(w, b, X, Y)
        matplotlib.pyplot.show()

    def draw_scores(self):
        out_dir = os.path.join(self._env['project_root'], 'build', 'scores')
        os.system('mkdir -p ' + out_dir)

        for f in self._replays:
            out_png = os.path.join(
                out_dir, os.path.split(f)[1] + '.png'
            )

            if self._env['verbose']:
                print(f)

            if os.path.exists(out_png):
                if self._env['verbose']:
                    print('skip %s' % f)
                continue

            r = self.parse_replay(f)

            fig = matplotlib.pyplot.figure()
            matplotlib.pyplot.plot([x['frame']['score'] for x in r['entries']])
            fig.savefig(out_png)
            matplotlib.pyplot.close(fig)


class TestStatistics(unittest.TestCase):
    def test_should_be_consistent_with_replay_format(self):
        s = Statistics()

        for f in s._replays:
            sys.stderr.write(f + '\n')

            replay = s.parse_replay(f)

            self.assertTrue(
                numpy.all(
                    numpy.array(
                        replay['entries'][0]['frame']['score'],
                        dtype=numpy.uint64
                    ) == 0
                )
            )

            for entry in replay['entries']:
                for player_id, player in entry['players'].items():
                    if player.get('joint') is not None:
                        self.assertTrue(
                            numpy.logical_and(
                                numpy.all(
                                    numpy.array(
                                        player['joint']['states'],
                                        dtype=numpy.uint64
                                    ) <= 4
                                ),
                                numpy.all(
                                    numpy.array(
                                        player['joint']['states'],
                                        dtype=numpy.uint64
                                    ) >= 1
                                )
                            )
                        )

    def test_dumb_search_separates(self):
        s = Statistics()

        ss = s.helper_26(10)[1]
        sp = s.helper_27_search_dumb(*ss)
        self.assertGreater(s.helper_27_check(*ss, sp).index.size, 0)


def unittest_main_wrapper(argv=sys.argv):
    class DebugRunner:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, test):
            a = unittest.TextTestRunner()
            return test(a._makeResult(), debug=True)

    _testRunner = None
    _argv = argv.copy()

    if len(_argv) > 1 and _argv[1] == 'debug':
        del _argv[1]
        _testRunner = DebugRunner

    unittest.main(argv=_argv, testRunner=_testRunner)


if __name__ == '__main__':
    unittest_main_wrapper()
