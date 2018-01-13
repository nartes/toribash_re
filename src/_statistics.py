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

    def helper_10(self, a):
        res = dict([(x, []) for x in
                    ['r_id', 'e_id', 'p_id', 'pos',
                     'action_states', 'action_ids', 'timestamp', 'score']])

        r_id = 0
        for r in a:
            e_id = 0
            for e in r['replay']['entries']:
                for p_id in e['players'].keys():
                    p = e['players'][p_id]
                    res['r_id'].append(r_id)
                    res['e_id'].append(e_id)
                    res['p_id'].append(p_id)
                    res['pos'].append(numpy.array(
                        p['pos']['floats']).reshape(-1, 3))
                    res['action_ids'].append(
                        p.get('joint') and p['joint']['ids'] or [])
                    res['action_states'].append(
                        p.get('joint') and p['joint']['states'] or [])
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

    def helper_13(self, b):
        c = b
        d = c[['pos', 'timestamp', 'e_id', 'r_id', 'p_id']]
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
        w = pandas.merge(b[self.minus_list(list(b.columns), ['timestamp'])], y,
                         on=self.minus_list(list(y.columns), ['mean_pos', 'timestamp']), how='outer')
        u = w.sort_values(by=['r_id', 'e_id', 'p_id'])

        return u

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

    def helper_19(self, c):
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

    def helper_20(self):
        a = pickle.load(io.open("build/replays_1332_10.dat", 'rb'))
        b = _statistics.Statistics().helper_10(a)
        c = _statistics.Statistics().helper_13(b)
        e = _statistics.Statistics().helper_19(c)
        matplotlib.pyplot.scatter(e['distance'], e['score'])
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


def unittest_main_wrapper(argv=sys.argv):
    class DebugRunner:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, test):
            a = unittest.TextTestRunner()
            test(a._makeResult(), debug=True)

    _testRunner = None
    _argv = argv.copy()

    if len(_argv) > 1 and _argv[1] == 'debug':
        del _argv[1]
        _testRunner = DebugRunner

    unittest.main(argv=_argv, testRunner=_testRunner)


if __name__ == '__main__':
    unittest_main_wrapper()
