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
import pickle
import copy
import tempfile


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
