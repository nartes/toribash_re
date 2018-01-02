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
