import io
import os
import sys
import subprocess
import tempfile
import multiprocessing
import functools
import unittest
import importlib
import optparse
import json
import pprint
import glob
import numpy
import re
import scipy.stats
import threading
import signal
import pandas
import _statistics


os.environ['TORIBASH_PROJECT_ROOT'] =\
    os.environ.get('TORIBASH_PROJECT_ROOT') or\
    os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')


os.environ['R2_PIPE_GIT'] = os.path.join(
    os.environ['TORIBASH_PROJECT_ROOT'], 'deps', 'radare2-r2pipe', 'python')

os.environ['RPDB_GIT'] = os.path.join(
    os.environ['TORIBASH_PROJECT_ROOT'], '..', 'rpdb')

os.environ['AUTOPEP8_BINARY'] = os.environ.get('AUTOPEP8_BINARY') or\
    sys.executable + " -m autopep8"

for p, e in [(os.environ['R2_PIPE_GIT'], 'Warning: not found GIT r2pipe'),
             (os.environ['RPDB_GIT'], 'Warning: not found GIT rpdb')]:
    try:
        if not os.path.exists(p):
            raise ValueError(r"The Path doesn't exists %s" % k)
        else:
            sys.path.insert(0, p)

    except Exception as ex:
        print(e)
        raise(ex)

import r2pipe

os.environ['MAKE_JOBS'] = os.environ.get('MAKE_JOBS') or\
    str(os.cpu_count())


class Utils:
    def sub_shell(cmds,
                  communicate=False,
                  stderr_to_stdout=False,
                  verbose=False,
                  wait=True,
                  env=dict(list(os.environ.items())),
                  critical=True):
        ret = None

        tf = tempfile.mktemp()
        f = io.open(tf, 'w')
        f.write(u'' + cmds)
        f.close()

        if verbose:
            print('*' * 9 + 'BEGIN_COMAND' + '*' * 9)
            print(tf)
            print('*' * 9 + '************' + '*' * 9)
            print(cmds)
            print('*' * 9 + 'END_COMAND' + '*' * 9)

        _env = dict([(k.upper(), str(v)) for (k, v) in env.items()])

        if verbose:
            pprint.pprint(_env)

        if communicate:
            inp = subprocess.PIPE
            outp = subprocess.PIPE
            if stderr_to_stdout:
                errp = outp
            else:
                errp = subprocess.PIPE

            proc = subprocess.Popen(['zsh', tf],
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,
                                    env=_env)
            try:
                proc.wait()
                out, err = proc.communicate()
                ret = out.decode()
            except:
                proc.kill()
        else:
            proc = subprocess.Popen(['zsh', tf],
                                    stdin=sys.stdin,
                                    stdout=sys.stdout,
                                    stderr=sys.stderr,
                                    env=_env)
            try:
                if wait:
                    proc.wait()
            except:
                proc.kill()

        if wait:
            if proc.returncode != 0 and critical:
                raise ValueError(proc.returncode)

        return ret


def sub_shell(cmds,
              communicate=False,
              stderr_to_stdout=False,
              verbose=False,
              wait=True,
              critical=True):
    ret = None

    if verbose:
        print('*' * 9 + 'BEGIN_COMAND' + '*' * 9)
        print(cmds)
        print('*' * 9 + 'END_COMAND' + '*' * 9)

    tf = tempfile.mktemp()
    f = io.open(tf, 'w')
    f.write(u'' + cmds)
    f.close()

    if communicate:
        inp = subprocess.PIPE
        outp = subprocess.PIPE
        if stderr_to_stdout:
            errp = outp
        else:
            errp = subprocess.PIPE

        proc = subprocess.Popen(['zsh', tf],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE)
        try:
            proc.wait()
            out, err = proc.communicate()
            ret = out.decode()
        except:
            proc.kill()
    else:
        proc = subprocess.Popen(['zsh', tf],
                                stdin=sys.stdin,
                                stdout=sys.stdout,
                                stderr=sys.stderr)
        try:
            if wait:
                proc.wait()
        except:
            proc.kill()

    if wait:
        if proc.returncode != 0 and critical:
            raise ValueError(proc.returncode)

    return ret


os.environ['toribash_out'] = sub_shell('echo -E $PWD/build/toribash_out',
                                       communicate=True).strip()
os.environ['toribash_common'] = sub_shell('echo -E $PWD/build/toribash',
                                          communicate=True).strip()
os.environ['radare2_http_port'] = os.environ.get('radare2_http_port') or '9998'
os.environ['radare2_tcp_port'] = os.environ.get('radare2_tcp_port') or '9997'


class Rctx:
    def __init__(self, url):
        self.url = url
        self.timeout_error = False

    @staticmethod
    def query(cmds, attr, url, e, q, l):
        try:
            res = getattr(r2pipe.open(url), attr)(cmds)
            if not e.is_set():
                l.acquire()
                q.put(res)
                e.set()
                l.release()
        except:
            return None

    @staticmethod
    def wait(e, q, l, timeout):
        os.system("sleep %d" % timeout)
        if not e.is_set():
            l.acquire()
            q.put(None)
            e.set()
            l.release()

    def timeout_cmd(self, cmds, attr, timeout):
        q = multiprocessing.Queue()
        l = multiprocessing.Lock()
        e = multiprocessing.Event()

        query_process = multiprocessing.Process(
            target=functools.partial(
                self.query,
                cmds,
                attr,
                self.url,
                e,
                q,
                l))

        wait_process = multiprocessing.Process(
            target=functools.partial(
                self.wait,
                e, q, l, timeout))

        wait_process.start()
        query_process.start()

        e.wait()

        res = q.get()

        if res is None:
            self.timeout_error = True

        os.kill(wait_process.pid, signal.SIGKILL)
        os.kill(query_process.pid, signal.SIGKILL)

        return res

    def cmd(self, cmds, timeout=0.5):
        return self.timeout_cmd(cmds, attr='cmd', timeout=timeout)

    def cmdj(self, cmds, timeout=0.5):
        return self.timeout_cmd(cmds, attr='cmdj', timeout=timeout)


class Algos:
    def __init__(self):
        self.rctx = Rctx("tcp://127.0.0.1:%s" % os.environ['radare2_tcp_port'])
        #self.rctx = r2pipe.open("http://127.0.0.1:%s" % os.environ['radare2_http_port'])

    def run_lines(self, cmds, with_output=False, **kwargs):
        for l in cmds.split('\n'):
            l = l.strip()

            if l and not l.startswith('#'):
                print("Consequent command: %s" % l)
                ret = self.rctx.cmd(l, **kwargs)
                if with_output:
                    print(ret)
            else:
                print("Ignored: %s" % l)

    def toribash_manual_functions(self):
        self.run_lines(r"""
            af man.steam_init_v2 @ 0x081fb240
            af man.steam_networking @ 0x081fb590
            af man.steam_init @ 0x81fae20
            af man.steam_callbacks @ 0x081fdf10
            af @ 0x080c0bf0
            af man.toribash_core @ 0x080b4d10
            af @ 0x080c0c50
            af man.gl_init @ 0x080ea3c0
            af man.lua_init @ 0x080fda40
            af @ 0x081046d0
            af @ 0x081fcd90
            af @ 0x081fce00
            af man.toribash_login_init @ 0x80914a0
            af man.toribash_entry1 @ 0x80b49b0
            af man.toribash_steam_hell @ 0x81fb620
            afr man.toribash_steam_hell_prelude @ 0x080c0448
            af man.curl_related_logic @ 0x8105940
            af man.curl_related_logic_2 @ 0x81f5ab0
            af man.curl_related_logic_3 @ 0x81f6510
            af man.curl_related_logic_4 @ 0x81f54d0

            af man.curl_related_logic_4_1 @ 0x81f54d0
            af man.curl_related_logic_4_2 @ 0x80c0100
            #af man.curl_related_logic_4_2 @ 0x80b9000
            #afr man.curl_related_logic_4_3 @ 0x80bd790
""" +

                       # hit breakpoint at: 81f54d0
                       #:> dbt
                       # 0  0x81f54d0  sp: 0x0         0    [man.curl_related_logic_4]  man.curl_related_logic_4 man.curl_related_logic_40
                       # 1  0x80b9230  sp: 0xffd689fc  0    [??]  fcn.080b5980+14512
                       # 2  0x80bd790  sp: 0xffd68a1c  32   [??]  loc.080b9ab0+15584
                       # 3  0xf7d19146 sp: 0xffd68a34  24   [??]  section_end..bss-295031030
                       # 4  0xf7e5b19b sp: 0xffd68a38  4    [??]  section_end..bss-293712033
                       # 5  0x80c01b4  sp: 0xffd68a4c  20   [fcn.080c0100]  fcn.080c0100+180
                       # 6  0x80c04b4  sp: 0xffd68a8c  64   [man.toribash_steam_hell_prelude]  man.toribash_steam_hell_prelude+108
                       # 7  0x80c15c0  sp: 0xffd68abc  48   [fcn.080c1590]  fcn.080c1590+48
                       # 8  0xf7bd4abc sp: 0xffd68ac0  4    [??]  section_end..bss-296359808
                       # 9  0x80b51f5  sp: 0xffd68acc  12   [man.toribash_core]  man.toribash_core+1253
                       # 10  0xf7928bf9 sp: 0xffd68bd8  268  [??]  section_end..bss-299161155
                       # 11  0xf792a930 sp: 0xffd68bec  20   [??]  section_end..bss-299153676
                       # 12  0xf7fbb2b0 sp: 0xffd68bf0  4    [??]  eip+83536
                       # 13  0xf792a8c9 sp: 0xffd68bf8  8    [??]  section_end..bss-299153779

                       #:> dbt
                       # 0  0x80b9000  sp: 0x0         0    [man.curl_related_logic_4_2]  man.curl_related_logic_4_2 man.curl_related_logic_4_20
                       # 1  0xf7c61146 sp: 0xff9205a4  8    [??]  section_end..bss-295784694
                       # 2  0xf7da319b sp: 0xff9205a8  4    [??]  section_end..bss-294465697
                       # 3  0x80c01b4  sp: 0xff9205bc  20   [fcn.080c0100]  fcn.080c0100+180
                       # 4  0x80c04b4  sp: 0xff9205fc  64   [man.toribash_steam_hell_prelude]  man.toribash_steam_hell_prelude+108
                       # 5  0x80c15c0  sp: 0xff92062c  48   [??]  fcn.080c0c50+2416
                       # 6  0xf7b1cabc sp: 0xff920630  4    [??]  section_end..bss-297113472
                       # 7  0x80b51f5  sp: 0xff92063c  12   [man.toribash_core]  man.toribash_core+1253
                       # 8  0xf7870bf9 sp: 0xff920748  268  [??]  section_end..bss-299914819
                       # 9  0xf7872930 sp: 0xff92075c  20   [??]  section_end..bss-299907340
                       # 10  0xf7f032b0 sp: 0xff920760  4    [??]  map.usr_lib32_ld_2.26.so._r_x+86704
                       # 11  0xf78728c9 sp: 0xff920768  8    [??]  section_end..bss-299907443
                       # 12  0xf7f032b0 sp: 0xff920784  28   [??]  map.usr_lib32_ld_2.26.so._r_x+86704
                       # 13  0xf7b1cabc sp: 0xff920790  12   [??]  section_end..bss-297113472


                       r"""
            #0x083b8000 lua static code
            #0x083bb770 lua stack cmd
            #0x083bb390 lua stack cmd
            #0x080fdf00 population of lua functions
        """)

        self.rctx.cmd("af man.steam_class_init @ 0x081fdf10")

    def random_walker(self):
        freq_stat = {}

        self.run_lines("aa")

        for f in self.rctx.cmd("f~sym.imp[2]").split('\n'):
            self.run_lines("db @@=%s" % f)

        while True:
            self.rctx.cmd("ds %d" % numpy.random.randint(1, 10 ** 3 + 1, 1))

            dr = self.rctx.cmdj("drj")
            print('\n'.join(["% 10s %10x" % (r, v) for r, v in dr.items()]))

            # TODO: I'm in an incredible pain! pdj 10 @ eip does something wrong
            # and radare doesn't crash. Seems eip has some meaning as a flag or symbol
            def my_pd(cmd_s="pdj 10 @ `eip`"):
                asm_l = self.rctx.cmdj(cmd_s)
                print('\n'.join([
                    "%10x %s" % (o['offset'], o['opcode']) for o in asm_l
                ]))
            # my_pd()
            my_pd(cmd_s="pdj 10 @r:eip")
            #print(self.rctx.cmd("pd 10 @ eip"))
            #my_pd(cmd_s="pdj 10 @r,eip")

            def update_freqs():
                for k in dr:
                    freq_stat[k] = freq_stat.get(k, {})
                    freq_stat[k][dr[k]] = freq_stat[k].get(dr[k], 0) + 1

            update_freqs()

            def entropy(b):
                b = b / (numpy.sum(b) + 1e-6)
                b = numpy.maximum(b, 1e-6)
                return numpy.sum(numpy.log(1 / (1 - b)) * (1 - b) + numpy.log(1 / (b)) * b)

            # for k in freq_stat.keys():
            def plot_freqs(key_list=['eip']):
                for k in key_list:
                    h = numpy.histogram(list(freq_stat[k].keys()), weights=list(
                        freq_stat[k].values()), bins=10)
                    print("%s entropy: %10f" % (k, entropy(h[0])))
                    print(
                        ' '.join(['%10.2f' % o for o in h[0] / numpy.sum(h[0])]))
                    print(' '.join(['%10x' % o for o in numpy.uint64(h[1])]))

                    if entropy(h[0]) < 1.0:
                        if numpy.random.randint(0, 2):
                            self.rctx.cmd("dc")

            plot_freqs()

            print(self.rctx.cmd("dbt~:[1]"))

            os.system("sleep 1")

    def toribash_search_string_usage(self):
        assert int(self.rctx.cmd("f~retrie:0[0]"), 16) == 0x847d38c

        self.run_lines(r"""
            s 0x847d38c
            /x 8cd34708
        """)

        assert int(self.rctx.cmd("f~hit[0]").split('\n')[-1], 16) == 0x80915d8

        self.rctx.cmd("f man.print_unable_login @ 0x80915d5")

        self.run_lines(r"""
            fs searches
            f-*
            fs *
        """)

    def call_inject(self):
        self.run_lines(r"""
aa
db main
        """, with_output=True)

        while int(self.rctx.cmd("dr~eip[1]"), 16) > 0x90000000:
            self.rctx.cmd("dc")

        self.run_lines(r"""
dm
        """, with_output=True)

        assert self.rctx.cmd("dm~inject:0").find('-rwx') >= 0

        self.run_lines(r"""
s `dm~inject:0[0]`
/x 5589e55383
        """, with_output=True)

        assert int(self.rctx.cmd("f~hit[0]"), 16) > 0x60000000

        self.run_lines(r"""
afr sym.abc @ `f~hit[0]`
        """, with_output=True)

        assert int(self.rctx.cmd("f~sym.abc[0]"), 16) > 0

        self.run_lines(r"""
ds 6
.--
        """, with_output=True)

    def call_inject_2(self):
        _old_cmd = self.rctx.cmdj("aoj @@ eip")[0]['bytes']
        self.run_lines(r"""
wa call `f~sym.abc[0]`@@=eip
        """, with_output=True)

        assert self.rctx.cmdj("aoj @@ eip")[0]['mnemonic'] == 'call'
        assert self.rctx.cmdj("aoj @@ eip")[0]['jump'] == int(
            self.rctx.cmd("f~sym.abc[0]"), 16)

        #_old_pos = int(self.rctx.cmd("dr~eip[1]"), 16)

        _cur_pos = self.rctx.cmdj("aoj 2 @@ eip")[0]['addr']
        _next_pos = self.rctx.cmdj("aoj 2 @@ eip")[1]['addr']
        self.rctx.cmd("s eip")
        self.rctx.cmd("db %d" % _next_pos)
        self.rctx.cmd("ds")
        print(self.rctx.cmd("pd 10 @@ eip"))
        self.rctx.cmd("dc")
        assert int(self.rctx.cmd("dr~eip[1]"), 16) == _next_pos
        #self.rctx.cmd("wx %s @@ %d" % (_old_cmd, _old_pos))
        self.rctx.cmd("dr eip = %d" % _cur_pos)
        self.rctx.cmd("wx %s @@ eip" % _old_cmd)
        self.rctx.cmd(".--")

    def toribash_set_b_for_steam_api(self):
        # TODO(radare): it doesn't accept ~<prefix>*
        # what it does accept is ~<prefix>, the star is not parsed,
        # it may be presumed to stay there.

        # for s in re.compile(r'[\s\n\r]+').split(self.rctx.cmd("f~sym.imp.Steam*[2]")):
        for s in re.compile(r'[\s\n\r]+').split(self.rctx.cmd("f~sym.imp.Steam[2]")):
            self.rctx.cmd("db %s" % s)

    def af1d(self, fcn='man.toribash_core'):
        refs = [r['addr']
                for r in self.rctx.cmdj("afij @ %s" % fcn)[0]['callrefs']]
        self.rctx.cmd("af @@=%s" % ' '.join([str(i) for i in refs]))

    def r2_init(self):
        self.init()
        self.manual_functions()

    def init(self):
        self.run_lines(r"""
            aa
        """)

    def kill_server(self):
        self.run_lines(r"""
            #=h--
            .--
        """)

    def test(self):
        self.run_lines(r"""
            db man.steam_init
            dc
            db 0x0805bf88
            dc
            db man.toribash_core
            dc
            #!echo The work is out there.
            #db 0x080e4d86
            #db 0x08104c73
            #wa mov eax, 0@@=\`axt @@ sym.imp.Steam*~man.steam_init[1]\`
            #k a=\`f~sym.imp.Steam:2[0]\`
            #k n=\`f~sym.imp.Steam:2[2]\`
            #db \`k a\`
            #dbc \`k a\` dbt
            #dc
            #axt @@ sym.imp.Steam*
            #pdj 2~{}..
            #wao nop @@ \`axt @@ sym.imp.Steam*~[1]\`
        """)

    def patch_steam_init(self):
        self.rctx.cmd("!echo Patch man.steam_init")
        assert '0x081fae20' in self.rctx.cmd("afo man.steam_init")
        assert 'e8db2ee5ff' in self.rctx.cmdj("aoj @ 0x081fae90")[0]['bytes']

        self.rctx.cmd("wa mov eax, 0 @ 0x081fae33")
        self.rctx.cmd("wa mov eax, 1 @ 0x081fae40")
        self.rctx.cmd("wao nop @@=%s" % " ".join([str(x) for x in [
            0x081fae50, 0x081fae5d, 0x081fae5b, 0x081fae60, 0x081fae63, 0x081fae65,
            0x081fae67, 0x081fae6f, 0x081fae74, 0x081fae78,
            0x081fae90, 0x081faea1, 0x081faeb1, 0x081faeb4, 0x081faeca, 0x081faece
        ]]))

        print(self.rctx.cmd("i"))

    def final(self):
        self.r2_init()
        self.patch_steam_init()
        self.af1d()
        self.test()
        self.kill_server()


class RandomWalkerAgent:
    def __init__(self):
        pass
        self.a = {}
        self.p = {}
        self.w = {0: numpy.zeros(100)}
        self.X_q = {}
        self.X_u = {}
        self.q = {}
        self.n_q = {}
        self.index = 0
        self.k = 1

    def get_parameters(self):
        return self.generate_sample()

    def update(self, *args, **kwargs):
        pass

    def n_sample_generator(self):
        def add_uniform(_range):
            params = [ \
                numpy.ones(len(_range)) / len(_range),
                _range]

            return scipy.stats.rv_discrete(name='custom', values=params[::-1])

        g = [ add_uniform(o) for o \
            in [ \
                numpy.arange(2),
                numpy.arange(6),
                numpy.arange(3),
                numpy.arange(1, 10 ** 3 + 1),
                numpy.array([1.0]),
                numpy.arange(2),
                numpy.arange(2)]]

        def generate_sample():
            return [ \
                r.rvs(size=1)[0] for r in g]

        return generate_sample

    def generate_classification_sample(self, cmd_id, cmd_args, regs, d_regs, d_eps, index):
        return self.generate_training_sample(
            cmd_id, cmd_args, regs, d_regs, d_eps, index)

    def generate_training_sample(self, cmd_id, cmd_args, regs, d_regs, d_eps, index):
        # Why list doesn't support vectorized sum!!!! Good python3.
        def sum_list(*args):
            res = []
            for o in args:
                res.extend(o)

            return res

        return pandas.DataFrame(numpy.hstack([
            cmd_id.values,
            regs.values,
            d_regs.values,
            cmd_args.values,
            d_eps.values,
            numpy.ones((1, 1))]),
            columns=sum_list(*[list(o) for o in [ \
                cmd_id.columns,
                regs.columns,
                d_regs.columns,
                cmd_args.columns,
                d_eps.columns,
                ['perceptron_offset']]]))

    def utility(self, X_kplus1_u):
        def utility_1():
            u_k = pandas.DataFrame({'u': \
                (X_kplus1_u[['d_eps_eip']] > 1e-6).sum(axis=1).values + \
                (X_kplus1_u[X_kplus1_u.columns[X_kplus1_u.columns.str.startswith('d_regs_')]] \
                    .abs() < 1e-6).sum(axis=1).values})

            i_ = u_k > 2
            u_k[i_] = 1
            u_k[~i_] = -1

            return u_k

        def utility_2():
            u_k = pandas.DataFrame({'u': \
                (X_kplus1_u[['d_eps_eip']]).sum(axis=1).values})

            i_ = u_k > 0.1
            u_k[i_] = 1
            u_k[~i_] = -1

            return u_k

        #def utility_3():
        #    u_k = X_kplus1_u[['d_eps_eip']].sum(axis=1) +


        return utility_2().values.reshape(-1)

    def train_perceptron(self, w_kminus1, a_kminus1, p, X_kminus1_q, n_6):
        X_k_u = self.generate_training_sample(
            self.cmd_id(a_kminus1), self.cmd_args(a_kminus1),
            self.regs(p).iloc[[-1]], self.d_regs(p).iloc[[-1]],
            self.d_eps(p).iloc[[-1]],
            self.index(a_kminus1))
        u_k = self.utility(X_k_u)
        if n_6 == 1:
            if not hasattr(self, '_statistics'):
                self._statistics = _statistics.Statistics()

            w_k = self._statistics.helper_31_perceptron_update(
                X_kminus1_q.values, u_k, w_kminus1)
        else:
            w_k = w_kminus1.copy()

        return w_k, u_k, X_k_u


class RandomWalkerCore:
    def __init__(self):
        self._algos = Algos()
        self._agent = RandomWalkerAgent()

        self._log = tempfile.mktemp(
            dir='build', prefix='rw-log-', suffix='.json')

        pprint.pprint({'log_': self.log_})

        #pandas.set_option('display.expand_frame_repr', False)

    def trace_plots(self, **kwargs):
        pprint.pprint(pandas.DataFrame({
            'u': kwargs['u'],
            'q': kwargs['q'],
            'n_q': kwargs['n_q'],
            }).iloc[-20:])

        try:
            pprint.pprint(pandas.concat(kwargs['X_q']).iloc[-5:])
        except:
            pass

        try:
            pprint.pprint(pandas.concat(kwargs['X_u']).iloc[-5:])
        except:
            pass

    def get_a_perception(self):
        res = {}

        bt = self._algos.rctx.cmdj("dbtj", timeout=1)

        res['bt'] = bt

        try:
            regs = json.loads(self._algos.rctx.cmd("drj", timeout=1))
        except:
            regs = None

        if regs is None and self._def_perception is not None:
            regs = dict([(o, -1) for o in self._def_perception['regs'].keys()])

        res['regs'] = regs

        ops = self._algos.rctx.cmdj("aoj 10 @r:eip", timeout=1)

        res['ops'] = ops

        errs = {'timeout_error': self._algos.rctx.timeout_error}
        res['errs'] = errs

        self._algos.rctx.timeout_error = False

        return res

    def get_possible_actions(self):
        def log_action(actn):
            self.trace(actn, 'actions')

        def custom_run_lines(*args, **kwargs):
            self._algos.run_lines(with_output=False, *args, **kwargs)

        def ds_0(timeout):
            return ds(1, timeout=timeout)

        def ds(num, timeout):
            assert num > 0

            return [{
                'log_entry': {
                    'act': 'ds', 'num': int(num), 'timeout': float(timeout)
                },
                'callback': functools.partial(
                    custom_run_lines, "ds %d" % num, timeout=timeout)}]

        def dso(num, timeout):
            assert num > 0

            return [{
                'log_entry': {
                    'act': 'dso', 'num': int(num), 'timeout': float(timeout)
                },
                'callback': functools.partial(
                    custom_run_lines, "dso %d" % num, timeout=timeout)}]

        def dcs(num, timeout):
            assert num > 0

            return [{
                'log_entry': {
                    'act': 'dcs', 'num': int(num), 'timeout': float(timeout)
                },
                'callback': functools.partial(
                    custom_run_lines, "dcs %d" % num, timeout=timeout)}]

        def dcr(timeout):
            return [{
                'log_entry': {
                    'act': 'dcr', 'timeout': float(timeout)
                },
                'callback': functools.partial(
                    custom_run_lines, "dcr", timeout=timeout)}]

        def dcf(timeout):
            return [{
                'log_entry': {
                    'act': 'dcf', 'timeout': float(timeout)
                },
                'callback': functools.partial(
                    custom_run_lines, "dcf", timeout=timeout)}]

        def dcc(timeout):
            return [{
                'log_entry': {
                    'act': 'dcc', 'timeout': float(timeout)
                },
                'callback': functools.partial(
                    custom_run_lines, "dcc", timeout=timeout)}]

        def kill_toribash(timeout):
            def _callback():
                cmd_ = "pkill -9 toribash_steam"
                print("Consequent command: %s" % cmd_)
                os.system(cmd_)

            return [{
                'log_entry': {
                    'act': 'kill toribash_steam', 'timeout': float(timeout)
                },
                'callback': _callback}]

        def kill_radare(timeout):
            def _callback():
                cmd_ = "pkill -9 radare2"
                print("Consequent command: %s" % cmd_)
                os.system(cmd_)

            return [{
                'log_entry': {
                    'act': 'kill radare', 'timeout': float(timeout)
                },
                'callback': _callback}]

        def ood(timeout):
            return [{
                'log_entry': {
                    'act': 'ood', 'timeout': float(timeout)
                },
                'callback': functools.partial(
                    custom_run_lines, "ood", timeout=timeout)}]

        res = {}

        res[0] = [ds_0, ds_0, ds_0, ds_0, kill_toribash, kill_radare, ood]
        res[1] = [dso, ds, ds]

        return res

    def pick_up_with_partial(self, actions, n):
        _act = None

        if n[0] == 0:
            _act = actions[n[0]][n[1]](n[4])
        elif n[0] == 1:
            _act = actions[n[0]][n[2]](n[3], n[4])

        return _act

    def actuators_log_action_and_execute(self, action):
        for e in action:
            self.trace(e['log_entry'], 'actions')
            if e['callback'] is not None:
                e['callback']()

    def generate_action(self, n, p, w_kminus1, index):
        pass

    def cmd_id(self, a_k):
        res = []

        if not hasattr(self, '_cmd_id_cmds'):
            self._cmd_id_cmds = {}

        for a in a_k:
            if self._cmd_id_cmds.get(a['log_entry']['act']) is None:
                self._cmd_id_cmds[a['log_entry']['act']] = len(self._cmd_id_cmds)

            res.append(self._cmd_id_cmds[a['log_entry']['act']])

        return pandas.DataFrame({'cmd_id': res})

    def cmd_args(self, a_k):
        res =  {
            'timeout': [],
            'num': []
        }
        for a in a_k:
            res['timeout'].append(a['log_entry']['timeout'])
            res['num'].append(a['log_entry'].get('num', -1))

        return pandas.DataFrame(res)

    def regs(self, p):
        d = pandas.DataFrame([o.get('regs', {}) for o in p], index=p.index)

        return d

    def d_regs(self, p):
        d = self.regs(p)

        d.loc[0] = -1
        d.sort_index(inplace=True)

        d = d.diff(axis=0)
        d.loc[0] = 0
        d.sort_index(inplace=True)

        return d.rename(columns=lambda x: 'd_regs_' + str(x)).iloc[1:]

    def regs_hist(self, p):
        regs_hist_ = {}

        for r in ['eip']:
            regs_hist_[r] = {}
            for k in range(len(p)):
                regs_hist_[r][k] = \
                    numpy.histogram(self.regs(p)[r].values[:k + 1], bins=10)[0]

        return pandas.DataFrame(regs_hist_)

    def eps(self, p):
        regs_hist_ = self.regs_hist(p)

        def entropy(a):
            a = numpy.maximum(a, 1e-6)
            a /= numpy.sum(a)
            return numpy.sum(a * numpy.log(1 / a) + (1 - a) * numpy.log(1 / (1 - a)))

        eps_ = regs_hist_.copy().apply(lambda x: [entropy(o) for o in x])

        return eps_

    def d_eps(self, p):
        eps = self.eps(p).rename(columns=lambda x: 'd_eps_' + str(x)).diff(axis=0)
        eps.loc[1] = 0

        return eps.iloc[1:]

    def index(self, a_k):
        return pandas.DataFrame({'index': [o['log_entry']['index'] for o in a_k]})

    def run(self):
        while True:
            p = self.get_a_perception()
            acts = self.get_possible_actions()
            params = self._agent.get_parameters()

            

        pass


class TestJobsProcessor(unittest.TestCase):
    def setUp(self):
        self.job = functools.partial(sub_shell, "sleep 0.1")
        self.jobs1 = [self.job, ] * 4
        self.jobs2 = [{'job': self.job,
                       'deps': [
                           {'job': self.job, 'deps': []},
                           {'job': self.job, 'deps': []}
                       ]},
                      {'job': self.job,
                       'deps': [
                           {'job': self.job,
                            'deps': [
                                {'job': self.job, 'deps': []},
                                {'job': self.job, 'deps': []}
                            ]},
                           {'job': self.job,
                            'deps': [
                                {'job': self.job, 'deps': []},
                                {'job': self.job, 'deps': []}
                            ]},
                       ]},
                      {'job': self.job,
                       'deps': []}
                      ]
        self.jobs3 = [self.job, ] * 5
        self.jobs4 = JobsProcessor.make_dependant(self.jobs3)

    def test_make_dependant(self):
        self.assertEqual(self.jobs4[0]['job'], self.job)
        self.assertTrue(len(self.jobs4[0]['deps'][0]['deps'][0]['deps']) == 1)

    def test_empty_jobs(self):
        jp = JobsProcessor(1)
        jp.add_jobs([])

    def test_topological_sort(self):
        sem = multiprocessing.Semaphore(1)
        jp = JobsProcessor(1)
        jp.add_jobs(self.jobs2 + self.jobs4)
        jp.add_jobs(self.jobs3)
        seq = JobsProcessor._topological_sort(jp.jobs)

        for j in seq:
            jp._run_job(j, sem=sem)

    def test_run_job(self):
        JobsProcessor(1)._run_job(self.job, sem=multiprocessing.Semaphore(1))

    def test_should_finish(self):
        jobs = self.jobs1 + self.jobs2

        for k in [1, 3, 10]:
            jp = JobsProcessor(k)
            jp.add_jobs(jobs)
            jp.add_jobs(self.jobs4)

            jp.start()

            jp.join()

            self.assertEqual(jp.exitcode, 0)


class JobsProcessor(multiprocessing.Process):
    def __init__(self, num_of_workers):
        multiprocessing.Process.__init__(self)

        self.jobs = []

        assert num_of_workers > 0

        self.num_of_workers = num_of_workers

    @staticmethod
    def make_dependant(jobs):
        jobs_new = []

        for j in jobs:
            jobs_new = [{'job': j, 'deps': jobs_new}]

        return jobs_new

    def add_jobs(self, jobs):
        for j in jobs:
            if not isinstance(j, dict):
                self.jobs.append({'job': j, 'deps': []})
            else:
                self.jobs.append(j)

    @classmethod
    def _run_job(cls, job, sem):
        sem.acquire()
        job()
        sem.release()

    @staticmethod
    def _topological_sort(jobs):
        seq = []
        q = list(jobs)

        while len(q) > 0:
            j = q[0]
            del q[0]

            seq.append(j['job'])

            q.extend(j['deps'])

        seq.reverse()

        return seq

    def run(self):
        seq = self._topological_sort(self.jobs)
        self.jobs = []

        self.sem = multiprocessing.Semaphore(self.num_of_workers)

        workers = []

        for j in seq:
            self.sem.acquire()
            w = multiprocessing.Process(
                target=functools.partial(self._run_job, j, self.sem))
            w.start()
            self.sem.release()

        for w in workers:
            w.join()


class Tasks:
    def __init__(self, args):
        parser = optparse.OptionParser()
        parser.add_option("-t", "--task", dest="task",
                          help="a name of the main task")
        parser.add_option("--stage", dest="stage", type="int", default=1,
                          help="an intermediate stage number to continue from")
        parser.add_option("--env", dest="env", default=json.dumps({}),
                          help="additional environmental variables as a json string")
        parser.add_option("-V", "--verbose", dest="verbose", action="store_true", default=False,
                          help="enable the verbose mode")

        self._options, self._args = parser.parse_args(args)

        self._initial_args = args

        self.setup()

        if self._options.task in ['lua_test', 'inject']:
            getattr(self, self._options.task)()
        else:
            raise ValueError('Unknown command %s' % ' '.join(args))

    def setup(self):
        self._env = {}

        self._env.update(dict([
            (k.lower(), v) for (k, v) in json.loads(self._options.env).items()
        ]))

        self._env['project_root'] = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..'))
        self._env['lua_cli'] = os.path.join(
            self._env['project_root'],
            'deps',
            'lua',
            'src',
            'lua.py'
        )
        self._env['python_executable'] = self._env.get(
            'python_executable',
            os.path.join(self._env['project_root'],
                         'tmp', 'env', 'bin', 'python')
        )

    def _sub_shell(self, cmds, env={}, *args, **kwargs):
        return Utils.sub_shell(
            cmds=cmds,
            env=dict([
                (k.lower(), v) for (k, v) in
                list(os.environ.items()) +
                list(self._env.items()) +
                list(env.items())
            ]),
            verbose=self._options.verbose,
            *args,
            **kwargs
        )

    def inject(self):
        self._sub_shell(r"""
            cd $PROJECT_ROOT;
            make inject;
            """)

    def check(self):
        sub_shell(r"""
            cd $TORIBASH_PROJECT_ROOT;\
            make check;
        """, critical=True, verbose=True)

    def recover(self):
        sub_shell(r"""
            cd $TORIBASH_PROJECT_ROOT;\
            make recover;
        """, critical=True, verbose=True)

    def _stage_init(self, start_count):
        self._start_count = start_count
        self._stage_count = 1
        self._stage_stack = []

        if self._options.verbose:
            print("_stage_init, _stage_count=%d" % self._stage_count)

    def _stage_step(self):
        self._stage_count += 1

        if self._options.verbose:
            print("_stage_step, _stage_count=%d" % self._stage_count)

    def _stage_save(self, name):
        self._stage_stack.append([name, self._stage_count])

        if self._options.verbose:
            print("_stage_save, _stage_count=%d" % self._stage_count)

    def _stage_load(self, name):
        while self._stage_stack[-1][0] != name:
            del self._stage_stack[-1]

        self._stage_count = self._stage_stack[-1][1]

        if self._options.verbose:
            print("_stage_load, _stage_count=%d" % self._stage_count)

    def _stage_filter(self):
        if self._stage_count < self._start_count:
            return True
        else:
            return False

    def lua_test(self):
        self._stage_init(self._options.stage)

        self._stage_save('a')

        for abi in ['ms', 'sysv']:
            for opt in ['-O0', '-O2', '-O3']:
                for bit in [32, 64]:
                    _env = None

                    self._stage_load('a')

                    for t in ['custom_prefix', 'environment']:
                        if self._stage_filter():
                            self._stage_step()
                            continue

                        _prefix = '%d_%s_%s' % (bit, opt[1:].lower(), abi)

                        ret = self._sub_shell(r"""
    cd $PROJECT_ROOT;
    CC=clang CFLAGS="$OPT -g -mabi=$ABI -m$BIT" LDFLAGS="-mabi=$ABI -m$BIT"\
        $PYTHON_EXECUTABLE $LUA_CLI -t $TASK {args};
                            """.format(
                            args=' '.join([
                                self._options.verbose and '-V' or '',
                                '--env=\'{env}\''.format(env=json.dumps({
                                    'prefix': os.path.join(
                                        self._env['project_root'], 'build', 'lua_' + _prefix
                                    ),
                                    'pkgname': 'lua_' + _prefix
                                }))
                            ])
                        ),
                            env={
                                'bit': bit,
                                'opt': opt,
                                'abi': abi,
                                'task': t
                        },
                            communicate=(t == 'environment')
                        )

                        if t == 'environment':
                            _env = json.loads(ret)

                        self._stage_step()

                    self._sub_shell(r"""
    cd $PROJECT_ROOT;
    export LD_LIBRARY_PATH=$_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
    export PKG_CONFIG_PATH=$_PKG_CONFIG_PATH:$PKG_CONFIG_PATH
    $CC -g -mabi=$ABI -m$BIT $OPT\
        -o build/lua_test_${_PREFIX}.o -c src/lua.cpp `pkg-config $PKGNAME --cflags`;
    $CC -mabi=$ABI -m$BIT\
        -o build/lua_test_${_PREFIX} build/lua_test_${_PREFIX}.o `pkg-config $PKGNAME --libs`;
    $CC -mabi=$ABI -m$BIT\
        -o build/lua_test_static_${_PREFIX}\
        build/lua_test_${_PREFIX}.o `pkg-config --variable=static_libs $PKGNAME`;
                        """,
                                    env={
                                        '_prefix':
                                            '%d_%s_%s_%s' % (
                                                bit, 'clang', opt[1:].lower(), abi),
                                        'opt': opt,
                                        'bit': bit,
                                        'abi': abi,
                                        'cc': 'clang',
                                        '_ld_library_path': _env['ld_library_path'],
                                        '_pkg_config_path': _env['pkg_config_path'],
                                        'pkgname': _env['pkgname']
                                    }
                                    )

                    self._stage_step()

                self._stage_step()

            self._stage_step()

    def _older(self, path1, path2):
        res = False

        for p2 in path2:
            if not os.path.exists(path2):
                res = True
                break

            if not os.path.exists(path1):
                raise ValueError('The source is not valid: %s!' % path1)

            if os.stat(path1).st_mtime > os.stat(path2).st_mtime:
                res = True
                break

        return res

    @classmethod
    def _logged_job(cls, job):
        print(sub_shell(job, critical=True, verbose=True, communicate=True,
                        stderr_to_stdout=True))

    def _build(self, lang, name, s, tb, to, bit, abi, cc, optimize, parallel=False):
        flags = []
        cflags = (os.environ.get('CFLAGS') or '').split(' ')
        cxxflags = (os.environ.get('CXXFLAGS') or '').split(' ')
        ldflags = (os.environ.get('LDFLAGS') or '').split(' ')

        flags += ['-m' + bit]
        ldflags += ['-m' + bit]

        if name == 'c_constructs':
            flags += ['-Dfactorial_attributes=' + cc]

        # if lang == 'g++':
        #    cxxflags += ['-mabi=' + abi]

        if optimize != 'NO_OPT':
            flags += ['-' + optimize]

        cflags += flags
        cxxflags += flags

        compiler = None
        linker = None
        strip = None

        _cflags = ' '.join(cflags)
        _cxxflags = ' '.join(cxxflags)
        _ldflags = ' '.join(ldflags)
        _srcs = ' '.join(s)

        if self._older(tb, tb + '_striped'):
            strip = r"""
                cp {target} {target}_striped;\
                strip {target}_striped;
            """.format(target=tb)

        if lang == 'gcc':
            if self._older(_srcs, to):
                compiler = 'gcc {cflags} -o {cobject} -c {srcs}'\
                    .format(
                        cflags=_cflags,
                        cobject=to,
                        srcs=_srcs)
            if self._older(to, tb):
                linker = 'gcc {ldflags} -o {ctarget} {cobject}'\
                    .format(
                        ldflags=_ldflags,
                        ctarget=tb,
                        cobject=to)
        elif lang == 'g++':
            if self._older(_srcs, to):
                compiler = 'g++ {cxxflags} -o {cxxobject} -c {srcs}'\
                    .format(
                        cxxflags=_cxxflags,
                        cxxobject=to,
                        srcs=_srcs)
            if self._older(to, tb):
                linker = 'g++ {ldflags} -o {cxxtarget} {cxxobject}'\
                    .format(
                        ldflags=_ldflags,
                        cxxtarget=tb,
                        cxxobject=to)

        jobs = []

        for c in [compiler, linker, strip]:
            if c is not None:
                job = functools.partial(self._logged_job,
                                        r"""
                        cd $TORIBASH_PROJECT_ROOT;
                        {command}
                    """.format(command=c))

                if not parallel:
                    job()
                else:
                    jobs.append(job)

        return jobs

    def experiments(self):
        jp = JobsProcessor(int(os.environ['MAKE_JOBS']))

        src = ['src/experiments', 'build',
               ['gcc', 'c_constructs', 'c_constructs.c'],
               ['g++', 'cpp_constructs', 'cpp_constructs.cpp']]

        for t in src[2:]:
            s = [os.path.join(src[0], x) for x in t[2:]]
            tb = os.path.join(src[1], t[1])

            cc_l = [None]

            abi_l = [None]

            if t[1] == 'c_constructs':
                cc_l = ['cdecl', 'fastcall', 'thiscall',
                        'ms_abi', 'sysv_abi']

            # abi_l = ['ms_abi', 'sysv'_abi']

            for bit in ['32', '64']:
                for abi in abi_l:
                    for o in ['NO_OPT', 'O1', 'O2', 'O3']:
                        for cc in cc_l:
                            suffix =\
                                '_' + '_'.join([x for x in [bit, abi, o, cc]
                                                if x is not None])
                            jobs = self._build(
                                lang=t[0],
                                name=t[1],
                                s=s,
                                tb=tb + suffix,
                                to=tb + suffix + '.o',
                                bit=bit,
                                abi=abi,
                                optimize=o,
                                cc=cc,
                                parallel=True)

                            jp.add_jobs(jp.make_dependant(jobs))

        jp.start()
        jp.join()

    def clean(self):
        sub_shell(r"""
            cd $TORIBASH_PROJECT_ROOT;
            rm -fr build/*;
        """, critical=True, verbose=True)


class Radare:
    def __init__(self, args):
        parser = optparse.OptionParser()
        parser.add_option("-t", "--task", dest="task", default="run",
                          help="a name of the main task")
        parser.add_option("--env", dest="env", default=json.dumps({}),
                          help="additional environmental variables as a json string")
        parser.add_option("--rarun", dest="rarun", default=json.dumps({}),
                          help="specify rarun options as a json string")
        parser.add_option("--tcp_server_port", dest="tcp_server_port", type="int",
                          help="set a port to listen for the tcp server")
        parser.add_option("--args", dest="args", default="-",
                          help="set the direct r2 arguments")
        parser.add_option("-c", "--cmds", dest="cmds", action="append", default=[],
                          help="specify r2 commands as multiline strings," +
                               "or put - to consume the commands fron stdin")
        parser.add_option("-V", "--verbose", dest="verbose", action="store_true", default=False,
                          help="enable the verbose mode")

        self._cmds = []

        self._options, self._args = parser.parse_args(args)

        self._initial_args = args

        self.setup()

        if self._options.task in ['run', 'custom']:
            getattr(self, self._options.task)()
        else:
            raise ValueError('Unknown command %s' % ' '.join(args))

    def _join_paths(self, *args):
        f = [p for p in args if p is not None and p != '']

        return os.path.pathsep.join(f)

    def setup(self):
        self._env = {}

        self._env.update(dict([
            (k.lower(), v) for (k, v) in
            list(os.environ.items()) +
            list(json.loads(self._options.env).items())
        ]))

        self._env['r2_valgrind'] = self._env.get('r2_valgrind', '')

        self._env['project_root'] = self._env.get('project_root',
            os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

        self._env['prefix'] = self._env.get('prefix',
            os.path.join(self._env['project_root'], 'deps', 'radare2', 'tmp', 'install'))

        self._env['path'] = self._join_paths(
            os.path.join(self._env['prefix'], 'bin'),
            self._env.get('path')
        )

        self._env['ld_library_path'] = self._join_paths(
            os.path.join(self._env['prefix'], 'lib'),
            self._env.get('ld_library_path')
        )

        self._env['pkg_config_path'] = self._join_paths(
            os.path.join(self._env['prefix'], 'lib', 'pkgconfig'),
            self._env.get('pkg_config_path')
        )

        self._rarun = {
            'params': json.loads(self._options.rarun)
        }

        if len(self._rarun['params']) > 0:
            self._env['rr2profile'] = tempfile.mktemp()
            if self._options.verbose:
                print('rr2profile %s' % self._env['rr2profile'])
            with io.open(self._env['rr2profile'], 'w') as f:
                f.write(u'' +
                        '\n'.join(['%s=%s' % (k, v)
                                   for (k, v) in self._rarun['params'].items()])
                        )

        self._cmds = []
        for c in self._options.cmds:
            if c == '-':
                self._cmds.append(sys.stdin.read())
            else:
                self._cmds.append(c)

        if self._options.tcp_server_port is not None:
            self._cmds.append('e tcp.islocal=true')
            self._cmds.append('.:%d' % self._options.tcp_server_port)

        self._env['r2cmds'] = tempfile.mktemp()
        with io.open(self._env['r2cmds'], 'w') as f:
            f.write(u'' + '\n'.join(self._cmds))

        if self._options.verbose:
            print('r2cmds %s' % self._env['r2cmds'])

        self._env['radare2_args'] = self._options.args

    def _sub_shell(self, cmds, env={}, *args, **kwargs):
        return Utils.sub_shell(
            cmds=cmds,
            env=dict([
                (k.lower(), v) for (k, v) in
                list(os.environ.items()) +
                list(self._env.items()) +
                list(env.items())
            ]),
            verbose=self._options.verbose,
            *args,
            **kwargs
        )

    def run(self):
        self._sub_shell(r"""
            {valgrind} radare2 {rarun} -c ".!cat $R2CMDS" {args};
            """.replace(
            '{args}', self._env['radare2_args'])
            .replace(
                '{rarun}',
                len(self._rarun['params']) and r'-e "dbg.profile=$RR2PROFILE"' or '')
            .replace(
                '{valgrind}',
                len(self._env['r2_valgrind']) and r'valgrind --vgdb-stop-at=startup' or ''))

    def custom(self):
        self._sub_shell(r"""
            {args};
            """.replace('{args}', self._env['radare2_args'])
                        )


if __name__ == '__main__':
    if 'run' == sys.argv[1]:
        sub_shell(r'rm -I $toribash_out', verbose=True, critical=False)

        sub_shell(r"""
            cat <<EOF > $TORIBASH_PROJECT_ROOT/build/toribash.rr2""" + '\n'
                  + r'program=$toribash_common/toribash_steam' + '\n'
                  + r'chdir=$toribash_common' + '\n'
                  + r'setenv=LD_PRELOAD=$toribash_common/libsteam_api.so' + '\n'
                  + r'#stdout=$toribash_out'
                  + '\nEOF' + r"""

            cat <<EOF > $TORIBASH_PROJECT_ROOT/build/toribash.r2.cmd
                #e http.sandbox=false
                #e http.no_pipe=true
                #e http.bind=127.0.0.1
                #e http.port=""" + os.environ['radare2_http_port'] + r"""
                #=h
                e tcp.islocal=true
                .:""" + os.environ['radare2_tcp_port'] + r"""
                """
                  + '\nEOF' + r"""
        """, verbose=True)

        sub_shell(r"""
            if [ -n "${VALGRIND}" ]; then
                valgrind --vgdb-stop-at=startup\
                   r2 -e "dbg.profile=$TORIBASH_PROJECT_ROOT/build/toribash.rr2"\
                      -c ".!cat $TORIBASH_PROJECT_ROOT/build/toribash.r2.cmd "\
                      -d $toribash_common/toribash_steam;
            else
                r2 -e "dbg.profile=$TORIBASH_PROJECT_ROOT/build/toribash.rr2"\
                   -c ".!cat $TORIBASH_PROJECT_ROOT/build/toribash.r2.cmd "\
                   -d $toribash_common/toribash_steam;
            fi
        """, verbose=True)
    elif 'script' == sys.argv[1]:
        algos = Algos()

        if len(sys.argv) == 3:
            getattr(algos, sys.argv[2])()
    elif 'rw' == sys.argv[1]:
        RandomWalkerCore().run()
    elif 'make' == sys.argv[1]:
        Tasks(sys.argv[2:])
    elif 'unit_test' == sys.argv[1]:
        sys.argv = sys.argv[:1] + sys.argv[2:]
        unittest.main()
    elif 'run_single_unit_test' == sys.argv[1]:
        if len(sys.argv) == 3:
            suite = unittest.TestSuite()
            (cls, meth) = sys.argv[2].split('.')
            suite.addTest(globals()[cls](meth))
            suite.debug()
    elif 'autopep8' == sys.argv[1]:
        for f in glob.glob(
                os.path.join(os.path.abspath(
                    os.path.dirname(__file__)), '..', 'src', '*.py')
        ):
            sub_shell(r"""
                %s -i %s
            """ % (os.environ['AUTOPEP8_BINARY'], f), verbose=True)
    elif 'radare2' == sys.argv[1]:
        Radare(sys.argv[2:])
    else:
        raise ValueError("\n\tUnknown command:\n\t\t%s\n" %
                         ' '.join(sys.argv[1:]))
