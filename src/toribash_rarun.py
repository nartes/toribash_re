import io
import os
import sys
import subprocess
import tempfile
import multiprocessing
import functools
import unittest
import importlib


os.environ['TORIBASH_PROJECT_ROOT'] =\
    os.environ.get('TORIBASH_PROJECT_ROOT') or\
    os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')


os.environ['R2_PIPE_GIT'] = os.path.join(
    os.environ['TORIBASH_PROJECT_ROOT'], '..', 'radare2-r2pipe', 'python')

os.environ['RPDB_GIT'] = os.path.join(
    os.environ['TORIBASH_PROJECT_ROOT'], '..', 'rpdb')

os.environ['AUTOPEP8_BINARY'] = os.environ.get('AUTOPEP8_BINARY') or\
    sys.executable + " -m autopep8"

os.environ['RADARE2_GIT'] = os.environ.get('RADARE2_GIT') or\
    os.path.join(
        os.environ['TORIBASH_PROJECT_ROOT'], '..', 'radare2', 'tmp',
        'install')

if not os.path.exists(os.environ['RADARE2_GIT']):
    raise ValueError("Can not find GIT radare2 installation:\n%s" %
                     os.environ['RADARE2_GIT'])
else:
    os.environ['PATH'] = os.path.join(os.environ['RADARE2_GIT'], 'bin') +\
        os.path.pathsep + (os.environ.get('PATH') or '')

    os.environ['LD_LIBRARY_PATH'] = os.path.join(os.environ['RADARE2_GIT'], 'lib') +\
        os.path.pathsep + (os.environ.get('LD_LIBRARY_PATH') or '')


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

    def _ctx(self):
        return r2pipe.open(self.url)

    def cmd(self, cmds):
        return self._ctx().cmd(cmds)

    def cmdj(self, cmds):
        return self._ctx().cmdj(cmds)


class Algos:
    def __init__(self):
        self.rctx = Rctx("tcp://127.0.0.1:%s" % os.environ['radare2_tcp_port'])
        #self.rctx = r2pipe.open("http://127.0.0.1:%s" % os.environ['radare2_http_port'])

    def run_lines(self, cmds):
        for l in cmds.split('\n'):
            l = l.strip()

            if l and not l.startswith('#'):
                print("Consequent command: %s" % l)
                self.rctx.cmd(l)
            else:
                print("Ignored: %s" % l)

    def manual_functions(self):
        self.run_lines(r"""
            af man.steam_init_v2 @ 0x081fb240
            af man.steam_networking @ 0x081fb590
            af man.steam_init @ 0x81fae20
            afr man.steam_callbacks @ fcn.081fdf10
            af @ 0x080c0bf0
            af man.toribash_core @ 0x080b4d10
            af @ 0x080c0c50
            af man.gl_init @ 0x080ea3c0
            af man.lua_init @ 0x080fda40
            af @ 0x081046d0
            af @ 0x081fcd90
            af @ 0x081fce00

            #0x083b8000 lua static code
            #0x083bb770 lua stack cmd
            #0x083bb390 lua stack cmd
            #0x080fdf00 population of lua functions
        """)

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
    def __init__(self):
        pass

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

    def lua_test(self):
        sub_shell(r"""
            cd $TORIBASH_PROJECT_ROOT;\
            make lua_test;
        """, critical=True, verbose=True)

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
    elif 'make' == sys.argv[1]:
        tasks = Tasks()

        if len(sys.argv) == 3:
            getattr(tasks, sys.argv[2])()
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
        for f in [os.path.abspath(__file__)]:
            sub_shell(r"""
                %s -i %s
            """ % (os.environ['AUTOPEP8_BINARY'], f), verbose=True)
    elif 'r2' == sys.argv[1]:
        sub_shell(r"""
            %s
        """ % ' '.join(sys.argv[2:]), verbose=True)
    else:
        raise ValueError("\n\tUnknown command:\n\t\t%s\n" % ' '.join(sys.argv[1:]))
