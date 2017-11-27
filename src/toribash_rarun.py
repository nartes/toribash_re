import io, os, sys
import subprocess
import tempfile

os.environ['TORIBASH_PROJECT_ROOT'] =\
        os.environ.get('TORIBASH_PROJECT_ROOT') or\
        os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')

R2_PIPE_GIT = os.path.join(os.environ['TORIBASH_PROJECT_ROOT'], '..', 'radare2-r2pipe', 'python')

if os.path.exists(R2_PIPE_GIT):
    sys.path.insert(0, R2_PIPE_GIT)
else:
    print('Warning: not found GIT r2pipe!')

import r2pipe

def sub_shell(cmds,
              communicate = False,
              verbose = False,
              wait = True,
              critical = True):
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
        proc = subprocess.Popen(['zsh', tf],
                                stdin = subprocess.PIPE,
                                stdout = subprocess.PIPE,
                                stderr = subprocess.PIPE)
        try:
            proc.wait()
            out, err = proc.communicate()
            ret = out.decode()
        except:
            proc.kill()
    else:
        proc = subprocess.Popen(['zsh', tf],
                                stdin = sys.stdin,
                                stdout = sys.stdout,
                                stderr = sys.stderr)
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
                                       communicate = True).strip()
os.environ['toribash_common'] = sub_shell('echo -E $PWD/build/toribash',
                                          communicate = True).strip()
os.environ['radare2_http_port'] = os.environ.get('radare2_http_port') or '9998'
os.environ['radare2_tcp_port'] = os.environ.get('radare2_tcp_port') or '9997'

if 'run' == sys.argv[-1]:
    sub_shell(r'rm -I $toribash_out', verbose = True, critical = False)

    sub_shell(r"""
        cat <<EOF > $TORIBASH_PROJECT_ROOT/build/toribash.rr2""" + '\n'\
        + r'program=$toribash_common/toribash_steam' + '\n'\
        + r'chdir=$toribash_common' + '\n'\
        + r'setenv=LD_PRELOAD=$toribash_common/libsteam_api.so' + '\n'\
        + r'#stdout=$toribash_out'\
        + '\nEOF' + r"""

        cat <<EOF > $TORIBASH_PROJECT_ROOT/build/toribash.r2.cmd
            #e http.sandbox=false
            #e http.no_pipe=true
            #e http.bind=127.0.0.1
            #e http.port=""" + os.environ['radare2_http_port'] + r"""
            #=h
            .:""" + os.environ['radare2_tcp_port'] + r"""
            """\
        + '\nEOF' + r"""
    """, verbose = True)

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
    """, verbose = True)
elif 'script' == sys.argv[1]:
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

        def af1d(self, fcn = 'man.toribash_core'):
            refs = [r['addr'] for r in self.rctx.cmdj("afij @ %s" % fcn)[0]['callrefs']]
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
            self.rctx.cmd("wao nop @@=%s" % " ".join([str(x) for x in [\
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

    algos = Algos()

    if len(sys.argv) == 3:
        getattr(algos, sys.argv[2])()
elif 'make' == sys.argv[1]:
    class Tasks:
        def __init__(self):
            pass

        def check(self):
            sub_shell(r"""
                cd $TORIBASH_PROJECT_ROOT;\
                make check;
            """, critical = True, verbose = True)

        def recover(self):
            sub_shell(r"""
                cd $TORIBASH_PROJECT_ROOT;\
                make recover;
            """, critical = True, verbose = True)

        def lua_test(self):
            sub_shell(r"""
                cd $TORIBASH_PROJECT_ROOT;\
                make lua_test;
            """, critical = True, verbose = True)

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

        def _build(self, lang, name, s, tb, to, bit, abi, cc, optimize):
            flags = []
            cflags = (os.environ.get('CFLAGS') or '').split(' ')
            cxxflags = (os.environ.get('CXXFLAGS') or '').split(' ')
            ldflags = (os.environ.get('LDFLAGS') or '').split(' ')

            flags += ['-m' + bit]
            ldflags += ['-m' + bit]

            if name == 'c_constructs':
                flags += ['-Dfactorial_attributes=' + cc]

            #if lang == 'g++':
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
                """.format(target = tb)


            if lang == 'gcc':
                if self._older(_srcs, to):
                    compiler = 'gcc {cflags} -o {cobject} -c {srcs}'\
                            .format(
                                cflags = _cflags,
                                cobject = to,
                                srcs = _srcs)
                if self._older(to, tb):
                    linker = 'gcc {ldflags} -o {ctarget} {cobject}'\
                            .format(
                                ldflags = _ldflags,
                                ctarget = tb,
                                cobject = to)
            elif lang == 'g++':
                if self._older(_srcs, to):
                    compiler = 'g++ {cxxflags} -o {cxxobject} -c {srcs}'\
                            .format(
                                cxxflags = _cxxflags,
                                cxxobject = to,
                                srcs = _srcs)
                if self._older(to, tb):
                    linker = 'g++ {ldflags} -o {cxxtarget} {cxxobject}'\
                            .format(
                                ldflags = _ldflags,
                                cxxtarget = tb,
                                cxxobject = to)

            for c in [compiler, linker, strip]:
                if c is not None:
                    sub_shell(r"""
                        cd $TORIBASH_PROJECT_ROOT;
                        {command}
                    """.format(command = c),
                    critical = True, verbose = True)

        def experiments(self):
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

                #abi_l = ['ms_abi', 'sysv'_abi']

                for bit in ['32', '64']:
                    for abi in abi_l:
                        for o in ['NO_OPT', 'O1', 'O2', 'O3']:
                            for cc in cc_l:
                                suffix =\
                                    '_' + '_'.join([x for x in [bit, abi, o, cc]\
                                                    if x is not None])
                                self._build(
                                    lang = t[0],
                                    name = t[1],
                                    s = s,
                                    tb = tb + suffix,
                                    to = tb + suffix + '.o',
                                    bit = bit,
                                    abi = abi,
                                    optimize = o,
                                    cc = cc)
        def clean(self):
            sub_shell(r"""
                cd $TORIBASH_PROJECT_ROOT;
                rm -fr build/*;
            """, critical = True, verbose = True)

    tasks = Tasks()

    if len(sys.argv) == 3:
        getattr(tasks, sys.argv[2])()
