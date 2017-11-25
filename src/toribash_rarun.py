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