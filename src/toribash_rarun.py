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
        proc = subprocess.Popen(r"""
            zsh %s
        """ % tf, shell = True,
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
        proc = subprocess.Popen(cmds, shell = True,
                                stdin = sys.stdin,
                                stdout = sys.stdout,
                                stderr = sys.stderr)
        if wait:
            proc.wait()

    if proc.returncode != 0 and critical:
        raise ValueError(proc.returncode)

    return ret


os.environ['toribash_out'] = sub_shell('echo -E $PWD/build/toribash_out',
                                       communicate = True).strip()
os.environ['toribash_common'] = sub_shell('echo -E $PWD/build/toribash',
                                          communicate = True).strip()

if 'run' == sys.argv[-1]:
    sub_shell(r'rm -I $toribash_out', verbose = True, critical = False)

    sub_shell(r"""
        cat <<EOF > $TORIBASH_PROJECT_ROOT/build/toribash.rr2""" + '\n'\
        + r'program=$toribash_common/toribash_steam' + '\n'\
        + r'chdir=$toribash_common' + '\n'\
        + r'setenv=LD_PRELOAD=$toribash_common/libsteam_api.so' + '\n'\
        + r'#stdout=$toribash_out'\
        + '\nEOF' + r"""

        cat <<EOF > $TORIBASH_PROJECT_ROOT/build/toribash.r2.rc
            af man.steam_init_v2 @ 0x081fb240
            af man.steam_networking @ 0x081fb590
            af man.steam_init @ 0x81fae20
            afr man.steam_callbacks @ fcn.081fdf10"""\
        + '\nEOF' + r"""

        cat <<EOF > $TORIBASH_PROJECT_ROOT/build/toribash.r2.cmd
            aa
            .!cat $TORIBASH_PROJECT_ROOT/build/toribash.r2.rc
            db man.steam_init
            dc
            #wa mov eax, 0@@=\`axt @@ sym.imp.Steam*~man.steam_init[1]\`
            e http.sandbox=false
            e http.bind=127.0.0.1
            e http.port=9999
            =h&
            #k a=\`f~sym.imp.Steam:2[0]\`
            #k n=\`f~sym.imp.Steam:2[2]\`
            #db \`k a\`
            #dbc \`k a\` dbt
            #dc
            #axt @@ sym.imp.Steam*
            #pdj 2~{}..
            #wao nop @@ \`axt @@ sym.imp.Steam*~[1]\`"""\
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
elif 'script' == sys.argv[-1]:
    rctx = r2pipe.open("http://127.0.0.1:9999")
    print(rctx.cmd("i"))
#
#
##pdj 1 @ 0x081fae33 | python -c "import json, io, sys, os; print('?y ',json.load(sys.stdin)[0]['size']); sys.stdout.flush();" > tmp/1.txt
##.! cat tmp/1.txt
#
#
