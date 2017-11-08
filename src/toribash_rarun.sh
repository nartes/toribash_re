#!/usr/bin/sh

toribash_out=$PWD/build/toribash_out
toribash_common=$PWD/build/toribash

rm -I $toribash_out

cat <<EOF > build/toribash.rr2
program=$toribash_common/toribash_steam
chdir=$toribash_common
setenv=LD_PRELOAD=$toribash_common/libsteam_api.so
#stdout=$toribash_out
EOF

cat <<EOF > build/toribash.r2.rc
af man.steam_init_v2 @ 0x081fb240
af man.steam_networking @ 0x081fb590
af man.steam_init @ 0x81fae20
EOF

cat <<EOF > build/toribash.r2.cmd
aa
.!cat build/toribash.r2.rc
db man.steam_init
dc
EOF

r2 -e "dbg.profile=./build/toribash.rr2"\
   -c ".!cat build/toribash.r2.cmd "\
   -d $toribash_common/toribash_steam
