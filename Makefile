check:
	sha256sum -c src/toribash.tar.sha256

recover:
	rm -Ifr build/toribash
	mkdir -p build/toribash
	tar --strip-components 1 -C build/toribash -xvf $$PWD/tmp/toribash.tar

clean:
	rm -fr build/*

inject: src/mylib/inject.cpp
	g++ -c -mabi=ms -fPIC -o build/inject.o $< -g -O0 -m32
	g++ -mabi=ms -o build/inject.so -shared build/inject.o -m32
	for k in .bss .data; do\
		echo Patch rwx to $${k};\
		$$PYTHON_EXECUTABLE src/toribash_rarun.py \
		radare2 -t custom --args="rabin2 -O p/$${k}/rwx build/inject.so";\
	done
	$$PYTHON_EXECUTABLE src/toribash_rarun.py \
		radare2 -t custom --args="r2 -qc \"oo+; wx 07 @@=0x4c 0x6c\" build/inject.so";\

patch_toribash_init:
	mkdir -p build/patch_toribash &&\
	cd build/patch_toribash &&\
	cmake ../../src/patch_toribash

patch_toribash_build:
	cd build/patch_toribash &&\
	make -j3 VERBOSE=1
	$$PYTHON_EXECUTABLE src/toribash_rarun.py \
	  radare2 -t custom \
	  --args="r2 -qc \"oo+; wx 07 @@=0x4c 0x6c\" build/patch_toribash/libpatch_toribash.so";
	mkdir -p build/patch_toribash/dummy_libs
	cp build/patch_toribash/libdummy.so build/patch_toribash/dummy_libs/libsteam_api.so


run_lua_test:
	export DEBUG_PROGRAM=$$PWD/build/lua_test_32_clang_o0_ms;\
	LD_LIBRARY_PATH=$$PWD/build/lua_32_o0_ms/lib:$$LD_LIBRARY_PATH \
	$$PYTHON_EXECUTABLE src/toribash_rarun.py \
	radare2 \
	--rarun='{"setenv": "LD_PRELOAD=${PWD}/build/inject.so", "program": "$$DEBUG_PROGRAM"}' \
	--cmds=".:12345" --args='-d $$DEBUG_PROGRAM' -V

run_toribash:
	export DEBUG_PROGRAM=$$PWD/build/toribash/toribash_steam; \
	export DEBUG_CHDIR=$$PWD/build/toribash/; \
	LD_LIBRARY_PATH=$$PWD/build/patch_toribash/dummy_libs:$$LD_LIBRARY_PATH \
	export INJECT_SO=${PWD}/build/patch_toribash/libpatch_toribash.so; \
	$$PYTHON_EXECUTABLE src/toribash_rarun.py \
	radare2 \
	--rarun='{"setenv": "LD_PRELOAD='$$INJECT_SO'", "chdir": "'$$DEBUG_CHDIR'", "program": "$$DEBUG_PROGRAM"}' \
	--cmds=".:12345" --args='-d $$DEBUG_PROGRAM' -V

src/mylib/inject.cpp: src/mylib/inject.hpp
