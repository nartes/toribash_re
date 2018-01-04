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

run_lua_test:
	LD_LIBRARY_PATH=$$PWD/build/lua_32_o0/lib:$$LD_LIBRARY_PATH \
	$$PYTHON_EXECUTABLE src/toribash_rarun.py \
	radare2 \
	--rarun='{"setenv": "LD_PRELOAD=build/inject.so", "program": "build/lua_test_32_clang_o0"}' \
	--cmds=".:12345" --args='-d build/lua_test_32_clang_o0' -V

src/mylib/inject.cpp: src/mylib/inject.hpp
