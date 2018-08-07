check:
	sha256sum -c res/large_files.sha256.txt

recover:
	rm -Ifr build/toribash
	mkdir -p build/toribash
	tar --strip-components 1 -C build/toribash -xvf $$PWD/tmp/toribash.tar.xz

clean:
	rm -fr build/*

run_toribash:
	cp scripts/toribash.lua build/toribash/data/script; \
	export LD_LIBRARY_PATH=$$PWD/build/patch_toribash/dummy_libs:$$LD_LIBRARY_PATH; \
	export LD_PRELOAD=$$PWD/build/patch_toribash/libpatch_toribash.so; \
	cd build/toribash; \
	./toribash_steam


src/mylib/inject.cpp: src/mylib/inject.hpp
