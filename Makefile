check:
	sha256sum -c src/toribash.tar.sha256

recover:
	rm -Ifr build/toribash
	mkdir -p build/toribash
	tar --strip-components 1 -C build/toribash -xvf $$PWD/tmp/toribash.tar

clean:
	rm -fr build/*

inject: src/mylib/inject.cpp
	gcc -c -fPIC -o build/inject.o $< -g -O0 -m32
	g++ -o build/inject.so -shared build/inject.o -m32


src/mylib/inject.cpp: src/mylib/inject.hpp
