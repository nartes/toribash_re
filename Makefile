check:
	sha256sum -c src/toribash.tar.sha256

recover:
	rm -Ifr build/toribash
	mkdir -p build/toribash
	tar --strip-components 1 -C build/toribash -xvf $$PWD/tmp/toribash.tar

lua_test: src/lua.cpp
	gcc -o build/lua.o -c src/lua.cpp `pkg-config lua --cflags` $$CXXFLAGS
	gcc -o build/lua_test build/lua.o `pkg-config lua --libs` $$LDFLAGS

clean:
	rm -fr build/*
