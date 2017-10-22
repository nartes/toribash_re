lua_test: src/lua.cpp
	gcc -o build/lua.o -c src/lua.cpp `pkg-config lua --cflags` $$CXXFLAGS
	gcc -o build/lua_test build/lua.o `pkg-config lua --libs` $$LDFLAGS

clean:
	rm -fr build/*
