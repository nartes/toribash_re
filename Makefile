lua_test: src/lua.cpp
	gcc -o build/lua.o -c src/lua.cpp `pkg-config lua --cflags`
	gcc -o build/lua_test build/lua.o `pkg-config lua --libs`

clean:
	rm -fr build/*
