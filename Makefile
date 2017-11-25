check:
	sha256sum -c src/toribash.tar.sha256

recover:
	rm -Ifr build/toribash
	mkdir -p build/toribash
	tar --strip-components 1 -C build/toribash -xvf $$PWD/tmp/toribash.tar

lua_test: src/lua.cpp
	gcc -o build/lua.o -c src/lua.cpp `pkg-config lua --cflags` $$CXXFLAGS
	gcc -o build/lua_test build/lua.o `pkg-config lua --libs` $$LDFLAGS

experiments: src/experiments/c_constructs.c
	for bit in 32 64; do\
		for c in cdecl fastcall thiscall ms_abi sysv_abi; do\
			for o in NO_OPT O1 O2 O3; do\
				[ "NO_OPT" != "$$o" ] &&  _a=-$$o;\
				[ "NO_OPT" == "$$o" ] &&  _a= ;\
				_b=_$${o}_$${c}_$${bit};\
				echo "CC src/c_constructs.c -> build/c_constructs$$_b";\
				gcc $$_a -m$${bit}\
				         -Dfactorial_attributes=$$c\
					 -o build/c_constructs$$_b.o\
					 -c src/experiments/c_constructs.c $$CFLAGS;\
				echo "LD src/c_constructs.c -> build/c_constructs$$_b";\
				gcc -m$${bit}\
				     -o build/c_constructs$$_b\
				     build/c_constructs$$_b.o $$LDFLAGS;\
				echo "strip build/c_constructs$$_b to build/c_constructs$${_b}_striped";\
				cp build/c_constructs$$_b build/c_constructs$${_b}_striped;\
				strip build/c_constructs$${_b}_striped;\
			done;\
		done;\
	done;

clean:
	rm -fr build/*
