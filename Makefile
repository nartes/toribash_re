check:
	sha256sum -c src/toribash.tar.sha256

recover:
	rm -Ifr build/toribash
	mkdir -p build/toribash
	tar --strip-components 1 -C build/toribash -xvf $$PWD/tmp/toribash.tar

clean:
	rm -fr build/*
