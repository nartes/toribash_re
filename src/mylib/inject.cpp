#include "inject.hpp"

#include <cstdio>


void _zero_area_for_injects_2() {
	static char _zero_area_for_injects[0x400000] = {};

	int i = 0;
	i += 1;
}

int abc(char *buf, int n) {
	printf("Hello, World!\n");
	fflush(stdout);

	for (int i = 0; i < n; ++i)
	{
		printf("%02xhh", buf);
	}
}
