#include "inject.hpp"

#include <cstdio>

int abc(char *buf, int n) {
	printf("Hello, World!\n");
	fflush(stdout);

	for (int i = 0; i < n; ++i)
	{
		printf("%02xhh", buf);
	}
}
