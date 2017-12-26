#include "inject.hpp"

#include <cstdio>

int abc(char *buf, int n) {
	for (int i = 0; i < n; ++i)
	{
		printf("%02xhh", buf);
	}
}
