#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>

#ifndef FACTORIAL_ATTRIBUTES
#define FACTORIAL_ATTRIBUTES
#endif

FACTORIAL_ATTRIBUTES
long long int factorial(int n) {
	long long int res = 0x1;

	if (n < 0) {
		return -1;
	}

	while (n > 0) {
		res *= n;
		--n;
	}

	return res;
}

int main(int argc, char *argv[]) {
	if (argc >= 2 && !strcmp("factorial", argv[1])) {
		if (argc < 3) {
			printf("No argument has been given for factorial!\n");
			return -0x1;
		}

		errno = 0;

		char *tail;
		int n = strtol(argv[2], &tail, 0);

		if (errno || *tail != '\0') {
			printf("Incorrect long int argument detected: %s\n", argv[2]);
			return -0x1;
		}

		long long int r = factorial(n);

		printf("Factorial of %d is %d\n", n, factorial(n));
	}

	return 0;
}
