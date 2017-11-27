#include <iostream>
#include <vector>
#include <list>
#include <unordered_map>
#include <string>

using namespace std;

class A {
public:
	A() {
		a = 0xf234;
		e.assign(10, {{{{0x2, {-1, 0xff}}}}});
		d = new char*[3];
		d[0] = "asdfasdfasdf";
		d[1] = "Hello, World!";
		d[2] = 0;
	}
protected:
	int a;
	int b;
	int c;
	vector<vector<list<pair<int, pair<long, long>>>>> e;
	char **d;
};

class B: public A {
public:
	B() {
		a = 0xff;
	}
};

int main(int argc, char *argv[]) {
	string line;

	B b = B();
	A a = A();

	cout << "Hello, World!" << endl;
	while (getline(cin, line)) {
		cout << line << endl
		     << "The line has been read" << endl;
	}

	return 0;
}
