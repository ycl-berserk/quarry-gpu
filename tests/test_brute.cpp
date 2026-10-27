#include <cstdio>

#include "cpu/selftest.h"

int main() {
    std::printf("== quarry CPU reference tests ==\n");
    return runSelfTest() ? 0 : 1;
}
