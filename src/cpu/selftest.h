#pragma once

// 自测集合。CLI 的 `quarry selftest` 和 tests/test_brute.exe 共用同一套检查，
// 保证"能跑起来的验证"和"CI 里的验证"不是两份东西。
bool runSelfTest();
