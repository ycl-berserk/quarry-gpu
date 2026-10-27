#pragma once

#include <cstdint>

// PCG32。选它是因为：实现短、状态只有 16 字节、跨平台比特级一致。
// 场景和射线都由固定 seed 生成，这样任何引擎（CPU/GPU）跑出来的输入完全相同。
class Pcg32 {
public:
    explicit Pcg32(uint64_t seed = 0, uint64_t stream = 0) {
        state_ = 0;
        inc_ = (stream << 1u) | 1u;
        nextU32();
        state_ += 0x853c49e6748fea9bull;
        nextU32();
        state_ += seed;
        nextU32();
    }

    uint32_t nextU32() {
        const uint64_t old = state_;
        state_ = old * 6364136223846793005ull + inc_;
        const uint32_t xorshifted = static_cast<uint32_t>(((old >> 18u) ^ old) >> 27u);
        const uint32_t rot = static_cast<uint32_t>(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((0u - rot) & 31u));
    }

    float uniform01() {
        return static_cast<float>(nextU32() >> 8) * (1.0f / 16777216.0f);
    }

    float uniform(float lo, float hi) {
        return lo + (hi - lo) * uniform01();
    }

private:
    uint64_t state_ = 0;
    uint64_t inc_ = 1;
};
