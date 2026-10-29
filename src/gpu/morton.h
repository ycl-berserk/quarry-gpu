#pragma once

#include <cstdint>

// 10 位/轴量化 + Morton 交错。射线排序和点查询排序共用同一套编码：
// 相邻下标的元素落在同一块空间里，一个 warp 的遍历路径高度重合。
// 10 位已经和地形格点同量级，再加位数只多排序轮次；排序开销随键宽线性涨。
__device__ __forceinline__ uint32_t expandBits10(uint32_t v) {
    v &= 0x3ffu;
    v = (v | (v << 16)) & 0x030000ffu;
    v = (v | (v << 8)) & 0x0300f00fu;
    v = (v | (v << 4)) & 0x030c30c3u;
    v = (v | (v << 2)) & 0x09249249u;
    return v;
}

__device__ __forceinline__ uint32_t quantize10(float v, float lo, float hi) {
    if (!(hi > lo)) return 0;
    float f = (v - lo) / (hi - lo);
    f = fminf(fmaxf(f, 0.0f), 1.0f);
    return static_cast<uint32_t>(f * 1023.0f + 0.5f);
}

__device__ __forceinline__ uint32_t morton3D10(float x, float y, float z, float loX, float hiX,
                                               float loY, float hiY, float loZ, float hiZ) {
    return expandBits10(quantize10(x, loX, hiX)) |
           (expandBits10(quantize10(y, loY, hiY)) << 1) |
           (expandBits10(quantize10(z, loZ, hiZ)) << 2);
}
