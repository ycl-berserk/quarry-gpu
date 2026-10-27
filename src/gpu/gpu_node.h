#pragma once

#include "core/geometry.h"

// 32 字节，正好半个缓存行：遍历时一个 128B 事务能装两个节点。
// 内部节点：left/right 是子节点下标（>= 0）。
// 叶子：left = -(区间起点 + 1)，right = 三角形个数（LBVH 里恒为 1）。
// 叶子编号固定为 n-1 + slot，和 learn/03 的布局一致。
struct GpuBVHNode {
    float mnx, mny, mnz;
    float mxx, mxy, mxz;
    int left;
    int right;
};

QUARRY_HD inline AABB nodeBox(const GpuBVHNode& n) {
    AABB b;
    b.mn = {n.mnx, n.mny, n.mnz};
    b.mx = {n.mxx, n.mxy, n.mxz};
    return b;
}

QUARRY_HD inline void setNodeBox(GpuBVHNode& n, const AABB& b) {
    n.mnx = b.mn.x;
    n.mny = b.mn.y;
    n.mnz = b.mn.z;
    n.mxx = b.mx.x;
    n.mxy = b.mx.y;
    n.mxz = b.mx.z;
}

QUARRY_HD inline bool sameBox(const GpuBVHNode& a, const GpuBVHNode& b) {
    return a.mnx == b.mnx && a.mny == b.mny && a.mnz == b.mnz &&
           a.mxx == b.mxx && a.mxy == b.mxy && a.mxz == b.mxz;
}
