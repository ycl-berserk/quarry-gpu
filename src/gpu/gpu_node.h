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

// ---------------------------------------------------------------------------
// 4 宽节点：把二叉 LBVH 里偶数深度的内部节点折叠成"孩子 + 孙子"最多 4 个槽。
// 一个 128B 节点正好一条缓存行，一次读换四层判断，树深减半。
//   child >= 0      子节点是另一个 4 宽节点（编号即 wide 下标）
//   child <  0      叶子，取值与二叉节点的 left 相同：-(叶子槽 + 1)
//   child == INT_MIN 空槽（孩子是叶子时腾出的位置），盒写成空盒，遍历时先跳过
// 叶子在二叉树里深度为奇数时是直接孩子，偶数时是孙子；两条规则覆盖全部叶子，
// 折叠不会漏掉任何图元。
// ---------------------------------------------------------------------------
constexpr int kWideEmpty = (-2147483647 - 1);

struct GpuBVH4Entry {
    float mnx, mny, mnz;
    float mxx, mxy, mxz;
    int child;
    int pad;
};

struct alignas(128) GpuBVH4Node {
    GpuBVH4Entry c[4];
};

QUARRY_HD inline AABB wideBox(const GpuBVH4Entry& e) {
    AABB b;
    b.mn = {e.mnx, e.mny, e.mnz};
    b.mx = {e.mxx, e.mxy, e.mxz};
    return b;
}
