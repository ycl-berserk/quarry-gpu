#pragma once

#include <cstdint>
#include <vector>

#include "core/gaussian.h"
#include "core/geometry.h"

// 椭球的中位数切分 BVH（单线程）。布局与三角形的 BVH 相同：
//   count > 0 → 叶子，start/count 是 prims 里的区间；count == 0 → 内部节点。
// 节点盒是图元外接盒的并集，遍历用"点到盒"的平方距离当下界。
struct GaussBVHNode {
    AABB box;
    int start = 0;
    int count = 0;
    int right = -1;
};

struct GaussBVH {
    std::vector<GaussBVHNode> nodes;
    std::vector<Gaussian> prims;  // 按叶子顺序重排
    std::vector<int> primIds;     // prims[i] 对应的场景索引
};

struct GaussBVHStats {
    uint64_t nodeTests = 0;
    uint64_t primTests = 0;
};

GaussBVH buildGaussBVH(const GaussianCloud& cloud);
ClosestHit closestGaussianBVH(const GaussBVH& bvh, const Vec3& p, GaussBVHStats* stats = nullptr);
std::vector<ClosestHit> kNearestGaussianBVH(const GaussBVH& bvh, const Vec3& p, int k,
                                            GaussBVHStats* stats = nullptr);
std::vector<ClosestHit> radiusGaussianBVH(const GaussBVH& bvh, const Vec3& p, float radius,
                                          GaussBVHStats* stats = nullptr);
