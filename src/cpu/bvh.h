#pragma once

#include <cstdint>
#include <vector>

#include "core/geometry.h"
#include "core/scene.h"

// 隐式二叉树布局：
//   count > 0  → 叶子，start/count 是在 bvh.tris 里的区间
//   count == 0 → 内部节点，左孩子恒为 node + 1，右孩子在 right
struct BVHNode {
    AABB box;
    int start = 0;
    int count = 0;
    int right = -1;
};

struct BVH {
    std::vector<BVHNode> nodes;
    std::vector<Tri> tris;      // 已按叶子顺序重排，遍历时连续访问
    std::vector<int> primIds;   // tris[i] 对应的场景索引，结果对外一律报场景索引
};

struct BVHStats {
    uint64_t nodeTests = 0;
    uint64_t triTests = 0;
};

BVH buildBVH(const Scene& scene);
RayHit raycastBVH(const BVH& bvh, const Ray& ray, BVHStats* stats = nullptr);

// 最近点用最优优先遍历：优先队列按"点到节点的平方距离"弹出，这个值是严格下界，
// 弹到一个下界已经大于当前最优的节点，后面的都不用看了。
ClosestHit closestPointBVH(const BVH& bvh, const Vec3& p, BVHStats* stats = nullptr);

// kNN 用大小固定为 k 的大顶堆：堆顶是当前第 k 近，同时也是剪枝阈值。
std::vector<ClosestHit> kNearestBVH(const BVH& bvh, const Vec3& p, int k,
                                    BVHStats* stats = nullptr);

// 半径查询没有"当前最优"可收紧，就是按固定半径做球-AABB 剔除。
std::vector<ClosestHit> radiusBVH(const BVH& bvh, const Vec3& p, float radius,
                                  BVHStats* stats = nullptr);

size_t bvhDepth(const BVH& bvh);
