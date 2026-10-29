#include "cpu/gauss_bvh.h"

#include <algorithm>
#include <cassert>

namespace {

constexpr int kLeafSize = 4;
constexpr int kMaxDepth = 64;

// 中位数切分：与三角形 BVH 同一套策略，只是重心换成椭球中心、盒换成外接盒。
// 椭球的"重心"就是均值——用外接盒中心会把长轴方向拉偏。
class Builder {
public:
    explicit Builder(const GaussianCloud& cloud) : cloud_(cloud) {
        bounds_.resize(cloud.gaussians.size());
        order_.resize(cloud.gaussians.size());
        for (size_t i = 0; i < cloud.gaussians.size(); ++i) {
            bounds_[i] = gaussianBounds(cloud.gaussians[i]);
            order_[i] = static_cast<int>(i);
        }
    }

    GaussBVH build() {
        if (cloud_.gaussians.empty()) return GaussBVH{};

        buildRange(0, static_cast<int>(order_.size()), 0);

        GaussBVH bvh;
        bvh.nodes = std::move(nodes_);
        bvh.prims.resize(order_.size());
        for (size_t i = 0; i < order_.size(); ++i) bvh.prims[i] = cloud_.gaussians[order_[i]];
        bvh.primIds = order_;
        return bvh;
    }

private:
    int buildRange(int start, int end, int depth) {
        assert(depth < kMaxDepth);

        const int nodeIndex = static_cast<int>(nodes_.size());
        nodes_.push_back(GaussBVHNode{});

        AABB box;
        AABB centroidBox;
        for (int i = start; i < end; ++i) {
            const int prim = order_[i];
            box.extend(bounds_[prim]);
            centroidBox.extend(cloud_.gaussians[prim].mean);
        }
        nodes_[nodeIndex].box = box;

        const int count = end - start;
        if (count <= kLeafSize) {
            nodes_[nodeIndex].start = start;
            nodes_[nodeIndex].count = count;
            return nodeIndex;
        }

        const Vec3 ext = centroidBox.extent();
        const int axis = (ext.x >= ext.y && ext.x >= ext.z) ? 0 : (ext.y >= ext.z ? 1 : 2);

        int mid = start + count / 2;
        std::nth_element(order_.begin() + start, order_.begin() + mid, order_.begin() + end,
                         [&](int lhs, int rhs) {
                             return component(cloud_.gaussians[lhs].mean, axis) <
                                    component(cloud_.gaussians[rhs].mean, axis);
                         });

        if (mid == start || mid == end) mid = start + count / 2;
        if (mid == start || mid == end) mid = start + 1;

        const int left = buildRange(start, mid, depth + 1);
        const int right = buildRange(mid, end, depth + 1);
        assert(left == nodeIndex + 1);
        (void)left;

        nodes_[nodeIndex].right = right;
        return nodeIndex;
    }

    const GaussianCloud& cloud_;
    std::vector<AABB> bounds_;
    std::vector<int> order_;
    std::vector<GaussBVHNode> nodes_;
};

struct NodeBound {
    float bound;
    int node;
};

struct NearerFirst {
    bool operator()(const NodeBound& a, const NodeBound& b) const { return a.bound > b.bound; }
};

struct FartherFirst {
    bool operator()(const ClosestHit& a, const ClosestHit& b) const { return closerHit(a, b); }
};

// 与三角形 BVH 的 Scratch 同样的理由：每次查询都新建容器会被 malloc 吃掉一半时间。
struct Scratch {
    std::vector<NodeBound> frontier;
    std::vector<ClosestHit> knnHeap;
    std::vector<int> stack;
};

thread_local Scratch g_scratch;

void frontierPush(std::vector<NodeBound>& frontier, float bound, int node) {
    frontier.push_back({bound, node});
    std::push_heap(frontier.begin(), frontier.end(), NearerFirst{});
}

NodeBound frontierPop(std::vector<NodeBound>& frontier) {
    std::pop_heap(frontier.begin(), frontier.end(), NearerFirst{});
    const NodeBound top = frontier.back();
    frontier.pop_back();
    return top;
}

}  // namespace

GaussBVH buildGaussBVH(const GaussianCloud& cloud) { return Builder(cloud).build(); }

ClosestHit closestGaussianBVH(const GaussBVH& bvh, const Vec3& p, GaussBVHStats* stats) {
    ClosestHit best;
    if (bvh.nodes.empty()) return best;

    auto& frontier = g_scratch.frontier;
    frontier.clear();
    frontierPush(frontier, distanceSquaredToAABB(p, bvh.nodes[0].box), 0);

    while (!frontier.empty()) {
        const NodeBound cur = frontierPop(frontier);
        if (best.found && cur.bound > best.dist2) break;

        const GaussBVHNode& node = bvh.nodes[cur.node];
        if (stats) ++stats->nodeTests;

        if (node.count > 0) {
            for (int i = 0; i < node.count; ++i) {
                const int slot = node.start + i;
                if (stats) ++stats->primTests;
                ClosestHit h;
                h.found = true;
                h.dist2 = gaussianDist2(p, bvh.prims[slot]);
                h.triangle = bvh.primIds[slot];
                if (!best.found || closerHit(h, best)) best = h;
            }
            continue;
        }

        const int children[2] = {cur.node + 1, node.right};
        for (int child : children) {
            const float bound = distanceSquaredToAABB(p, bvh.nodes[child].box);
            if (!best.found || bound <= best.dist2) frontierPush(frontier, bound, child);
        }
    }

    return best;
}

std::vector<ClosestHit> kNearestGaussianBVH(const GaussBVH& bvh, const Vec3& p, int k,
                                            GaussBVHStats* stats) {
    std::vector<ClosestHit> out;
    if (k <= 0 || bvh.nodes.empty()) return out;

    auto& heap = g_scratch.knnHeap;
    heap.clear();

    auto& frontier = g_scratch.frontier;
    frontier.clear();
    frontierPush(frontier, distanceSquaredToAABB(p, bvh.nodes[0].box), 0);

    auto full = [&]() { return static_cast<int>(heap.size()) >= k; };
    auto worst = [&]() -> const ClosestHit& { return heap.front(); };

    while (!frontier.empty()) {
        const NodeBound cur = frontierPop(frontier);
        if (full() && cur.bound > worst().dist2) break;

        const GaussBVHNode& node = bvh.nodes[cur.node];
        if (stats) ++stats->nodeTests;

        if (node.count > 0) {
            for (int i = 0; i < node.count; ++i) {
                const int slot = node.start + i;
                if (stats) ++stats->primTests;
                ClosestHit h;
                h.found = true;
                h.dist2 = gaussianDist2(p, bvh.prims[slot]);
                h.triangle = bvh.primIds[slot];

                if (full() && !closerHit(h, worst())) continue;
                if (full()) {
                    std::pop_heap(heap.begin(), heap.end(), FartherFirst{});
                    heap.pop_back();
                }
                heap.push_back(h);
                std::push_heap(heap.begin(), heap.end(), FartherFirst{});
            }
            continue;
        }

        const int children[2] = {cur.node + 1, node.right};
        for (int child : children) {
            const float bound = distanceSquaredToAABB(p, bvh.nodes[child].box);
            if (!full() || bound <= worst().dist2) frontierPush(frontier, bound, child);
        }
    }

    out.assign(heap.begin(), heap.end());
    std::sort(out.begin(), out.end(), closerHit);
    return out;
}

std::vector<ClosestHit> radiusGaussianBVH(const GaussBVH& bvh, const Vec3& p, float radius,
                                          GaussBVHStats* stats) {
    std::vector<ClosestHit> out;
    if (radius < 0.0f || bvh.nodes.empty()) return out;

    const float r2 = radius * radius;
    auto& stack = g_scratch.stack;
    stack.clear();
    stack.push_back(0);

    while (!stack.empty()) {
        const int index = stack.back();
        stack.pop_back();

        const GaussBVHNode& node = bvh.nodes[index];
        if (distanceSquaredToAABB(p, node.box) > r2) continue;
        if (stats) ++stats->nodeTests;

        if (node.count > 0) {
            for (int i = 0; i < node.count; ++i) {
                const int slot = node.start + i;
                if (stats) ++stats->primTests;
                ClosestHit h;
                h.found = true;
                h.dist2 = gaussianDist2(p, bvh.prims[slot]);
                if (h.dist2 > r2) continue;
                h.triangle = bvh.primIds[slot];
                out.push_back(h);
            }
        } else {
            stack.push_back(node.right);
            stack.push_back(index + 1);
        }
    }

    std::sort(out.begin(), out.end(), closerHit);
    return out;
}
