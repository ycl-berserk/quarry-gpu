#include "cpu/bvh.h"

#include <algorithm>
#include <cassert>

namespace {

constexpr int kLeafSize = 4;
constexpr int kMaxDepth = 64;

class Builder {
public:
    explicit Builder(const Scene& scene) : scene_(scene) {
        bounds_.resize(scene.tris.size());
        order_.resize(scene.tris.size());
        for (size_t i = 0; i < scene.tris.size(); ++i) {
            bounds_[i] = scene.tris[i].bounds();
            order_[i] = static_cast<int>(i);
        }
    }

    BVH build() {
        if (scene_.tris.empty()) return BVH{};

        buildRange(0, static_cast<int>(order_.size()), 0);

        BVH bvh;
        bvh.nodes = std::move(nodes_);
        bvh.tris.resize(order_.size());
        for (size_t i = 0; i < order_.size(); ++i) bvh.tris[i] = scene_.tris[order_[i]];
        bvh.primIds = order_;
        return bvh;
    }

private:
    // 返回节点下标。用中位数切分（非 SAH）：构建简单、并行友好，
    // 质量略低于 SAH，但对 2D/3D 均匀场景足够。SAH 留到 M1 再替换。
    int buildRange(int start, int end, int depth) {
        assert(depth < kMaxDepth);

        const int nodeIndex = static_cast<int>(nodes_.size());
        nodes_.push_back(BVHNode{});

        AABB box;
        AABB centroidBox;
        for (int i = start; i < end; ++i) {
            const int tri = order_[i];
            box.extend(bounds_[tri]);
            centroidBox.extend(bounds_[tri].center());
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
                             return component(bounds_[lhs].center(), axis) <
                                    component(bounds_[rhs].center(), axis);
                         });

        // 所有三角形重心重合时 nth_element 无法保证两侧非空，退化为按数量对半分。
        if (mid == start || mid == end) mid = start + count / 2;
        if (mid == start || mid == end) mid = start + 1;

        const int left = buildRange(start, mid, depth + 1);
        const int right = buildRange(mid, end, depth + 1);
        (void)left;
        assert(left == nodeIndex + 1);

        nodes_[nodeIndex].right = right;
        return nodeIndex;
    }

    const Scene& scene_;
    std::vector<AABB> bounds_;
    std::vector<int> order_;
    std::vector<BVHNode> nodes_;
};

}  // namespace

BVH buildBVH(const Scene& scene) { return Builder(scene).build(); }

RayHit raycastBVH(const BVH& bvh, const Ray& ray, BVHStats* stats) {
    RayHit best;
    if (bvh.nodes.empty()) return best;

    float bestT = ray.tMax;
    int stack[kMaxDepth];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        const int nodeIndex = stack[--sp];
        const BVHNode& node = bvh.nodes[nodeIndex];

        float tEnter;
        if (!intersectAABB(ray, node.box, bestT, tEnter)) continue;
        if (stats) ++stats->nodeTests;

        if (node.count > 0) {
            for (int i = 0; i < node.count; ++i) {
                const int tri = node.start + i;
                Ray sub = ray;
                sub.tMax = bestT;

                float t;
                if (stats) ++stats->triTests;
                if (intersectTriangle(sub, bvh.tris[tri], t)) {
                    bestT = t;
                    best.hit = true;
                    best.t = t;
                    best.triangle = bvh.primIds[tri];
                }
            }
        } else {
            // 先压右再压左：左子树先出栈，命中率更高时能更早收紧 bestT。
            stack[sp++] = node.right;
            stack[sp++] = nodeIndex + 1;
        }
    }

    return best;
}

namespace {

struct NodeBound {
    float bound;
    int node;
};

// 小顶堆比较：堆顶永远是下界最小的节点。
struct NearerFirst {
    bool operator()(const NodeBound& a, const NodeBound& b) const { return a.bound > b.bound; }
};

// 大顶堆比较：堆顶永远是"最差"的候选（最远；距离并列时场景索引最大）。
struct FartherFirst {
    bool operator()(const ClosestHit& a, const ClosestHit& b) const { return closerHit(a, b); }
};

// 一次查询只有几十个节点，但每次新建一个 priority_queue 的 malloc 成本比遍历本身还高
// （实测 20000 次查询里堆操作占 69ms，真正遍历只占 21ms）。
// 缓冲区按线程复用，clear() 保留容量，之后的查询不再分配。
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

ClosestHit closestPointBVH(const BVH& bvh, const Vec3& p, BVHStats* stats) {
    ClosestHit best;
    if (bvh.nodes.empty()) return best;

    auto& frontier = g_scratch.frontier;
    frontier.clear();
    frontierPush(frontier, distanceSquaredToAABB(p, bvh.nodes[0].box), 0);

    while (!frontier.empty()) {
        const NodeBound cur = frontierPop(frontier);

        // 队列按下界升序弹出；当前下界都超过最优解了，剩下的一定更远。
        if (best.found && cur.bound > best.dist2) break;

        const BVHNode& node = bvh.nodes[cur.node];
        if (stats) ++stats->nodeTests;

        if (node.count > 0) {
            for (int i = 0; i < node.count; ++i) {
                const int slot = node.start + i;
                if (stats) ++stats->triTests;
                ClosestHit h = closestPointOnTriangle(p, bvh.tris[slot]);
                h.triangle = bvh.primIds[slot];
                if (!best.found || closerHit(h, best)) best = h;
            }
            continue;
        }

        const int children[2] = {cur.node + 1, node.right};
        for (int child : children) {
            const float bound = distanceSquaredToAABB(p, bvh.nodes[child].box);
            // 用 <=：下界恰好等于最优距离时仍可能藏着索引更小的并列解。
            if (!best.found || bound <= best.dist2) frontierPush(frontier, bound, child);
        }
    }

    return best;
}

std::vector<ClosestHit> kNearestBVH(const BVH& bvh, const Vec3& p, int k, BVHStats* stats) {
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

        // 堆满之后，堆顶就是第 k 近；节点下界超过它就不可能进入结果。
        if (full() && cur.bound > worst().dist2) break;

        const BVHNode& node = bvh.nodes[cur.node];
        if (stats) ++stats->nodeTests;

        if (node.count > 0) {
            for (int i = 0; i < node.count; ++i) {
                const int slot = node.start + i;
                if (stats) ++stats->triTests;
                ClosestHit h = closestPointOnTriangle(p, bvh.tris[slot]);
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

std::vector<ClosestHit> radiusBVH(const BVH& bvh, const Vec3& p, float radius, BVHStats* stats) {
    std::vector<ClosestHit> out;
    if (radius < 0.0f || bvh.nodes.empty()) return out;

    const float r2 = radius * radius;
    auto& stack = g_scratch.stack;
    stack.clear();
    stack.push_back(0);

    while (!stack.empty()) {
        const int index = stack.back();
        stack.pop_back();

        const BVHNode& node = bvh.nodes[index];
        if (distanceSquaredToAABB(p, node.box) > r2) continue;
        if (stats) ++stats->nodeTests;

        if (node.count > 0) {
            for (int i = 0; i < node.count; ++i) {
                const int slot = node.start + i;
                if (stats) ++stats->triTests;
                ClosestHit h = closestPointOnTriangle(p, bvh.tris[slot]);
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

size_t bvhDepth(const BVH& bvh) {
    if (bvh.nodes.empty()) return 0;

    size_t maxDepth = 0;
    std::vector<std::pair<int, size_t>> stack{{0, 1}};
    while (!stack.empty()) {
        const auto [index, depth] = stack.back();
        stack.pop_back();
        maxDepth = std::max(maxDepth, depth);

        const BVHNode& node = bvh.nodes[index];
        if (node.count == 0) {
            stack.emplace_back(index + 1, depth + 1);
            stack.emplace_back(node.right, depth + 1);
        }
    }
    return maxDepth;
}
