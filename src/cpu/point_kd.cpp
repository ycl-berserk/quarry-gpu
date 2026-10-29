#include "cpu/point_kd.h"

#include <algorithm>
#include <cstddef>

namespace {

constexpr int kLeafSize = 8;

ClosestHit makeHit(const Vec3& p, const Vec3& v, int index) {
    ClosestHit h;
    h.found = true;
    h.point = v;
    const Vec3 d = p - v;
    h.dist2 = dot(d, d);
    h.triangle = index;
    return h;
}

// 建树时点和原始索引必须一起搬，否则重排后 ids 对不上 pts。
struct Slot {
    Vec3 p;
    int id;
};

int buildNode(std::vector<Slot>& slots, std::vector<KdNode>& nodes, int start, int count) {
    const int id = static_cast<int>(nodes.size());
    nodes.push_back(KdNode{});
    nodes[static_cast<size_t>(id)].start = start;
    nodes[static_cast<size_t>(id)].count = count;
    for (int i = start; i < start + count; ++i) {
        nodes[static_cast<size_t>(id)].box.extend(slots[static_cast<size_t>(i)].p);
    }

    const AABB box = nodes[static_cast<size_t>(id)].box;

    if (count <= kLeafSize) return id;

    // 沿最长轴切中位数：比轮换轴稳定，退化成链的概率低。
    const Vec3 ext = box.extent();
    const int axis = (ext.x >= ext.y && ext.x >= ext.z) ? 0 : (ext.y >= ext.z ? 1 : 2);
    const int mid = start + count / 2;

    auto begin = slots.begin() + start;
    auto nth = slots.begin() + mid;
    auto end = slots.begin() + start + count;
    std::nth_element(begin, nth, end, [axis](const Slot& a, const Slot& b) {
        return component(a.p, axis) < component(b.p, axis);
    });

    // 子递归会 push_back 节点，可能让 nodes 扩容；这里必须在递归之后再按下标
    // 写回，不能持有引用跨过递归。
    nodes[static_cast<size_t>(id)].count = 0;
    nodes[static_cast<size_t>(id)].axis = axis;
    const int left = buildNode(slots, nodes, start, mid - start);
    const int right = buildNode(slots, nodes, mid, start + count - mid);
    nodes[static_cast<size_t>(id)].left = left;
    nodes[static_cast<size_t>(id)].right = right;
    return id;
}

void closestRec(const PointKdTree& tree, int id, const Vec3& p, ClosestHit& best) {
    const KdNode& node = tree.nodes[static_cast<size_t>(id)];
    // 下界等于当前最优时仍要进：可能藏着距离相同、索引更小的点。
    if (best.found && distanceSquaredToAABB(p, node.box) > best.dist2) return;

    if (node.count > 0) {
        for (int i = node.start; i < node.start + node.count; ++i) {
            const ClosestHit h = makeHit(p, tree.pts[static_cast<size_t>(i)],
                                         tree.ids[static_cast<size_t>(i)]);
            // 并列时取索引更小者，和暴力法 / GPU 的 (dist2, index) 字典序保持一致；
            // 只比距离的话，kd 的访问顺序会决定并列时留下谁。
            if (!best.found || closerHit(h, best)) best = h;
        }
        return;
    }

    const float dl = distanceSquaredToAABB(p, tree.nodes[static_cast<size_t>(node.left)].box);
    const float dr = distanceSquaredToAABB(p, tree.nodes[static_cast<size_t>(node.right)].box);
    if (dl <= dr) {
        closestRec(tree, node.left, p, best);
        closestRec(tree, node.right, p, best);
    } else {
        closestRec(tree, node.right, p, best);
        closestRec(tree, node.left, p, best);
    }
}

void insertKnn(std::vector<ClosestHit>& best, int k, const ClosestHit& h) {
    int pos = static_cast<int>(best.size());
    while (pos > 0 && closerHit(h, best[static_cast<size_t>(pos - 1)])) --pos;
    if (static_cast<int>(best.size()) == k && pos == k) return;
    if (static_cast<int>(best.size()) < k) best.push_back(h);
    for (int j = static_cast<int>(best.size()) - 1; j > pos; --j) {
        best[static_cast<size_t>(j)] = best[static_cast<size_t>(j - 1)];
    }
    best[static_cast<size_t>(pos)] = h;
}

void knnRec(const PointKdTree& tree, int id, const Vec3& p, int k,
            std::vector<ClosestHit>& best) {
    const KdNode& node = tree.nodes[static_cast<size_t>(id)];
    if (static_cast<int>(best.size()) == k &&
        distanceSquaredToAABB(p, node.box) > best.back().dist2) {
        return;
    }

    if (node.count > 0) {
        for (int i = node.start; i < node.start + node.count; ++i) {
            const ClosestHit h = makeHit(p, tree.pts[static_cast<size_t>(i)],
                                         tree.ids[static_cast<size_t>(i)]);
            if (static_cast<int>(best.size()) == k && !closerHit(h, best.back())) continue;
            insertKnn(best, k, h);
        }
        return;
    }

    const float dl = distanceSquaredToAABB(p, tree.nodes[static_cast<size_t>(node.left)].box);
    const float dr = distanceSquaredToAABB(p, tree.nodes[static_cast<size_t>(node.right)].box);
    if (dl <= dr) {
        knnRec(tree, node.left, p, k, best);
        knnRec(tree, node.right, p, k, best);
    } else {
        knnRec(tree, node.right, p, k, best);
        knnRec(tree, node.left, p, k, best);
    }
}

void radiusRec(const PointKdTree& tree, int id, const Vec3& p, float r2,
               std::vector<ClosestHit>& out) {
    const KdNode& node = tree.nodes[static_cast<size_t>(id)];
    if (distanceSquaredToAABB(p, node.box) > r2) return;

    if (node.count > 0) {
        for (int i = node.start; i < node.start + node.count; ++i) {
            const ClosestHit h = makeHit(p, tree.pts[static_cast<size_t>(i)],
                                         tree.ids[static_cast<size_t>(i)]);
            if (h.dist2 <= r2) out.push_back(h);
        }
        return;
    }

    radiusRec(tree, node.left, p, r2, out);
    radiusRec(tree, node.right, p, r2, out);
}

}  // namespace

PointKdTree buildPointKdTree(const PointCloud& cloud) {
    PointKdTree tree;
    if (cloud.pts.empty()) return tree;

    std::vector<Slot> slots;
    slots.reserve(cloud.pts.size());
    for (size_t i = 0; i < cloud.pts.size(); ++i) {
        slots.push_back(Slot{cloud.pts[i], static_cast<int>(i)});
    }

    tree.nodes.reserve(cloud.pts.size() / kLeafSize * 2 + 1);
    buildNode(slots, tree.nodes, 0, static_cast<int>(slots.size()));

    tree.pts.reserve(slots.size());
    tree.ids.reserve(slots.size());
    for (const Slot& s : slots) {
        tree.pts.push_back(s.p);
        tree.ids.push_back(s.id);
    }
    return tree;
}

ClosestHit kdClosest(const PointKdTree& tree, const Vec3& p) {
    ClosestHit best;
    if (tree.nodes.empty()) return best;
    closestRec(tree, 0, p, best);
    return best;
}

std::vector<ClosestHit> kdKNearest(const PointKdTree& tree, const Vec3& p, int k) {
    std::vector<ClosestHit> best;
    if (tree.nodes.empty() || k <= 0) return best;
    best.reserve(static_cast<size_t>(k));
    knnRec(tree, 0, p, k, best);
    return best;
}

std::vector<ClosestHit> kdRadius(const PointKdTree& tree, const Vec3& p, float radius) {
    std::vector<ClosestHit> out;
    if (tree.nodes.empty()) return out;
    radiusRec(tree, 0, p, radius * radius, out);
    std::sort(out.begin(), out.end(), closerHit);
    return out;
}
