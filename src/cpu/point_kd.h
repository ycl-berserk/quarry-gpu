#pragma once

#include <vector>

#include "core/geometry.h"
#include "core/point_cloud.h"

// CPU 侧的点云基线：中位数切分的 kd-tree。
//
// 为什么不是 Embree：Embree 4.4 的 rtcPointQuery 是"对场景里的几何找最近点"，
// 没有点云图元，喂不进几十万个孤立点；Warp 也只有网格最近点。所以点云查询没有
// 现成的第三方单线程入口，这里用最标准的 kd-tree 当对照，口径必须在文档里写清。
struct KdNode {
    AABB box;
    int start = 0;    // 叶子：pts 里的区间 [start, start + count)
    int count = 0;    // count == 0 表示内部节点
    int left = -1;
    int right = -1;
    int axis = 0;     // 内部节点按哪个轴切分（只用于调试）
};

struct PointKdTree {
    std::vector<Vec3> pts;   // 按叶序重排
    std::vector<int> ids;    // 每个槽位对应的场景索引
    std::vector<KdNode> nodes;
};

PointKdTree buildPointKdTree(const PointCloud& cloud);

ClosestHit kdClosest(const PointKdTree& tree, const Vec3& p);
std::vector<ClosestHit> kdKNearest(const PointKdTree& tree, const Vec3& p, int k);
std::vector<ClosestHit> kdRadius(const PointKdTree& tree, const Vec3& p, float radius);
