#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/geometry.h"

// 点云：M2 的第一种非三角图元。每个图元是一个零体积的点，包围盒退化成
// mn == mx == p；构建/查询骨架与三角形共用，只有"叶子里测什么"不同。
struct PointCloud {
    std::vector<Vec3> pts;
    AABB bounds;

    void updateBounds() {
        bounds = AABB{};
        for (const Vec3& p : pts) bounds.extend(p);
    }
};

struct PointCloudSpec {
    // sheet  —— 落在与 terrain 场景相同的高度场上（带微小抖动），像 LiDAR 扫面；
    // volume —— 立方体内均匀分布，节点重叠多，是剪枝的下限。
    std::string kind = "sheet";
    int count = 131072;
    uint64_t seed = 1;
};

PointCloud buildPointCloud(const PointCloudSpec& spec);
