#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/geometry.h"

struct Scene {
    std::vector<Tri> tris;
    AABB bounds;

    void updateBounds() {
        bounds = AABB{};
        for (const Tri& t : tris) bounds.extend(t.bounds());
    }
};

struct SceneSpec {
    std::string kind = "terrain";  // terrain | random | mixed | city
    int grid = 256;                // terrain 的边长（格数），三角形数 = 2 * grid^2
    int randomTris = 0;            // random / mixed 的随机三角形数
    uint64_t seed = 1;
    // city：楼的数量，每栋 10 个三角形（4 面墙 + 屋顶）；地面固定 128x128 地形。
    // 场地边长随 sqrt(buildings) 放大，楼间距基本固定（8 个世界单位）。
    int buildings = 65536;
};

Scene buildScene(const SceneSpec& spec);

// 两个基准场景：
//   Down   —— 从上方垂直向下，全部命中，几何分布均匀，用来测理想吞吐；
//   Random —— 相机在远处随机朝向，命中率远低于 100%，用来暴露遍历剪枝的差异。
enum class RayMode { Down, Random };

std::vector<Ray> makeRays(const AABB& bounds, RayMode mode, int count, uint64_t seed);

const char* rayModeName(RayMode mode);

// 地形高度场。点云生成器也用它，好让"扫面"点云和 terrain 场景重合。
float terrainHeight(float x, float z);

// 最近点 / kNN / 半径查询的查询点：均匀落在包围盒放大 1.2 倍的范围内，
// 一部分在物体内部、一部分在外部，两种剪枝路径都会被走到。
std::vector<Vec3> makeQueryPoints(const AABB& bounds, int count, uint64_t seed);
