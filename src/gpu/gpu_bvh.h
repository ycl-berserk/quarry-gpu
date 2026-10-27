#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/fingerprint.h"
#include "core/geometry.h"
#include "core/scene.h"

// GPU 后端的主机侧 API。头文件里不出现任何 CUDA 类型，设备指针一律用 void*：
// 没装 CUDA 的机器（CI、纯 CPU 构建）也能包含它、链接到空实现之外的分支。
//
// 树是 LBVH（Morton + 基数排序 + Karras 建树），节点布局与 learn/03 一致：
//   内部节点 0 .. n-2，叶子 n-1 .. 2n-2。
struct GpuBVH {
    void* d_nodes = nullptr;    // GpuBVHNode[]，共 2n-1 个
    void* d_tris = nullptr;     // Tri[]，按叶子顺序重排
    void* d_primIds = nullptr;  // int[]，tris[i] 对应的场景索引
    int triCount = 0;
    int nodeCount = 0;
    int depth = 0;
};

struct GpuBuildTimings {
    double mortonMs = 0.0;   // 重心 + Morton 码
    double sortMs = 0.0;     // cub::DeviceRadixSort
    double treeMs = 0.0;     // Karras 内部节点二分
    double scatterMs = 0.0;  // 三角形按叶子顺序重排
    double refitMs = 0.0;    // 自底向上合并 AABB
    double totalMs = 0.0;
};

// 射线与结果都常驻显存：计时只包住 kernel，不含 H2D/D2H。
struct GpuRayBuffer {
    void* d_rays = nullptr;     // Ray[]
    void* d_results = nullptr;  // uint64_t[]，每条射线 (hit<<32 | t 的位表示)
    int count = 0;
};

// 返回设备名与计算能力；没有可用设备时返回 false 并写 why。
bool gpuDescribe(std::string* why);

// 可能抛 std::runtime_error（CUDA 调用失败时）。
GpuBVH buildBVHGPU(const Scene& scene, GpuBuildTimings* timings = nullptr);
void destroyBVHGPU(GpuBVH& bvh);

GpuRayBuffer uploadRaysGPU(const std::vector<Ray>& rays);
void destroyRayBufferGPU(GpuRayBuffer& buf);

// 跑一遍遍历，返回 kernel 毫秒数（CUDA event 计时，不含拷贝）。
double raycastGPU(const GpuBVH& bvh, GpuRayBuffer& buf, bool collectStats = false);

// 把显存里的逐射线结果按射线顺序折叠成指纹；perRay 非空时顺便拷回主机。
RayStats summarizeGPU(const GpuRayBuffer& buf, std::vector<uint64_t>* perRay = nullptr);

// 上一次带 collectStats 的遍历的节点/三角形测试总数。
struct GpuTraversalStats {
    uint64_t nodeTests = 0;
    uint64_t triTests = 0;
    uint64_t stackOverflows = 0;
};
GpuTraversalStats lastTraversalStatsGPU();

// 检查树的结构不变量：父盒逐位等于两个子盒的并、叶子盒逐位等于三角形包围盒、
// 每个非根节点被恰好引用一次、根盒等于场景包围盒。返回空串表示通过。
std::string validateBVHGPU(const GpuBVH& bvh, const Scene& scene);
