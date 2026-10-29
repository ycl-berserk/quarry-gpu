#pragma once

#include <vector>

#include "core/gaussian.h"
#include "core/geometry.h"
#include "core/point_cloud.h"
#include "core/scene.h"

// 暴力求交：每条射线遍历全部三角形。慢，但它是所有加速结构的真值来源。
RayHit raycastBrute(const Scene& scene, const Ray& ray);
void raycastBruteAll(const Scene& scene, const std::vector<Ray>& rays, std::vector<RayHit>& hits);

// 暴力最近点 / kNN / 半径查询。同样只当参考实现：
//   - 最近点在距离并列时取场景索引最小的那个；
//   - kNN 与 radius 的结果按 closerHit 排序。
ClosestHit closestPointBrute(const Scene& scene, const Vec3& p);
std::vector<ClosestHit> kNearestBrute(const Scene& scene, const Vec3& p, int k);
std::vector<ClosestHit> radiusBrute(const Scene& scene, const Vec3& p, float radius);

// 点云的暴力查询：O(点数) 扫一遍，距离就是两点间平方距离。与三角形版本共用
// ClosestHit 与 closerHit，所以 (dist2, 索引) 的并列规则天然一致。
ClosestHit closestVertexBrute(const PointCloud& cloud, const Vec3& p);
std::vector<ClosestHit> kNearestVertexBrute(const PointCloud& cloud, const Vec3& p, int k);
std::vector<ClosestHit> radiusVertexBrute(const PointCloud& cloud, const Vec3& p, float radius);

// 高斯的暴力查询：距离是"到椭球实体"的平方距离（内部为 0），与 GPU 叶子测试
// 用同一个 gaussianDist2。
ClosestHit closestGaussianBrute(const GaussianCloud& cloud, const Vec3& p);
std::vector<ClosestHit> kNearestGaussianBrute(const GaussianCloud& cloud, const Vec3& p, int k);
std::vector<ClosestHit> radiusGaussianBrute(const GaussianCloud& cloud, const Vec3& p,
                                            float radius);
