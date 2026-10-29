#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/geometry.h"

// 高斯图元：均值 + 旋转（四元数）+ 半轴。查询语义是**椭球实体**：
// 距离为 0 表示查询点落在椭球内，否则是到椭球表面的欧氏距离。
// 中心密度、截断阈值这类渲染语义不进查询层——那是着色器的事。
struct Quat {
    float x = 0.0f, y = 0.0f, z = 0.0f, w = 1.0f;
};

struct Gaussian {
    Vec3 mean;
    Quat rot;    // 使用时归一化，允许调用方传未归一化的
    Vec3 scale;  // 三个半轴，> 0
};

// 旋转矩阵的三列（世界系 → 局部系的转置）。四元数先归一化再展开。
QUARRY_HD inline void gaussianAxes(const Quat& q, Vec3& u, Vec3& v, Vec3& w) {
    // 所有乘加都用 QUARRY_FMA 写死顺序：MSVC 不会收紧成 FMA，nvcc 默认会，
    // 只有显式融合两边才是同一串 IEEE 运算。
    const float lenSq = QUARRY_FMA(q.w, q.w, QUARRY_FMA(q.z, q.z, QUARRY_FMA(q.y, q.y, q.x * q.x)));
    float inv = 1.0f;
    if (lenSq > 0.0f) inv = 1.0f / sqrtf(lenSq);
    const float qx = q.x * inv, qy = q.y * inv, qz = q.z * inv, qw = q.w * inv;

    const float yyzz = QUARRY_FMA(qy, qy, qz * qz);
    const float xxzz = QUARRY_FMA(qx, qx, qz * qz);
    const float xxyy = QUARRY_FMA(qx, qx, qy * qy);

    u = {QUARRY_FMA(-2.0f, yyzz, 1.0f), 2.0f * QUARRY_FMA(qx, qy, qz * qw),
         2.0f * QUARRY_FMA(qx, qz, -(qy * qw))};
    v = {2.0f * QUARRY_FMA(qx, qy, -(qz * qw)), QUARRY_FMA(-2.0f, xxzz, 1.0f),
         2.0f * QUARRY_FMA(qy, qz, qx * qw)};
    w = {2.0f * QUARRY_FMA(qx, qz, qy * qw), 2.0f * QUARRY_FMA(qy, qz, -(qx * qw)),
         QUARRY_FMA(-2.0f, xxyy, 1.0f)};
}

// 椭球的外接 AABB：半宽_i = sqrt((a·u_i)² + (b·v_i)² + (c·w_i)²)。
// 末尾按坐标量级补一点余量：盒是剪枝用的**下界**，短一个 ulp 就可能把真正更近的
// 图元挡在门外，而盒略大只是多访问几个节点。
QUARRY_HD inline AABB gaussianBounds(const Gaussian& g) {
    Vec3 u, v, w;
    gaussianAxes(g.rot, u, v, w);
    const float a2 = g.scale.x * g.scale.x;
    const float b2 = g.scale.y * g.scale.y;
    const float c2 = g.scale.z * g.scale.z;

    AABB box;
    box.mn = g.mean;
    box.mx = g.mean;
    for (int axis = 0; axis < 3; ++axis) {
        const float ui = component(u, axis), vi = component(v, axis), wi = component(w, axis);
        const float half = sqrtf(QUARRY_FMA(c2, wi * wi, QUARRY_FMA(b2, vi * vi, a2 * ui * ui)));
        float pad = half + 1e-6f * (1.0f + fabsf(component(g.mean, axis)) + half);
        (axis == 0 ? box.mn.x : axis == 1 ? box.mn.y : box.mn.z) -= pad;
        (axis == 0 ? box.mx.x : axis == 1 ? box.mx.y : box.mx.z) += pad;
    }
    return box;
}

// 点到椭球实体的平方距离。内部为 0。
// 外部时解 Eberly 的一维方程 F(t) = Σ (a_i·y_i / (t + a_i²))² − 1 = 0（t > 0），
// 最近点 x_i = a_i²·y_i / (t + a_i²)。F 在 t > 0 上单调递减且凸。
//
// 求根策略：牛顿 + 二分兜底，括号 [0, 2·a_max·|y|] 两端符号有保证——
//   F(0) > 0 由上面的内部判定给出；F(2·a_max·|y|) < 0 因为每项
//   (a_i y_i/(t + a_i²))² ≤ (y_i/(2|y|))² ≤ 1/4，三项和不超过 3/4。
// 牛顿步只在括号内才采用，否则取中点，所以迭代永远收敛（最坏退化成二分）。
// 不能只用牛顿：球体近似给出的初值 t0 = a_max(|y| − a_max) 在扁椭球上会落在根
// 的**上方**，从上方起步时第一步可能冲到负值；纯牛顿从 0 起步又会在远点处
// 线性收敛（12 轮仍差 12%）。括号保证了两个方向都不会跑飞。
// 迭代次数写死、不提前退出：CPU 参考实现与 GPU 内核必须走同一串运算。
constexpr int kGaussianNewtonIters = 14;

QUARRY_HD inline float gaussianDist2(const Vec3& p, const Gaussian& g) {
    Vec3 u, v, w;
    gaussianAxes(g.rot, u, v, w);
    const Vec3 d = p - g.mean;
    const float y0 = dot(d, u), y1 = dot(d, v), y2 = dot(d, w);

    const float a = g.scale.x, b = g.scale.y, c = g.scale.z;
    const float aa = a * a, bb = b * b, cc = c * c;
    const float gy0 = a * y0, gy1 = b * y1, gy2 = c * y2;

    // F(0) = Σ(y_i/a_i)² − 1；≤ 0 即点在椭球内（含表面），最近距离为 0。
    {
        const float r0 = y0 / a, r1 = y1 / b, r2 = y2 / c;
        if (QUARRY_FMA(r2, r2, QUARRY_FMA(r1, r1, r0 * r0)) <= 1.0f) return 0.0f;
    }

    const float lenSq = QUARRY_FMA(y2, y2, QUARRY_FMA(y1, y1, y0 * y0));
    const float len = sqrtf(lenSq);
    const float amax = a > b ? (a > c ? a : c) : (b > c ? b : c);

    float lo = 0.0f;
    float hi = 2.0f * amax * len;
    float t = 0.0f;
    if (len > amax) t = amax * (len - amax);  // 球体时就是精确根，其余当参考点

    for (int iter = 0; iter < kGaussianNewtonIters; ++iter) {
        // 每轮 3 次除法：先取倒数再乘，比 g/d 少一半除法；两侧都是 IEEE 除法，
        // 位模式依旧一致。
        const float ir0 = 1.0f / (t + aa), ir1 = 1.0f / (t + bb), ir2 = 1.0f / (t + cc);
        const float s0 = gy0 * ir0, s1 = gy1 * ir1, s2 = gy2 * ir2;
        const float f = QUARRY_FMA(s2, s2, QUARRY_FMA(s1, s1, s0 * s0)) - 1.0f;
        if (f > 0.0f) {
            lo = t;  // 还在根的下方
        } else {
            hi = t;
        }
        const float fp = -2.0f * (s0 * s0 * ir0 + s1 * s1 * ir1 + s2 * s2 * ir2);
        float next = 0.5f * (lo + hi);  // 二分兜底
        if (fp < 0.0f) {
            const float newton = t - f / fp;
            if (newton > lo && newton < hi) next = newton;
        }
        t = next;
    }

    // x_i = gy_i · a_i / (t + a_i²)；e 是最近点指向查询点的向量。
    const float ir0 = 1.0f / (t + aa), ir1 = 1.0f / (t + bb), ir2 = 1.0f / (t + cc);
    const float e0 = y0 - gy0 * ir0 * a;
    const float e1 = y1 - gy1 * ir1 * b;
    const float e2 = y2 - gy2 * ir2 * c;
    return QUARRY_FMA(e2, e2, QUARRY_FMA(e1, e1, e0 * e0));
}

struct GaussianCloud {
    std::vector<Gaussian> gaussians;
    AABB bounds;

    void updateBounds() {
        bounds = AABB{};
        for (const Gaussian& g : gaussians) bounds.extend(gaussianBounds(g));
    }
};

struct GaussianCloudSpec {
    // sheet  —— 中心贴在 terrain 高度场上，模拟 3DGS 训练出的表面高斯；
    // volume —— 立方体内均匀，重叠最重，是剪枝的下限。
    std::string kind = "sheet";
    int count = 131072;
    uint64_t seed = 1;
    float scaleMin = 0.08f;   // 主轴半轴下界（场景边长 100）
    float scaleMax = 0.50f;   // 主轴半轴上界
    float aspectMin = 0.40f;  // 次轴 / 主轴 的比值下界，控制各向异性程度
};

GaussianCloud buildGaussianCloud(const GaussianCloudSpec& spec);
