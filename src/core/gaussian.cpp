#include "core/gaussian.h"

#include <cmath>

#include "core/rng.h"
#include "core/scene.h"

namespace {

// Shoemake：三个均匀数给出均匀分布的随机旋转（单位四元数）。
Quat uniformQuat(Pcg32& rng) {
    const float u1 = rng.uniform01();
    const float u2 = rng.uniform01();
    const float u3 = rng.uniform01();
    const float twoPi = 6.28318530717958647692f;
    const float s1 = sqrtf(1.0f - u1);
    const float s2 = sqrtf(u1);
    Quat q;
    q.x = s1 * sinf(twoPi * u2);
    q.y = s1 * cosf(twoPi * u2);
    q.z = s2 * sinf(twoPi * u3);
    q.w = s2 * cosf(twoPi * u3);
    return q;
}

}  // namespace

GaussianCloud buildGaussianCloud(const GaussianCloudSpec& spec) {
    GaussianCloud cloud;
    if (spec.count <= 0) return cloud;
    cloud.gaussians.reserve(static_cast<size_t>(spec.count));

    Pcg32 rng(spec.seed, 5);
    const float half = 50.0f;

    for (int i = 0; i < spec.count; ++i) {
        Gaussian g;
        if (spec.kind == "volume") {
            g.mean = {rng.uniform(-half, half), rng.uniform(-half, half),
                      rng.uniform(-half, half)};
        } else {
            const float x = rng.uniform(-half, half);
            const float z = rng.uniform(-half, half);
            g.mean = {x, terrainHeight(x, z), z};
        }
        g.rot = uniformQuat(rng);

        const float major = rng.uniform(spec.scaleMin, spec.scaleMax);
        g.scale = {major, major * rng.uniform(spec.aspectMin, 1.0f),
                   major * rng.uniform(spec.aspectMin, 1.0f)};
        cloud.gaussians.push_back(g);
    }

    cloud.updateBounds();
    return cloud;
}
