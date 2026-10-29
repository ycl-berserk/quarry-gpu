#include "core/point_cloud.h"

#include "core/rng.h"
#include "core/scene.h"

PointCloud buildPointCloud(const PointCloudSpec& spec) {
    PointCloud cloud;
    if (spec.count <= 0) return cloud;
    cloud.pts.reserve(static_cast<size_t>(spec.count));

    Pcg32 rng(spec.seed, 4);
    const float half = 50.0f;

    if (spec.kind == "volume") {
        for (int i = 0; i < spec.count; ++i) {
            cloud.pts.push_back({rng.uniform(-half, half), rng.uniform(-half, half),
                                 rng.uniform(-half, half)});
        }
    } else {
        // 扫面点云：x/z 均匀铺开，高度取地形面 + 2 cm 抖动，避免所有点共面。
        for (int i = 0; i < spec.count; ++i) {
            const float x = rng.uniform(-half, half);
            const float z = rng.uniform(-half, half);
            const float y = terrainHeight(x, z) + rng.uniform(-0.02f, 0.02f);
            cloud.pts.push_back({x, y, z});
        }
    }

    cloud.updateBounds();
    return cloud;
}
