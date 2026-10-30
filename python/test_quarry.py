"""对拍 Python 绑定：同一份 demo 场景，quarry.Scene 的结果必须和 numpy 暴力法一致。

用法：
    py -3.13 python/test_quarry.py

依赖 numpy；扩展模块在 build/python/ 下（build.bat 会自动构建）。
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build" / "python"))
import quarry  # noqa: E402


def closest_point_on_triangles(p, a, b, c):
    """Ericson, Real-Time Collision Detection 5.1.5，p: (3,), a/b/c: (M,3)。"""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)

    out = np.empty_like(a)
    # 顶点区域
    mA = (d1 <= 0) & (d2 <= 0)
    mB = (d3 >= 0) & (d4 <= d3)
    mC = (d6 >= 0) & (d5 <= d6)
    # 边区域
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    mAB = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    mAC = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    mBC = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)

    out[mA] = a[mA]
    out[mB] = b[mB]
    out[mC] = c[mC]
    t = np.zeros(len(a))
    t[mAB] = d1[mAB] / (d1[mAB] - d3[mAB])
    out[mAB] = a[mAB] + ab[mAB] * t[mAB, None]
    t[mAC] = d2[mAC] / (d2[mAC] - d6[mAC])
    out[mAC] = a[mAC] + ac[mAC] * t[mAC, None]
    t[mBC] = (d4[mBC] - d3[mBC]) / ((d4[mBC] - d3[mBC]) + (d5[mBC] - d6[mBC]))
    out[mBC] = b[mBC] + (c[mBC] - b[mBC]) * t[mBC, None]

    # 内部区域（剩下的全是）
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    rest = ~(mA | mB | mC | mAB | mAC | mBC)
    out[rest] = a[rest] + ab[rest] * v[rest, None] + ac[rest] * w[rest, None]
    return out


def brute_raycast(origins, dirs, tris, tmax):
    n = len(origins)
    hit = np.zeros(n, dtype=bool)
    t_out = np.full(n, np.inf)
    a, b, c = (tris[:, k].astype(np.float64) for k in range(3))
    e1 = b - a
    e2 = c - a
    for i in range(n):
        o = origins[i].astype(np.float64)
        d = dirs[i].astype(np.float64)
        pv = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, pv)
        ok = np.abs(det) >= 1e-7
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / det
            tv = o - a
            u = np.einsum("ij,ij->i", tv, pv) * inv
            q = np.cross(tv, e1)
            v = np.einsum("j,ij->i", d, q) * inv
            t = np.einsum("ij,ij->i", e2, q) * inv
        ok &= (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1)
        ok &= (t >= 1e-6) & (t <= tmax)
        if np.any(ok):
            tt = np.where(ok, t, np.inf)
            j = int(np.argmin(tt))
            hit[i] = True
            t_out[i] = tt[j]
    return hit, t_out


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name} failed {detail}")
    print(f"[PASS] {name}")


def quat_axes(rot):
    """把 (N,4) xyzw 四元数展开成旋转矩阵的三列（世界系 → 局部系），double。"""
    q = rot.astype(np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    u = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)], axis=1)
    v = np.stack([2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)], axis=1)
    t = np.stack([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)], axis=1)
    return u, v, t


def ellipsoid_dist2(queries, means, rot, scale, iters=80):
    """numpy 暴力椭球距离（double）：内部为 0，外部二分求 Eberly 方程的根。

    返回 (nq, n) 的平方距离矩阵。
    """
    u, v, w = quat_axes(rot)
    a, b, c = (scale[:, j].astype(np.float64) for j in range(3))
    d = queries[:, None, :].astype(np.float64) - means[None, :, :].astype(np.float64)
    y0, y1, y2 = (np.einsum("qki,ki->qk", d, col) for col in (u, v, w))

    norm = (y0 / a) ** 2 + (y1 / b) ** 2 + (y2 / c) ** 2
    out = np.zeros_like(norm)

    def F(t):
        return ((a * y0) / (t + a * a)) ** 2 + ((b * y1) / (t + b * b)) ** 2 + \
               ((c * y2) / (t + c * c)) ** 2 - 1.0

    lo = np.zeros_like(norm)
    hi = 2.0 * np.maximum(np.maximum(a, b), c) * np.sqrt(y0 * y0 + y1 * y1 + y2 * y2)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pos = F(mid) > 0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    t = 0.5 * (lo + hi)
    x0 = a * a * y0 / (t + a * a)
    x1 = b * b * y1 / (t + b * b)
    x2 = c * c * y2 / (t + c * c)
    outside = norm > 1.0
    return np.where(outside,
                    (y0 - x0) ** 2 + (y1 - x1) ** 2 + (y2 - x2) ** 2,
                    0.0)


def closest_brute(points, tris):
    """每个点到全部三角形的 (平方距离矩阵, 最小距离, argmin)。"""
    a64, b64, c64 = (tris[:, j].astype(np.float64) for j in range(3))
    D = np.empty((len(points), len(tris)))
    for i, p in enumerate(points):
        q = closest_point_on_triangles(p.astype(np.float64), a64, b64, c64)
        D[i] = np.einsum("ij,ij->i", q - p, q - p)
    return D, D.min(axis=1), D.argmin(axis=1).astype(np.int32)


def check_closest(scene, points, tris, label):
    """距离按容差、ID 按"并列允许"判定：地形上相邻三角形的距离经常只差 1 ulp。"""
    D, ref_d2, ref_tri = closest_brute(points, tris)
    dist2, tri = scene.closest(points)
    check(f"{label} distance", np.allclose(dist2, ref_d2, rtol=2e-4, atol=1e-6),
          f"max abs diff {np.max(np.abs(dist2 - ref_d2)):.3e}")
    tie_ok = D[np.arange(len(points)), tri] <= ref_d2 * (1 + 2e-4) + 1e-6
    check(f"{label} ids (ties allowed)", np.all(tie_ok),
          f"{np.sum(~tie_ok)} points pick a strictly farther triangle; "
          f"{np.sum(tri != ref_tri)} ids differ at ties")
    return dist2, tri, D


def main():
    np.set_printoptions(precision=6, suppress=True)
    device = quarry.describe()
    check("GPU available", bool(device), device)
    print(f"       {device}")

    vertices, indices = quarry.demo_mesh("terrain", grid=32)
    check("demo_mesh shapes", vertices.shape == (2048 * 3, 3) and indices.shape == (2048, 3),
          (vertices.shape, indices.shape))

    scene = quarry.Scene(vertices, indices)
    check("triangle_count", scene.triangle_count == 2048, scene.triangle_count)
    check("node_count", scene.node_count == 2 * 2048 - 1, scene.node_count)
    check("validate", scene.validate() == "", scene.validate())
    check("build_ms", scene.build_ms > 0.0, scene.build_ms)

    lo, hi = np.array(scene.bounds["min"]), np.array(scene.bounds["max"])
    check("bounds order", np.all(lo <= hi))

    rng = np.random.default_rng(7)
    points = (lo + (hi - lo) * rng.random((300, 3))).astype(np.float32)
    tris = vertices[indices].astype(np.float32)

    # 暴力参考：每个点到全部三角形的平方距离（double）。地形上点到相邻三角形的
    # 距离经常只差 1 ulp，ID 按"距离在容差内并列"判定，而不是要求整数完全相同。
    dist2, tri, D = check_closest(scene, points, tris, "closest")
    ref_d2 = D.min(axis=1)
    check("last_query_ms", scene.last_query_ms > 0.0, scene.last_query_ms)

    k = 4
    kd, kid = scene.knn(points, k)
    check("knn shapes", kd.shape == (len(points), k) and kid.shape == (len(points), k))
    check("knn first = closest", np.array_equal(kid[:, 0], tri),
          f"{(kid[:, 0] != tri).sum()} mismatches at rank 0")
    bench_d2 = np.sort(D, axis=1)[:, :k]
    check("knn dist2 sorted", np.all(kd >= kd[:, :1]) and np.all(np.diff(kd, axis=1) >= -1e-6))
    check("knn dist2 values", np.allclose(kd, bench_d2, rtol=2e-4, atol=1e-6),
          f"max abs diff {np.max(np.abs(kd - bench_d2)):.3e}")

    radius = 0.3 * float(np.max(hi - lo))
    offsets, rd2, rid = scene.radius(points, radius)
    check("radius offsets", offsets.shape == (len(points) + 1,) and offsets[0] == 0
          and offsets[-1] == len(rid))
    check("radius dist2 sorted per segment",
          all(np.all(np.diff(rd2[offsets[i]:offsets[i + 1]]) >= 0) for i in range(len(points))))
    check("radius bound", np.all(rd2 <= radius * radius + 1e-5))
    # 集合对拍（浮点边界可能让个别点差一条，容忍 1 条）
    bad = 0
    r2 = radius * radius
    for i in range(len(points)):
        want = set(np.nonzero(D[i] <= r2)[0].tolist())
        got = set(rid[offsets[i]:offsets[i + 1]].tolist())
        if want != got:
            bad += 1
    check("radius sets", bad == 0, f"{bad} points differ at the boundary")

    # 查询调度：Morton 序重排查询点只改执行顺序，三条查询的结果必须逐位不变
    m_dist2, m_tri = scene.closest(points, order="morton")
    check("closest morton bitwise",
          np.array_equal(m_tri, tri)
          and np.array_equal(m_dist2.view(np.uint32), dist2.view(np.uint32)),
          f"{(m_tri != tri).sum()} ids differ")
    m_kd, m_kid = scene.knn(points, k, order="morton")
    check("knn morton bitwise",
          np.array_equal(m_kid, kid)
          and np.array_equal(m_kd.view(np.uint32), kd.view(np.uint32)))
    m_off, m_rd2, m_rid = scene.radius(points, radius, order="morton")
    check("radius morton bitwise",
          np.array_equal(m_off, offsets) and np.array_equal(m_rid, rid)
          and np.array_equal(m_rd2.view(np.uint32), rd2.view(np.uint32)))

    # 建树器只改树形，不改查询语义：SAH 树的结果必须和 LBVH 逐位一致
    sah_scene = quarry.Scene(vertices, indices, builder="sah")
    check("sah validate", sah_scene.validate() == "", sah_scene.validate())
    check("sah node_count", sah_scene.node_count == 2 * 2048 - 1, sah_scene.node_count)
    s_dist2, s_tri = sah_scene.closest(points)
    check("sah closest bitwise",
          np.array_equal(s_tri, tri)
          and np.array_equal(s_dist2.view(np.uint32), dist2.view(np.uint32)),
          f"{(s_tri != tri).sum()} ids differ")
    s_kd, s_kid = sah_scene.knn(points, k)
    check("sah knn bitwise",
          np.array_equal(s_kid, kid)
          and np.array_equal(s_kd.view(np.uint32), kd.view(np.uint32)))
    s_off, s_rd2, s_rid = sah_scene.radius(points, radius)
    check("sah radius bitwise",
          np.array_equal(s_off, offsets) and np.array_equal(s_rid, rid)
          and np.array_equal(s_rd2.view(np.uint32), rd2.view(np.uint32)))
    check("sah build_stats has bin breakdown", "sah_bin_ms" in sah_scene.build_stats)

    origins = (lo + (hi - lo) * rng.random((200, 3))).astype(np.float32)
    origins[:, 1] = hi[1] + 1.0
    dirs = np.tile(np.array([0.0, -1.0, 0.0], dtype=np.float32), (200, 1))
    hit, t = scene.raycast(origins, dirs)
    ref_hit, ref_t = brute_raycast(origins, dirs, tris, np.inf)
    check("raycast hits", np.array_equal(hit, ref_hit), f"{(hit != ref_hit).sum()} mismatches")
    both = hit & ref_hit
    check("raycast t", np.allclose(t[both], ref_t[both], rtol=2e-4, atol=1e-5),
          f"max rel {np.max(np.abs(t[both] - ref_t[both]) / ref_t[both]):.3e}")
    check("raycast miss = inf", np.all(np.isinf(t[~hit])))

    face_scene = quarry.Scene(tris)
    check("triangles constructor", face_scene.triangle_count == 2048)

    # ---- 增量更新 ----
    lifted = vertices.copy()
    lifted[:, 1] += 0.1
    scene.update_vertices(lifted)
    check("update keeps triangle count", scene.triangle_count == 2048)
    check("update validate", scene.validate() == "", scene.validate())
    check("update refit_ms", scene.refit_ms > 0.0, scene.refit_ms)
    check("update uploaded everything", scene.refit_stats["updated"] == 2048,
          scene.refit_stats["updated"])
    new_lo, new_hi = np.array(scene.bounds["min"]), np.array(scene.bounds["max"])
    check("update bounds", np.allclose(new_hi, lifted.max(axis=0), atol=1e-5),
          (new_hi, lifted.max(axis=0)))
    check_closest(scene, points, lifted[indices].astype(np.float32), "update closest")

    # 只声明改动的三角形：未列出的在树里保持旧值，结果仍须与整份新数组一致
    partial = lifted[indices].astype(np.float32)
    dirty = np.array([0, 5, 7, 2047], dtype=np.int32)
    partial[dirty] += np.array([0.0, 3.0, 0.0], dtype=np.float32)
    scene.update_triangles(partial, dirty=dirty)
    check("partial update validate", scene.validate() == "", scene.validate())
    check("partial updated count", scene.refit_stats["updated"] == len(dirty),
          scene.refit_stats["updated"])
    check_closest(scene, points, partial, "partial update closest")

    # 第二次增量：与上一轮共享叶子和祖先，epoch 去重不能吃掉新一轮的标记
    again = partial.copy()
    dirty2 = np.array([0, 5, 7, 2047, 100], dtype=np.int32)
    again[dirty2] += np.array([0.0, 3.0, 0.0], dtype=np.float32)
    scene.update_triangles(again, dirty=dirty2)
    check("second partial update validate", scene.validate() == "", scene.validate())
    check_closest(scene, points, again, "second partial update closest")

    # ---- 点云：与 numpy 暴力法逐位对拍 ----
    npts = 20000
    nq = 200
    pts = (lo + (hi - lo) * rng.random((npts, 3))).astype(np.float32)
    cloud = quarry.PointCloud(pts)
    check("point_count", cloud.point_count == npts, cloud.point_count)
    check("point node_count", cloud.node_count == 2 * npts - 1, cloud.node_count)
    check("point validate", cloud.validate() == "", cloud.validate())
    check("point build_ms", cloud.build_ms > 0.0, cloud.build_ms)
    plo, phi = np.array(cloud.bounds["min"]), np.array(cloud.bounds["max"])
    check("point bounds", np.allclose(plo, pts.min(axis=0), atol=1e-6)
          and np.allclose(phi, pts.max(axis=0), atol=1e-6))

    q = (lo + (hi - lo) * rng.random((nq, 3))).astype(np.float32)
    diff = pts[None, :, :] - q[:, None, :]
    # float32 逐点平方距离：和 GPU 的 dot(d,d) 同顺序，可以要求位模式一致。
    d2 = (diff * diff).sum(axis=-1, dtype=np.float32)
    rows = np.arange(nq)

    ref_best = d2.argmin(axis=1)
    ref_d2 = d2[rows, ref_best]
    cdist2, cidx = cloud.closest(q)
    check("point closest dist2 bitwise",
          np.array_equal(cdist2.view(np.uint32), ref_d2.view(np.uint32)),
          f"{(cdist2.view(np.uint32) != ref_d2.view(np.uint32)).sum()} differ")
    check("point closest ids", np.array_equal(cidx, ref_best.astype(np.int32)),
          f"{(cidx != ref_best).sum()} mismatches")
    check("point last_query_ms", cloud.last_query_ms > 0.0, cloud.last_query_ms)

    ck = 4
    ckd, ckid = cloud.knn(q, ck)
    kd_ok = True
    for i in range(nq):
        # stable sort：距离并列时保持原始索引升序，正是 (dist2, index) 字典序。
        order = np.argsort(d2[i], kind="stable")[:ck]
        kd_ok &= np.array_equal(ckid[i], order.astype(np.int32))
        kd_ok &= np.array_equal(ckd[i].view(np.uint32), d2[i][order].view(np.uint32))
    check("point knn bitwise", kd_ok)

    cradius = 0.08 * float(np.linalg.norm(phi - plo))
    coff, crd2, crid = cloud.radius(q, cradius)
    check("point radius offsets", coff.shape == (nq + 1,) and coff[0] == 0
          and coff[-1] == len(crid))
    cr_ok = True
    for i in range(nq):
        want = np.nonzero(d2[i] <= cradius * cradius)[0]
        want = want[np.argsort(d2[i][want], kind="stable")]
        got = crid[coff[i]:coff[i + 1]]
        cr_ok &= np.array_equal(got, want.astype(np.int32))
        cr_ok &= np.array_equal(crd2[coff[i]:coff[i + 1]].view(np.uint32),
                                d2[i][want].view(np.uint32))
    check("point radius bitwise", cr_ok)

    m_cdist2, m_cidx = cloud.closest(q, order="morton")
    check("point closest morton bitwise",
          np.array_equal(m_cidx, cidx)
          and np.array_equal(m_cdist2.view(np.uint32), cdist2.view(np.uint32)))

    # 增量更新：先整体平移，再只改一个子集
    moved = pts.copy()
    moved[:, 1] += np.float32(0.1)
    cloud.update_points(moved)
    check("point update validate", cloud.validate() == "", cloud.validate())
    check("point update refit_ms", cloud.refit_ms > 0.0, cloud.refit_ms)
    check("point update uploaded everything", cloud.refit_stats["updated"] == npts,
          cloud.refit_stats["updated"])
    move_diff = moved[None, :, :] - q[:, None, :]
    d2_moved = (move_diff * move_diff).sum(axis=-1, dtype=np.float32)
    best_moved = d2_moved.argmin(axis=1)
    mdist2, midx = cloud.closest(q)
    check("point update closest bitwise",
          np.array_equal(midx, best_moved.astype(np.int32))
          and np.array_equal(mdist2.view(np.uint32),
                             d2_moved[rows, best_moved].view(np.uint32)))

    partial_pts = moved.copy()
    dirty_pts = np.array([0, 3, 17, npts - 1], dtype=np.int32)
    partial_pts[dirty_pts] += np.array([0.0, 0.5, 0.0], dtype=np.float32)
    cloud.update_points(partial_pts, dirty=dirty_pts)
    check("point partial validate", cloud.validate() == "", cloud.validate())
    check("point partial updated count", cloud.refit_stats["updated"] == len(dirty_pts),
          cloud.refit_stats["updated"])
    p_diff = partial_pts[None, :, :] - q[:, None, :]
    d2_partial = (p_diff * p_diff).sum(axis=-1, dtype=np.float32)
    best_partial = d2_partial.argmin(axis=1)
    pdist2, pidx = cloud.closest(q)
    check("point partial closest bitwise",
          np.array_equal(pidx, best_partial.astype(np.int32))
          and np.array_equal(pdist2.view(np.uint32),
                             d2_partial[rows, best_partial].view(np.uint32)))

    # ---- 高斯：椭球实体距离，与 numpy 暴力法对拍 ----
    ngs, ngq = 2000, 200
    gmeans, grot, gscale = quarry.demo_gaussians("sheet", ngs, seed=3)
    gcloud = quarry.GaussianCloud(gmeans, grot, gscale)
    check("gaussian_count", gcloud.gaussian_count == ngs, gcloud.gaussian_count)
    check("gaussian node_count", gcloud.node_count == 2 * ngs - 1, gcloud.node_count)
    check("gaussian validate", gcloud.validate() == "", gcloud.validate())
    check("gaussian build_ms", gcloud.build_ms > 0.0, gcloud.build_ms)

    gq = np.array(gcloud.bounds["min"]) + (np.array(gcloud.bounds["max"]) -
                                           np.array(gcloud.bounds["min"])) * rng.random((ngq, 3))
    gq = gq.astype(np.float32)
    gd2 = ellipsoid_dist2(gq, gmeans, grot, gscale)
    grows = np.arange(ngq)
    gref_best = gd2.argmin(axis=1)
    gref_d2 = gd2[grows, gref_best]

    gdist2, gidx = gcloud.closest(gq)
    check("gaussian closest dist2", np.allclose(gdist2, gref_d2, rtol=1e-4, atol=1e-6),
          f"max abs diff {np.max(np.abs(gdist2 - gref_d2)):.3e}")
    tie_ok = gd2[grows, gidx] <= gref_d2 * (1 + 1e-4) + 1e-6
    check("gaussian closest ids (ties allowed)", np.all(tie_ok),
          f"{np.sum(tie_ok == False)} points pick a strictly farther gaussian")
    check("gaussian last_query_ms", gcloud.last_query_ms > 0.0, gcloud.last_query_ms)

    m_gdist2, m_gidx = gcloud.closest(gq, order="morton")
    check("gaussian closest morton bitwise",
          np.array_equal(m_gidx, gidx)
          and np.array_equal(m_gdist2.view(np.uint32), gdist2.view(np.uint32)))

    gk = 4
    gkd, gkid = gcloud.knn(gq, gk)
    check("gaussian knn shapes", gkd.shape == (ngq, gk) and gkid.shape == (ngq, gk))
    bench_gd2 = np.sort(gd2, axis=1)[:, :gk]
    check("gaussian knn dist2", np.allclose(gkd, bench_gd2, rtol=1e-4, atol=1e-6),
          f"max abs diff {np.max(np.abs(gkd - bench_gd2)):.3e}")

    gradius = 0.03 * float(np.linalg.norm(np.array(gcloud.bounds["max"]) -
                                          np.array(gcloud.bounds["min"])))
    goff, grd2, grid_ = gcloud.radius(gq, gradius)
    check("gaussian radius offsets", goff.shape == (ngq + 1,) and goff[0] == 0
          and goff[-1] == len(grid_))
    gr2 = gradius * gradius
    band = 1e-3 * gr2
    gr_ok = True
    for i in range(ngq):
        got = set(grid_[goff[i]:goff[i + 1]].tolist())
        inside = set(np.nonzero(gd2[i] <= gr2 - band)[0].tolist())
        outside = set(np.nonzero(gd2[i] > gr2 + band)[0].tolist())
        gr_ok &= inside <= got and not (got & outside)
        gr_ok &= np.all(np.diff(grd2[goff[i]:goff[i + 1]]) >= 0)
    check("gaussian radius sets", gr_ok)

    # 增量更新：整体平移后再只改一个子集
    gmoved = gmeans.copy()
    gmoved[:, 1] += np.float32(0.1)
    gcloud.update(gmoved, grot, gscale)
    check("gaussian update validate", gcloud.validate() == "", gcloud.validate())
    check("gaussian update uploaded everything", gcloud.refit_stats["updated"] == ngs,
          gcloud.refit_stats["updated"])
    gd2_moved = ellipsoid_dist2(gq, gmoved, grot, gscale)
    best_moved = gd2_moved.argmin(axis=1)
    mdist2, midx = gcloud.closest(gq)
    check("gaussian update closest",
          np.allclose(mdist2, gd2_moved[grows, best_moved], rtol=1e-4, atol=1e-6)
          and np.all(gd2_moved[grows, midx] <= gd2_moved[grows, best_moved] * (1 + 1e-4) + 1e-6))

    gpartial = gmoved.copy()
    gdirty = np.array([0, 7, ngs - 1], dtype=np.int32)
    gpartial[gdirty] += np.array([0.0, 0.5, 0.0], dtype=np.float32)
    gcloud.update(gpartial, grot, gscale, dirty=gdirty)
    check("gaussian partial validate", gcloud.validate() == "", gcloud.validate())
    check("gaussian partial updated count", gcloud.refit_stats["updated"] == len(gdirty),
          gcloud.refit_stats["updated"])
    gd2_part = ellipsoid_dist2(gq, gpartial, grot, gscale)
    best_part = gd2_part.argmin(axis=1)
    pdist2, pidx = gcloud.closest(gq)
    check("gaussian partial closest",
          np.allclose(pdist2, gd2_part[grows, best_part], rtol=1e-4, atol=1e-6)
          and np.all(gd2_part[grows, pidx] <= gd2_part[grows, best_part] * (1 + 1e-4) + 1e-6))

    for bad_call in (lambda: quarry.Scene(np.zeros((5, 3), dtype=np.float32)),
                     lambda: quarry.PointCloud(np.zeros((1, 3), dtype=np.float32)),
                     lambda: scene.knn(points, 0),
                     lambda: scene.knn(points, 33),
                     lambda: cloud.knn(q, 33),
                     lambda: scene.raycast(points, dirs[:-1]),
                     lambda: scene.update_triangles(partial[:10]),
                     lambda: cloud.update_points(pts[:10]),
                     lambda: gcloud.update(gmoved[:10], grot, gscale),
                     lambda: quarry.GaussianCloud(gmeans, grot, gscale[:10]),
                     lambda: quarry.GaussianCloud(gmeans, grot, -gscale),
                     lambda: face_scene.update_vertices(lifted)):
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError("bad input did not raise ValueError")
    print("[PASS] bad inputs raise ValueError")

    print("\nPYTHON BINDINGS PASS")


if __name__ == "__main__":
    main()
