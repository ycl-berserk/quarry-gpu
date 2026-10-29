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

    for bad_call in (lambda: quarry.Scene(np.zeros((5, 3), dtype=np.float32)),
                     lambda: quarry.PointCloud(np.zeros((1, 3), dtype=np.float32)),
                     lambda: scene.knn(points, 0),
                     lambda: scene.knn(points, 33),
                     lambda: cloud.knn(q, 33),
                     lambda: scene.raycast(points, dirs[:-1]),
                     lambda: scene.update_triangles(partial[:10]),
                     lambda: cloud.update_points(pts[:10]),
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
