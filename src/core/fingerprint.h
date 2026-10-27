#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/geometry.h"

// 每条射线的结果压成一个 64 位值（1 位命中标志 + 命中距离的 32 位浮点表示），
// 再按射线顺序做 FNV-1a。这样：
//   - 同样的射线顺序 → 同样的 hash，不受求和顺序影响；
//   - GPU 版本可以分块并发算，最后按块顺序合并，仍然可复现。
// sumT 是浮点累加，只适合和同一实现多次运行对比；跨引擎用 hits + hash 比。
struct RayStats {
    uint64_t rays = 0;
    uint64_t hits = 0;
    uint64_t hash = 0xcbf29ce484222325ull;  // FNV-1a offset basis
    double sumT = 0.0;

    void add(const RayHit& h) {
        ++rays;
        uint32_t bits = 0;
        if (h.hit) {
            ++hits;
            sumT += static_cast<double>(h.t);
            std::memcpy(&bits, &h.t, sizeof(bits));
        }
        const uint64_t v = (static_cast<uint64_t>(h.hit ? 1u : 0u) << 32) | bits;
        hash ^= v;
        hash *= 0x100000001b3ull;  // FNV-1a prime
    }

    std::string summary() const {
        char buf[192];
        std::snprintf(buf, sizeof(buf), "rays=%llu hits=%llu sum_t=%.6f hash=0x%016llx",
                      static_cast<unsigned long long>(rays),
                      static_cast<unsigned long long>(hits), sumT,
                      static_cast<unsigned long long>(hash));
        return buf;
    }
};

// 最近点 / kNN / 半径查询的指纹。单条查询折叠 (命中, dist2, 场景索引)，
// 列表查询先折叠长度再逐条折叠——长度进 hash，空结果和少一个结果不会撞车。
// 距离并列时 closerHit 保证顺序唯一，所以这个 hash 对 CPU 和 GPU 是同一把尺子。
struct QueryStats {
    uint64_t queries = 0;
    uint64_t results = 0;
    uint64_t hash = 0xcbf29ce484222325ull;  // FNV-1a offset basis
    double sumDist2 = 0.0;

    void add(const ClosestHit& h) {
        ++queries;
        results += h.found ? 1u : 0u;
        mix((static_cast<uint64_t>(h.found ? 1u : 0u) << 32) | dist2Bits(h));
        mix(h.found ? static_cast<uint64_t>(h.triangle) + 1u : 0u);
        if (h.found) sumDist2 += static_cast<double>(h.dist2);
    }

    void add(const std::vector<ClosestHit>& list) {
        ++queries;
        results += list.size();
        mix(static_cast<uint64_t>(list.size()));
        for (const ClosestHit& h : list) {
            mix(dist2Bits(h));
            mix(static_cast<uint64_t>(h.triangle) + 1u);
            sumDist2 += static_cast<double>(h.dist2);
        }
    }

    std::string summary() const {
        char buf[224];
        std::snprintf(buf, sizeof(buf),
                      "queries=%llu results=%llu sum_dist2=%.6f hash=0x%016llx",
                      static_cast<unsigned long long>(queries),
                      static_cast<unsigned long long>(results), sumDist2,
                      static_cast<unsigned long long>(hash));
        return buf;
    }

private:
    static uint64_t dist2Bits(const ClosestHit& h) {
        uint32_t bits = 0;
        if (h.found) std::memcpy(&bits, &h.dist2, sizeof(bits));
        return bits;
    }

    void mix(uint64_t v) {
        hash ^= v;
        hash *= 0x100000001b3ull;  // FNV-1a prime
    }
};
