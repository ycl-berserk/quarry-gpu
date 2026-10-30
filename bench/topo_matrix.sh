#!/usr/bin/env bash
# 拓扑更新基准矩阵：规模 × 建树器 × 编辑分布 × 比例 × 编辑种类 (+宽节点)。
# 输出按配置分段，配合 out/topo_matrix.txt 存档。
set -u
cd "$(dirname "$0")/.."
mkdir -p out
OUT=out/topo_matrix.txt
: > "$OUT"

run() {
    echo "=== $*" >> "$OUT"
    ./build/gpu_query_bench.exe "$@" 2>&1 | grep -E "^scene|^gpu build|^topo" >> "$OUT"
}

for grid in 256 1024; do
    for b in "" "--sah"; do
        for k in cluster scatter; do
            run --grid "$grid" --no-cpu --topo --topo-kind "$k" \
                --topo-frac 0.01 --topo-mix mixed --topo-rounds 5 $b
        done
    done
done

for mix in insert delete move mixed; do
    run --grid 256 --no-cpu --topo --topo-kind cluster --topo-frac 0.01 \
        --topo-mix "$mix" --topo-rounds 5
done

for f in 0.001 0.005 0.05; do
    run --grid 1024 --no-cpu --topo --topo-kind cluster --topo-frac "$f" \
        --topo-mix mixed --topo-rounds 5
done

run --grid 1024 --no-cpu --topo --topo-kind cluster --topo-frac 0.01 --topo-mix mixed \
    --topo-rounds 5 --wide
run --grid 256 --no-cpu --topo --topo-kind cluster --topo-frac 0.01 --topo-mix mixed \
    --topo-rounds 5 --wide
echo "matrix done" >> "$OUT"
