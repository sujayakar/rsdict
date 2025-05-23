use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::prelude::*;
use rsdict::RsDict;

fn prepare_rsdict(_density: f32) -> RsDict {
    let mut rng = StdRng::seed_from_u64(0xFEE5EED);
    let blocks = std::iter::repeat_with(|| rng.gen()).take(100_000);
    let dict = RsDict::from_blocks(blocks);
    return dict;
}

fn rank_bench(c: &mut Criterion) {
    let dict = prepare_rsdict(0.5);
    c.bench_function("rsdict::rank", |b| {
        let mut i = 0;
        b.iter(|| {
            let r = dict.rank(black_box(i), true);
            i = (i + 13337) % (100_000 * 64);
            black_box(r);
        })
    });
}

fn select_bench(c: &mut Criterion) {
    let dict = prepare_rsdict(0.5);
    c.bench_function("rsdict::select", |b| {
        let mut i = 0;
        b.iter(|| {
            let r = dict.select(black_box(i), true);
            i = (i + 211) % 3_200_000;
            black_box(r);
        })
    });
}

fn simple_bench(c: &mut Criterion) {
    let dict = prepare_rsdict(0.5);
    c.bench_function("rsdict::getbit", |b| {
        let mut i = 0;
        b.iter(|| {
            let r = dict.get_bit(black_box(i));
            i = (i + 13337) % (100_000 * 64);
            black_box(r);
        })
    });
}

criterion_group!(benches, rank_bench, select_bench, simple_bench);
criterion_main!(benches);
