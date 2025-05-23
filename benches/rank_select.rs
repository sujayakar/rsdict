use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::prelude::*;
use rsdict::RsDict;

fn prepare_rsdict(_density: f32) -> RsDict {
    let mut rng = StdRng::seed_from_u64(0xFEE5EED);
    let blocks = std::iter::repeat_with(|| rng.gen()).take(100_000);
    let dict = RsDict::from_blocks(blocks);
    return dict;
}

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
fn prepare_avx512_rsdict() -> Option<rsdict::avx512_rsdict::Avx512RsDict> {
    if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512vpopcntdq") {
        return None;
    }
    
    let mut dict = rsdict::avx512_rsdict::Avx512RsDict::with_capacity(100_000 * 64);
    let mut rng = StdRng::seed_from_u64(0xFEE5EED);
    
    for _ in 0..100_000 {
        let block = rng.gen::<u64>();
        for i in 0..64 {
            dict.push((block >> i) & 1 == 1);
        }
    }
    dict.finalize();
    
    Some(dict)
}

#[cfg(not(all(target_arch = "x86_64", feature = "simd")))]
fn prepare_avx512_rsdict() -> Option<()> {
    None
}

fn rank_select_benchmark(c: &mut Criterion) {
    // Standard benchmarks
    c.bench_function("rsdict::rank", |b| {
        let dict = prepare_rsdict(0.5);
        let mut i = 0u64;
        b.iter(|| {
            i = (i + 1111) % dict.len() as u64;
            dict.rank(black_box(i), true)
        });
    });

    c.bench_function("rsdict::select", |b| {
        let dict = prepare_rsdict(0.5);
        let max_rank = dict.rank(dict.len() as u64 - 1, true);
        let mut i = 0u64;
        b.iter(|| {
            i = (i + 97) % max_rank.max(1);
            dict.select(black_box(i), true).unwrap()
        });
    });

    c.bench_function("rank_acceleration::rank", |b| {
        let dict = prepare_rsdict(0.5);
        let mut i = 0u64;
        b.iter(|| {
            i = (i + 1111) % dict.len() as u64;
            dict.rank(black_box(i), true)
        });
    });

    c.bench_function("rank_acceleration::select", |b| {
        let dict = prepare_rsdict(0.5);
        let max_rank = dict.rank(dict.len() as u64 - 1, true);
        let mut i = 0u64;
        b.iter(|| {
            i = (i + 97) % max_rank.max(1);
            dict.select(black_box(i), true).unwrap()
        });
    });
    
    // AVX-512 benchmarks if available
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if let Some(dict) = prepare_avx512_rsdict() {
            c.bench_function("avx512::rank", |b| {
                let mut i = 0usize;
                b.iter(|| {
                    i = (i + 1111) % dict.len();
                    dict.rank(black_box(i), true)
                });
            });

            c.bench_function("avx512::select", |b| {
                let max_rank = dict.rank(dict.len() - 1, true);
                let mut i = 0u64;
                b.iter(|| {
                    i = (i + 97) % max_rank.max(1);
                    dict.select(black_box(i), true).unwrap()
                });
            });
        } else {
            println!("AVX-512 not available on this CPU, skipping AVX-512 benchmarks");
        }
    }
}

criterion_group!(benches, rank_select_benchmark);
criterion_main!(benches);
