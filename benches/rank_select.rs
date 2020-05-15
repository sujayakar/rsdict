use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rsdict::RsDict;
use succinct::bit_vec::{BitVecPush, BitVector};
use succinct::rank::{JacobsonRank, Rank9, RankSupport};
use succinct::select::{BinSearchSelect, Select0Support, Select1Support};

const NUM_BITS: usize = 1_000_000;
const SEED: u64 = 88004802264174740;

fn random_bits(len: usize) -> BitVector<u64> {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut bv = BitVector::with_capacity(len as u64);
    for _ in 0..len {
        bv.push_bit(rng.gen());
    }
    bv
}

fn random_indices(count: usize, range: usize) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(SEED);
    (0..count).map(|_| rng.gen_range(0, range)).collect()
}


fn bench_one_rank<T, F, G>(c: &mut Criterion, name: &str, create: F, rank: G)
    where F: FnOnce(BitVector<u64>) -> T,
          G: Fn(&T, u64) -> u64
{
    let r = create(random_bits(NUM_BITS));
    let indices = random_indices(1000, NUM_BITS);
    c.bench_function(name, |b| {
        b.iter(|| {
            for &ix in &indices {
                rank(&r, black_box(ix as u64));
            }
        })
    });
}

fn bench_rank(c: &mut Criterion) {
    bench_one_rank(
        c,
        "rsdict::rank",
        |bits| {
            let mut rs_dict = RsDict::with_capacity(NUM_BITS);
            for b in bits.iter() {
                rs_dict.push(b);
            }
            rs_dict
        },
        |r, i| r.rank(i, true)
    );
    bench_one_rank(
        c,
        "jacobson::rank",
        JacobsonRank::new,
        |r, i| r.rank(i, true)
    );
    bench_one_rank(
        c,
        "rank9::rank",
        Rank9::new,
        |r, i| r.rank(i, true)
    );
}

fn bench_one_select<T, F, G, H>(c: &mut Criterion, name: &str, create: F, select0: G, select1: H)
where
    F: Fn(BitVector<u64>) -> T,
    G: Fn(&T, u64) -> Option<u64>,
    H: Fn(&T, u64) -> Option<u64>
{
    let bits = random_bits(NUM_BITS);
    let num_set = bits.iter().filter(|&b| b).count();
    let r = create(bits);
    let indices = random_indices(1000, num_set);

    c.bench_function(&format!("{}::select0", name), |b| {
        b.iter(|| {
            for &ix in &indices {
                select0(&r, black_box(ix as u64));
            }
        })
    });
    c.bench_function(&format!("{}::select1", name), |b| {
        b.iter(|| {
            for &ix in &indices {
                select1(&r, black_box(ix as u64));
            }
        })
    });
}

fn bench_select(c: &mut Criterion) {
    bench_one_select(
        c,
        "rsdict",
        |bits| {
            let mut rs_dict = RsDict::with_capacity(NUM_BITS);
            for b in bits.iter() {
                rs_dict.push(b);
            }
            rs_dict
        },
        |r, i| r.select0(i),
        |r, i| r.select1(i),
    );
    bench_one_select(
        c,
        "rank9::binsearch",
        |b| BinSearchSelect::new(Rank9::new(b)),
        |r, i| r.select0(i),
        |r, i| r.select1(i),
    );
}

criterion_group!(benches, bench_rank, bench_select);
criterion_main!(benches);
