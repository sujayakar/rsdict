# RsDict: Fast rank/select over bitmaps
Rank and select are two useful operations on bitmaps for building more sophisticated data
structures.  First, the *rank* at a given index `i` counts the number of set bits left of `i`. For
example, a sparse array can be represented as a dense array of the values present and a bitmap
indicating which indices are present. Then, rank provides a function from an index into the sparse
array to an index into the dense one.

*Select* is the inverse of rank, where `select(B, k)` returns the index of the `k`th set bit. To make
the two inverses, we use zero-indexing for select (so `select(B, 0)` returns the index of the first
bit set in `B`) and rank only counts indices strictly to the left of `i`. From our previous example,
`select` allows going from an index in the dense array to the original sparse array.

This data structure implements these two operations efficiently on top of an append-only bitmap. It's
an implementation of [Navarro and Providel, "Fast, Small, Simple Rank/Select On
Bitmaps"](https://users.dcc.uchile.cl/~gnavarro/ps/sea12.1.pdf), with heavy inspiration from a [Go
implementation](https://github.com/hillbig/rsdic). The underlying bitmap is stored in compressed
form, so long runs of zeros and ones do not take up much space. The indices for rank and select add
about 28% overhead over the uncompressed bitmap.

For more examples on how to use rank and select to build succinct datastructures, see Navarro's book
on [Compact Data
Structures](https://www.cambridge.org/core/books/compact-data-structures/68A5983E6F1176181291E235D0B7EB44)
for an overview.

## Implementation notes
This library is mostly a port of the Go implementation with a few additional optimizations.

### SSE acceleration for rank
With the nightly-only `simd` feature and a CPU with SSSE3 support, the final step of rank is computed
in a few steps without any loops. Turning this feature on improves the `rsdict::rank` benchmark by
about 40% on my computer. See `rank_acceleration.rs` for more details.

### Optimized routines for rank and select within a `u64`
With a CPU that supports `popcnt`, computing rank within a small block of 64 bits will use this
instruction to efficiently count the number of bits set.  Select uses an adapted version of an [an
algorithm from Daniel Lemire's
blog](https://lemire.me/blog/2018/02/21/iterating-over-set-bits-quickly/) that uses `tzcnt` to
quickly skip over runs of trailing zeros.

### Compact binomial coefficient lookup table
Encoding and decoding blocks of the compressed bitmap requires computing the binomial coefficient
`B(n, k)` where `0 <= k <= n <= 64`. Computing this on-the-fly is too expensive, so we store a
precomputed lookup table of the coefficients. However, we exploit the symmetry of `B` in `k` to
store only half the values. See `build.rs` for more details.

## Performance
Here's some results from running the benchmark on my 2018 MacBook Pro with `-C target-cpu=native`.
```
rsdict::rank            time:   [10.330 us 10.488 us 10.678 us]
Found 4 outliers among 100 measurements (4.00%)
  4 (4.00%) high mild

jacobson::rank          time:   [17.958 us 18.335 us 18.740 us]
Found 6 outliers among 100 measurements (6.00%)
  1 (1.00%) high mild
  5 (5.00%) high severe

rank9::rank             time:   [6.8907 us 7.0768 us 7.2940 us]
Found 1 outliers among 100 measurements (1.00%)
  1 (1.00%) high severe

rsdict::select0         time:   [37.124 us 37.505 us 37.991 us]
Found 3 outliers among 100 measurements (3.00%)
  3 (3.00%) high severe

rsdict::select1         time:   [29.782 us 29.918 us 30.067 us]
Found 7 outliers among 100 measurements (7.00%)
  5 (5.00%) high mild
  2 (2.00%) high severe

rank9::binsearch::select0
                        time:   [229.64 us 231.54 us 233.87 us]
Found 5 outliers among 100 measurements (5.00%)
  2 (2.00%) high mild
  3 (3.00%) high severe

rank9::binsearch::select1
                        time:   [253.69 us 255.84 us 258.19 us]
Found 9 outliers among 100 measurements (9.00%)
  4 (4.00%) high mild
  5 (5.00%) high severe
```
So for rank queries, this implementation is faster than `succinct-rs`'s Jacobson and slightly slower
than its Rank9.  However for select queries, it's *much* faster than doing binary search over these
rank structures, so consider using this library if select is an important operation for your algorithm.

## Testing
We use QuickCheck for testing data structure invariants.  In addition, there's basic AFL fuzz integration
to find interesting test cases using program coverage.  Install [cargo-afl](https://github.com/rust-fuzz/afl.rs)
and run the `rsdict_fuzz` binary with the `fuzz` feature set.
```
$ cargo install afl
$ cargo afl build --release --test rsdict_fuzz --features fuzz

# Create some starting bitsets within target/fuzz/in and create an empty directory target/fuzz/out.
$ cargo afl fuzz -i target/fuzz/in -o target/fuzz/out target/release/rsdict_fuzz
```
