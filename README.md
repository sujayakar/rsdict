# RsDict: Fast rank/select over bitmaps
This data structure implements [Navarro and Providel, "Fast, Small, Simple
Rank/Select On Bitmaps"](https://users.dcc.uchile.cl/~gnavarro/ps/sea12.1.pdf),
with heavy inspiration from a [Go implementation](https://github.com/hillbig/rsdic).

This data structure stores an append-only bitmap and provides two queries: rank and select.  First,
for some bitmap `B`, `rank(i)` counts the number of bits set to the left of `i`.  Then, `select(i)`
returns the index of the `i`th set bit, providing an inverse to `rank`.  These operations are useful
for building many different succinct data structures.  See Navarro's book on [Compact Data Structures](https://www.cambridge.org/core/books/compact-data-structures/68A5983E6F1176181291E235D0B7EB44) for an overview.

This library ports the Go implementation to Rust and adds a few optimizations.  First, the final phase
of computing a rank involves scanning over the compressed bitmap, decompressing it one block at a time
and keeping a running total of set bits.  For CPUs with SSSE3 support, this library performs this final
step without looping by using vectorized instructions.  Second, we use optimized routes for computing
`rank` and `select` within a single `u64`.  Rank uses `popcnt`, if available, and select implements
[this algorithm](https://lemire.me/blog/2018/02/21/iterating-over-set-bits-quickly/) to quickly skip over
unset bits.
