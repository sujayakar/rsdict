# Fast combined rank/select data structure for bitmaps
This data structure implements [Navarro and Providel, "Fast, Small, Simple
Rank/Select On Bitmaps,"](https://users.dcc.uchile.cl/~gnavarro/ps/sea12.1.pdf),
with heavy inspiration from a [Go implementation](https://github.com/hillbig/rsdic).

Most data structures accelerate *rank* and *select* queries separately, and the
main idea of the paper is to design structures that can be used for both
simultaneously.
