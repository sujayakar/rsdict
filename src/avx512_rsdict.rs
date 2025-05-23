//! AVX-512 optimized rank/select dictionary
//! 
//! This implementation uses 512-bit blocks and AVX-512 instructions
//! for much better performance on modern CPUs.

#![cfg(all(target_arch = "x86_64", feature = "simd"))]

use std::arch::x86_64::*;

/// Block size in bits - matches AVX-512 register width
const BLOCK_SIZE: usize = 512;
/// Block size in bytes
const BLOCK_BYTES: usize = BLOCK_SIZE / 8;
/// Block size in u64s
const BLOCK_U64S: usize = BLOCK_SIZE / 64;

/// Number of small blocks per large block
const SMALL_BLOCKS_PER_LARGE: usize = 64;
/// Large block size in bits
const LARGE_BLOCK_SIZE: usize = BLOCK_SIZE * SMALL_BLOCKS_PER_LARGE;

/// AVX-512 optimized rank/select dictionary
#[derive(Clone, Debug)]
pub struct Avx512RsDict {
    /// Raw bit data, aligned to 64 bytes for AVX-512
    blocks: Vec<Block>,
    
    /// Rank (popcount) for each small block
    /// Stored contiguously for SIMD scanning
    small_block_ranks: Vec<u16>,
    
    /// Cumulative rank at each large block boundary
    large_block_ranks: Vec<u64>,
    
    /// Total number of bits
    len: usize,
    
    /// Total number of ones
    num_ones: usize,
}

/// A 512-bit block aligned for AVX-512
#[repr(align(64))]
#[derive(Clone, Debug)]
struct Block {
    data: [u64; BLOCK_U64S],
}

impl Block {
    fn new() -> Self {
        Block { data: [0; BLOCK_U64S] }
    }
    
    /// Count ones using AVX-512 popcnt
    #[inline]
    #[target_feature(enable = "avx512f", enable = "avx512vpopcntdq")]
    unsafe fn popcount(&self) -> u32 {
        let v = _mm512_loadu_si512(self.data.as_ptr() as *const __m512i);
        let counts = _mm512_popcnt_epi64(v);
        _mm512_reduce_add_epi64(counts) as u32
    }
    
    /// Get bit at position (0..511)
    #[inline]
    fn get_bit(&self, pos: usize) -> bool {
        debug_assert!(pos < BLOCK_SIZE);
        let word_idx = pos / 64;
        let bit_idx = pos % 64;
        (self.data[word_idx] >> bit_idx) & 1 == 1
    }
    
    /// Set bit at position (0..511)
    #[inline]
    fn set_bit(&mut self, pos: usize, bit: bool) {
        debug_assert!(pos < BLOCK_SIZE);
        let word_idx = pos / 64;
        let bit_idx = pos % 64;
        if bit {
            self.data[word_idx] |= 1u64 << bit_idx;
        } else {
            self.data[word_idx] &= !(1u64 << bit_idx);
        }
    }
}

impl Avx512RsDict {
    /// Create a new empty AVX-512 optimized dictionary
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            small_block_ranks: Vec::new(),
            large_block_ranks: Vec::new(),
            len: 0,
            num_ones: 0,
        }
    }
    
    /// Create with expected capacity
    pub fn with_capacity(bits: usize) -> Self {
        let n_blocks = (bits + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let n_large = (n_blocks + SMALL_BLOCKS_PER_LARGE - 1) / SMALL_BLOCKS_PER_LARGE;
        
        Self {
            blocks: Vec::with_capacity(n_blocks),
            small_block_ranks: Vec::with_capacity(n_blocks),
            large_block_ranks: Vec::with_capacity(n_large),
            len: 0,
            num_ones: 0,
        }
    }
    
    /// Push a bit to the end
    pub fn push(&mut self, bit: bool) {
        let block_idx = self.len / BLOCK_SIZE;
        let bit_idx = self.len % BLOCK_SIZE;
        
        // Need a new block?
        if bit_idx == 0 {
            self.blocks.push(Block::new());
            
            // Update large block ranks
            if block_idx % SMALL_BLOCKS_PER_LARGE == 0 {
                self.large_block_ranks.push(self.num_ones as u64);
            }
        }
        
        // Set the bit
        if let Some(block) = self.blocks.get_mut(block_idx) {
            block.set_bit(bit_idx, bit);
        }
        
        if bit {
            self.num_ones += 1;
        }
        self.len += 1;
        
        // If we completed a block, compute its rank
        if self.len % BLOCK_SIZE == 0 {
            unsafe {
                let rank = self.blocks[block_idx].popcount() as u16;
                self.small_block_ranks.push(rank);
            }
        }
    }
    
    /// Finalize the structure after all pushes
    pub fn finalize(&mut self) {
        // Handle the last partial block if any
        let last_block_bits = self.len % BLOCK_SIZE;
        if last_block_bits > 0 {
            let block_idx = self.len / BLOCK_SIZE;
            unsafe {
                let rank = self.blocks[block_idx].popcount() as u16;
                self.small_block_ranks.push(rank);
            }
        }
    }
    
    /// Get the bit at position
    pub fn get_bit(&self, pos: usize) -> bool {
        assert!(pos < self.len, "Position {} out of bounds", pos);
        let block_idx = pos / BLOCK_SIZE;
        let bit_idx = pos % BLOCK_SIZE;
        self.blocks[block_idx].get_bit(bit_idx)
    }
    
    /// Count the number of `bit` values before position `pos`
    pub fn rank(&self, pos: usize, bit: bool) -> u64 {
        assert!(pos <= self.len, "Position {} out of bounds", pos);
        
        if pos == 0 {
            return 0;
        }
        
        let block_idx = (pos - 1) / BLOCK_SIZE;
        let bit_idx = (pos - 1) % BLOCK_SIZE + 1;
        
        // Start with large block rank
        let large_idx = block_idx / SMALL_BLOCKS_PER_LARGE;
        let mut rank = if large_idx > 0 {
            self.large_block_ranks[large_idx]
        } else {
            0
        };
        
        // Add small block ranks using SIMD
        let start_block = large_idx * SMALL_BLOCKS_PER_LARGE;
        if block_idx > start_block {
            rank += unsafe { self.sum_ranks_simd(start_block, block_idx) };
        }
        
        // Add rank within the final block
        if bit_idx > 0 && bit_idx < BLOCK_SIZE {
            rank += unsafe { self.rank_within_block(block_idx, bit_idx) };
        } else if bit_idx == BLOCK_SIZE && block_idx < self.small_block_ranks.len() {
            rank += self.small_block_ranks[block_idx] as u64;
        }
        
        if bit {
            rank
        } else {
            pos as u64 - rank
        }
    }
    
    /// Sum ranks using AVX-512
    #[inline]
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    unsafe fn sum_ranks_simd(&self, start: usize, end: usize) -> u64 {
        let ranks = &self.small_block_ranks[start..end];
        let mut sum = 0u64;
        
        // Process 32 ranks at a time (512 bits / 16 bits per rank)
        let chunks = ranks.chunks_exact(32);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            // Load 32 u16 values
            let v = _mm512_loadu_si512(chunk.as_ptr() as *const __m512i);
            
            // Convert lower 8 u16 to u64 and sum
            let low_256 = _mm512_castsi512_si256(v);
            let low_128 = _mm256_castsi256_si128(low_256);
            let low = _mm512_cvtepu16_epi64(low_128);
            
            // Convert next 8 u16 to u64 and sum  
            let mid_low = _mm256_extracti128_si256(low_256, 1);
            let mid1 = _mm512_cvtepu16_epi64(mid_low);
            
            // Convert upper 16 u16 values
            let high_256 = _mm512_extracti32x8_epi32(v, 1);
            let high_128 = _mm256_castsi256_si128(high_256);
            let mid2 = _mm512_cvtepu16_epi64(high_128);
            
            let high_high = _mm256_extracti128_si256(high_256, 1);
            let high = _mm512_cvtepu16_epi64(high_high);
            
            // Sum all
            sum += _mm512_reduce_add_epi64(low) as u64;
            sum += _mm512_reduce_add_epi64(mid1) as u64;
            sum += _mm512_reduce_add_epi64(mid2) as u64;
            sum += _mm512_reduce_add_epi64(high) as u64;
        }
        
        // Handle remainder
        for &rank in remainder {
            sum += rank as u64;
        }
        
        sum
    }
    
    /// Rank within a single block up to bit_idx
    #[inline]
    #[target_feature(enable = "avx512f", enable = "avx512vpopcntdq", enable = "avx512bw")]
    unsafe fn rank_within_block(&self, block_idx: usize, bit_idx: usize) -> u64 {
        let block = &self.blocks[block_idx];
        
        // For partial words, mask and count
        let full_words = bit_idx / 64;
        let mut count = 0u64;
        
        // Count full words
        for i in 0..full_words {
            count += block.data[i].count_ones() as u64;
        }
        
        // Count partial word
        let remainder_bits = bit_idx % 64;
        if remainder_bits > 0 {
            let mask = (1u64 << remainder_bits) - 1;
            count += (block.data[full_words] & mask).count_ones() as u64;
        }
        
        count
    }
    
    /// Find the position of the `rank`-th occurrence of `bit`
    pub fn select(&self, rank: u64, bit: bool) -> Option<usize> {
        let total = if bit { self.num_ones } else { self.len - self.num_ones };
        if rank >= total as u64 {
            return None;
        }
        
        // Binary search on large blocks
        let large_idx = self.find_large_block(rank, bit);
        let large_start = if large_idx > 0 {
            self.large_block_ranks[large_idx]
        } else {
            0
        };
        
        let mut remaining = if bit {
            rank - large_start
        } else {
            let large_pos = large_idx * SMALL_BLOCKS_PER_LARGE * BLOCK_SIZE;
            rank - (large_pos as u64 - large_start)
        };
        
        // Linear search within large block
        let start_block = large_idx * SMALL_BLOCKS_PER_LARGE;
        let end_block = ((large_idx + 1) * SMALL_BLOCKS_PER_LARGE).min(self.blocks.len());
        
        for block_idx in start_block..end_block {
            let block_rank = if bit {
                self.small_block_ranks[block_idx] as u64
            } else {
                BLOCK_SIZE as u64 - self.small_block_ranks[block_idx] as u64
            };
            
            if remaining < block_rank {
                // Found the block, search within it
                return Some(block_idx * BLOCK_SIZE + self.select_within_block(block_idx, remaining, bit));
            }
            
            remaining -= block_rank;
        }
        
        None
    }
    
    /// Binary search for large block containing rank
    fn find_large_block(&self, rank: u64, bit: bool) -> usize {
        if self.large_block_ranks.is_empty() {
            return 0;
        }
        
        let mut left = 0;
        let mut right = self.large_block_ranks.len();
        
        while left < right {
            let mid = left + (right - left) / 2;
            let block_rank = if bit {
                self.large_block_ranks[mid]
            } else {
                (mid * SMALL_BLOCKS_PER_LARGE * BLOCK_SIZE) as u64 - self.large_block_ranks[mid]
            };
            
            if block_rank <= rank {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        left.saturating_sub(1)
    }
    
    /// Find the position within a block
    fn select_within_block(&self, block_idx: usize, rank: u64, bit: bool) -> usize {
        let block = &self.blocks[block_idx];
        let mut count = 0u64;
        
        for word_idx in 0..BLOCK_U64S {
            let word = if bit {
                block.data[word_idx]
            } else {
                !block.data[word_idx]
            };
            
            let word_count = word.count_ones() as u64;
            if count + word_count > rank {
                // The bit is in this word
                let mut remaining = rank - count;
                for bit_idx in 0..64 {
                    if (word >> bit_idx) & 1 == 1 {
                        if remaining == 0 {
                            return word_idx * 64 + bit_idx;
                        }
                        remaining -= 1;
                    }
                }
            }
            count += word_count;
        }
        
        // Should not reach here
        0
    }
    
    /// Total length in bits
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    /// Count ones
    pub fn count_ones(&self) -> usize {
        self.num_ones
    }
    
    /// Count zeros
    pub fn count_zeros(&self) -> usize {
        self.len - self.num_ones
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    
    #[test]
    fn test_basic_operations() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512vpopcntdq") {
            println!("Skipping AVX-512 tests on this CPU");
            return;
        }
        
        let mut dict = Avx512RsDict::new();
        
        // Test empty
        assert_eq!(dict.len(), 0);
        assert_eq!(dict.count_ones(), 0);
        
        // Push some bits
        dict.push(true);
        dict.push(false);
        dict.push(true);
        dict.push(true);
        dict.finalize();
        
        // Test get_bit
        assert_eq!(dict.get_bit(0), true);
        assert_eq!(dict.get_bit(1), false);
        assert_eq!(dict.get_bit(2), true);
        assert_eq!(dict.get_bit(3), true);
        
        // Test rank
        assert_eq!(dict.rank(0, true), 0);
        assert_eq!(dict.rank(1, true), 1);
        assert_eq!(dict.rank(2, true), 1);
        assert_eq!(dict.rank(3, true), 2);
        assert_eq!(dict.rank(4, true), 3);
        
        assert_eq!(dict.rank(0, false), 0);
        assert_eq!(dict.rank(1, false), 0);
        assert_eq!(dict.rank(2, false), 1);
        assert_eq!(dict.rank(3, false), 1);
        assert_eq!(dict.rank(4, false), 1);
        
        // Test select
        assert_eq!(dict.select(0, true), Some(0));
        assert_eq!(dict.select(1, true), Some(2));
        assert_eq!(dict.select(2, true), Some(3));
        assert_eq!(dict.select(3, true), None);
        
        assert_eq!(dict.select(0, false), Some(1));
        assert_eq!(dict.select(1, false), None);
    }
    
    #[test]
    fn test_large_structure() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512vpopcntdq") {
            return;
        }
        
        let mut dict = Avx512RsDict::with_capacity(100_000);
        
        // Fill with pattern
        for i in 0..100_000 {
            dict.push(i % 3 == 0);
        }
        dict.finalize();
        
        // Verify structure
        assert_eq!(dict.len(), 100_000);
        assert_eq!(dict.count_ones(), 33_334);
        
        // Test some ranks
        assert_eq!(dict.rank(0, true), 0);
        assert_eq!(dict.rank(3, true), 1);
        assert_eq!(dict.rank(30, true), 10);
        assert_eq!(dict.rank(300, true), 100);
        
        // Test some selects
        assert_eq!(dict.select(0, true), Some(0));
        assert_eq!(dict.select(1, true), Some(3));
        assert_eq!(dict.select(100, true), Some(300));
    }
    
    proptest! {
        #[test]
        fn prop_rank_select_consistency(bits in prop::collection::vec(any::<bool>(), 1..10000)) {
            if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512vpopcntdq") {
                return Ok(());
            }
            
            let mut dict = Avx512RsDict::with_capacity(bits.len());
            for &bit in &bits {
                dict.push(bit);
            }
            dict.finalize();
            
            // Verify all bits match
            for (i, &bit) in bits.iter().enumerate() {
                prop_assert_eq!(dict.get_bit(i), bit);
            }
            
            // Verify rank correctness
            let mut ones = 0;
            let mut zeros = 0;
            for (i, &bit) in bits.iter().enumerate() {
                prop_assert_eq!(dict.rank(i, true), ones);
                prop_assert_eq!(dict.rank(i, false), zeros);
                
                if bit {
                    ones += 1;
                } else {
                    zeros += 1;
                }
            }
            
            // Verify select correctness
            let mut one_positions = vec![];
            let mut zero_positions = vec![];
            for (i, &bit) in bits.iter().enumerate() {
                if bit {
                    one_positions.push(i);
                } else {
                    zero_positions.push(i);
                }
            }
            
            for (rank, &pos) in one_positions.iter().enumerate() {
                prop_assert_eq!(dict.select(rank as u64, true), Some(pos));
            }
            
            for (rank, &pos) in zero_positions.iter().enumerate() {
                prop_assert_eq!(dict.select(rank as u64, false), Some(pos));
            }
        }
        
        #[test]
        fn prop_compare_with_naive(bits in prop::collection::vec(any::<bool>(), 1..5000)) {
            if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512vpopcntdq") {
                return Ok(());
            }
            
            // Build AVX512 version
            let mut avx_dict = Avx512RsDict::with_capacity(bits.len());
            for &bit in &bits {
                avx_dict.push(bit);
            }
            avx_dict.finalize();
            
            // Build original version
            let mut orig_dict = crate::RsDict::with_capacity(bits.len());
            for &bit in &bits {
                orig_dict.push(bit);
            }
            
            // Compare all operations
            for i in 0..bits.len() {
                prop_assert_eq!(avx_dict.get_bit(i), orig_dict.get_bit(i as u64));
                prop_assert_eq!(avx_dict.rank(i, true), orig_dict.rank(i as u64, true));
                prop_assert_eq!(avx_dict.rank(i, false), orig_dict.rank(i as u64, false));
            }
            
            // Compare select operations
            let num_ones = avx_dict.count_ones();
            let num_zeros = avx_dict.count_zeros();
            
            for rank in 0..num_ones.min(100) {
                prop_assert_eq!(
                    avx_dict.select(rank as u64, true),
                    orig_dict.select(rank as u64, true).map(|x| x as usize)
                );
            }
            
            for rank in 0..num_zeros.min(100) {
                prop_assert_eq!(
                    avx_dict.select(rank as u64, false),
                    orig_dict.select(rank as u64, false).map(|x| x as usize)
                );
            }
        }
    }
}