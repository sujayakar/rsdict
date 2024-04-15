use crate::enum_code::ENUM_CODE_LENGTH;

fn scan_block_naive(classes: &[u8], start: usize, end: usize) -> (u64, u64) {
    let mut class_sum = 0;
    let mut length_sum = 0;
    for &class in &classes[start..end] {
        class_sum += class as u64;
        length_sum += ENUM_CODE_LENGTH[class as usize] as u64;
    }
    (class_sum, length_sum)
}

#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
pub fn scan_block(classes: &[u8], start: usize, end: usize) -> (u64, u64) { scan_block_naive(classes, start, end) }

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod accelerated {
    use super::scan_block_naive;
    use crate::enum_code::ENUM_CODE_LENGTH;
    use std::arch::x86_64::{__m128i, _mm_sad_epu8, _mm_setzero_si128};
    use std::simd::{num::SimdUint, u64x2, u8x16, Simd};
    use std::simd::prelude::SimdOrd;
    use std::slice;
    use std::u64;

    // Scan a prefix of a large block of small block classes, returning the
    // sum of the classes and their total encoded length.
    //
    // Preconditions:
    // * start <= end <= start + 16
    // * classes[start] must be 128-bit aligned
    //
    // Returns:
    // * class_sum: classes[start..end].sum()
    // * length_sum: classes[start.end].map(|i| ENUM_CODE_LENGTH[i]).sum()
    pub fn scan_block(classes: &[u8], start: usize, end: usize) -> (u64, u64) {
        if is_x86_feature_detected!("ssse3") {
            unsafe { scan_block_ssse3(classes, start, end) }
        } else {
            scan_block_naive(classes, start, end)
        }
    }

    #[target_feature(enable = "ssse3")]
    unsafe fn scan_block_ssse3(classes: &[u8], start: usize, end: usize) -> (u64, u64) {
        // Step 1: Load the classes into a u8x16.  Our approach here is to do a
        // single load and then mask off the elements past `len`.  This is unsafe
        // since we're potentially reading past the end of the slice, but we're
        // masking off the extraneous elements before processing them.
        let len = end - start;
        debug_assert!(len <= 16);

        // Step 1a: Start with all bits on, shift to turn off the lowest 8n bits,
        // and then negate to have the lowest 8n bits on.
        let lo_shift = len as u32 * 8;
        let lo_mask = !u64::MAX.checked_shl(lo_shift).unwrap_or(0);
        // Step 1b: Do the same for the remaining 8 bytes.
        let hi_shift = len.saturating_sub(8) as u32 * 8;
        let hi_mask = !u64::MAX.checked_shl(hi_shift).unwrap_or(0);
        let ix_mask: u8x16 = core::mem::transmute(u64x2::from([lo_mask, hi_mask]));

        let classes = {
            let start = classes.as_ptr().offset(start as isize);
            let block = slice::from_raw_parts(start, 16);

            let block = u8x16::from_slice(block);
            block & ix_mask
        };

        // Step 2: We want to be able to pack the `ENUM_CODE_LENGTH` table of 65
        // entries into a single u8x16 vector.  We can do this with two insights:
        //
        // 1) The table is symmetric, so we only need to store half of it if we can
        //    transform the indices.
        // 2) The table "caps" out at 64 for most of the range in the middle, which
        //    is the length of the 15th element.  If we just truncate indices greater
        //    than 15 (after reflection), we'll not change the value.
        //
        // Putting this together, we have f(i) = min(i, 64 - i, 15) such that
        //
        //    ENUM_CODE_LENGTH[i] == ENUM_CODE_LENGTH[f(i)] for i in [0, 64].
        //
        let indices = classes.simd_min(u8x16::splat(64) - classes).simd_min(u8x16::splat(15));
        let enum_code_vector: u8x16 = u8x16::from([
            ENUM_CODE_LENGTH[0], ENUM_CODE_LENGTH[1], ENUM_CODE_LENGTH[2], ENUM_CODE_LENGTH[3],
            ENUM_CODE_LENGTH[4], ENUM_CODE_LENGTH[5], ENUM_CODE_LENGTH[6], ENUM_CODE_LENGTH[7],
            ENUM_CODE_LENGTH[8], ENUM_CODE_LENGTH[9], ENUM_CODE_LENGTH[10], ENUM_CODE_LENGTH[11],
            ENUM_CODE_LENGTH[12], ENUM_CODE_LENGTH[13], ENUM_CODE_LENGTH[14], ENUM_CODE_LENGTH[15],
        ]);

        // Step 3: This is the real magic.  Now that we've packed our table into
        // a vector and transformed our classes into indices into this packed vector,
        // we can use `pshufb` to index into our table in parallel.
        let code_lengths = Simd::swizzle_dyn(enum_code_vector, indices);

        // Step 4: Compute our sums and return.
        let class_sum = sum_u8x16(classes);
        let length_sum = sum_u8x16(code_lengths);

        (class_sum, length_sum)
    }

    // In case `std::simd` supports `psadbw`, that could be a
    // great way to sum a u8x16 into a u64x2 in a single SSE2 instruction.
    unsafe fn sum_u8x16(xs: u8x16) -> u64 {
        let zero_m128: __m128i = _mm_setzero_si128();
        let xs_m128: __m128i = __m128i::from(xs);
        let sum_m128 = _mm_sad_epu8(zero_m128, xs_m128);
        u64x2::from(sum_m128).reduce_sum()
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub use self::accelerated::scan_block;
