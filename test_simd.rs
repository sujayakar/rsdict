#![cfg_attr(feature = "simd", feature(portable_simd))]

use rsdict::RsDict;

fn main() {
    // Create a large enough dataset to trigger SIMD paths
    let mut dict = RsDict::new();
    
    // Add many bits to ensure we go through the SIMD-accelerated paths
    for i in 0..100000 {
        dict.push(i % 3 == 0);
    }
    
    // Perform rank operations that will use scan_block
    let mut sum = 0u64;
    for i in (0..100000).step_by(1000) {
        sum += dict.rank(i, true);
    }
    
    println!("Sum of ranks: {}", sum);
}