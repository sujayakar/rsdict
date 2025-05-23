#![cfg_attr(feature = "simd", feature(portable_simd))]

use rsdict::RsDict;

fn main() {
    let mut dict = RsDict::new();
    
    // Create a large dataset
    for i in 0..1_000_000 {
        dict.push(i % 3 == 0);
    }
    
    // Time rank operations
    let start = std::time::Instant::now();
    let mut sum = 0u64;
    for i in (0..1_000_000).step_by(100) {
        sum += dict.rank(i, true);
    }
    let elapsed = start.elapsed();
    
    println!("Total rank sum: {}", sum);
    println!("Time: {:?}", elapsed);
}