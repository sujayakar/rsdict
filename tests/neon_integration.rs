//! Integration tests for NEON acceleration on AArch64

#![cfg_attr(feature = "simd", feature(portable_simd))]

use rsdict::RsDict;

#[test]
fn test_neon_acceleration_available() {
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            println!("NEON acceleration is available and enabled");
        } else {
            println!("NEON feature not detected on this AArch64 CPU");
        }
    }
    
    #[cfg(not(target_arch = "aarch64"))]
    {
        println!("Not running on AArch64, NEON tests skipped");
    }
}

#[test]
fn test_rank_with_potential_neon() {
    // This test verifies that rank operations work correctly
    // whether or not NEON acceleration is available
    
    let mut dict = RsDict::with_capacity(10000);
    
    // Create a pattern
    for i in 0..10000 {
        dict.push(i % 5 == 0);
    }
    
    // Test various positions
    assert_eq!(dict.rank(0, true), 0);
    assert_eq!(dict.rank(5, true), 1);
    assert_eq!(dict.rank(10, true), 2);
    assert_eq!(dict.rank(100, true), 20);
    assert_eq!(dict.rank(1000, true), 200);
    assert_eq!(dict.rank(9999, true), 2000);
    
    // Test false bits
    assert_eq!(dict.rank(5, false), 4);
    assert_eq!(dict.rank(10, false), 8);
    assert_eq!(dict.rank(100, false), 80);
}

#[test]
fn benchmark_comparison() {
    use std::time::Instant;
    
    let mut dict = RsDict::with_capacity(1_000_000);
    for i in 0..1_000_000 {
        dict.push(i % 3 == 0);
    }
    
    // Warm up
    for _ in 0..100 {
        let _ = dict.rank(500_000, true);
    }
    
    // Measure performance
    let iterations = 10_000;
    let start = Instant::now();
    let mut sum = 0u64;
    
    for i in 0..iterations {
        let pos = (i * 9973) % 1_000_000;
        sum += dict.rank(pos, true);
    }
    
    let elapsed = start.elapsed();
    let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();
    
    println!("Rank operations per second: {:.0}", ops_per_sec);
    
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    if std::arch::is_aarch64_feature_detected!("neon") {
        println!("(Using NEON acceleration)");
        // On NEON-capable hardware, we expect good performance
        // but we can't assert specific numbers as they vary by CPU
    }
    
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if is_x86_feature_detected!("ssse3") {
        println!("(Using SSSE3 acceleration)");
    }
    
    #[cfg(not(feature = "simd"))]
    println!("(Using naive implementation)");
    
    // Ensure the sum is used to prevent optimization
    assert!(sum > 0);
}