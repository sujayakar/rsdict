// other library used for correctness checks
extern crate fid;
use fid::{FID, BitVector};

#[test]
/// Check that elias fano runs considering a lot of possible combinations.
fn test_all_ones() {
    let mut r = rsdict::RsDict::new();
    for _ in 0..65 {
        r.push(true);
    }
}
#[test]
/// Check that elias fano runs considering a lot of possible combinations.
fn test_select_ones() {  
    let mut r = rsdict::RsDict::new();
    let mut bv = BitVector::new();
    for _ in 0..65 {
        r.push(true);
        bv.push(true);
    }
    
    // these fail
    for i in 0..65 {
        assert_eq!(r.select(i, true).unwrap(), bv.select(true, i));
    }
}

#[test]
/// Check that elias fano runs considering a lot of possible combinations.
fn test_get_bits() {  
    let mut r = rsdict::RsDict::new();
    let mut bv = BitVector::new();
    for _ in 0..65 {
        r.push(true);
        bv.push(true);
    }
    
    // these fail
    for i in 0..65 {
        assert_eq!(r.get_bit(i), bv.get(i));
    }
}
