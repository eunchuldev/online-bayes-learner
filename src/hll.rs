pub use hyperloglogplus::HyperLogLog as Hll;
use hyperloglogplus::HyperLogLogPlus;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::BuildHasher;

#[derive(Serialize, Deserialize)]
pub struct HasherBuilder;

impl BuildHasher for HasherBuilder {
    type Hasher = DefaultHasher;
    #[inline]
    fn build_hasher(&self) -> DefaultHasher {
        DefaultHasher::new()
    }
}

pub type Hllp = HyperLogLogPlus<[u8], HasherBuilder>;
