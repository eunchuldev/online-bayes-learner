use super::{Store, StoreError};
use crate::serializer::Serializer;
use rocksdb::{BlockBasedOptions, Options, DB};
use serde::{de::DeserializeOwned, Serialize};
use std::path::Path;

fn rocksdb_default_opts() -> Options {
    let mut opts = Options::default();
    // https://github.com/facebook/rocksdb/wiki/Setup-Options-and-Basic-Tuning
    #[allow(deprecated)]
    opts.set_max_background_compactions(4);
    #[allow(deprecated)]
    opts.set_max_background_flushes(2);
    opts.set_level_compaction_dynamic_level_bytes(true);
    opts.set_bytes_per_sync(1048576);
    opts.create_if_missing(true);

    let mut table_opts = BlockBasedOptions::default();
    table_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
    table_opts.set_cache_index_and_filter_blocks(true);
    table_opts.set_cache_index_and_filter_blocks(true);
    table_opts.set_block_size(16 * 1024);
    table_opts.set_format_version(5);

    // options.compaction_pri = kMinOverlappingRatio;
    opts.set_block_based_table_factory(&table_opts);
    opts
}

pub struct StoreImpl<S: Serializer> {
    inner: DB,
    _marker: std::marker::PhantomData<fn() -> S>,
}

impl<S: Serializer> Store<S> for StoreImpl<S> {
    fn open<P: AsRef<Path>>(path: P) -> Result<Self, StoreError> {
        Ok(Self {
            inner: DB::open(&rocksdb_default_opts(), path)?,
            _marker: Default::default(),
        })
    }
    fn get<K: Serialize, V: DeserializeOwned>(&self, k: &K) -> Result<Option<V>, StoreError> {
        Ok(self
            .inner
            .get(S::to_vec(k)?)?
            .map(|s| S::from_slice(&s))
            .transpose()?)
    }
    fn put<K: Serialize, V: Serialize>(&mut self, k: &K, v: &V) -> Result<(), StoreError> {
        Ok(self.inner.put(S::to_vec(k)?, S::to_vec(v)?)?)
    }
    fn save(&self) -> Result<(), StoreError> {
        Ok(())
    }
}
