use crate::serializer::{Serializer, SerializerError};
use serde::{de::DeserializeOwned, Serialize};
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum StoreError {
    #[error("rocksdb error")]
    Rocksdb(#[from] rocksdb::Error),
    #[error("serializer error")]
    Serializer(#[from] SerializerError),
}

pub trait Store<S: Serializer>: Sized {
    fn open<P: AsRef<Path>>(path: P) -> Result<Self, StoreError>;
    fn get<K: Serialize, V: DeserializeOwned>(&self, k: &K) -> Result<Option<V>, StoreError>;
    fn put<K: Serialize, V: Serialize>(&mut self, k: &K, v: &V) -> Result<(), StoreError>;
    fn save(&self) -> Result<(), StoreError>;
}

#[cfg(feature = "rocksdb")]
mod rocksdb_impl;
#[cfg(feature = "rocksdb")]
pub use super::store::rocksdb_impl::StoreImpl;

#[cfg(feature = "hashmap")]
mod hashmap;
