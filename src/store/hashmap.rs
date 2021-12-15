use super::store::{Store, StoreError};
use rocksdb::{BlockBasedOptions, Options, DB};
use serde::{de::DeserializeOwned, Serialize};
use std::path::Path;

pub struct StoreImpl<K, V>
where
    K: Eq + Hash,
{
    inner: HashMap<K, V>,
    path: PathBuf,
}

impl<K, V> Store<K, V> for StoreImpl<K, V>
where
    K: Eq + Hash + PartialEq + PartialOrd + Serialize + DeserializeOwned,
    V: Serialize + DeserializeOwned,
{
    fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let inner = match std::fs::read(&(*path.as_ref())) {
            Ok(bytes) => HashMap::<K, V>::try_from_slice(&bytes)?,
            Err(_) => HashMap::<K, V>::new(),
        };
        Ok(Self {
            inner,
            path: path.as_ref().to_path_buf(),
        })
    }
    fn get(&self, k: &K) -> Result<Option<V>> {
        Ok(self.inner.get(&k).copied())
    }
    fn put(&mut self, k: K, v: V) -> Result<()> {
        self.inner.insert(k, v);
        Ok(())
    }
    fn save(&self) -> Result<()> {
        if let Some(p) = self.path.parent() {
            std::fs::create_dir_all(p)?;
        }
        std::fs::write(self.path.clone(), &rmp_serde::to_vec(&self.inner)?)?;
        Ok(())
    }
}
