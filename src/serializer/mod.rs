use serde::{de::DeserializeOwned, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SerializerError {
    #[cfg(feature = "postcard")]
    #[error("error")]
    Postcard(#[from] postcard::Error),
}

pub trait Serializer {
    fn to_vec<T: Serialize>(t: &T) -> Result<Vec<u8>, SerializerError>;
    fn from_slice<T: DeserializeOwned>(t: &[u8]) -> Result<T, SerializerError>;
}

#[cfg(feature = "postcard")]
mod postcard_impl;

#[cfg(feature = "postcard")]
pub use super::serializer::postcard_impl::SerializerImpl;
