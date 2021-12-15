use super::{Serializer, SerializerError};
use postcard::{from_bytes, to_stdvec};
use serde::{de::DeserializeOwned, Serialize};

pub struct SerializerImpl;

impl Serializer for SerializerImpl {
    fn to_vec<T: Serialize>(t: &T) -> Result<Vec<u8>, SerializerError> {
        Ok(to_stdvec(t)?)
    }
    fn from_slice<T: DeserializeOwned>(t: &[u8]) -> Result<T, SerializerError> {
        Ok(from_bytes(t)?)
    }
}
