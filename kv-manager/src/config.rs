use crate::manager::KvCacheError;

/// Shape of the KV cache, fixed at engine start-up.
///
/// Together these fields determine the size of one cache block, which is the
/// unit of allocation for the whole system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheConfig {
    /// Tokens stored per block. Small blocks waste less tail space; large
    /// blocks mean shorter block tables and better kernel locality.
    /// PagedAttention-style engines typically use 16 or 32.
    pub block_size: usize,
    /// Transformer layers in the model.
    pub num_layers: usize,
    /// KV heads per layer (less than the number of query heads under
    /// grouped-query attention).
    pub num_kv_heads: usize,
    /// Dimension of one head.
    pub head_dim: usize,
    /// Bytes per element, e.g. 2 for f16/bf16, 1 for fp8.
    pub dtype_size: usize,
}

impl CacheConfig {
    /// Bytes needed to cache one token: a K vector and a V vector
    /// (`num_kv_heads * head_dim` elements each) for every layer.
    pub fn token_bytes(&self) -> usize {
        2 * self.num_layers * self.num_kv_heads * self.head_dim * self.dtype_size
    }

    /// Bytes of one cache block (`block_size` tokens).
    pub fn block_bytes(&self) -> usize {
        self.block_size * self.token_bytes()
    }

    pub(crate) fn validate(&self) -> Result<(), KvCacheError> {
        let all_nonzero = self.block_size > 0
            && self.num_layers > 0
            && self.num_kv_heads > 0
            && self.head_dim > 0
            && self.dtype_size > 0;
        if all_nonzero {
            Ok(())
        } else {
            Err(KvCacheError::InvalidConfig(
                "all CacheConfig fields must be non-zero",
            ))
        }
    }
}
