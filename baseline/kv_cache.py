import torch

class KVCache:
    """
    Per-layer naive contiguous key/value cache.
    """

    def __init__(self, cfg, batch_size, device, dtype, max_seq_len=None):
        self.cfg = cfg
        self.cur_len = 0
        self.max_seq_len = max_seq_len if max_seq_len is not None else cfg.max_seq_len
        # Why preallocated rather than naively append vectors to end of cache?
        # Appending requires copying over elements into a brand new tensor
        # Preallocation allocates one and with simple indexing writes to memory exactly once
        self.k_cache = [
            torch.zeros(batch_size, cfg.n_kv_heads, self.max_seq_len, cfg.d_head,
                        device=device, dtype=dtype)
            for _ in range(cfg.n_layers)
        ]

        self.v_cache = [
            torch.zeros(batch_size, cfg.n_kv_heads, self.max_seq_len, cfg.d_head,
                        device=device, dtype=dtype)
            for _ in range(cfg.n_layers)
        ]

    def write(self, layer, start, k, v): # k is (batch, n_kv_heads, n_kv, d_head)
        # Write the new key and value vectors into kv cache
        n = k.shape[2]
        self.k_cache[layer][:, :, start:start+n, :] = k
        self.v_cache[layer][:, :, start:start+n, :] = v
        
    def read(self, layer):
        # Return the cache up to what we have already processed
        return (self.k_cache[layer][:, :, :self.cur_len, :], self.v_cache[layer][:, :, :self.cur_len, :])

    def reset(self):
        # Reuse the preallocated buffers for a new sequence. Data past cur_len is
        # unreachable (read() only returns [:cur_len]), so no need to zero.
        self.cur_len = 0