"""Paged KV cache used by the inference engine."""

import torch


class CacheFullError(RuntimeError):
    """Raised when the block pool is exhausted. The scheduler catches this to
    preempt/evict a sequence and retry -- it is a capacity signal, not a bug."""


class PagedKVCache():
    """
    Paged KV cache class with batch awareness
    """

    def __init__(self, cfg, batch_size, num_blocks, block_size, device, dtype):
        # Need to first allocate the super big chunk of memory
        # Need to initialize the free list
        # Keep separate key and value lists

        self.cfg = cfg
        self.n_kv_heads = cfg.n_kv_heads
        self.d_head = cfg.d_head
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_blocks = num_blocks   # total physical blocks -> lets reset() rebuild the free list
        self.device = device

        self.cur_lens = [0] * batch_size # <- keeps the current sequence lengths, i.e. where we need to append new kv vectors to

        self.k_pool = [torch.zeros(num_blocks, block_size, cfg.n_kv_heads, cfg.d_head,
                                   device=device, dtype=dtype)
                       for _ in range(cfg.n_layers)]
        
        self.v_pool = [torch.zeros(num_blocks, block_size, cfg.n_kv_heads, cfg.d_head,
                                   device=device, dtype=dtype)
                       for _ in range(cfg.n_layers)]

        self.free_blocks = list(range(num_blocks)) 
        self.block_tables = [[] for _ in range(batch_size)]

    def write(self, layer, starts, k, v): # (batch, n_kv_heads, seq_id, d_head)
        # Should also do a check on whether or not we have hit the 32k token maximum
        for i in range(k.shape[0]):
            for j in range(k.shape[2]):
                # Loop through every token in the incoming KV tensors
                start_new = starts[i] + j
                # Need to allocate coresponding element for KV vector
                offset = start_new % self.block_size
                # Compute which block to start at (this sequence's table)
                pid = self.block_tables[i][start_new // self.block_size]
                # Write to KV pool
                self.k_pool[layer][pid, offset] = k[i, :, j, :] # (n_kv_heads, d_head)
                self.v_pool[layer][pid, offset] = v[i, :, j, :]
                
    def allocate_block(self, n_news):
        # Per-sequence scalar arithmetic (cur_lens / n_news are Python lists, so
        # `list + list` would concatenate, not add element-wise).
        #
        # ATOMIC: validate + count the total new blocks needed across ALL sequences
        # FIRST, and raise before mutating anything. Otherwise a partial allocation
        # (grow seq 0, then run out on seq 3) would leave cur_lens advanced with no
        # token written -- and the scheduler's preempt-and-retry would corrupt state.
        total_needed = 0
        for b in range(self.batch_size):
            new_len = self.cur_lens[b] + n_news[b]

            # Per-sequence cap: the model cannot attend past max_seq_len (RoPE /
            # context window), so refuse to grow a sequence beyond it.
            if new_len > self.cfg.max_seq_len:
                raise ValueError(
                    f"sequence {b} would reach {new_len} tokens, exceeds "
                    f"max_seq_len={self.cfg.max_seq_len}")

            needed = (new_len - 1) // self.block_size + 1   # ceil(new_len/bs); 0 when new_len==0
            total_needed += max(0, needed - len(self.block_tables[b]))

        # Pool exhaustion is a capacity signal for the scheduler, not a crash.
        if total_needed > len(self.free_blocks):
            raise CacheFullError(
                f"need {total_needed} new blocks, only {len(self.free_blocks)} free")

        # Second pass: guaranteed to succeed, so this half can't leave partial state.
        for b in range(self.batch_size):
            new_len = self.cur_lens[b] + n_news[b]
            needed = (new_len - 1) // self.block_size + 1
            while len(self.block_tables[b]) < needed:
                self.block_tables[b].append(self.free_blocks.pop())
            self.cur_lens[b] = new_len

    def free(self, b):
        """
        To free one single sequence 'b'
        """

        # Loop through & add
        self.free_blocks.extend(self.block_tables[b])
        self.block_tables[b] = [] # Clear block table b
        self.cur_lens[b] = 0

    def truncate(self, b, new_len):
        """Roll one logical sequence back, releasing pages beyond ``new_len``."""
        if not 0 <= new_len <= self.cur_lens[b]:
            raise ValueError(
                f"cannot truncate slot {b} from {self.cur_lens[b]} to {new_len}"
            )
        needed = (new_len + self.block_size - 1) // self.block_size
        released = self.block_tables[b][needed:]
        self.block_tables[b] = self.block_tables[b][:needed]
        self.free_blocks.extend(released)
        self.cur_lens[b] = new_len

    def reset(self):
        """
        Fully reset the cache
        """

        # Give all blocks back to free
        self.free_blocks = list(range(self.num_blocks))
        # Build empty list of lists
        self.block_tables = [[] for _ in range(self.batch_size)]
        # Build list of 0s
        self.cur_lens = [0] * self.batch_size

    def read(self, layer):
        # First we need to find max_blocks and pad the rest of the 
        # First we need to identify which blocks are in use from block_table
        max_blocks = max(len(bt) for bt in self.block_tables)
        padded_blocks = [bt + [0] * (max_blocks - len(bt)) for bt in self.block_tables]
        idx = torch.tensor(padded_blocks, device=self.device)

        keys_in_use = self.k_pool[layer][idx] # (batch_size, max_blocks, block_size, n_kv_heads, d_head)
        values_in_use = self.v_pool[layer][idx] # (batch_size, max_blocks, block_size, n_kv_heads, d_head)

        # Now we need to reshape into (batch, seq_len, n_kv_heads, d_head)
        # (batch_size, max_seq_len, n_kv_heads, d_head) <- max_seq_len because padding
        k = keys_in_use.reshape(self.batch_size, max_blocks * self.block_size, self.n_kv_heads, self.d_head)
        v = values_in_use.reshape(self.batch_size, max_blocks * self.block_size, self.n_kv_heads, self.d_head)

        # Transpose so that our attention kernel recieves in expected shape
        # We need to return attetion mask for batched case
        return (k.transpose(1, 2), v.transpose(1, 2))
