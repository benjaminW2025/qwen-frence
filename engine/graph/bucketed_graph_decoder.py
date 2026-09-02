"""
Bucketed CUDA-graph decode for the continuous-batching scheduler.

The scheduler supports adaptive batch size, but we run a fixed CUDA
graph batch size. We fix this by capturing buckets of CUDA graphs
across different batch sizes.
"""

import torch

from paged_graph_decoder import CUDAGraphDecoder


def default_buckets(max_running):
    """Log-spaced buckets {1,2,4,...} capped at max_running (plus max_running itself)."""
    b, out = 1, []
    while b < max_running:
        out.append(b)
        b *= 2
    out.append(max_running)
    return sorted(set(out))


class BucketedGraphDecoder:
    def __init__(self, model, cache, max_running, max_blocks, device, dtype, buckets=None):
        self.max_blocks = max_blocks
        self.buckets = default_buckets(max_running) if buckets is None else sorted(set(buckets))

        # Capture one graph per bucket, all against the SAME shared cache pool.
        self.decoders = {}
        for b in self.buckets:
            dec = CUDAGraphDecoder(model, cache, batch_size=b, max_blocks=max_blocks,
                                   device=device, dtype=dtype)
            dec.capture()
            self.decoders[b] = dec

    def _pick_bucket(self, n):
        for b in self.buckets:
            if b >= n:
                return b
        raise ValueError(f"num_active {n} exceeds largest bucket {self.buckets[-1]}")

    @torch.no_grad()
    def decode(self, input_ids, positions, seq_lens, block_table, slot_mapping):
        """Dense (num_active, ...) inputs. block_table is (num_active, <= max_blocks).
        Pads batch -> bucket and block_table -> max_blocks, replays, returns (num_active,1,vocab)."""
        n = input_ids.shape[0]
        if n == 0:
            raise ValueError("cannot decode an empty batch")

        bucket = self._pick_bucket(n)
        decoder = self.decoders[bucket]

        # The block-table width is also fixed at capture time. Unused entries can point
        # at block zero because seq_lens prevents the attention kernel from reading them.
        block_width = block_table.shape[1]
        if block_width > self.max_blocks:
            raise ValueError(
                f"block table width {block_width} exceeds captured width {self.max_blocks}"
            )
        if block_width < self.max_blocks:
            zeros = block_table.new_zeros(n, self.max_blocks - block_width)
            block_table = torch.cat((block_table, zeros), dim=1)

        if bucket > n:
            # A captured graph must execute every row. Duplicate the first active row so
            # padding has valid positions/cache metadata and writes the same KV values to
            # the same slot as that row. The padded logits are discarded after replay.
            n_padding = bucket - n

            def pad_batch(tensor):
                repeats = (n_padding,) + tensor.shape[1:]
                return torch.cat((tensor, tensor[:1].expand(repeats)), dim=0)

            input_ids = pad_batch(input_ids)
            positions = pad_batch(positions)
            seq_lens = pad_batch(seq_lens)
            block_table = pad_batch(block_table)
            slot_mapping = pad_batch(slot_mapping)

        logits = decoder.decode(
            input_ids, positions, seq_lens, block_table, slot_mapping
        )
        return logits[:n]
