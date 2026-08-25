"""KV-cache backends: per-request contiguous buffers and a paged block pool.

Two authoritative designs live here:

* ``ContiguousKVCache``  — one preallocated [max_len, kv_heads, head_dim] pair of
  tensors per request (modes: ``contiguous``, ``batched``).
* ``PagedKVCache``       — one shared physical block pool per K and V, a
  free-list allocator, and per-request block tables (modes: ``paged``,
  ``triton``). The pools are the *only* runtime KV source of truth; attention
  reads through the block tables, never a shadow copy (PRD FR6, acceptance #7).

``BlockAllocator`` is pure Python so allocator invariants are unit-tested on a
laptop without Torch (PRD §13.1).
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

try:  # Torch is optional at import time so pure-Python parts run anywhere.
    import torch as _torch
except ImportError:  # pragma: no cover - exercised only on lightweight installs
    _torch = None


class CacheCapacityFull(Exception):
    """Internal signal: not enough free blocks to satisfy a reservation."""

    is_capacity_error = True


class DoubleFreeError(ValueError):
    """A block was released twice or by a non-owner (invariant violation)."""


@dataclass(frozen=True)
class CacheStats:
    kind: str
    blocks_total: int = 0
    blocks_used: int = 0
    utilization: float = 0.0
    reserved_bytes: int = 0
    occupied_bytes: int = 0
    internal_fragmentation_bytes: int = 0
    temporary_gather_bytes: int = 0
    request_blocks_used: int = 0
    prefix_blocks_used: int = 0
    prefix_cache_hits: int = 0
    prefix_cache_misses: int = 0

    def as_metrics(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "blocks_total": self.blocks_total,
            "blocks_used": self.blocks_used,
            "utilization": round(self.utilization, 4),
            "reserved_bytes": self.reserved_bytes,
            "occupied_bytes": self.occupied_bytes,
            "internal_fragmentation_bytes": self.internal_fragmentation_bytes,
            "temporary_gather_bytes": self.temporary_gather_bytes,
            "request_blocks_used": self.request_blocks_used,
            "prefix_blocks_used": self.prefix_blocks_used,
            "prefix_cache_hits": self.prefix_cache_hits,
            "prefix_cache_misses": self.prefix_cache_misses,
        }


@dataclass
class CacheView:
    """Logical [seq_len, kv_heads, head_dim] window over a backend."""

    keys: Any = None
    values: Any = None


class BlockAllocator:
    """Free-list allocator that owns every physical block exactly once.

    Invariants asserted in code (PRD §13.4):
      * free_blocks + allocated_blocks == total_blocks
      * a physical block has zero or one owner
      * double free / foreign free raises instead of corrupting state
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        if block_size < 1:
            raise ValueError("block_size must be positive")
        self.total_blocks = num_blocks
        self.block_size = block_size
        self._free: list[int] = list(range(num_blocks))
        self._owner: dict[int, str] = {}

    # -- accounting ---------------------------------------------------------
    @property
    def free_count(self) -> int:
        return len(self._free)

    @property
    def used_count(self) -> int:
        return len(self._owner)

    def blocks_for(self, tokens: int) -> int:
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        return math.ceil(tokens / self.block_size)

    def can_satisfy(self, n_blocks: int) -> bool:
        return n_blocks <= self.free_count

    def owned_by(self, owner: str) -> int:
        return sum(1 for o in self._owner.values() if o == owner)

    # -- mutation -----------------------------------------------------------
    def allocate(self, owner: str, n_blocks: int) -> list[int]:
        if n_blocks > self.free_count:
            raise CacheCapacityFull(
                f"requested {n_blocks} blocks, only {self.free_count} free"
            )
        ids = [self._free.pop() for _ in range(n_blocks)]
        for block_id in ids:
            self._owner[block_id] = owner
        return ids

    def free(self, block_ids: Sequence[int], owner: str) -> None:
        if len(set(block_ids)) != len(block_ids):
            raise DoubleFreeError("same block freed twice in one call")
        for block_id in block_ids:
            actual_owner = self._owner.get(block_id)
            if actual_owner is None:
                raise DoubleFreeError(f"block {block_id} is already free")
            if actual_owner != owner:
                raise DoubleFreeError(
                    f"block {block_id} owned by {actual_owner!r}, freed by {owner!r}"
                )
        for block_id in block_ids:
            del self._owner[block_id]
            self._free.append(block_id)

    def free_for(self, owner: str) -> list[int]:
        owned = [b for b, o in self._owner.items() if o == owner]
        self.free(owned, owner)
        return owned

    def assert_invariants(self) -> None:
        assert len(self._owner) + len(self._free) == self.total_blocks, (
            "allocator leak: free + allocated != total"
        )
        assert len(set(self._owner)) == len(self._owner), "duplicate ownership"
        assert all(b >= 0 and b < self.total_blocks for b in self._free)


class ContiguousKVCache:
    """Per-request preallocated KV tensors sized to worst-case sequence length.

    Fragmentation is visible by construction: capacity minus occupied tokens
    is internal fragmentation (PRD FR4).
    """

    kind = "contiguous"

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: Any = None,
        device: str = "cuda",
    ) -> None:
        if _torch is None:
            raise RuntimeError("ContiguousKVCache requires torch")
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype or _torch.float16
        self.device = device
        self._slot_bytes = 2 * num_kv_heads * head_dim * _torch.tensor([], dtype=self.dtype).element_size()
        self._buffers: dict[str, list[list[Any]]] = {}
        self._capacity: dict[str, int] = {}
        self._length: dict[str, int] = {}

    def reserve(self, request_id: str, token_capacity: int) -> None:
        if request_id in self._buffers:
            raise ValueError(f"duplicate reservation for {request_id}")
        shape = (token_capacity, self.num_kv_heads, self.head_dim)
        self._buffers[request_id] = [
            [_torch.empty(shape, dtype=self.dtype, device=self.device) for _ in range(2)]
            for _ in range(self.num_layers)
        ]
        self._capacity[request_id] = token_capacity
        self._length[request_id] = 0

    def append(
        self,
        request_id: str,
        layer: int,
        keys: Any,
        values: Any,
        start_pos: int,
    ) -> None:
        buffers = self._buffers[request_id][layer]
        end = start_pos + keys.shape[0]
        buffers[0][start_pos:end].copy_(keys)
        buffers[1][start_pos:end].copy_(values)
        self._length[request_id] = max(self._length[request_id], end)

    def view(self, request_id: str, layer: int) -> CacheView:
        length = self._length[request_id]
        k, v = self._buffers[request_id][layer]
        return CacheView(keys=k[:length], values=v[:length])

    def release(self, request_id: str) -> None:
        self._buffers.pop(request_id, None)
        self._capacity.pop(request_id, None)
        self._length.pop(request_id, None)

    def stats(self) -> CacheStats:
        reserved = sum(self._capacity.values()) * self._slot_bytes * self.num_layers
        occupied = sum(self._length.values()) * self._slot_bytes * self.num_layers
        fragmented = max(0, reserved - occupied)
        return CacheStats(
            kind=self.kind,
            reserved_bytes=reserved,
            occupied_bytes=occupied,
            internal_fragmentation_bytes=fragmented,
            temporary_gather_bytes=0,
        )


class PagedKVCache:
    """Shared physical KV block pools addressed through per-request block tables.

    Layout (PRD FR6): ``[num_layers, num_blocks, block_size, num_kv_heads, head_dim]``
    for each of K and V. Legacy paged modes reserve worst-case capacity through
    ``reserve``. The ragged engine admits empty tables and grows them with
    ``ensure_capacity_batch`` so a scheduler iteration either acquires every
    required block or changes nothing.
    """

    kind = "paged"

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        kv_cache_bytes: int,
        dtype: Any = None,
        device: str = "cuda",
        prefix_cache_max_blocks: int = 0,
    ) -> None:
        if _torch is None:
            raise RuntimeError("PagedKVCache requires torch")
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype or _torch.float16
        self.device = device
        element_size = _torch.tensor([], dtype=self.dtype).element_size()
        self.slot_bytes = 2 * num_kv_heads * head_dim * element_size
        bytes_per_block_all_layers = self.slot_bytes * block_size * num_layers
        self.num_blocks = kv_cache_bytes // bytes_per_block_all_layers
        if self.num_blocks < 1:
            raise ValueError("kv_cache_bytes too small for even one block")
        shape = (num_layers, self.num_blocks, block_size, num_kv_heads, head_dim)
        self.key_pool = _torch.empty(shape, dtype=self.dtype, device=device)
        self.value_pool = _torch.empty(shape, dtype=self.dtype, device=device)
        self.allocator = BlockAllocator(self.num_blocks, block_size)
        self._tables: dict[str, list[int]] = {}
        self._length: dict[str, int] = {}
        self.gathered_bytes = 0
        self.prefix_cache_max_blocks = prefix_cache_max_blocks
        self._prefixes: OrderedDict[tuple[int, ...], tuple[str, list[int], int]] = (
            OrderedDict()
        )
        self._prefix_counter = 0
        self.prefix_cache_hits = 0
        self.prefix_cache_misses = 0

    def _evict_prefixes(self, needed: int, protect: tuple[int, ...] | None = None) -> None:
        while self._prefixes and self.allocator.free_count < needed:
            key = next(iter(self._prefixes))
            if key == protect:
                self._prefixes.move_to_end(key)
                if all(candidate == protect for candidate in self._prefixes):
                    return
                continue
            owner, blocks, _ = self._prefixes.pop(key)
            self.allocator.free(blocks, owner)

    def restore_prefix(self, request_id: str, token_ids: Sequence[int]) -> int:
        """Copy the longest cached block-aligned prefix into a new request."""
        if not self.prefix_cache_max_blocks:
            return 0
        aligned = ((len(token_ids) - 1) // self.block_size) * self.block_size
        for length in range(aligned, 0, -self.block_size):
            key = tuple(token_ids[:length])
            entry = self._prefixes.get(key)
            if entry is None:
                continue
            _, source_blocks, _ = entry
            self._evict_prefixes(len(source_blocks), protect=key)
            if not self.allocator.can_satisfy(len(source_blocks)):
                break
            self.ensure_capacity_batch({request_id: length})
            target_blocks = self._tables[request_id]
            source = _torch.tensor(source_blocks, dtype=_torch.long, device=self.device)
            target = _torch.tensor(target_blocks, dtype=_torch.long, device=self.device)
            self.key_pool.index_copy_(1, target, self.key_pool.index_select(1, source))
            self.value_pool.index_copy_(1, target, self.value_pool.index_select(1, source))
            self._length[request_id] = length
            self._prefixes.move_to_end(key)
            self.prefix_cache_hits += 1
            return length
        if aligned:
            self.prefix_cache_misses += 1
        return 0

    def store_prefix(self, request_id: str, token_ids: Sequence[int]) -> int:
        """Retain a reusable full-block prompt prefix in the physical pool."""
        if not self.prefix_cache_max_blocks:
            return 0
        length = ((len(token_ids) - 1) // self.block_size) * self.block_size
        blocks_needed = length // self.block_size
        if not length or blocks_needed > self.prefix_cache_max_blocks:
            return 0
        key = tuple(token_ids[:length])
        if key in self._prefixes:
            self._prefixes.move_to_end(key)
            return length
        if self._length.get(request_id, 0) < length:
            return 0
        cached_blocks = sum(len(entry[1]) for entry in self._prefixes.values())
        while self._prefixes and cached_blocks + blocks_needed > self.prefix_cache_max_blocks:
            _, (owner, blocks, _) = self._prefixes.popitem(last=False)
            self.allocator.free(blocks, owner)
            cached_blocks -= len(blocks)
        self._evict_prefixes(blocks_needed)
        if not self.allocator.can_satisfy(blocks_needed):
            return 0
        self._prefix_counter += 1
        owner = f"prefix-{self._prefix_counter}"
        target_blocks = self.allocator.allocate(owner, blocks_needed)
        source_blocks = self._tables[request_id][:blocks_needed]
        source = _torch.tensor(source_blocks, dtype=_torch.long, device=self.device)
        target = _torch.tensor(target_blocks, dtype=_torch.long, device=self.device)
        self.key_pool.index_copy_(1, target, self.key_pool.index_select(1, source))
        self.value_pool.index_copy_(1, target, self.value_pool.index_select(1, source))
        self._prefixes[key] = (owner, target_blocks, length)
        return length

    # -- admission ----------------------------------------------------------
    def reserve(self, request_id: str, token_capacity: int) -> None:
        if request_id in self._tables:
            raise ValueError(f"duplicate reservation for {request_id}")
        needed = self.allocator.blocks_for(token_capacity)
        if not self.allocator.can_satisfy(needed):
            raise CacheCapacityFull(
                f"need {needed} blocks, {self.allocator.free_count} free"
            )
        self._tables[request_id] = self.allocator.allocate(request_id, needed)
        self._length[request_id] = 0

    def ensure_capacity_batch(self, required_tokens: dict[str, int]) -> None:
        """Transactionally grow several block tables to their required lengths."""
        additions: dict[str, int] = {}
        for request_id, tokens in required_tokens.items():
            if request_id not in self._tables:
                raise KeyError(f"request {request_id!r} has no block table")
            required_blocks = self.allocator.blocks_for(tokens)
            additions[request_id] = max(0, required_blocks - len(self._tables[request_id]))
        total = sum(additions.values())
        self._evict_prefixes(total)
        if not self.allocator.can_satisfy(total):
            raise CacheCapacityFull(
                f"need {total} additional blocks, {self.allocator.free_count} free"
            )

        allocated: dict[str, list[int]] = {}
        try:
            for request_id, count in additions.items():
                if count:
                    allocated[request_id] = self.allocator.allocate(request_id, count)
            for request_id, block_ids in allocated.items():
                self._tables[request_id].extend(block_ids)
        except Exception:
            for request_id, block_ids in allocated.items():
                self.allocator.free(block_ids, request_id)
            raise

    def append_slots(
        self,
        layer: int,
        keys: Any,
        values: Any,
        slot_mapping: Any,
    ) -> None:
        """Write a flat packed token axis into physical cache slots."""
        keys_view = self.key_pool[layer].view(
            self.num_blocks * self.block_size, self.num_kv_heads, self.head_dim
        )
        values_view = self.value_pool[layer].view(
            self.num_blocks * self.block_size, self.num_kv_heads, self.head_dim
        )
        keys_view.index_copy_(0, slot_mapping, keys.to(keys_view.dtype))
        values_view.index_copy_(0, slot_mapping, values.to(values_view.dtype))

    def slot_mapping(self, request_ids: Sequence[str], positions: Sequence[int]) -> list[int]:
        if len(request_ids) != len(positions):
            raise ValueError("request_ids and positions must have equal length")
        slots: list[int] = []
        for request_id, position in zip(request_ids, positions, strict=True):
            table = self._tables[request_id]
            slots.append(
                table[position // self.block_size] * self.block_size
                + position % self.block_size
            )
        return slots

    def commit_lengths(self, lengths: dict[str, int]) -> None:
        for request_id, length in lengths.items():
            if self.allocator.blocks_for(length) > len(self._tables[request_id]):
                raise RuntimeError(f"length {length} exceeds allocated table for {request_id}")
            self._length[request_id] = max(self._length[request_id], length)

    def allocated_tokens(self, request_id: str) -> int:
        return len(self._tables.get(request_id, ())) * self.block_size

    # -- writes -------------------------------------------------------------
    def append(
        self,
        request_id: str,
        layer: int,
        keys: Any,
        values: Any,
        start_pos: int,
    ) -> None:
        table = self._tables[request_id]
        flat_indices = [
            table[position // self.block_size] * self.block_size + position % self.block_size
            for position in range(start_pos, start_pos + keys.shape[0])
        ]
        index = _torch.tensor(flat_indices, dtype=_torch.long, device=self.device)
        keys_view = self.key_pool[layer].view(self.num_blocks * self.block_size, self.num_kv_heads, self.head_dim)
        values_view = self.value_pool[layer].view(
            self.num_blocks * self.block_size, self.num_kv_heads, self.head_dim
        )
        keys_view[index] = keys.to(keys_view.dtype)
        values_view[index] = values.to(values_view.dtype)
        self._length[request_id] = max(self._length[request_id], start_pos + keys.shape[0])

    # -- reads ----------------------------------------------------------------
    def view(self, request_id: str, layer: int, length: int | None = None) -> CacheView:
        """Gather a temporary logical tensor for reference attention.

        The gather exists only inside this call; the pools remain the sole
        storage. Its size is reported as ``temporary_gather_bytes`` (PRD FR6).
        """
        seq_len = self._length[request_id] if length is None else length
        if self.allocator.blocks_for(seq_len) > len(self._tables[request_id]):
            raise RuntimeError(f"requested view length {seq_len} exceeds allocated block table")
        table = self._tables[request_id]
        flat_indices = [
            block_id * self.block_size + slot
            for block_id in table
            for slot in range(self.block_size)
        ]
        index = _torch.tensor(
            flat_indices[: math.ceil(seq_len / self.block_size) * self.block_size],
            dtype=_torch.long,
            device=self.device,
        )
        keys_flat = self.key_pool[layer].view(
            self.num_blocks * self.block_size, self.num_kv_heads, self.head_dim
        )
        values_flat = self.value_pool[layer].view(
            self.num_blocks * self.block_size, self.num_kv_heads, self.head_dim
        )
        gathered_keys = keys_flat[index][:seq_len]
        gathered_values = values_flat[index][:seq_len]
        self.gathered_bytes += seq_len * self.slot_bytes
        return CacheView(keys=gathered_keys, values=gathered_values)

    def block_table(self, request_id: str) -> list[int]:
        return list(self._tables[request_id])

    def sequence_length(self, request_id: str) -> int:
        return self._length[request_id]

    def release(self, request_id: str) -> None:
        if request_id in self._tables:
            self.allocator.free_for(request_id)
            del self._tables[request_id]
        self._length.pop(request_id, None)

    def stats(self) -> CacheStats:
        used = self.allocator.used_count
        total = self.allocator.total_blocks
        reserved = used * self.slot_bytes * self.block_size * self.num_layers
        prefix_blocks = sum(len(entry[1]) for entry in self._prefixes.values())
        request_blocks = used - prefix_blocks
        prefix_tokens = sum(entry[2] for entry in self._prefixes.values())
        occupied = (
            sum(self._length.values()) + prefix_tokens
        ) * self.slot_bytes * self.num_layers
        return CacheStats(
            kind=self.kind,
            blocks_total=total,
            blocks_used=used,
            utilization=used / total if total else 0.0,
            reserved_bytes=reserved,
            occupied_bytes=occupied,
            internal_fragmentation_bytes=max(0, reserved - occupied),
            temporary_gather_bytes=self.gathered_bytes,
            request_blocks_used=request_blocks,
            prefix_blocks_used=prefix_blocks,
            prefix_cache_hits=self.prefix_cache_hits,
            prefix_cache_misses=self.prefix_cache_misses,
        )

    def assert_invariants(self) -> None:
        self.allocator.assert_invariants()
