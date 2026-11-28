# VMM Prefix Cache

Crisp technical specification of physical VMM prefix caching and zero-copy context reuse in `plowrt`.

---

## 1. Core Principles

* **Content-Addressed Hash Tree:** Prompt token sequences map to physical VMM memory blocks using cryptographic rolling hashes.
* **Zero-Copy Sharing:** Multiple sessions sharing a common prompt prefix reference the identical physical memory pages without duplication.
* **Tier 2 Memory Boundary:** Prefix cache blocks reside strictly within Tier 2 dynamic memory, preventing eviction of Tier 1 nominal sliding rings.

---

## 2. Allocation & Lifecycle

1. **Lookup:** On prompt arrival, the scheduler matches the longest prefix hash in the cache index.
2. **Hit Path:** Matched KV blocks are bound directly to the new session's page table. Prefill computation is skipped for all cached tokens.
3. **Eviction:** When free Tier 2 memory falls below headroom thresholds (`PLOW_VMM_CACHE_MIN_FREE_MIB`), pages are evicted via LRU policy.
4. **Deferred Reclaim:** Freed blocks are returned asynchronously via `PLOW_VMM_DEFERRED_RECLAIM` to avoid blocking active decode steps.

---

## 3. Verification

* Property tests validate hash consistency and zero memory corruption:
  ```bash
  cargo test -p plowrt --lib prefix_cache
  ```
