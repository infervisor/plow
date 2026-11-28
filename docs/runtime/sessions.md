# Session Management & Context State

Crisp technical specification of multi-turn session state, context lifecycle, and eviction policies in `plowrt`.

---

## 1. Session Architecture

* **Session Identity:** Unique session ID mapped via HTTP header (`X-Session-Id` or OpenAI `session_id`).
* **Context State:** Tracks prompt tokens, generated tokens, active KV page pointers, and attention mask boundaries.
* **TTL & Expiration:** Sessions expire after inactivity duration (`PLOW_SESSION_TTL_MS`, default 300,000 ms), releasing all bound pages to the free pool.

---

## 2. Memory Tier Allocation

* **Nominal Ring (Tier 1):** Sessions admitted within nominal concurrency reserve a dedicated circular sliding-window ring buffer.
* **Dynamic Extension (Tier 2):** Context extending beyond sliding window capacity allocates uniform blocks (`PLOW_KV_PAGE_SIZE`) from Tier 2.
* **Zero Leaks:** On client disconnect or graceful completion, session state teardown returns all physical pages immediately.

---

## 3. Diagnostic Commands

* Query active sessions and VMM memory distribution via administrative endpoint:
  ```bash
  curl http://localhost:8100/admin/status
  ```
