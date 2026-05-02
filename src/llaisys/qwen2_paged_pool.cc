// PagedAttention-style KV cache pool: shared physical pages, per-request
// block tables, ref-counted CoW, per-tenant quotas (reservation_floor /
// max_pages / burst_pages), per-tenant chain-hash prefix index (16-token
// chunks, FNV-1a). Cross-tenant page takeover zero-wipes.

#include "llaisys/models/qwen2.h"
#include "llaisys/runtime.h"
#include "llaisys/tensor.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

// Must match the definition in qwen2.cc (ODR).
struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    llaisysDeviceType_t device_type;
    int device_id;
    llaisysTensor_t zero_bias_hs;
    llaisysTensor_t zero_bias_di;
    llaisysTensor_t zero_bias_voc;
};

namespace {

constexpr size_t kPrefixTrack = 4096;          // tokens kept for diagnostics + chain rebuild
constexpr uint64_t kHashSeed = 0xcbf29ce484222325ULL;
constexpr uint64_t kHashPrime = 0x100000001b3ULL;

inline uint64_t chain_hash(uint64_t prev, const int64_t *tokens, size_t n) {
    uint64_t h = prev ? prev : kHashSeed;
    for (size_t i = 0; i < n; ++i) {
        h ^= static_cast<uint64_t>(tokens[i]);
        h *= kHashPrime;
    }
    return h;
}

inline size_t dtype_size(llaisysDataType_t dtype) {
    switch (dtype) {
        case LLAISYS_DTYPE_F32: return 4;
        case LLAISYS_DTYPE_F16: return 2;
        case LLAISYS_DTYPE_BF16: return 2;
        case LLAISYS_DTYPE_I64: return 8;
        case LLAISYS_DTYPE_I32: return 4;
        default: return 4;
    }
}

struct Page {
    uint16_t ref_count = 0;          // 0 = free
    uint64_t tenant_id = 0;          // 0 = unowned (in global free list)
    uint64_t last_used_seq = 0;      // monotonic LRU stamp
    uint64_t chain_hash = 0;         // hash of cumulative prefix up to and incl. this page
    bool sealed = false;             // true once a page is fully filled and committed
};

struct TenantState {
    uint64_t tenant_id = 0;
    size_t reservation_floor = 0;
    size_t max_pages = SIZE_MAX;
    size_t burst_pages = SIZE_MAX;

    size_t pages_in_use = 0;         // ref_count > 0 OR in this tenant's free pool
    std::unordered_map<uint64_t /*chain_hash*/, int32_t /*page_id*/> hash_to_page;
    std::deque<int32_t> lru_free;    // released pages still tagged as this tenant
};

struct RequestBlock {
    uint64_t tenant_id = 0;
    bool in_use = false;
    size_t pos = 0;                  // committed token count
    std::vector<int32_t> page_table; // size = ceil(pos / page_size)
    std::vector<int64_t> prefix;     // up to kPrefixTrack tokens (for re-hashing on Commit)
};

} // anonymous namespace

struct LlaisysQwen2PagedPool {
    LlaisysQwen2Model *model = nullptr;
    size_t page_size = 16;
    size_t n_pages = 0;
    size_t nlayer = 0;
    size_t nkvh = 0;
    size_t dh = 0;
    size_t max_pages_per_request = 0;
    llaisysDataType_t dtype = LLAISYS_DTYPE_F32;
    llaisysDeviceType_t device = LLAISYS_DEVICE_CPU;
    int device_id = 0;

    // One big K and V tensor per layer of shape [n_pages * page_size, nkvh, dh].
    // A page's slice is at offset [page_id * page_size, (page_id+1) * page_size).
    std::vector<llaisysTensor_t> big_k;
    std::vector<llaisysTensor_t> big_v;

    std::vector<Page> pages;
    std::vector<RequestBlock> request_blocks;
    std::deque<int32_t> global_free_list;   // pages with tenant_id == 0
    std::unordered_map<uint64_t, TenantState> tenants;

    std::mutex mu;
    uint64_t lru_counter = 0;
};

namespace {

// Caller must NOT hold pool->mu (D2D copy can be slow on CUDA).
void copy_page(LlaisysQwen2PagedPool *pool, int32_t src_page_id, int32_t dst_page_id) {
    const size_t src_off_tokens = static_cast<size_t>(src_page_id) * pool->page_size;
    const size_t dst_off_tokens = static_cast<size_t>(dst_page_id) * pool->page_size;
    const size_t row_bytes = pool->nkvh * pool->dh * dtype_size(pool->dtype);
    const size_t bytes_per_layer = pool->page_size * row_bytes;
    if (bytes_per_layer == 0) return;

    if (pool->device == LLAISYS_DEVICE_CPU) {
        for (size_t l = 0; l < pool->nlayer; ++l) {
            std::byte *kbase = static_cast<std::byte *>(tensorGetData(pool->big_k[l]));
            std::byte *vbase = static_cast<std::byte *>(tensorGetData(pool->big_v[l]));
            std::memcpy(kbase + dst_off_tokens * row_bytes,
                        kbase + src_off_tokens * row_bytes, bytes_per_layer);
            std::memcpy(vbase + dst_off_tokens * row_bytes,
                        vbase + src_off_tokens * row_bytes, bytes_per_layer);
        }
    } else {
        const auto *rt = llaisysGetRuntimeAPI(pool->device);
        for (size_t l = 0; l < pool->nlayer; ++l) {
            std::byte *kbase = static_cast<std::byte *>(tensorGetData(pool->big_k[l]));
            std::byte *vbase = static_cast<std::byte *>(tensorGetData(pool->big_v[l]));
            rt->memcpy_sync(kbase + dst_off_tokens * row_bytes,
                            kbase + src_off_tokens * row_bytes,
                            bytes_per_layer, LLAISYS_MEMCPY_D2D);
            rt->memcpy_sync(vbase + dst_off_tokens * row_bytes,
                            vbase + src_off_tokens * row_bytes,
                            bytes_per_layer, LLAISYS_MEMCPY_D2D);
        }
    }
}

// Caller must NOT hold pool->mu (memset can be slow on CUDA).
void wipe_page(LlaisysQwen2PagedPool *pool, int32_t page_id) {
    const size_t offset_tokens = static_cast<size_t>(page_id) * pool->page_size;
    const size_t bytes_per_layer =
        pool->page_size * pool->nkvh * pool->dh * dtype_size(pool->dtype);
    if (bytes_per_layer == 0) return;

    if (pool->device == LLAISYS_DEVICE_CPU) {
        for (size_t l = 0; l < pool->nlayer; ++l) {
            std::byte *kp = static_cast<std::byte *>(tensorGetData(pool->big_k[l]))
                            + offset_tokens * pool->nkvh * pool->dh * dtype_size(pool->dtype);
            std::byte *vp = static_cast<std::byte *>(tensorGetData(pool->big_v[l]))
                            + offset_tokens * pool->nkvh * pool->dh * dtype_size(pool->dtype);
            std::memset(kp, 0, bytes_per_layer);
            std::memset(vp, 0, bytes_per_layer);
        }
    } else {
        void *zeros = std::calloc(1, bytes_per_layer);
        const auto *rt = llaisysGetRuntimeAPI(pool->device);
        for (size_t l = 0; l < pool->nlayer; ++l) {
            std::byte *kp = static_cast<std::byte *>(tensorGetData(pool->big_k[l]))
                            + offset_tokens * pool->nkvh * pool->dh * dtype_size(pool->dtype);
            std::byte *vp = static_cast<std::byte *>(tensorGetData(pool->big_v[l]))
                            + offset_tokens * pool->nkvh * pool->dh * dtype_size(pool->dtype);
            rt->memcpy_sync(kp, zeros, bytes_per_layer, LLAISYS_MEMCPY_H2D);
            rt->memcpy_sync(vp, zeros, bytes_per_layer, LLAISYS_MEMCPY_H2D);
        }
        std::free(zeros);
    }
}

// Caller holds mu. Defaults to 25% of pool when SetTenantQuota hasn't run.
TenantState &get_or_init_tenant(LlaisysQwen2PagedPool *pool, uint64_t tenant_id) {
    auto it = pool->tenants.find(tenant_id);
    if (it == pool->tenants.end()) {
        TenantState ts;
        ts.tenant_id = tenant_id;
        ts.max_pages = std::max<size_t>(1, pool->n_pages / 4);
        ts.burst_pages = ts.max_pages / 2;
        ts.reservation_floor = 0;
        it = pool->tenants.emplace(tenant_id, std::move(ts)).first;
    }
    return it->second;
}

// Caller holds mu.
void chain_index_remove_page(LlaisysQwen2PagedPool *pool, uint64_t tenant_id, int32_t page_id) {
    auto it = pool->tenants.find(tenant_id);
    if (it == pool->tenants.end()) return;
    auto &index = it->second.hash_to_page;
    Page &p = pool->pages[page_id];
    if (p.chain_hash != 0) {
        auto e = index.find(p.chain_hash);
        if (e != index.end() && e->second == page_id) {
            index.erase(e);
        }
    }
    p.chain_hash = 0;
    p.sealed = false;
}

// Allocate one page for `tenant_id`. Caller holds mu. Order:
//   (1) tenant's own LRU free, (2) global free list,
//   (3) evict another tenant's pages above their reservation_floor.
// need_wipe is set iff path (3) crossed a tenant boundary.
struct AllocResult {
    int32_t page_id = -1;
    bool need_wipe = false;
    bool from_other_tenant = false;
};

AllocResult alloc_one_page(LlaisysQwen2PagedPool *pool, uint64_t tenant_id) {
    AllocResult r{};
    TenantState &ts = get_or_init_tenant(pool, tenant_id);
    if (ts.pages_in_use >= ts.max_pages) {
        return r; // hard quota
    }

    // (1) tenant's own LRU free.
    if (!ts.lru_free.empty()) {
        r.page_id = ts.lru_free.front();
        ts.lru_free.pop_front();
        Page &p = pool->pages[r.page_id];
        p.ref_count = 0;
        p.last_used_seq = ++pool->lru_counter;
        return r;
    }

    // (2) global free list.
    if (!pool->global_free_list.empty()) {
        r.page_id = pool->global_free_list.front();
        pool->global_free_list.pop_front();
        Page &p = pool->pages[r.page_id];
        p.tenant_id = tenant_id;
        p.last_used_seq = ++pool->lru_counter;
        ts.pages_in_use++;
        return r;
    }

    // (3) evict another tenant's LRU burst page (above their floor).
    int32_t victim = -1;
    uint64_t victim_seq = UINT64_MAX;
    uint64_t victim_owner = 0;
    for (auto &kv : pool->tenants) {
        if (kv.first == tenant_id) continue;
        TenantState &ots = kv.second;
        if (ots.pages_in_use <= ots.reservation_floor) continue;
        for (int32_t pid : ots.lru_free) {
            const Page &p = pool->pages[pid];
            if (p.last_used_seq < victim_seq) {
                victim_seq = p.last_used_seq;
                victim = pid;
                victim_owner = kv.first;
            }
        }
    }
    if (victim < 0) return r;

    {
        TenantState &ots = pool->tenants.at(victim_owner);
        auto it = std::find(ots.lru_free.begin(), ots.lru_free.end(), victim);
        if (it != ots.lru_free.end()) ots.lru_free.erase(it);
        chain_index_remove_page(pool, victim_owner, victim);
        ots.pages_in_use--;
    }

    Page &p = pool->pages[victim];
    p.tenant_id = tenant_id;
    p.last_used_seq = ++pool->lru_counter;
    ts.pages_in_use++;
    r.page_id = victim;
    r.need_wipe = true;
    r.from_other_tenant = true;
    return r;
}

} // anonymous namespace

extern "C" {

__export struct LlaisysQwen2PagedPool *llaisysQwen2PagedPoolCreate(
    struct LlaisysQwen2Model *model,
    size_t n_pages, size_t page_size,
    size_t max_pages_per_request) {
    if (!model || n_pages == 0 || page_size == 0 || max_pages_per_request == 0) {
        return nullptr;
    }
    auto *pool = new LlaisysQwen2PagedPool();
    pool->model = model;
    pool->page_size = page_size;
    pool->n_pages = n_pages;
    pool->nlayer = model->meta.nlayer;
    pool->nkvh = model->meta.nkvh;
    pool->dh = model->meta.dh;
    pool->max_pages_per_request = max_pages_per_request;
    pool->dtype = model->meta.dtype;
    pool->device = model->device_type;
    pool->device_id = model->device_id;

    pool->big_k.reserve(pool->nlayer);
    pool->big_v.reserve(pool->nlayer);
    size_t shape[3] = {n_pages * page_size, pool->nkvh, pool->dh};
    for (size_t l = 0; l < pool->nlayer; ++l) {
        pool->big_k.push_back(tensorCreate(shape, 3, pool->dtype, pool->device, pool->device_id));
        pool->big_v.push_back(tensorCreate(shape, 3, pool->dtype, pool->device, pool->device_id));
    }
    // Zero-init so first-time reads never see garbage.
    const size_t total_bytes = n_pages * page_size * pool->nkvh * pool->dh * dtype_size(pool->dtype);
    if (total_bytes > 0) {
        if (pool->device == LLAISYS_DEVICE_CPU) {
            for (size_t l = 0; l < pool->nlayer; ++l) {
                std::memset(tensorGetData(pool->big_k[l]), 0, total_bytes);
                std::memset(tensorGetData(pool->big_v[l]), 0, total_bytes);
            }
        } else {
            void *zeros = std::calloc(1, total_bytes);
            const auto *rt = llaisysGetRuntimeAPI(pool->device);
            for (size_t l = 0; l < pool->nlayer; ++l) {
                rt->memcpy_sync(tensorGetData(pool->big_k[l]), zeros, total_bytes, LLAISYS_MEMCPY_H2D);
                rt->memcpy_sync(tensorGetData(pool->big_v[l]), zeros, total_bytes, LLAISYS_MEMCPY_H2D);
            }
            std::free(zeros);
        }
    }

    pool->pages.resize(n_pages);
    for (size_t i = 0; i < n_pages; ++i) {
        pool->global_free_list.push_back(static_cast<int32_t>(i));
    }
    return pool;
}

__export void llaisysQwen2PagedPoolDestroy(struct LlaisysQwen2PagedPool *pool) {
    if (!pool) return;
    for (auto t : pool->big_k) tensorDestroy(t);
    for (auto t : pool->big_v) tensorDestroy(t);
    delete pool;
}

__export void llaisysQwen2PagedPoolSetTenantQuota(
    struct LlaisysQwen2PagedPool *pool, uint64_t tenant_id,
    size_t reservation_floor, size_t max_pages, size_t burst_pages) {
    if (!pool || tenant_id == 0) return;
    std::lock_guard<std::mutex> guard(pool->mu);
    TenantState &ts = get_or_init_tenant(pool, tenant_id);
    ts.reservation_floor = reservation_floor;
    ts.max_pages = max_pages == 0 ? SIZE_MAX : max_pages;
    ts.burst_pages = burst_pages;
}

__export int32_t llaisysQwen2PagedPoolAcquire(
    struct LlaisysQwen2PagedPool *pool,
    uint64_t tenant_id,
    const int64_t *prefix_tokens, size_t nprefix,
    size_t *matched_prefix_len) {
    if (matched_prefix_len) *matched_prefix_len = 0;
    if (!pool || tenant_id == 0) return -1;

    int32_t block_id = -1;
    std::vector<int32_t> wipe_pages;  // pages we need to zero outside the lock

    {
        std::lock_guard<std::mutex> guard(pool->mu);

        // Reuse a freed RequestBlock slot if any.
        for (size_t i = 0; i < pool->request_blocks.size(); ++i) {
            if (!pool->request_blocks[i].in_use && pool->request_blocks[i].page_table.empty()) {
                block_id = static_cast<int32_t>(i);
                break;
            }
        }
        if (block_id < 0) {
            block_id = static_cast<int32_t>(pool->request_blocks.size());
            pool->request_blocks.emplace_back();
        }

        RequestBlock &rb = pool->request_blocks[block_id];
        rb = RequestBlock{};
        rb.tenant_id = tenant_id;
        rb.in_use = true;

        // Prefix lookup via chain index; matched pages are ref++ (lift back
        // from lru_free if needed). Unmatched tail is left to Append.
        TenantState &ts = get_or_init_tenant(pool, tenant_id);
        size_t matched_pages = 0;
        uint64_t h = 0;
        size_t off = 0;
        while (off + pool->page_size <= nprefix) {
            h = chain_hash(h, prefix_tokens + off, pool->page_size);
            auto it = ts.hash_to_page.find(h);
            if (it == ts.hash_to_page.end()) break;
            int32_t page_id = it->second;
            Page &p = pool->pages[page_id];
            if (p.tenant_id != tenant_id || !p.sealed) break;
            if (p.ref_count == 0) {
                auto fit = std::find(ts.lru_free.begin(), ts.lru_free.end(), page_id);
                if (fit != ts.lru_free.end()) ts.lru_free.erase(fit);
            }
            p.ref_count++;
            p.last_used_seq = ++pool->lru_counter;
            rb.page_table.push_back(page_id);
            matched_pages++;
            off += pool->page_size;
        }

        rb.pos = matched_pages * pool->page_size;
        if (matched_prefix_len) *matched_prefix_len = rb.pos;
    } // mutex released

    for (int32_t pid : wipe_pages) wipe_page(pool, pid);
    return block_id;
}

__export void llaisysQwen2PagedPoolRelease(
    struct LlaisysQwen2PagedPool *pool, int32_t block_id) {
    if (!pool || block_id < 0 || block_id >= static_cast<int32_t>(pool->request_blocks.size()))
        return;
    std::lock_guard<std::mutex> guard(pool->mu);
    RequestBlock &rb = pool->request_blocks[block_id];
    if (!rb.in_use) return;

    auto tenant_it = pool->tenants.find(rb.tenant_id);
    for (int32_t pid : rb.page_table) {
        Page &p = pool->pages[pid];
        if (p.ref_count > 0) p.ref_count--;
        if (p.ref_count == 0 && tenant_it != pool->tenants.end()) {
            tenant_it->second.lru_free.push_back(pid);
            p.last_used_seq = ++pool->lru_counter;
        }
    }
    rb.in_use = false;
    rb.page_table.clear();
    rb.prefix.clear();
    rb.pos = 0;
}

__export int32_t llaisysQwen2PagedPoolAppend(
    struct LlaisysQwen2PagedPool *pool, int32_t block_id,
    size_t n_new_tokens,
    int32_t *out_slot_mapping) {
    if (!pool || block_id < 0 || block_id >= static_cast<int32_t>(pool->request_blocks.size()))
        return -1;
    if (n_new_tokens == 0) return 0;

    std::vector<int32_t> wipe_pages;
    // (src, dst) deferred to outside the mutex to keep slow D2D copies off
    // the critical path.
    std::vector<std::pair<int32_t, int32_t>> cow_copies;

    {
        std::lock_guard<std::mutex> guard(pool->mu);
        RequestBlock &rb = pool->request_blocks[block_id];
        if (!rb.in_use) return -1;

        size_t cursor = rb.pos;
        for (size_t i = 0; i < n_new_tokens; ++i, ++cursor) {
            const size_t logical_page = cursor / pool->page_size;
            const size_t offset_in_page = cursor % pool->page_size;

            // Ensure the page exists; allocate if needed.
            if (logical_page >= rb.page_table.size()) {
                if (rb.page_table.size() >= pool->max_pages_per_request) return -1;
                AllocResult ar = alloc_one_page(pool, rb.tenant_id);
                if (ar.page_id < 0) return -1;
                Page &p = pool->pages[ar.page_id];
                p.ref_count = 1;
                p.sealed = false;
                p.chain_hash = 0;
                rb.page_table.push_back(ar.page_id);
                if (ar.need_wipe) wipe_pages.push_back(ar.page_id);
            }

            // CoW: fork a shared page before writing.
            int32_t page_id = rb.page_table[logical_page];
            Page &p = pool->pages[page_id];
            if (p.ref_count > 1) {
                AllocResult ar = alloc_one_page(pool, rb.tenant_id);
                if (ar.page_id < 0) return -1;
                Page &np = pool->pages[ar.page_id];
                np.ref_count = 1;
                np.sealed = false;
                np.chain_hash = 0;

                // copy_page overwrites every byte → ar.need_wipe is moot.
                cow_copies.emplace_back(page_id, ar.page_id);

                p.ref_count--;
                if (p.ref_count == 0) {
                    pool->tenants.at(rb.tenant_id).lru_free.push_back(page_id);
                }
                rb.page_table[logical_page] = ar.page_id;
                page_id = ar.page_id;
            }

            if (out_slot_mapping) {
                out_slot_mapping[i] =
                    (static_cast<int32_t>(page_id) << 16) |
                    (static_cast<int32_t>(offset_in_page) & 0xFFFF);
            }
        }
        rb.pos = cursor;
    } // mutex released

    for (int32_t pid : wipe_pages) wipe_page(pool, pid);
    for (auto &c : cow_copies) copy_page(pool, c.first, c.second);
    return 0;
}

__export void llaisysQwen2PagedPoolCommit(
    struct LlaisysQwen2PagedPool *pool, int32_t block_id,
    size_t new_pos,
    const int64_t *tokens, size_t ntokens) {
    if (!pool || block_id < 0 || block_id >= static_cast<int32_t>(pool->request_blocks.size()))
        return;
    std::lock_guard<std::mutex> guard(pool->mu);
    RequestBlock &rb = pool->request_blocks[block_id];
    if (!rb.in_use) return;
    rb.pos = new_pos;

    if (tokens && ntokens > 0) {
        const size_t keep = std::min(ntokens, kPrefixTrack);
        rb.prefix.assign(tokens, tokens + keep);
    }

    // Seal each fully-filled, write-owned page and register its chain hash.
    if (!rb.prefix.empty()) {
        TenantState &ts = get_or_init_tenant(pool, rb.tenant_id);
        uint64_t h = 0;
        size_t off = 0;
        for (size_t pi = 0; pi < rb.page_table.size(); ++pi) {
            const size_t end_in_block = (pi + 1) * pool->page_size;
            if (end_in_block > rb.prefix.size()) break;   // prefix doesn't cover this page
            if (end_in_block > rb.pos) break;             // page not committed yet
            h = chain_hash(h, rb.prefix.data() + off, pool->page_size);
            off = end_in_block;
            int32_t page_id = rb.page_table[pi];
            Page &p = pool->pages[page_id];
            if (!p.sealed) {
                p.chain_hash = h;
                p.sealed = true;
                ts.hash_to_page.emplace(h, page_id);  // first-writer wins
            }
        }
    }
}

__export size_t llaisysQwen2PagedPoolBlockPos(
    struct LlaisysQwen2PagedPool *pool, int32_t block_id) {
    if (!pool || block_id < 0 || block_id >= static_cast<int32_t>(pool->request_blocks.size()))
        return 0;
    std::lock_guard<std::mutex> guard(pool->mu);
    return pool->request_blocks[block_id].pos;
}

__export size_t llaisysQwen2PagedPoolPageTable(
    struct LlaisysQwen2PagedPool *pool, int32_t block_id, int32_t *out_pages) {
    if (!pool || block_id < 0 || block_id >= static_cast<int32_t>(pool->request_blocks.size()))
        return 0;
    std::lock_guard<std::mutex> guard(pool->mu);
    const auto &pt = pool->request_blocks[block_id].page_table;
    if (out_pages) {
        for (size_t i = 0; i < pt.size(); ++i) out_pages[i] = pt[i];
    }
    return pt.size();
}

__export llaisysTensor_t llaisysQwen2PagedPoolPageK(
    struct LlaisysQwen2PagedPool *pool, int32_t page_id, size_t layer) {
    if (!pool || page_id < 0 || page_id >= static_cast<int32_t>(pool->n_pages)) return nullptr;
    if (layer >= pool->nlayer) return nullptr;
    const size_t start = static_cast<size_t>(page_id) * pool->page_size;
    const size_t end = start + pool->page_size;
    return tensorSlice(pool->big_k[layer], 0, start, end);
}

__export llaisysTensor_t llaisysQwen2PagedPoolPageV(
    struct LlaisysQwen2PagedPool *pool, int32_t page_id, size_t layer) {
    if (!pool || page_id < 0 || page_id >= static_cast<int32_t>(pool->n_pages)) return nullptr;
    if (layer >= pool->nlayer) return nullptr;
    const size_t start = static_cast<size_t>(page_id) * pool->page_size;
    const size_t end = start + pool->page_size;
    return tensorSlice(pool->big_v[layer], 0, start, end);
}

__export llaisysTensor_t llaisysQwen2PagedPoolBigK(
    struct LlaisysQwen2PagedPool *pool, size_t layer) {
    if (!pool || layer >= pool->nlayer) return nullptr;
    return pool->big_k[layer];
}
__export llaisysTensor_t llaisysQwen2PagedPoolBigV(
    struct LlaisysQwen2PagedPool *pool, size_t layer) {
    if (!pool || layer >= pool->nlayer) return nullptr;
    return pool->big_v[layer];
}

__export size_t llaisysQwen2PagedPoolNumPages(struct LlaisysQwen2PagedPool *pool) {
    return pool ? pool->n_pages : 0;
}
__export size_t llaisysQwen2PagedPoolPageSize(struct LlaisysQwen2PagedPool *pool) {
    return pool ? pool->page_size : 0;
}
__export size_t llaisysQwen2PagedPoolTenantPagesUsed(
    struct LlaisysQwen2PagedPool *pool, uint64_t tenant_id) {
    if (!pool || tenant_id == 0) return 0;
    std::lock_guard<std::mutex> guard(pool->mu);
    auto it = pool->tenants.find(tenant_id);
    return it == pool->tenants.end() ? 0 : it->second.pages_in_use;
}
__export size_t llaisysQwen2PagedPoolGlobalPagesFree(struct LlaisysQwen2PagedPool *pool) {
    if (!pool) return 0;
    std::lock_guard<std::mutex> guard(pool->mu);
    return pool->global_free_list.size();
}

} // extern "C"

// ----- Scatter K/V into pages -----
// CUDA path lives in qwen2_paged_pool_scatter.cu (under ENABLE_NVIDIA_API).
#ifdef ENABLE_NVIDIA_API
namespace llaisys::paged {
void scatter_kv_cuda(
    void *big_k, void *big_v,
    const void *src_k, const void *src_v,
    const int32_t *slot_mapping,
    size_t n_tokens, size_t nkvh, size_t dh, size_t page_size,
    size_t elem_size,
    llaisysDeviceType_t device);
}
#endif

extern "C" {

__export int32_t llaisysQwen2PagedPoolScatterKV(
    struct LlaisysQwen2PagedPool *pool,
    size_t layer,
    llaisysTensor_t k_new,
    llaisysTensor_t v_new,
    const int32_t *slot_mapping,
    size_t n_tokens) {
    if (!pool || !k_new || !v_new || !slot_mapping) return -1;
    if (layer >= pool->nlayer) return -1;
    if (n_tokens == 0) return 0;

    const size_t row_bytes = pool->nkvh * pool->dh * dtype_size(pool->dtype);
    void *big_k = tensorGetData(pool->big_k[layer]);
    void *big_v = tensorGetData(pool->big_v[layer]);
    void *src_k = tensorGetData(k_new);
    void *src_v = tensorGetData(v_new);

#ifdef ENABLE_NVIDIA_API
    if (pool->device == LLAISYS_DEVICE_NVIDIA) {
        llaisys::paged::scatter_kv_cuda(
            big_k, big_v, src_k, src_v, slot_mapping,
            n_tokens, pool->nkvh, pool->dh, pool->page_size,
            dtype_size(pool->dtype), pool->device);
        return 0;
    }
#endif

    if (pool->device != LLAISYS_DEVICE_CPU) {
        std::cerr << "[ERROR] PagedPool::ScatterKV: unsupported device "
                  << static_cast<int>(pool->device) << std::endl;
        return -1;
    }
    auto *kp = static_cast<std::byte *>(big_k);
    auto *vp = static_cast<std::byte *>(big_v);
    auto *sk = static_cast<const std::byte *>(src_k);
    auto *sv = static_cast<const std::byte *>(src_v);
    for (size_t i = 0; i < n_tokens; ++i) {
        const int32_t slot = slot_mapping[i];
        const size_t page_id = static_cast<size_t>((slot >> 16) & 0xFFFF);
        const size_t offset = static_cast<size_t>(slot & 0xFFFF);
        if (page_id >= pool->n_pages || offset >= pool->page_size) {
            std::cerr << "[ERROR] PagedPool::ScatterKV: bad slot 0x"
                      << std::hex << slot << std::dec
                      << " (page=" << page_id << ", offset=" << offset << ")"
                      << std::endl;
            return -1;
        }
        const size_t dst_off = (page_id * pool->page_size + offset) * row_bytes;
        const size_t src_off = i * row_bytes;
        std::memcpy(kp + dst_off, sk + src_off, row_bytes);
        std::memcpy(vp + dst_off, sv + src_off, row_bytes);
    }
    return 0;
}

} // extern "C"
