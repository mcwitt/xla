/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_STREAM_EXECUTOR_GPU_RAGGED_ALL_TO_ALL_DEVICE_KERNEL_LIB_CU_H_
#define XLA_STREAM_EXECUTOR_GPU_RAGGED_ALL_TO_ALL_DEVICE_KERNEL_LIB_CU_H_

#include <cstdint>

#if NCCL_VERSION_CODE >= 22900
#include "third_party/nccl/nccl_device.h"
#endif

namespace stream_executor::gpu {

template <int64_t kSize>
struct alignas(kSize) DeviceVec {
  uint8_t data[kSize];
};

#if NCCL_VERSION_CODE >= 22900

// Flag cells for the CTA-0-delegated LSA barrier (see the kernel's else
// branch). Launches of this kernel on one device are stream-serialized, so a
// plain flag + done-counter pair needs no epoch: CTA 0 resets both to zero
// before it runs the exit barrier, and the next launch cannot start until the
// current one completes.
static __device__ unsigned xla_ragged_dk_entry_flag = 0;
static __device__ unsigned xla_ragged_dk_done_count = 0;

template <int64_t kVectorSize>
struct RaggedAllToAllUpdateMetadata {
  int peer;
  int update;
  int64_t meta_idx;
  int64_t send_size;
  int64_t src_byte_offset;
  int64_t dst_byte_offset;
  int64_t byte_count;
};

template <int64_t kVectorSize>
__device__ bool LoadRaggedAllToAllUpdateMetadata(
    int64_t flat_idx, int64_t num_updates_per_replica, int64_t num_row_elements,
    int64_t input_buffer_offset_bytes, int64_t output_buffer_offset_bytes,
    const int64_t* __restrict__ input_offsets_ptr,
    const int64_t* __restrict__ send_sizes_ptr,
    const int64_t* __restrict__ output_offsets_ptr,
    RaggedAllToAllUpdateMetadata<kVectorSize>* meta) {
  meta->peer = flat_idx / num_updates_per_replica;
  meta->update = flat_idx % num_updates_per_replica;
  meta->meta_idx = meta->peer * num_updates_per_replica + meta->update;
  meta->send_size = send_sizes_ptr[meta->meta_idx];
  if (meta->send_size == 0) {
    return false;
  }

  const int64_t input_offset = input_offsets_ptr[meta->meta_idx];
  const int64_t output_offset = output_offsets_ptr[meta->meta_idx];
  meta->src_byte_offset =
      input_buffer_offset_bytes + input_offset * num_row_elements * kVectorSize;
  meta->dst_byte_offset = output_buffer_offset_bytes +
                          output_offset * num_row_elements * kVectorSize;
  meta->byte_count = meta->send_size * num_row_elements * kVectorSize;
  return true;
}

template <int64_t kVectorSize>
__device__ void RaggedAllToAllCopy(
    ncclWindow_t send_win, ncclWindow_t recv_win,
    const int64_t* __restrict__ input_offsets_ptr,
    const int64_t* __restrict__ send_sizes_ptr,
    const int64_t* __restrict__ output_offsets_ptr,
    int64_t num_updates_per_replica, int64_t num_row_elements,
    int64_t input_buffer_offset_bytes, int64_t output_buffer_offset_bytes,
    int start_lsa, int lsa_size, int num_ranks, ncclGin* gin, ncclTeam world,
    unsigned int signal_index) {
  using T = DeviceVec<kVectorSize>;

  if (lsa_size > 0) {
    const int grid = static_cast<int>(gridDim.x);
    const int64_t total_lsa_updates =
        static_cast<int64_t>(lsa_size) * num_updates_per_replica;
    const int ctas_per_unit =
        max(1, grid / static_cast<int>(total_lsa_updates));
    const int unit_step = grid / ctas_per_unit;
    const int my_unit_start = static_cast<int>(blockIdx.x) / ctas_per_unit;
    const int my_cta_in_unit = static_cast<int>(blockIdx.x) % ctas_per_unit;
    if (my_unit_start < total_lsa_updates) {
      const int unit_tid = my_cta_in_unit * static_cast<int>(blockDim.x) +
                           static_cast<int>(threadIdx.x);
      const int unit_nthreads = ctas_per_unit * static_cast<int>(blockDim.x);

      for (int unit = my_unit_start; unit < total_lsa_updates;
           unit += unit_step) {
        const int64_t flat_idx =
            static_cast<int64_t>(start_lsa) * num_updates_per_replica + unit;
        RaggedAllToAllUpdateMetadata<kVectorSize> meta;
        if (!LoadRaggedAllToAllUpdateMetadata<kVectorSize>(
                flat_idx, num_updates_per_replica, num_row_elements,
                input_buffer_offset_bytes, output_buffer_offset_bytes,
                input_offsets_ptr, send_sizes_ptr, output_offsets_ptr, &meta)) {
          continue;
        }

        const int lsa_peer = unit / num_updates_per_replica;
        const T* src = static_cast<const T*>(
            ncclGetLocalPointer(send_win, meta.src_byte_offset));
        T* dst = static_cast<T*>(
            ncclGetLsaPointer(recv_win, meta.dst_byte_offset, lsa_peer));

        const int64_t num_elements = meta.byte_count / kVectorSize;
        for (int64_t i = unit_tid; i < num_elements; i += unit_nthreads) {
          dst[i] = src[i];
        }
      }
    }
  }

  if (gin == nullptr) {
    return;
  }

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int nthreads = blockDim.x * gridDim.x;
  const int64_t total_updates = num_updates_per_replica * num_ranks;

  for (int64_t flat_idx = tid; flat_idx < total_updates; flat_idx += nthreads) {
    const int peer = flat_idx / num_updates_per_replica;
    if (peer >= start_lsa && peer < start_lsa + lsa_size) {
      continue;
    }

    RaggedAllToAllUpdateMetadata<kVectorSize> meta;
    if (!LoadRaggedAllToAllUpdateMetadata<kVectorSize>(
            flat_idx, num_updates_per_replica, num_row_elements,
            input_buffer_offset_bytes, output_buffer_offset_bytes,
            input_offsets_ptr, send_sizes_ptr, output_offsets_ptr, &meta)) {
      continue;
    }

    gin->put(world, meta.peer, recv_win, meta.dst_byte_offset, send_win,
             meta.src_byte_offset, meta.byte_count,
             ncclGin_SignalInc{signal_index});
  }
}

template <int64_t kVectorSize>
__global__ void __launch_bounds__(128) RaggedAllToAllDeviceKernelImpl(
    struct ncclDevComm dev_comm, ncclWindow_t send_win, ncclWindow_t recv_win,
    const int64_t* __restrict__ input_offsets_ptr,
    const int64_t* __restrict__ send_sizes_ptr,
    const int64_t* __restrict__ output_offsets_ptr,
    int64_t num_updates_per_replica, int64_t num_row_elements,
    int64_t input_buffer_offset_bytes, int64_t output_buffer_offset_bytes) {
  // NCCL device barrier/GIN APIs emit scope-qualified atomics that require
  // sm_60+. Lower architectures compile to an empty stub; the kernel is only
  // launched when the device supports NCCL device comms.
#if __CUDA_ARCH__ >= 600
  ncclTeam world = ncclTeamWorld(dev_comm);
  ncclTeam lsa = ncclTeamLsa(dev_comm);
  const int start_lsa = world.rank - lsa.rank;
  const int lsa_size = lsa.nRanks;
  const int num_ranks = world.nRanks;
  const bool has_remote_peers = (lsa_size < num_ranks);

  if (has_remote_peers) {
    const int gin_context = 0;
    const unsigned int signal_index = 0;

    ncclGin gin{dev_comm, gin_context};
    uint64_t signal_value =
        (blockIdx.x == 0) ? gin.readSignal(signal_index) : 0;

    ncclBarrierSession<ncclCoopCta> bar{ncclCoopCta(), ncclTeamTagWorld(), gin,
                                        blockIdx.x};
    bar.sync(ncclCoopCta(), ::cuda::memory_order_acquire,
             ncclGinFenceLevel::Relaxed);

    RaggedAllToAllCopy<kVectorSize>(
        send_win, recv_win, input_offsets_ptr, send_sizes_ptr,
        output_offsets_ptr, num_updates_per_replica, num_row_elements,
        input_buffer_offset_bytes, output_buffer_offset_bytes, start_lsa,
        lsa_size, num_ranks, &gin, world, signal_index);

    const int num_remote_peers =
        (num_ranks - lsa_size) * num_updates_per_replica;
    if (blockIdx.x == 0) {
      gin.waitSignal(ncclCoopCta(), signal_index,
                     signal_value + num_remote_peers);
    }

    gin.flush(ncclCoopCta());
    bar.sync(ncclCoopCta(), ::cuda::memory_order_release,
             ncclGinFenceLevel::Relaxed);
  } else {
    // Only CTA 0 participates in the cross-rank LSA barrier; the rest of the
    // grid synchronizes with it through device-scope flags. This removes the
    // per-CTA barrier slots whose cost scales with grid size x ranks, so the
    // copy grid can grow past the point where per-CTA sessions win it back.
    if (blockIdx.x == 0) {
      const long long t0 = clock64();
      ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), dev_comm,
                                             ncclTeamTagLsa{}, /*index=*/0};
      bar.sync(ncclCoopCta(), ::cuda::memory_order_relaxed);
      if (threadIdx.x == 0) {
        __threadfence();
        atomicExch(&xla_ragged_dk_entry_flag, 1u);
      }
      __syncthreads();
      const long long t1 = clock64();

      RaggedAllToAllCopy<kVectorSize>(
          send_win, recv_win, input_offsets_ptr, send_sizes_ptr,
          output_offsets_ptr, num_updates_per_replica, num_row_elements,
          input_buffer_offset_bytes, output_buffer_offset_bytes, start_lsa,
          lsa_size, num_ranks, /*gin=*/nullptr, world, /*signal_index=*/0);

      __threadfence_system();
      __syncthreads();
      const long long t2 = clock64();
      if (threadIdx.x == 0) {
        while (atomicAdd(&xla_ragged_dk_done_count, 0u) != gridDim.x - 1) {
        }
        const long long t3 = clock64();
        if (world.rank == 0) {
          printf("DKTIME entry=%lld copy0=%lld donewait=%lld cycles\n",
                 t1 - t0, t2 - t1, t3 - t2);
        }
        // Reset for the next launch before releasing the exit barrier; the
        // stream serializes launches, so no other grid can observe the
        // intermediate state.
        atomicExch(&xla_ragged_dk_entry_flag, 0u);
        atomicExch(&xla_ragged_dk_done_count, 0u);
      }
      __syncthreads();
      // Order the other CTAs' peer-buffer writes (observed via done_count)
      // before the release barrier's cross-rank signal.
      __threadfence_system();
      bar.sync(ncclCoopCta(), ::cuda::memory_order_release);
    } else {
      if (threadIdx.x == 0) {
        while (atomicAdd(&xla_ragged_dk_entry_flag, 0u) == 0u) {
        }
      }
      __syncthreads();

      RaggedAllToAllCopy<kVectorSize>(
          send_win, recv_win, input_offsets_ptr, send_sizes_ptr,
          output_offsets_ptr, num_updates_per_replica, num_row_elements,
          input_buffer_offset_bytes, output_buffer_offset_bytes, start_lsa,
          lsa_size, num_ranks, /*gin=*/nullptr, world, /*signal_index=*/0);

      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0) {
        atomicAdd(&xla_ragged_dk_done_count, 1u);
      }
    }
  }
#endif  // __CUDA_ARCH__ >= 600
}

#else  // NCCL_VERSION_CODE < 22900

template <int64_t kVectorSize>
__global__ void RaggedAllToAllDeviceKernelImpl(
    void* dev_comm, void* send_win, void* recv_win,
    const int64_t* input_offsets_ptr, const int64_t* send_sizes_ptr,
    const int64_t* output_offsets_ptr, int64_t num_updates_per_replica,
    int64_t num_row_elements, int64_t input_buffer_offset_bytes,
    int64_t output_buffer_offset_bytes) {}

#endif  // NCCL_VERSION_CODE >= 22900

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_RAGGED_ALL_TO_ALL_DEVICE_KERNEL_LIB_CU_H_
