# 
# minimalistic implementation of Flash Attention 3 for learning purposes.
# 
# what does this lack?
# - no attention masking
# - no persistent kernel scheduling
# - no GQA

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu.warpgroup import OperandSource
import cutlass.torch as cutlass_torch
import cutlass.pipeline as pipeline
import torch
import math
import cuda.bindings.driver as cuda
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from typing import List, Type, Tuple
import csv
import numpy as np

REDUCE_ALL_COORD = 0

def transpose(tensor: cutlass.Tensor, new_indices: List[int]):
    shape = tuple(tensor.shape[i] for i in new_indices)
    stride = tuple(tensor.layout.stride[i] for i in new_indices)
    return cute.make_tensor(tensor.iterator, cute.make_layout(shape, stride=stride))

# H100 kernel for Flash Attention 3
class HopperFA2:
    def __init__(
        self,
        cta_tile: Tuple[int, int, int],
    ):
        # tiler for the first part.
        self.cta_tile = cta_tile
        self.kv_pipeline_stages = 4
        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(
        self,
        q: cutlass.Tensor, # (B, H_qo, S_qo, D) [DANGER: transpose later on]
        k: cutlass.Tensor, # (B, H_kv, S_kv, D) [DANGER: transpose later on]
        v: cutlass.Tensor, # (B, H_kv, S_kv, D) [DANGER: transpose later on]
        o: cutlass.Tensor, # (B, H_qo, S_qo, D) [DANGER: transpose later on]
        stream: cuda.CUstream,
    ):
        q = transpose(q, [2, 3, 1, 0]) # (S_qo, D, H, B)
        k = transpose(k, [2, 3, 1, 0]) # (S_kv, D, H, B)
        v = transpose(v, [3, 2, 1, 0]) # (D, S_kv, H, B)
        o = transpose(o, [2, 3, 1, 0]) # (S_qo, D, H, B)

        q_smem_shape = o_smem_shape = (self.cta_tile[0], self.cta_tile[2])
        k_smem_shape = (self.cta_tile[1], self.cta_tile[2])
        v_smem_shape = (self.cta_tile[2], self.cta_tile[1])

        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type

        def make_smem_layout(smem_shape: Tuple[int, int], dtype: Type[cutlass.Numeric], stages: int, layout = cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128):
            smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
                layout,
                dtype,
            )
            return cute.tile_to_shape(
                smem_layout_atom,
                cute.append(smem_shape, stages),
                order=(0, 1, 2)
            )

        # 1. initialize shared-memory layout.
        self.q_smem_layout_staged = make_smem_layout(q_smem_shape, self.q_dtype, stages=1)
        self.k_smem_layout_staged = make_smem_layout(k_smem_shape, self.k_dtype, self.kv_pipeline_stages)
        self.v_smem_layout_staged = make_smem_layout(v_smem_shape, self.v_dtype, self.kv_pipeline_stages, layout=cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW128)
        self.o_smem_layout_staged = make_smem_layout(o_smem_shape, self.q_dtype, stages=1)
        self.q_smem_layout_unstaged = cute.slice_(self.q_smem_layout_staged, (None, None, 0))
        self.k_smem_layout_unstaged = cute.slice_(self.k_smem_layout_staged, (None, None, 0))
        self.v_smem_layout_unstaged = cute.slice_(self.v_smem_layout_staged, (None, None, 0))
        self.o_smem_layout_unstaged = cute.slice_(self.o_smem_layout_staged, (None, None, 0))

        print("q_smem_layout_staged", self.q_smem_layout_staged)
        print("o_smem_layout_unstaged", self.o_smem_layout_unstaged)

        # 2. setup tma atoms and tensors.
        g2s_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        s2g_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        q_tma_atom, q_tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s_op,
            q,
            self.q_smem_layout_unstaged,
            (self.cta_tile[0], self.cta_tile[2])
        )
        k_tma_atom, k_tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s_op,
            k,
            self.k_smem_layout_unstaged,
            (self.cta_tile[1], self.cta_tile[2])
        )
        v_tma_atom, v_tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s_op,
            v,
            self.v_smem_layout_unstaged,
            (self.cta_tile[2], self.cta_tile[1])
        )
        o_tma_atom, o_tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            s2g_op,
            o,
            self.o_smem_layout_unstaged,
            (self.cta_tile[0], self.cta_tile[2])
        )

        self.o_layout = utils.LayoutEnum.from_tensor(o)


        # 4. setup shared storage.

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(self.q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(self.k_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(self.v_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            mi_ptr: cute.struct.MemRange[cutlass.Int32, self.cta_tile[0]]
            li_ptr: cute.struct.MemRange[cutlass.Int32, self.cta_tile[0]]
            q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            kv_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.kv_pipeline_stages * 2]

        self.shared_storage = SharedStorage

        self.load_warp_id = 0
        self.mma_warp_ids = [4, 5, 6, 7]

        self.atom_layout_mnk = (1, 1, 1)
        self.acc_dtype = cutlass.Float32
        self.tiled_mma_qk = sm90_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.k_dtype,
            cute.nvgpu.warpgroup.OperandMajorMode.K,
            cute.nvgpu.warpgroup.OperandMajorMode.K,
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(self.cta_tile[0], self.cta_tile[1]),
        )

        self.tiled_mma_v = sm90_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.v_dtype,
            cute.nvgpu.warpgroup.OperandMajorMode.K,
            cute.nvgpu.warpgroup.OperandMajorMode.MN,
            self.acc_dtype,
            self.atom_layout_mnk,
            # gemm(P, V.T)
            tiler_mn=(self.cta_tile[1], self.cta_tile[2]),
            a_source=OperandSource.RMEM,
        )

        # num q blocks
        num_q_blocks = cute.size(cute.tiled_divide(q, (self.cta_tile[0], self.cta_tile[2])), mode=[1])
        # num kv blocks
        num_heads = cute.size(q, mode=[2])
        num_batches = cute.size(q, mode=[3])

        self.kernel(
            q_tma_tensor,
            k_tma_tensor,
            v_tma_tensor,
            o_tma_tensor,
            self.q_smem_layout_staged,
            self.k_smem_layout_staged,
            self.v_smem_layout_staged,
            self.o_smem_layout_staged,
            q_tma_atom,
            k_tma_atom,
            v_tma_atom,
            o_tma_atom,
            self.tiled_mma_qk,
            self.tiled_mma_v,
        ).launch(
            grid=[num_q_blocks, num_heads, num_batches],
            block=[256, 1, 1],
            cluster=[1, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        gQ: cutlass.Tensor,
        gK: cutlass.Tensor,
        gV: cutlass.Tensor,
        gO: cutlass.Tensor,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
        q_tma_atom: cute.Atom,
        k_tma_atom: cute.Atom,
        v_tma_atom: cute.Atom,
        o_tma_atom: cute.Atom,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_v: cute.TiledMma,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(q_tma_atom)
            cute.nvgpu.cpasync.prefetch_descriptor(k_tma_atom)
            cute.nvgpu.cpasync.prefetch_descriptor(v_tma_atom)
            cute.nvgpu.cpasync.prefetch_descriptor(o_tma_atom)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sQ: cute.Tensor = storage.sQ.get_tensor(
            q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner
        )
        sK: cute.Tensor = storage.sK.get_tensor(
            k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner
        )
        sV: cute.Tensor = storage.sV.get_tensor(
            v_smem_layout_staged.outer, swizzle=v_smem_layout_staged.inner
        )

        sO_ptr = cute.recast_ptr(
            sQ.iterator, o_smem_layout_staged.inner, dtype=self.o_dtype
        )
        sO = cute.make_tensor(sO_ptr, o_smem_layout_staged.outer)

        UNUSED = 1
        load_q_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len([self.load_warp_id]))
        load_q_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, UNUSED)
        load_q_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=1,
            producer_group=load_q_producer_group,
            consumer_group=load_q_consumer_group,
            tx_count=cute.size_in_bytes(self.q_dtype, cute.slice_(q_smem_layout_staged, (None, None, 0))),
            barrier_storage=storage.q_mbar_ptr.data_ptr(),
        )

        load_kv_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len([self.load_warp_id]))
        load_kv_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len(self.mma_warp_ids))
        load_kv_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.kv_pipeline_stages,
            producer_group=load_kv_producer_group,
            consumer_group=load_kv_consumer_group,
            tx_count=cute.size_in_bytes(self.k_dtype, cute.slice_(k_smem_layout_staged, (None, None, 0))) + cute.size_in_bytes(self.v_dtype, cute.slice_(v_smem_layout_staged, (None, None, 0))),
            barrier_storage=storage.kv_mbar_ptr.data_ptr(),
        )

        gQ_tiled = cute.flat_divide(gQ, (self.cta_tile[0], self.cta_tile[2])) # (S_qo, D, H_qo, B)
        gK_tiled = cute.flat_divide(gK, (self.cta_tile[1], self.cta_tile[2])) # (S_kv, D, H_kv, B)
        gV_tiled = cute.flat_divide(gV, (self.cta_tile[2], self.cta_tile[1])) # (D, S_kv, H_kv, B)
        gO_tiled = cute.flat_divide(gO, (self.cta_tile[0], self.cta_tile[2])) # (S_qo, D, H_qo, B)

        print("gV.shape", gV.shape)
        print("cta.tiler", (self.cta_tile[2], self.cta_tile[1]))
        print("gV_tiled.shape", gV_tiled.shape)

        gQ_local = gQ_tiled[*(None, None), *(bidx, None, bidy, bidz)] # (tile_M, tile_N, 1)
        gK_local = gK_tiled[*(None, None), *(None, 0, bidy, bidz)] # (tile_M, tile_N, S_kv_blocks)
        gV_local = gV_tiled[*(None, None), *(0, None, bidy, bidz)] # (tile_M, tile_N, S_kv_blocks)
        gO_local = gO_tiled[*(None, None), *(bidx, None, bidy, bidz)] # (tile_M, tile_N, 1)

        print("gK_local", gK_local)
        print("gV_local", gV_local)

        # load q into memory.
        is_producer_warp = warp_idx in [self.load_warp_id]
        is_consumer_warp = warp_idx in self.mma_warp_ids

        # partition tensors for TMA
        cluster_shape_mnk = (1, 1, 1)
        cta_layout = cute.make_layout(cluster_shape_mnk)
        cta_layout = cute.make_layout(cute.slice_(cta_layout, (0, None, 0)).shape)
        cta_crd = (0,)

        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
            q_tma_atom,
            cta_crd,
            cta_layout,
            cute.group_modes(sQ, 0, 2),
            cute.group_modes(gQ_local, 0, 2),
        )
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
            k_tma_atom,
            cta_crd,
            cta_layout,
            cute.group_modes(sK, 0, 2),
            cute.group_modes(gK_local, 0, 2),
        )
        tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
            v_tma_atom,
            cta_crd,
            cta_layout,
            cute.group_modes(sV, 0, 2),
            cute.group_modes(gV_local, 0, 2),
        )

        num_kv_blocks = cute.size(gK_local, mode=[2])

        if is_producer_warp:
            # load q 
            load_q_pipeline_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            load_q_pipeline.producer_acquire(load_q_pipeline_state)
            cute.copy(
                q_tma_atom,
                tQgQ[(None, 0)],
                tQsQ[(None, 0)],
                tma_bar_ptr=load_q_pipeline.producer_get_barrier(load_q_pipeline_state)
            )

            print("tKsK", tKsK)
            print("tVsV", tVsV)
            print("tKgK", tKgK)
            print("tVgV", tVgV)

            load_kv_pipeline_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_pipeline_stages)
            for i in cutlass.range(num_kv_blocks, unroll=1):
                load_kv_pipeline.producer_acquire(load_kv_pipeline_state)

                sK_chunk = tKsK[(None, load_kv_pipeline_state.index)]
                sV_chunk = tVsV[(None, load_kv_pipeline_state.index)]
                gK_chunk = tKgK[(None, load_kv_pipeline_state.count)]
                gV_chunk = tVgV[(None, load_kv_pipeline_state.count)]

                cute.copy(
                    k_tma_atom,
                    gK_chunk,
                    sK_chunk,
                    tma_bar_ptr=load_kv_pipeline.producer_get_barrier(load_kv_pipeline_state)
                )
                cute.copy(
                    v_tma_atom,
                    gV_chunk,
                    sV_chunk,
                    tma_bar_ptr=load_kv_pipeline.producer_get_barrier(load_kv_pipeline_state)
                )
                load_kv_pipeline_state.advance()


        qk_thr_mma = tiled_mma_qk.get_slice(tidx)
        tCsK = qk_thr_mma.partition_A(sK)
        tCsQ = qk_thr_mma.partition_B(sQ)
        tCrK = tiled_mma_qk.make_fragment_A(tCsK)
        tCrQ = tiled_mma_qk.make_fragment_B(tCsQ)
        qk_acc_shape = qk_thr_mma.partition_shape_C(
            (self.cta_tile[0], self.cta_tile[1])
        )
        tStS = cute.make_fragment(qk_acc_shape, self.acc_dtype)

        # note that we only properly partition B and C.
        # for A, we will just re-format the output later on.
        v_thr_mma = tiled_mma_v.get_slice(tidx)
        tCsV = v_thr_mma.partition_B(sV)

        tCrV = tiled_mma_v.make_fragment_B(tCsV)

        o_acc_shape = v_thr_mma.partition_shape_C(
            (self.cta_tile[1], self.cta_tile[2])
        )
        v_A_shape = v_thr_mma.partition_shape_A(
            (self.cta_tile[1], self.cta_tile[1])
        )
        tStO = cute.make_fragment(o_acc_shape, self.acc_dtype)
        tStO.fill(0.0)

        if is_consumer_warp:
            # run mma
            mma_kv_pipeline_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_pipeline_stages)
            mma_q_pipeline_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            load_q_pipeline.consumer_wait(mma_q_pipeline_state)
            load_q_pipeline.consumer_release(mma_q_pipeline_state)
            # each thread contains the accumulator outputs from 2 separate accumulators.
            # we need two register accumulators to store the running max and sum for each of the rows.
            # number of rows in tStS. (this is equivalent to 2 in Hopper)
            old_row_max = cute.make_fragment((tStS.shape[0][0],), cutlass.Float32)
            old_row_max.fill(-cutlass.Float32.inf)
            running_sum = cute.make_fragment_like(old_row_max, cutlass.Float32)
            running_sum.fill(0.0)

            for k_block in cutlass.range(num_kv_blocks, unroll=1):
                load_kv_pipeline.consumer_wait(mma_kv_pipeline_state)
                num_k_microtiles = cute.size(tCrK, mode=[2])

                # M * K = (64, 64)
                # N * K = (64, 128)
                tiled_mma_qk.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                cute.nvgpu.warpgroup.fence()
                for k_microtile_ix in cutlass.range(num_k_microtiles, unroll=1):
                    q_block_coord = (None, None, k_microtile_ix, 0)
                    k_block_coord = (None, None, k_microtile_ix, mma_kv_pipeline_state.index)
                    cute.gemm(
                        tiled_mma_qk,
                        tStS,
                        tCrQ[q_block_coord],
                        tCrK[k_block_coord],
                        tStS,
                    )
                    tiled_mma_qk.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.fence()

                # use a composition to get the desired matrices that you want.
                new_row_max = cute.make_fragment_like(old_row_max, cutlass.Float32)
                scaling_factor = cute.make_fragment_like(old_row_max, cutlass.Float32)

                # divide tStS by \sqrt{D}
                log2_e = math.log2(math.exp(1.0))  
                scale_log2 = 1.0 / math.sqrt(cute.size(gQ, mode=[1])) * log2_e

                for i in cutlass.range_constexpr(cute.size(old_row_max, mode=[0])):
                    row_tensor = tStS[(None, i, None), 0, 0]
                    row_output = tStO[(None, i, None), 0, 0]

                    row_ssa = row_tensor.load()
                    row_max = row_ssa.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, REDUCE_ALL_COORD)
                    for j in cutlass.range_constexpr(2):
                        row_max = cutlass.max(row_max, cute.arch.shuffle_sync_bfly(row_max, 1 << j))

                    new_row_max[i] = cutlass.max(old_row_max[i], row_max)

                    row_p = cute.exp2((row_ssa * scale_log2 - new_row_max[i] * scale_log2))

                    row_tensor.store(row_p)

                    row_sum = row_p.reduce(cute.ReductionOp.ADD, 0.0, REDUCE_ALL_COORD)
                    for j in cutlass.range_constexpr(2):
                        row_sum += cute.arch.shuffle_sync_bfly(row_sum, 1 << j)

                    scaling_factor[i] = cute.arch.exp2((old_row_max[i] * scale_log2 - new_row_max[i] * scale_log2))

                    if k_block != 0:
                        running_sum[i] = running_sum[i] * scaling_factor[i]

                    running_sum[i] += row_sum

                    if k_block != 0:
                        row_output.store(row_output.load() * scaling_factor[i])

                old_row_max.store(new_row_max.load())
                tPtP_fp32 = tStS
                tPtP = cute.make_fragment(tPtP_fp32.layout, self.k_dtype)
                tPtP.store(tPtP_fp32.load().to(self.k_dtype))

                # we re-format the tPtP we have into the shape that we need.
                p_A_layout = cute.make_layout(v_A_shape)
                p_A_tensor = cute.make_tensor(tPtP.iterator, p_A_layout)

                # output scaling
                num_k_microtiles = cute.size(p_A_tensor, mode=[2])
                for k_microtile_ix in cutlass.range(num_k_microtiles, unroll=1):
                    p_block_coord = (None, None, k_microtile_ix)
                    v_block_coord = (None, None, k_microtile_ix, mma_kv_pipeline_state.index)
                    cute.gemm(
                        tiled_mma_v,
                        tStO,
                        p_A_tensor[p_block_coord],
                        tCrV[v_block_coord],
                        tStO,
                    )
                    tiled_mma_v.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

                # output scaling
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.fence()

                mma_kv_pipeline_state.advance()
                load_kv_pipeline.consumer_release(mma_kv_pipeline_state)

            # write back the tile to global memory.
            for i in cutlass.range_constexpr(tStO.shape[0][1]):
                tStO_row = tStO[(None, i, None), 0, 0]
                tStO_row.store(tStO_row.load() / running_sum[i])

            copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                self.o_layout,
                elem_ty_d=self.o_dtype,
                elem_ty_acc=self.o_dtype,
            )

            tiled_copy_r2s = cute.make_tiled_copy_C(copy_atom_r2s, tiled_mma_v)
            # partition
            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx % 128)
            tRS_sD = thr_copy_r2s.partition_D(sO)
            tStO_fp16 = cute.make_fragment(tStO.layout, self.o_dtype)
            tStO_fp16.store(tStO.load().to(self.o_dtype))

            rD_shape = cute.shape(thr_copy_r2s.partition_S(sO))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_fragment_like(tRS_rD_layout, self.o_dtype)

            for i in range(cute.size(tRS_rD)):
                tRS_rD[i] = tStO_fp16[i]

            cute.copy(tiled_copy_r2s, tRS_rD, tRS_sD[(None, None, None, 0)])

            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            cute.arch.barrier()

            # # perform the smem -> gmem copy.
            sepi_for_tma_partition = cute.group_modes(sO, 0, 2)
            tcgc_for_tma_partition = cute.zipped_divide(gO_local, cute.slice_(self.cta_tile, (None, 0, None)))
            bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                o_tma_atom,
                0,
                cute.make_layout(1),
                sepi_for_tma_partition,
                tcgc_for_tma_partition,
            )

            cute.copy(
                o_tma_atom,
                bSG_sD[(None, 0)],
                bSG_gD[(None, 0)],
            )

if __name__ == "__main__":
    head_dims = 64
    batch_size = 3
    num_key_heads = 2
    qo_seq_len = 1024
    kv_seq_len = 1024

    B, S_qo, S_kv, H_kv, H_qo, D = batch_size, qo_seq_len, kv_seq_len, num_key_heads, num_key_heads, head_dims

    q_torch = torch.randn(B, H_qo, S_qo, D, dtype=torch.float16, device="cuda") 
    k_torch = torch.randn(B, H_kv, S_kv, D, dtype=torch.float16, device="cuda")
    v_torch = torch.randn(B, H_qo, S_qo, D, dtype=torch.float16, device="cuda")
    o_torch = torch.randn(B, H_qo, S_qo, D, dtype=torch.float16, device="cuda") / 50

    q = cutlass_torch.from_dlpack(q_torch, assumed_align=16)
    k = cutlass_torch.from_dlpack(k_torch, assumed_align=16)
    v = cutlass_torch.from_dlpack(v_torch, assumed_align=16)
    o = cutlass_torch.from_dlpack(o_torch, assumed_align=16)

    stream = cuda.CUstream()

    # compute the output of the kernel
    hopper_fa = HopperFA2(cta_tile=(64, 64, 64))
    compiled_kernel = cute.compile(hopper_fa, q, k, v, o, stream)
    compiled_kernel(q, k, v, o, stream)

    def save_tensor_to_csv(tensor, filename, D):
        tensor_np = tensor.cpu().numpy()
        tensor_reshaped = tensor_np.reshape(-1, D)
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = [f'dim_{i}' for i in range(D)]
            writer.writerow(header)
            for row in tensor_reshaped:
                writer.writerow(row.tolist())
    
    save_tensor_to_csv(o_torch, 'output_tensor.csv', D)
    o_torch_ref = torch.nn.functional.scaled_dot_product_attention(q_torch, k_torch, v_torch, scale=1.0 / math.sqrt(D))
    
    # Perform all-close check row-by-row and count failures
    failed_rows = 0
    failed_indices = []
    total_rows = 0
    for b in range(B):
        for h in range(H_qo):
            for s in range(S_qo):
                total_rows += 1
                try:
                    torch.testing.assert_close(o_torch[b, h, s, :], o_torch_ref[b, h, s, :], atol=1e-3, rtol=1e-3)
                except AssertionError:
                    failed_rows += 1
                    failed_indices.append(s)
    
    save_tensor_to_csv(o_torch_ref, 'output_tensor_ref.csv', D)

    torch.testing.assert_close(o_torch, o_torch_ref, atol=1e-3, rtol=1e-3)