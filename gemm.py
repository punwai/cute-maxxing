# 
# Gemm code utilizing the concept of worktiles.
# 

from typing import Tuple
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cuda.bindings.driver as cuda
import torch
import cutlass

WARP_SIZE = 32

class Gemm:
    def __init__(self, problem_shape: Tuple[int, int, int], cta_tile: Tuple[int, int, int]):
        self.problem_shape = problem_shape
        self.cta_tile = cta_tile
        self.pipeline_stages = 3
        self.buffer_align_bytes = 1024

    @cute.kernel
    def kernel(
        self,
        # global tma tensors
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        # tma atoms
        a_tma_atom: cute.CopyAtom,
        b_tma_atom: cute.CopyAtom,
        c_tma_atom: cute.CopyAtom,
        # shared storage
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        # mma
        tiled_mma: cute.TiledMma,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(a_tma_atom)
            cute.nvgpu.cpasync.prefetch_descriptor(b_tma_atom)

        # initialize smem tensors
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA: cute.Tensor = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB: cute.Tensor = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sc_ptr = cute.recast_ptr(
            sA.iterator, epi_smem_layout_staged.inner, dtype=self.c_dtype
        )
        sC = cute.make_tensor(sc_ptr, epi_smem_layout_staged.outer)

        # split global tensors
        gA = cute.flat_divide(gA, cute.slice_(self.cta_tile, (None, 0, None)))[None, None, bidx, None]
        gB = cute.flat_divide(gB, cute.slice_(self.cta_tile, (0, None, None)))[None, None, bidy, None]
        gC = cute.flat_divide(gC, cute.slice_(self.cta_tile, (None, None, 0)))[None, None, bidx, bidy]

        k_tile_cnt = cute.size(gA, mode=[2])
        assert k_tile_cnt == cute.size(gB, mode=[2])

        # initialize pipelines
        mainloop_pipeline_array_ptr = storage.mainloop_mbar_ptr.data_ptr()

        # actual pipeline
        a_smem_unstaged_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_unstaged_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(self.a_dtype, a_smem_unstaged_layout) \
            + cute.size_in_bytes(self.b_dtype, b_smem_unstaged_layout)
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)

        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=self.pipeline_stages,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
        )

        is_producer = warp_idx == 4
        is_consumer = warp_idx < 4

        # partition tma tensors
        cluster_shape_mnk = (1, 1, 1)
        cta_layout = cute.make_layout(cluster_shape_mnk)
        cta_layout = cute.make_layout(cute.slice_(cta_layout, (0, None, 0)).shape)
        cta_crd = (0,)
        tAsA, tAgA_mkl = cute.nvgpu.cpasync.tma_partition(
            a_tma_atom,
            cta_crd,
            cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        tBsB, tBgB_nkl = cute.nvgpu.cpasync.tma_partition(
            b_tma_atom,
            cta_crd,
            cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB, 0, 2),
        )

        # partition smem for mma
        thr_mma = tiled_mma.get_slice(0)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        tCrC = thr_mma.partition_C(gC)
        accumulators = cute.make_fragment(tCrC.shape, self.acc_dtype)

        tidx_in_warp = tidx % 32
        # producer -- i.e. loader
        if is_producer:
            mainloop_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.pipeline_stages
            )
            # keep loading tiles until we're done.
            for _ in cutlass.range(k_tile_cnt, unroll=1):
                # what acquire does is to signal that you are readying this barrier up for the TMA.
                mainloop_pipeline.producer_acquire(mainloop_producer_state)
                tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]
                tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                cute.copy(
                    a_tma_atom,
                    tAgA_k,
                    tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state)
                )
                cute.copy(
                    b_tma_atom,
                    tBgB_k,
                    tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state)
                )
                mainloop_producer_state.advance()

        num_k_microtiles = cute.size(tCrA, mode=[2])

        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)

        # tile consumer
        if is_consumer:
            mainloop_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.pipeline_stages
            )
            print("k_tile_cnt", k_tile_cnt)
            for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                # wait for the tile to be loaded.
                mainloop_pipeline.consumer_wait(mainloop_consumer_state)

                print("num_k_microtiles", num_k_microtiles)
                for k_microtile_ix in cutlass.range(num_k_microtiles, unroll=1):
                    k_block_coord = (None, None, k_microtile_ix, mainloop_consumer_state.index)
                    cute.gemm(
                        tiled_mma,
                        accumulators,
                        tCrA[k_block_coord],
                        tCrB[k_block_coord],
                        accumulators,
                    )

                    if k_tile == 0 and k_microtile_ix == 0:
                        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

                mainloop_pipeline.consumer_release(mainloop_consumer_state)
                mainloop_consumer_state.advance()

        # epilogue -- write back to global memory.
        if is_consumer:
            # 1. copy from rmem -> smem
            copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                self.c_layout,
                elem_ty_d=self.c_dtype,
                elem_ty_acc=self.acc_dtype,
            )
            tiled_copy_r2s = cute.make_tiled_copy_C(copy_atom_r2s, tiled_mma)
            # 1a. partition 
            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
            tRS_sD = thr_copy_r2s.partition_D(sC)
            tRS_rAcc = tiled_copy_r2s.retile(accumulators)

            print("tRS_sD", tRS_sD)
            print("tRS_rAcc", accumulators)

            # 1b. reformat the partition shape to match a shape that can be used in the cute.copy
            # despite the best of my efforts, i cannot figure out a way to use the existing accumulator
            # shapes with the shared tensor shape. so, we're going to make a new fragment that we know
            # has a valid shape, and we're going to just loop through tRS_rAcc to copy over the elements to tRS_rD.
            rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_fragment_like(tRS_rD_layout, self.c_dtype)


            for i in range(cute.size(tRS_rD)):
                tRS_rD[i] = accumulators[i]
            cute.copy(tiled_copy_r2s, tRS_rD, tRS_sD[(None, None, None, 0)])

            # perform the smem -> gmem copy.
            sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
            print("sC for tma partition", sepi_for_tma_partition)
            tcgc_for_tma_partition = cute.zipped_divide(gC, cute.slice_(self.cta_tile, (None, None, 0)))
            bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                c_tma_atom,
                0,
                cute.make_layout(1),
                sepi_for_tma_partition,
                tcgc_for_tma_partition,
            )

            print("bSG_sD", bSG_sD)
            print("bSG_gD", bSG_gD)

            cute.copy(
                c_tma_atom,
                bSG_sD[(None, 0)],
                bSG_gD[(None, 0)],
            )
            # 2. copy from smem -> gmem


            

    @cute.jit
    def __call__(
        self, 
        gA: cute.Tensor, 
        gB: cute.Tensor, 
        gC: cute.Tensor,
        stream: cuda.CUstream,
    ):
        a_smem_shape = (self.cta_tile[0], self.cta_tile[2])
        b_smem_shape = (self.cta_tile[1], self.cta_tile[2])
        c_smem_shape = (self.cta_tile[0], self.cta_tile[1])
        self.a_dtype = gA.element_type
        self.b_dtype = gB.element_type
        self.c_dtype = gC.element_type
        self.c_layout = utils.LayoutEnum.from_tensor(gC)
        
        def get_staged_layout(smem_shape: Tuple[int, int], type: type[cutlass.Numeric], stages: int, atom_kind: cute.nvgpu.warpgroup.SmemLayoutAtomKind = cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128):
            smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
                atom_kind,
                type,
            )
            return cute.tile_to_shape(
                smem_layout_atom,
                cute.append(smem_shape, stages),
                order=(0, 1, 2)
            )

        # 1. smem layouts for pipeline stages
        a_smem_layout_staged = get_staged_layout(a_smem_shape, gA.element_type, self.pipeline_stages)
        b_smem_layout_staged = get_staged_layout(b_smem_shape, gB.element_type, self.pipeline_stages)
        epi_smem_layout_staged = get_staged_layout(c_smem_shape, gC.element_type, self.pipeline_stages, cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW64)

        # 2. tma atoms & tensors
        g2s_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        s2g_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        a_tma_atom, a_tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s_op,
            gA,
            a_smem_layout,
            (self.cta_tile[0], self.cta_tile[2]),
        )

        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        b_tma_atom, b_tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s_op,
            gB,
            b_smem_layout,
            (self.cta_tile[1], self.cta_tile[2]),
        )

        c_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        c_tma_atom, c_tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            s2g_op,
            gC,
            c_smem_layout,
            (self.cta_tile[0], self.cta_tile[1]),
        )

        # 3. tiled mma
        mm = cute.nvgpu.warpgroup.OperandMajorMode.K

        self.atom_layout_mnk = (1, 1, 1)
        self.acc_dtype = cutlass.Float32
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            gA.element_type,
            gB.element_type,
            mm,
            mm,
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(self.cta_tile[0], self.cta_tile[1]),
        )


        @cute.struct
        class SharedStorage:
            sA: cute.struct.Align[
                cute.struct.MemRange[gA.element_type, cute.cosize(a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[gB.element_type, cute.cosize(b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            mainloop_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.pipeline_stages * 2]

        self.shared_storage = SharedStorage

        # how are you going to split this?
        num_m_blocks = self.problem_shape[0] // self.cta_tile[0]
        num_n_blocks = self.problem_shape[1] // self.cta_tile[1]
        self.kernel(
            a_tma_tensor, b_tma_tensor, c_tma_tensor,
            a_tma_atom, b_tma_atom, c_tma_atom,
            a_smem_layout_staged, b_smem_layout_staged, 
            epi_smem_layout_staged,
            tiled_mma,
        ).launch(
            grid=[num_m_blocks, num_n_blocks, 1],
            block=[8 * WARP_SIZE, 1, 1],
            cluster=[1, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )
        print("done")


if __name__ == "__main__":
    gemm = Gemm((1024, 1024, 64), (64, 64, 64))
    a_trch = torch.eye(64, 64, device="cuda", dtype=torch.float16)
    b_trch = torch.zeros(64, 64, device="cuda", dtype=torch.float16)
    for i in range(64):
        for j in range(64):
            b_trch[i, j] = float(f"{i}.{j:02d}")
    c_trch = torch.zeros(64, 64, device="cuda", dtype=torch.float32)
    
    print("a_trch", a_trch.dtype)
    a = cutlass_torch.from_dlpack(a_trch, assumed_align=16)
    b = cutlass_torch.from_dlpack(b_trch, assumed_align=16)
    c = cutlass_torch.from_dlpack(c_trch, assumed_align=16)

    stream = cuda.CUstream()

    compiled_kernel = cute.compile(gemm, a, b, c, stream)
    compiled_kernel(a, b, c, stream)

    c_ref = torch.matmul(a_trch.to(torch.float32), b_trch.to(torch.float32).T)

    print("c_trch", c_trch.dtype)
    
    # Extract tuples of (row_id, col_id) from the float values
    c_trch_tuples = []
    c_ref_tuples = []
    
    for i in range(c_trch.shape[0]):
        c_trch_row = []
        c_ref_row = []
        for j in range(c_trch.shape[1]):
            # Extract row_id and col_id from the float values like {row_id}.{col_id}
            c_trch_val = float(c_trch[i, j])
            c_ref_val = float(c_ref[i, j])
            
            # Parse the float to extract row_id and col_id
            c_trch_row_id = int(c_trch_val)
            c_trch_col_id = int(round((c_trch_val - c_trch_row_id) * 100))
            
            c_ref_row_id = int(c_ref_val)
            c_ref_col_id = int(round((c_ref_val - c_ref_row_id) * 100))
            
            c_trch_row.append(f"({c_trch_row_id},{c_trch_col_id})")
            c_ref_row.append(f"({c_ref_row_id},{c_ref_col_id})")
        
        c_trch_tuples.append(c_trch_row)
        c_ref_tuples.append(c_ref_row)
    
    # Save tuples to CSV files
    import pandas as pd
    
    pd.DataFrame(c_trch_tuples).to_csv("c_trch_tuples.csv", index=False, header=False)
    pd.DataFrame(c_ref_tuples).to_csv("c_ref_tuples.csv", index=False, header=False)
    
    print(torch.allclose(c_trch, c_ref, atol=1e-2))