import triton
import triton.language as tl


# ==========================================
# 1. The Custom Triton Accelerator
# ==========================================
@triton.jit
def _bm25_exhaustive_triton_kernel(
    query_tokens_ptr,  # [num_query_tokens]
    vocab_ptrs_ptr,  # [vocab_size + 1]
    doc_idxs_ptr,  # [total_nnz]
    scores_ptr,  # [total_nnz]
    score_buffer_ptr,  # [num_docs]
    num_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    """Zero-allocation brute-force gather. Ideal for dense/rerank triggers

    :param query_tokens_ptr: _description_
    :param BLOCK_SIZE: _description_
    """
    # 1. Each program block handles one token from the query
    pid = tl.program_id(axis=0)

    if pid < num_tokens:
        # 2. Get the specific token ID
        token_id = tl.load(query_tokens_ptr + pid)

        # 3. Look up the start and end pointers in the CSC index
        start_idx = tl.load(vocab_ptrs_ptr + token_id)
        end_idx = tl.load(vocab_ptrs_ptr + token_id + 1)

        # 4. Loop through the non-zero elements for this token
        # avoids all the repeat_interleave logic
        for i in range(start_idx, end_idx):
            # Read the document ID and the precomputed BM25 score
            doc_id = tl.load(doc_idxs_ptr + i)
            score = tl.load(scores_ptr + i)

            # 5. Direct Atomic Accumulation into the scratchpad
            tl.atomic_add(score_buffer_ptr + doc_id, score)
