import itertools
from typing import NamedTuple, Optional

import numpy as np
import torch

from remus.kernels.bm25_kernel import _bm25_exhaustive_triton_kernel
from remus.utils.log_utils import get_custom_logger

logger = get_custom_logger()


class TopKResult(NamedTuple):
    values: torch.Tensor
    indices: torch.Tensor


@torch.compile(mode="reduce-overhead", dynamic=True)
def _execute_wand_compiled(
    q_ids_sorted: torch.Tensor,
    bounds_sorted: torch.Tensor,
    vocab_ptrs: torch.Tensor,
    doc_idxs: torch.Tensor,
    scores: torch.Tensor,
    score_buffer: torch.Tensor,
    top_k: int,
) -> TopKResult:
    """Vectorized Term-at-a-Time WAND implementation

    Compiled TAAT WAND Engine (GPU/CPU Top-K)

    Leverages dynamic thresholding and suffix sums to prune the search
    space. By processing terms in descending order of their maximum potential
    contribution, we can skip document evaluations that mathematically cannot
    enter the Top-K.

    :param q_ids_sorted: 1D Tensor of query token IDs, sorted by descending upper bounds
    :param bounds_sorted: 1D Tensor of precomputed max BM25 scores for each token in q_ids_sorted
    :param vocab_ptrs: CSC index pointers defining start/end offsets for each token's posting list
    :param doc_idxs: flat CSC array of document IDs
    :param scores: flat CSC array of precomputed BM25 scores (float16)
    :param score_buffer: reusable 1D Tensor [num_docs] used as a scratchpad for score accumulation
    :param top_k: the number of highest-scoring documents to retrieve
    :return: a TopKResult NamedTuple containing the values and indices of the top results
    """
    # Initialize the scratchpad and set the entry-bar for the Top-K (threshold) to zero
    score_buffer.zero_()
    threshold = 0.0

    ## Precalculate Suffix Sums
    # represents the "Max Theoretical Future Gain."
    # suffix_sums[i] = the sum of max scores for all query terms AFTER term i
    suffix_sums = torch.zeros(len(q_ids_sorted), device=q_ids_sorted.device, dtype=torch.float32)
    if len(q_ids_sorted) > 1:
        # flip, cumsum, then flip back to get a right-to-left sum of the upper bounds
        suffix_sums[:-1] = torch.flip(
            torch.cumsum(torch.flip(bounds_sorted[1:], dims=[0]), dims=[0]), dims=[0]
        )

    ## Term at a Time Iteration (TAAT)
    for t_idx, token_id in enumerate(q_ids_sorted):
        # locate the posting list window for the current token
        start, end = vocab_ptrs[token_id], vocab_ptrs[token_id + 1]
        if start == end:
            continue

        # zero-copy slice of relevant document IDs and their corresponding BM25 scores
        d_ids = doc_idxs[start:end]
        s_vals = scores[start:end].to(torch.float32)

        ## WAND Pruning Step
        # only perform pruning once we have established a baseline Top-K threshold.
        if threshold > 0:
            # calculate the "Potential Score" for every document in this posting list
            # Potential Score = (Current Accumulated Score) + (This Term's Score) + (Max Future Terms)
            potential_scores = score_buffer[d_ids] + s_vals + suffix_sums[t_idx]

            # create a boolean mask: if a document's potential cannot beat the threshold, drop it.
            valid_mask = potential_scores > threshold

            # vectorized pruning via boolean indexing
            d_ids = d_ids[valid_mask]
            s_vals = s_vals[valid_mask]

        ## Accumulation and threshold update
        if len(d_ids) > 0:
            # Scatter-add the scores into our global document scratchpad
            score_buffer.index_add_(0, d_ids, s_vals)
            # Update the threshold: the new bar is the score of the k-th best document found so far
            # This makes pruning progressively more aggressive as the loop continues
            threshold = torch.topk(score_buffer, top_k).values[-1].item()

    # final extraction of the Top-K winners from the scratchpad
    return torch.topk(score_buffer, k=top_k)


class TorchBM25:
    """Matrix-Based GPU-accelerated Implementation of BM25 Keyword Search Algorithm.

    This engine utilizes a Compressed Sparse Column (CSC) index and vectorized
    Term-at-a-Time (TAAT) execution. It supports hardware-accelerated WAND
    pruning and custom Triton kernels for high-throughput lexical retrieval
    on NVIDIA GPUs.

    :param corpus_token_ids: A nested list where each inner list contains integer
        token IDs representing a document in the corpus
    :param vocab_size: The total size of the tokenizer vocabulary. Used to
        allocate fixed-width tensors for IDF and pointer arrays
    :param k1: BM25 saturation parameter. Controls non-linear term frequency scaling,
        defaults to 1.5
    :param b: BM25 length normalization parameter. Controls the impact of document length relative to average length,
        defaults to 0.75
    :param device: the target compute device (e.g., 'cuda:0' or 'cpu'),
        defaults to "cpu"
    :param ref_bm25: An optional 'seed' TorchBM25 instance used to impute
        background IDF weights from a larger reference corpus (e.g., MS MARCO), defaults to None
    :param min_df: Minimum document frequency for a term to be included in the index,
        defaults to 1
    :param max_vocab_size: Optional cap on the vocabulary size to truncate high-index tokens.
    :param build_index: Boolean flag to trigger the CSC index construction immediately upon initialization,
        defaults to True
    :param name: A descriptive string identifier for logging and multi-index tracking,
        defaults to "Base"
    """

    def __init__(
        self,
        corpus_token_ids: list[list[int]],
        vocab_size: int,
        k1: float = 1.5,
        b: float = 0.75,
        device: str = "cpu",
        ref_bm25: Optional["TorchBM25"] = None,
        min_df: int = 1,
        max_vocab_size: Optional[int] = None,
        build_index: bool = True,
        name: str = "Base",
    ) -> None:
        """Matrix-Based GPU-accelerated Implementation of BM25 Keyword Search Algorithm.

        :param corpus_token_ids: A nested list where each inner list contains integer
            token IDs representing a document in the corpus
        :param vocab_size: The total size of the tokenizer vocabulary. Used to
            allocate fixed-width tensors for IDF and pointer arrays
        :param k1: BM25 saturation parameter. Controls non-linear term frequency scaling,
            defaults to 1.5
        :param b: BM25 length normalization parameter. Controls the impact of document length relative to average length,
            defaults to 0.75
        :param device: the target compute device (e.g., 'cuda:0' or 'cpu'),
            defaults to "cpu"
        :param ref_bm25: An optional 'seed' TorchBM25 instance used to impute
            background IDF weights from a larger reference corpus (e.g., MS MARCO), defaults to None
        :param min_df: Minimum document frequency for a term to be included in the index,
            defaults to 1
        :param max_vocab_size: Optional cap on the vocabulary size to truncate high-index tokens.
        :param build_index: Boolean flag to trigger the CSC index construction immediately upon initialization,
            defaults to True
        :param name: A descriptive string identifier for logging and multi-index tracking,
            defaults to "Base"
        """

        self.device = device
        self.k1 = k1
        self.b = b
        self.min_df = min_df
        self.max_vocab_size = max_vocab_size
        self.build_index = build_index
        self.name = name

        # configure corpus based on provided corpus, max_vocab_size, etc.
        self._configure_corpus(corpus_token_ids, vocab_size)

        # calculate IDF/BM25 scores and build index if desired
        self._precompute_scores(ref_bm25)

    def _configure_corpus(
        self,
        corpus_token_ids: list[list[int]],
        vocab_size: int,
    ) -> None:
        """Internal method to configure BM25 based on provided corpus parameters

        :param corpus_token_ids: input corpus token ids - a list of documents where
            each document is a list of token ids
        :param vocab_size: provide tokenizer vocab size
        :raises ValueError: if corpus_token_ids not provided
        """

        if not corpus_token_ids:
            raise ValueError("'corpus_token_ids' must be provided.")

        self.corpus_token_ids = corpus_token_ids
        self.vocab_size = vocab_size

        logger.info(
            f"[{self.name}] Configured for {len(self.corpus_token_ids)} docs. "
            f"Vocab Boundary: {self.vocab_size}"
        )

    def _precompute_scores(
        self,
        ref_bm25: Optional["TorchBM25"] = None,
    ) -> None:
        """Internal method to precompute BM25 IDF weights (using reference BM25 if provided)

        :param ref_bm25: optional reference "seed" TorchBM25 instance, defaults to None
        """

        ## Corpus Document Statistics
        # num docs
        self.num_docs = len(self.corpus_token_ids)
        logger.info(
            f"[{self.name}] Start Bulk Vectorization for {self.num_docs} docs. Vocab: {self.vocab_size}"
        )

        # 1. Document Lengths (Vectorized)
        doc_lens = np.array([len(d) for d in self.corpus_token_ids], dtype=np.int32)
        doc_lengths_t = torch.from_numpy(doc_lens).to(self.device, dtype=torch.float32)
        mean_doc_len = doc_lengths_t.mean()

        # 2. Bulk Flattening (The RAM Heavy Phase)
        # np.fromiter is fastest way to drain a nested list into C-memory
        logger.info(f"[{self.name}] Flattening corpus...")
        flat_tokens = np.fromiter(
            itertools.chain.from_iterable(self.corpus_token_ids),
            dtype=np.uint32,
        )

        # Generate row indices corresponding to every token: [0, 0, 0, 1, 1, 2, 2...]
        row_indices = np.repeat(np.arange(self.num_docs, dtype=np.uint32), doc_lens)

        # 3. The 64-bit Packing Trick
        # shift doc id to the top 32 bits, keep token id in the bottom 32 bits
        # combined_key = (doc_id, token_id)
        # this allows numpy.unique to simultaneously calculate the exact Term Frequency (TF) per document
        # and setup the data for the Document Frequency (DF) in a single C-optimized pass
        logger.info(f"[{self.name}] Packing keys and calculating frequencies...")
        # Allocate exactly ONE 64-bit array and modify it in-place
        combined_key = row_indices.astype(np.uint64)
        combined_key <<= 32
        combined_key |= flat_tokens.astype(np.uint64)

        del flat_tokens, row_indices

        # 4. Global Unique Sort & Count
        # unique_keys = unique (doc_id, token_id) pairs
        # counts = exact Term Frequency (TF) of that token in that document
        unique_keys, counts = np.unique(combined_key, return_counts=True)

        # free the 64-bit array immediately
        del combined_key

        # 5. Unpack and Transfer to GPU
        # shift right by 32 bits to get doc_id, bitwise AND to get token_id
        logger.info(f"[{self.name}] Transferring {len(unique_keys)} unique term pairs to VRAM...")

        row_idxs_final = torch.from_numpy((unique_keys >> 32).astype(np.int64)).to(self.device)
        col_idxs_final = torch.from_numpy((unique_keys & 0xFFFFFFFF).astype(np.int64)).to(self.device)  # fmt: skip
        counts_final = torch.from_numpy(counts).to(self.device, dtype=torch.float32)

        del unique_keys, counts

        # 6. Exact Document Frequency (DF) Calculation
        # because pairs are exactly one per (document, token), a bincount
        # of the tokens gives us the exact Document Frequency across the corpus
        doc_freqs = torch.bincount(col_idxs_final, minlength=self.vocab_size).float()

        ## precompute IDF weights
        # IDF = log( (N - doc_freq + 0.5) / (doc_freq + 0.5) + 1 )
        self.idf = torch.log((self.num_docs - doc_freqs + 0.5) / (doc_freqs + 0.5) + 1.0)

        # impute 'seed' reference bm25 idf token weights
        if ref_bm25 is not None:
            logger.info(f"[{self.name}] Imputing IDF weights from reference BM25...")
            # since both engines use the same integer tokenizer, their IDs align perfectly.
            # just overwrite our native IDF values with the reference ones where they overlap
            overlap_size = min(self.vocab_size, ref_bm25.vocab_size)
            self.idf[:overlap_size] = ref_bm25.idf[:overlap_size]
            logger.info(
                f"[{self.name}] Successfully imputed {overlap_size} background IDF weights."
            )

        # 8. Dispatch to CSC Builder (index)
        # pass pre-allocated tensors straight to the indexer
        self._create_index(
            row_idxs_final,
            col_idxs_final,
            counts_final,
            doc_lengths_t,
            mean_doc_len,
        )

        return

    def _create_index(
        self,
        row_idxs: torch.Tensor,
        col_idxs: torch.Tensor,
        counts: torch.Tensor,
        doc_lengths: torch.Tensor,
        mean_doc_len: torch.Tensor,
    ) -> None:
        """Internal method to create the BM25 lookup index.

        Current strategy: Compressed Sparse Columns (CSC).
            Store sparsely, Lookup Sparsely, Compute densely if necessary.

        :param row_idxs: 1D tensors for document indexes (rows)
        :param col_idxs: 1D tensors for vocab/token indexes (columns)
        :param counts: 1D tensors for unique token counts
        :param doc_lengths: 1D tensor reprsenting document length
        :param mean_doc_len: tensor representing average document length
        """

        ## Base Case
        if not self.build_index:
            logger.info(f"[{self.name}] Not building index. Skipping matrix allocation.")
            return

        logger.info(f"[{self.name}] Calculating BM25 scores for {len(row_idxs)} elements...")

        ## Build search index
        # only build index for non-zero elements
        idf_vals = self.idf[col_idxs]
        doc_len_vals = doc_lengths[row_idxs]

        ## 1. Vectorized BM25 Calculation (Float32)
        # formula: IDF * ( (f * (k1 + 1)) / (f + k1 * (1 - b + b * (L / avgL))) )
        numerator = counts * (self.k1 + 1)
        denominator = counts + self.k1 * (1 - self.b + self.b * doc_len_vals / mean_doc_len)
        bm25_vals = idf_vals * (numerator / denominator)

        ## 2. Compressed Sparse Columns (CSC) Sorting
        # strategy: store sparsely, compute densely only as-necessary
        # sort by vocab token id (column) to allow faster doc (row) look up
        logger.info(f"[{self.name}] Sorting index into CSC format...")
        sort_idxs = torch.argsort(col_idxs)

        # reorder everything to be contiguous by token_id
        self.doc_idxs = row_idxs[sort_idxs]
        self.scores = bm25_vals[sort_idxs].to(torch.float16)

        ## 3. Boundary Pointer Construction
        logger.info(f"[{self.name}] Building vocabulary pointers...")

        # row pointers defining start/end window for each vocab token
        # 1D dense tensor
        # torch.bincount() returns counts by index
        vocab_counts = torch.bincount(col_idxs, minlength=self.vocab_size)

        # cumsum converts counts into absolute memory offsets
        self.vocab_ptrs = torch.zeros(self.vocab_size + 1, dtype=torch.long, device=self.device)
        torch.cumsum(vocab_counts, dim=0, out=self.vocab_ptrs[1:])

        ## 4. WAND Upper Bounds Precomputation
        logger.info(f"[{self.name}] Precomputing WAND upper bounds...")
        # segment_reduce performs an ultra-fast hardware reduction on contiguous blocks.
        # Since our scores are already sorted into CSC format, vocab_counts perfectly
        # defines the length of each token's segment.
        self.term_upper_bounds = torch.segment_reduce(
            self.scores.to(torch.float32),
            reduce="max",
            lengths=vocab_counts,
        ).to(torch.float16)

        # explicitly delete temp tensors to free VRAM
        del sort_idxs, bm25_vals, idf_vals, doc_len_vals

        logger.info(
            f"[{self.name}] CSC Index Complete. "
            f"Scores VRAM: {self.scores.element_size() * self.scores.nelement() / 1e6:.2f} MB | "
            f"WAND Bounds VRAM: {self.term_upper_bounds.element_size() * self.term_upper_bounds.nelement() / 1e6:.2f} MB"
        )

        return

    def _gather_sparse_data(
        self,
        query_ids: torch.Tensor,
        batch_ids: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Internal helper method to extract non-zero BM25 scores and their coordinates for a given set of query tokens.

        Strategy: use a pure sparse gathering technique on the underlying
            CSC (Compressed Sparse Column) index, avoiding any dense intermediate matrices or padding

        Terminology:
        * Row = query in batch
        * Column = document id containing query tokens

        :param query_ids: 1D tensor of unique token IDs present in the search query
        :param batch_ids: Optional 1D tensor mapping each token to its specific query in a batch, defaults to None
        :return: Tuple of (row_coords, col_coords, scores) for constructing a sparse COO tensor,
            or (None, None, None) if the query yields no results
        """

        # Base Case: query contains no recognizable tokens.
        if query_ids.numel() == 0:
            return None, None, None

        ## 1. Boundary Lookup (CSC Format)
        # self.vocab_ptrs acts as our column pointers. We look up exactly where
        # the data for each query token starts and ends in our flat storage arrays.
        # [n_query_tokens]
        starts = self.vocab_ptrs[query_ids]
        ends = self.vocab_ptrs[query_ids + 1]

        # calculate the exact number of documents containing each queried token.
        # [n_query_tokens]
        lengths = ends - starts

        # calculate total number of valid, non-zero scores to extract
        # total memory footprint needed for this extraction.
        total_elements = lengths.sum().item()

        # None of the queried tokens exist in any document in the corpus.
        if total_elements == 0:
            return None, None, None

        ## 2. Pure Sparse Index Extraction (no padding necessary)
        # generate a flat array of exact memory addresses (`flat_idxs`) to pull scores from
        # without using loops or allocating padded dense matrices

        # create a mapping that connects every element we are about to extract back to its parent token.
        # e.g., lengths = [2, 1, 3], repeats = [0, 0, 1, 2, 2, 2]
        token_idxs = torch.arange(len(query_ids), device=self.device)
        repeats = torch.repeat_interleave(token_idxs, lengths)

        # get absolute starting memory address of token block for every single element we need to extract
        # repeats the start address for every document that contains the token
        expanded_starts = starts[repeats]

        # Local Offset
        # e.g., if a token appears in 3 documents, the elements are stored at `start + 0`, `start + 1`, `start + 2`.
        # we need to calculate that `[0, 1, 2]` offset.

        # find where each token's block begins relative to our new extraction batch.
        # if lengths = [2, 1, 3], cum_lengths = [0, 2, 3, 6].
        cum_lengths = torch.zeros(len(query_ids) + 1, device=self.device, dtype=torch.long)
        cum_lengths[1:] = torch.cumsum(lengths, dim=0)

        # map those batch-relative group starts to every element.
        # group_starts = [0, 0, 2, 3, 3, 3]
        group_starts = cum_lengths[repeats]

        # Subtract the group start from a global counter to get the local 0-based offset for each element.
        # global_arange  =  [0, 1, 2, 3, 4, 5]
        # - group_starts =  [0, 0, 2, 3, 3, 3]
        # = local_offsets = [0, 1, 0, 0, 1, 2]
        global_arange = torch.arange(total_elements, device=self.device)
        local_offsets = global_arange - group_starts

        # base address (in self.scores) + Local Offset = Exact 1D Index to gather
        flat_idxs = expanded_starts + local_offsets

        ## 3. Gather & Map Coordinates
        # use 1D addresses to pull the data from the flat CSC arrays
        # column = document containing query tokens
        # row = query in batch
        col_coords = self.doc_idxs[flat_idxs]

        # actual BM25 float values
        scores = self.scores[flat_idxs]

        # map the batch (row) indices using repeats array
        # if batching multiple queries, this assigns the gathered scores to the correct query ID
        # else, it defaults them all to query 0
        if batch_ids is not None:
            row_coords = batch_ids[repeats]
        else:
            row_coords = torch.zeros_like(col_coords)

        # ([E], [E], [E])
        # where E = total number of non-zero BM25 scores extracted across all queried tokens
        return row_coords, col_coords, scores.to(torch.float32)

    @torch.inference_mode()
    def rank(
        self,
        query_ids: list[int],
        top_k: Optional[int] = 10,
    ) -> torch.Tensor | TopKResult:
        """Method to rank documents in the corpus against a single query

        :param query_ids: list of int token ids representing the query
        :param top_k: optional number of highest-scoring documents to return, defaults to 10
        :return: either a 1D tensor of all document scores (if top_k is None),
            or a namedtuple containing (.values, .indices) for the top k results
        """
        res = self.rank_batch([query_ids], top_k=top_k)
        if top_k is None:
            return res[0]
        return TopKResult(values=res.values[0], indices=res.indices[0])

    @torch.inference_mode()
    def rank_batch(
        self,
        batch_query_ids: list[list[int]],
        top_k: Optional[int] = 10,
        vram_threshold: int = 12,
    ) -> torch.Tensor | TopKResult:
        """Method to rank documents in the corpus against a batch of queries in parallel

        :param batch_query_ids: list of queries, where each query is a list of int token ids
        :param top_k: Optional number of highest-scoring documents to return per query, defaults to 5.
        :param vram_threshold: GB vram threshold, defaults to 12
        :raises MemoryError: if projected vram usage > vram threshold
        :return: A 2D tensor of shape [num_queries, num_docs] (if top_k is None),
            or a namedtuple containing (.values, .indices) of shape [num_queries, top_k]
        """
        num_queries = len(batch_query_ids)
        is_gpu = self.device.startswith("cuda")

        # 1. VRAM Safety Guard
        if top_k is None:
            # calculate size of the requested dense matrix in GB
            projected_gb = (num_queries * self.num_docs * 4) / 1e9
            logger.warning(f"[{self.name}] top_k is None. Requesting {projected_gb:.2f} GB dense matrix.")  # fmt: skip

            if projected_gb > vram_threshold:
                raise MemoryError(f"Batch too large for dense return. Requested {projected_gb:.2f}GB.")  # fmt: skip

            # pre-allocate full dense matrix
            results = torch.zeros((num_queries, self.num_docs), device=self.device, dtype=torch.float32)  # fmt: skip
            score_buffer = None
        else:
            # pre-allocate sparse output buffers
            k = min(top_k, self.num_docs)
            all_values = torch.zeros((num_queries, k), device=self.device, dtype=torch.float32)
            all_indices = torch.zeros((num_queries, k), device=self.device, dtype=torch.long)
            score_buffer = torch.zeros(self.num_docs, device=self.device, dtype=torch.float32)

        # 2. Master Dispatch Loop
        for i, q_ids in enumerate(batch_query_ids):
            if not q_ids:
                continue

            # filter out-of-bounds tokens on the CPU first (much faster than GPU masking for tiny arrays)
            valid_q_ids = [q for q in q_ids if q < self.vocab_size]
            if not valid_q_ids:
                continue

            # put it on GPU
            q_ids_t = torch.as_tensor(valid_q_ids, device=self.device, dtype=torch.long)

            # PATH A: Dense Extraction -> Custom Triton Kernel
            if top_k is None and is_gpu:
                num_tokens = q_ids_t.shape[0]
                _bm25_exhaustive_triton_kernel[(num_tokens,)](
                    q_ids_t,
                    self.vocab_ptrs,
                    self.doc_idxs,
                    self.scores,
                    results[i],
                    num_tokens,
                    BLOCK_SIZE=1024,
                )

            # PATH B: Top-K Vectorized WAND (GPU & CPU Compiled)
            elif top_k is not None:
                # sort query by max possible impact
                bounds = self.term_upper_bounds[q_ids_t].to(torch.float32)
                sorted_idx = torch.argsort(bounds, descending=True)

                # dispatch to compiled WAND engine
                res_vals, res_idx = _execute_wand_compiled(
                    q_ids_sorted=q_ids_t[sorted_idx],
                    bounds_sorted=bounds[sorted_idx],
                    vocab_ptrs=self.vocab_ptrs,
                    doc_idxs=self.doc_idxs,
                    scores=self.scores,
                    score_buffer=score_buffer,
                    top_k=k,
                )
                all_values[i], all_indices[i] = res_vals, res_idx

            # PATH C: Dense CPU Fallback
            else:
                _, col_coords, values = self._gather_sparse_data(q_ids_t)
                if col_coords is not None:
                    results[i].index_add_(0, col_coords, values)

        if score_buffer is not None:
            del score_buffer

        return results if top_k is None else TopKResult(values=all_values, indices=all_indices)
