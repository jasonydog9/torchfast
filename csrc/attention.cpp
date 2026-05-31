/*
 * attention.cpp — Scaled Dot-Product Attention (CPU)
 *
 * WHAT IS ATTENTION?
 * ------------------
 * Attention is the core operation in Transformer models (GPT, BERT, T5, etc.).
 * It allows each position in a sequence to "look at" all other positions and
 * decide which ones are most relevant for computing its own representation.
 *
 * INTUITION:
 *   Imagine you're reading a sentence and want to understand the word "it".
 *   To resolve "it", you need to attend to earlier nouns. Attention learns to
 *   weight earlier positions by how relevant they are to the current word.
 *
 * THE THREE COMPONENTS — Queries, Keys, Values:
 *   - Query (Q): "What am I looking for?" — the question each position asks.
 *   - Key   (K): "What do I contain?"    — the advertisement each position posts.
 *   - Value (V): "What do I offer?"      — the actual information to aggregate.
 *
 *   For each query position, we compute a similarity score against all key positions,
 *   turn those scores into a probability distribution (softmax), then take a
 *   weighted sum of the values. The result is a "context vector" for that position.
 *
 * THE FORMULA:
 *   Attention(Q, K, V) = softmax( Q @ K.T / sqrt(d_k) ) @ V
 *
 *   Step by step:
 *   1. scores = Q @ K.T          → raw similarity scores [S, S]
 *                                   (dot product of each query with each key)
 *   2. scores /= sqrt(d_k)       → scale to prevent vanishing softmax gradients
 *   3. weights = softmax(scores) → turn scores into attention probabilities [S, S]
 *   4. output = weights @ V      → weighted sum of values [S, d_k]
 *
 * WHY SCALE BY sqrt(d_k)?
 *   Without scaling, for large d_k (head dimension), the dot products grow large
 *   in magnitude. Large inputs to softmax produce near-one-hot distributions
 *   (extremely peaked), causing gradients to vanish during training.
 *   Scaling by 1/sqrt(d_k) keeps the dot products in a "reasonable" range.
 *   The d_k factor comes from the variance of a random dot product: if each
 *   component of q and k is N(0,1), then q·k has variance d_k.
 *
 * MULTI-HEAD ATTENTION:
 *   Rather than one large attention, we run H parallel "heads" with smaller
 *   dimensions (d_k = d_model / H). Each head attends to different aspects
 *   of the input, then results are concatenated.
 *   Input shape: [Batch, Heads, Seq_len, Head_dim]
 *
 * ATTENTION MASKING:
 *   Masks zero out attention scores for:
 *   - Causal masking (decoder): prevent attending to future positions
 *   - Padding masking: prevent attending to padding tokens
 *   Masked positions get score = -inf → softmax(−inf) = 0 (zero weight).
 *
 * COMPUTATIONAL COMPLEXITY:
 *   O(S² × d_k) per head — the S² term is why long sequences are expensive.
 *   FlashAttention addresses this by tiling the computation to avoid
 *   materializing the full [S, S] attention matrix. This CPU implementation
 *   is the straightforward (non-Flash) version.
 */

#include <torch/extension.h>
#include <cmath>  // std::sqrt


// ─── FORWARD PASS ─────────────────────────────────────────────────────────────
/*
 * Compute scaled dot-product attention.
 *
 * Q, K, V shapes: [batch, heads, seq_len, head_dim]
 * mask (optional): true = MASKED OUT (set to -inf), false = attend normally
 * Output shape:    [batch, heads, seq_len, head_dim]
 *
 * Implementation strategy:
 *   Reshape to [B*H, S, D] so we can use torch::bmm (batched matmul).
 *   bmm treats the first dim as an independent batch, computing the [S,D]×[D,S]
 *   matmul for each of the B*H heads in parallel.
 *   This lets us avoid nested Python/C++ loops over batch and heads.
 */
torch::Tensor sdp_attention_forward(
    const torch::Tensor& Q,                    // [B, H, S, D] queries
    const torch::Tensor& K,                    // [B, H, S, D] keys
    const torch::Tensor& V,                    // [B, H, S, D] values
    const c10::optional<torch::Tensor>& mask)  // optional [S,S] or [B,H,S,S] bool mask
{
    // ── Input validation ──────────────────────────────────────────────────────
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D [batch, heads, seq, head_dim]");
    TORCH_CHECK(K.dim() == 4, "K must be 4D");
    TORCH_CHECK(V.dim() == 4, "V must be 4D");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(K.scalar_type() == torch::kFloat32, "K must be float32");
    TORCH_CHECK(V.scalar_type() == torch::kFloat32, "V must be float32");

    const int64_t B = Q.size(0);  // batch size
    const int64_t H = Q.size(1);  // number of attention heads
    const int64_t S = Q.size(2);  // sequence length
    const int64_t D = Q.size(3);  // head dimension (d_k)

    // All three tensors must have the same shape for standard self-attention.
    // (Cross-attention with different K/V sequence lengths is a separate case.)
    TORCH_CHECK(K.size(0) == B && K.size(1) == H && K.size(2) == S && K.size(3) == D,
                "K shape must match Q");
    TORCH_CHECK(V.size(0) == B && V.size(1) == H && V.size(2) == S && V.size(3) == D,
                "V shape must match Q");

    // Ensure contiguous memory layout before calling .view() — view() requires
    // that the tensor's strides are compatible with the new shape.
    auto Qc = Q.contiguous();
    auto Kc = K.contiguous();
    auto Vc = V.contiguous();

    // ── Scaling factor ────────────────────────────────────────────────────────
    // scale = 1 / sqrt(D).  Applied BEFORE softmax to stabilize gradients.
    // Pre-compute as a float scalar to avoid recomputing for every element.
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // ── Reshape: [B, H, S, D] → [B*H, S, D] ─────────────────────────────────
    // Merging the batch and head dimensions lets torch::bmm process all
    // B*H "instances" in a single call. View is zero-copy — same memory,
    // just different shape metadata.
    auto Qr = Qc.view({B * H, S, D});
    auto Kr = Kc.view({B * H, S, D});
    auto Vr = Vc.view({B * H, S, D});

    // ── Step 1: Compute raw attention scores ─────────────────────────────────
    // scores[b, i, j] = (1/sqrt(D)) * Q[b, i, :] · K[b, j, :]
    //
    // bmm(A, B) computes A @ B for each element in the batch dimension.
    // Qr is [B*H, S, D], Kr.transpose(1,2) is [B*H, D, S].
    // Result: [B*H, S, S]  — for each head, an S×S score matrix.
    //
    // scores[b, i, j] tells us: "how much does query-position i want
    // to attend to key-position j?"
    auto scores = torch::bmm(Qr, Kr.transpose(1, 2)) * scale;  // [B*H, S, S]

    // ── Step 2: Apply attention mask (if provided) ────────────────────────────
    if (mask.has_value()) {
        auto m = mask.value();
        TORCH_CHECK(m.scalar_type() == torch::kBool, "mask must be bool tensor");

        // Broadcast mask to [B*H, S, S] regardless of input shape.
        if (m.dim() == 2) {
            // [S, S] mask (e.g., causal lower-triangular mask shared across all heads)
            // unsqueeze(0) → [1, S, S]; expand → [B*H, S, S]
            m = m.unsqueeze(0).expand({B * H, S, S});
        } else if (m.dim() == 4) {
            // [B, H, S, S] mask (different mask per sample/head)
            m = m.view({B * H, S, S});
        } else {
            TORCH_CHECK(false, "mask must be 2D [S,S] or 4D [B,H,S,S]");
        }
        // Wherever mask is TRUE: set score to -1e9 (≈ -infinity).
        // softmax(−1e9) ≈ 0, so masked positions receive zero attention weight.
        // Why -1e9 not -inf? To avoid NaN: softmax can produce 0/0 = NaN
        // if ALL positions are masked to -inf in a row.
        scores = scores.masked_fill(m, -1e9f);
    }

    // ── Step 3: Softmax — convert scores to attention weights ────────────────
    // softmax is applied over the LAST dimension (dim=-1 = the key dimension).
    // For each query position i, its S scores over all keys become probabilities
    // that sum to 1: Σ_j attn[b, i, j] = 1.
    //
    // This turns "similarity scores" into "how much to attend to each position".
    auto attn = torch::softmax(scores, /*dim=*/-1);  // [B*H, S, S]

    // ── Step 4: Weighted sum of values ────────────────────────────────────────
    // output[b, i, :] = Σ_j attn[b, i, j] * V[b, j, :]
    //
    // bmm(attn, Vr): attn is [B*H, S, S], Vr is [B*H, S, D]
    // Result: [B*H, S, D] — for each (head, query-position), a weighted
    // mixture of value vectors, where attention weights set the mixing coefficients.
    auto out = torch::bmm(attn, Vr);  // [B*H, S, D]

    // ── Reshape back to [B, H, S, D] ─────────────────────────────────────────
    // view() is again zero-copy — just changes the shape metadata.
    return out.view({B, H, S, D});
}


// ─── Python bindings ─────────────────────────────────────────────────────────
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sdp_attention_forward,
          "Scaled dot-product attention forward",
          py::arg("Q"),
          py::arg("K"),
          py::arg("V"),
          py::arg("mask") = c10::optional<torch::Tensor>());  // mask is optional
}
