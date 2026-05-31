/*
 * losses.cpp — Focal Loss Kernel
 *
 * THE PROBLEM: CLASS IMBALANCE
 * ----------------------------
 * Standard cross-entropy (CE) loss treats all training examples equally.
 * This is fine for balanced datasets, but breaks for skewed problems like:
 *   - Object detection: 1 foreground object vs 10,000 background regions
 *   - Medical imaging: 1 tumor pixel vs 1,000,000 healthy pixels
 *   - Fraud detection: 1 fraud case vs 10,000 legitimate transactions
 *
 * When 99.9% of examples are easy negatives, the CE loss is dominated by
 * them, and the model never learns to detect the rare hard positives.
 *
 * FOCAL LOSS — The Solution (Lin et al., 2017, RetinaNet):
 * ---------------------------------------------------------
 * Focal Loss adds a "focusing factor" (1 - p_t)^γ that down-weights
 * easy examples (high p_t) and up-weights hard examples (low p_t).
 *
 * Standard Cross-Entropy:
 *   CE(p_t) = -log(p_t)
 *
 * Focal Loss:
 *   FL(p_t) = -α * (1 - p_t)^γ * log(p_t)
 *
 * Parameters:
 *   p_t   = predicted probability for the TRUE class
 *   α     = class balance weight (typically 0.25 for the foreground class)
 *   γ     = focusing parameter (typically 2.0)
 *
 * HOW γ WORKS:
 *   When the model gets an example RIGHT (p_t ≈ 1.0):
 *     (1 - p_t)^γ ≈ 0   → near-zero weight → easy examples contribute little
 *   When the model gets it WRONG (p_t ≈ 0.0):
 *     (1 - p_t)^γ ≈ 1   → full weight → hard examples still drive learning
 *
 *   γ = 0: reduces to α-weighted cross-entropy (no focusing)
 *   γ = 2: empirically optimal for object detection (from the RetinaNet paper)
 *
 * HOW α WORKS:
 *   α re-weights the loss per class. For binary problems:
 *     positive class gets weight α
 *     negative class gets weight (1 - α)
 *   Setting α = 0.25 effectively 4× amplifies rare foreground vs background.
 *
 * NUMERICAL PIPELINE:
 *   Input: raw logits z ∈ ℝ^C  (one vector per sample)
 *   Step 1: probabilities p = softmax(z)   — convert to [0,1] probability dist
 *   Step 2: p_t = p[target_class]          — extract true-class probability
 *   Step 3: FL = -α * (1-p_t)^γ * log(p_t)
 *
 * WHY LOG DIRECTLY INSTEAD OF LOG(SOFTMAX)?
 *   We use log(p_t) where p_t = softmax(z)[t]. For numerical stability,
 *   torch::log_softmax would be better in production, but here we apply
 *   a 1e-7 floor on p_t to prevent log(0) = -inf.
 *
 * BACKWARD DERIVATION:
 * --------------------
 *   We need dFL/dz_j (gradient w.r.t. each logit z_j).
 *
 *   Chain rule:  dFL/dz_j = (dFL/dp_t) * (dp_t/dz_j)
 *
 *   Part 1 — dFL/dp_t:
 *     FL = -α * (1-p_t)^γ * log(p_t)
 *     dFL/dp_t = α * [ γ * (1-p_t)^(γ-1) * log(p_t)    ← chain rule on (1-p_t)^γ
 *                    - (1-p_t)^γ / p_t ]                ← chain rule on log(p_t)
 *
 *   Part 2 — dp_t/dz_j (softmax Jacobian):
 *     p_t = softmax(z)[t] = exp(z_t) / Σ_k exp(z_k)
 *
 *     The softmax Jacobian is:
 *       dp_i/dz_j = p_i * (δ_{ij} - p_j)
 *
 *     where δ_{ij} = 1 if i==j, else 0 (Kronecker delta).
 *
 *     Intuition: softmax normalizes over all classes, so changing z_j
 *     affects p_t both directly (if j==t) and indirectly via the normalization
 *     constant (always, since Σ p_k = 1 must hold).
 *
 *     For j == target:  dp_t/dz_j = p_t * (1 - p_t)  (positive push up)
 *     For j != target:  dp_t/dz_j = p_t * (-p_j)     (negative push down)
 *
 *   Final:
 *     dFL/dz_j = dFL/dp_t * p_t * (δ_{t,j} - p_j)
 *              = d_pt * dp_dz_j
 */

#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <cmath>  // std::pow, std::log, std::max


// ─── FORWARD PASS ─────────────────────────────────────────────────────────────
/*
 * Compute scalar mean focal loss over the batch.
 *
 * logits:  [N, C]  — raw unnormalized scores from the last linear layer
 * targets: [N]     — integer class indices in [0, C)
 * Returns: scalar loss tensor (mean over N samples)
 */
torch::Tensor focal_loss_forward(
    const torch::Tensor& logits,   // [N, C] raw (unnormalized) scores
    const torch::Tensor& targets,  // [N]    class indices
    float alpha,                   // class-balance weight (default 0.25)
    float gamma)                   // focusing exponent (default 2.0)
{
    // ── Input validation ──────────────────────────────────────────────────────
    TORCH_CHECK(logits.dim() == 2,   "logits must be 2D [batch, classes]");
    TORCH_CHECK(targets.dim() == 1,  "targets must be 1D [batch]");
    TORCH_CHECK(logits.is_contiguous(),  "logits must be contiguous");
    TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");
    TORCH_CHECK(logits.scalar_type()  == torch::kFloat32, "logits must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kInt64,   "targets must be int64");

    const int64_t N = logits.size(0);  // batch size
    const int64_t C = logits.size(1);  // number of classes
    TORCH_CHECK(targets.size(0) == N, "targets batch size must match logits");

    // ── Step 1: Convert logits to probabilities via softmax ───────────────────
    // softmax(z)_c = exp(z_c) / Σ_k exp(z_k)
    // This gives a probability distribution over classes for each sample.
    // PyTorch's softmax is numerically stable (subtracts max before exp).
    auto probs = torch::softmax(logits, /*dim=*/1);  // [N, C]

    // Get raw pointers for fast inner loop — avoid repeated tensor indexing overhead
    const float*   __restrict__ p = probs.data_ptr<float>();    // [N*C] flat
    const int64_t* __restrict__ t = targets.data_ptr<int64_t>(); // [N]

    auto loss_per_sample = torch::empty({N}, logits.options());  // [N] per-sample losses
    float* __restrict__ ls = loss_per_sample.data_ptr<float>();

    // ── Step 2: Compute focal loss for each sample ────────────────────────────
    // Each sample is independent — perfect for parallelism.
    at::parallel_for(0, N, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            int64_t cls = t[i];  // ground-truth class index for sample i
            TORCH_CHECK(cls >= 0 && cls < C, "target class index out of range");

            // p_t: predicted probability for the TRUE class.
            // Memory layout: probs[i, c] = p[i*C + c]
            float pt = p[i * C + cls];

            // Numerical stability floor: log(0) = -inf would produce NaN gradients.
            // 1e-7 is small enough to not affect correctly classified examples.
            pt = std::max(pt, 1e-7f);

            // Focal Loss: FL = -α * (1 - p_t)^γ * log(p_t)
            //
            // (1-p_t)^γ is the "focusing factor":
            //   When pt ≈ 1 (easy, correct): focus factor ≈ 0  → tiny loss
            //   When pt ≈ 0 (hard, wrong):   focus factor ≈ 1  → full loss
            //
            // log(p_t) is negative (since 0 < p_t ≤ 1), so the leading
            // minus sign makes the loss positive.
            float fl = -alpha * std::pow(1.0f - pt, gamma) * std::log(pt);
            ls[i] = fl;
        }
    });

    // Mean over the batch → scalar loss (matching CE convention)
    return loss_per_sample.mean();
}


// ─── BACKWARD PASS ───────────────────────────────────────────────────────────
/*
 * Compute dL/d(logits) for the focal loss.
 *
 * grad_out: scalar upstream gradient (from .backward() — usually 1.0)
 * Returns:  [N, C] gradient tensor w.r.t. logits
 *
 * See the file-level comment for the full derivation.
 * Summary:
 *   dFL/dz_j = scalar_grad * d_pt * dp_dz_j
 *   where:
 *     scalar_grad = grad_out / N       (mean over N, so divide by N)
 *     d_pt        = α * [γ*(1-pt)^(γ-1)*log(pt) - (1-pt)^γ/pt]
 *     dp_dz_j     = p_t * (δ_{t,j} - p_j)
 */
torch::Tensor focal_loss_backward(
    const torch::Tensor& grad_out,   // scalar — upstream gradient
    const torch::Tensor& logits,     // [N, C] — from forward
    const torch::Tensor& targets,    // [N]    — from forward
    float alpha,
    float gamma)
{
    TORCH_CHECK(logits.is_contiguous(),  "logits must be contiguous");
    TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");

    const int64_t N = logits.size(0);
    const int64_t C = logits.size(1);

    // Recompute softmax probabilities (same as forward).
    // Memory trade-off: not saving probs in forward saves N*C*4 bytes,
    // at the cost of recomputing softmax in backward.
    auto probs  = torch::softmax(logits, 1);   // [N, C]
    auto grad_l = torch::zeros_like(logits);   // [N, C] output, zero-initialized

    const float*   __restrict__ p  = probs.data_ptr<float>();
    const int64_t* __restrict__ t  = targets.data_ptr<int64_t>();
    float*         __restrict__ gl = grad_l.data_ptr<float>();

    // Incorporate the 1/N factor from the mean() in forward.
    // grad_out is the scalar upstream gradient (typically 1.0 from loss.backward()).
    float scalar_grad = grad_out.item<float>() / static_cast<float>(N);

    at::parallel_for(0, N, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            int64_t cls = t[i];                         // true class index
            float pt    = std::max(p[i * C + cls], 1e-7f);  // true-class prob (clamped)
            float fm1   = 1.0f - pt;                    // (1 - p_t), used multiple times

            // ── Compute dFL/dp_t ─────────────────────────────────────────────
            // Derivative of FL = -α*(1-pt)^γ*log(pt) w.r.t. p_t:
            //
            //   d/dpt [-α*(1-pt)^γ*log(pt)]
            //
            // Using product rule: d(u*v)/dpt = u'*v + u*v'
            //   u  = (1-pt)^γ        u' = -γ*(1-pt)^(γ-1)
            //   v  = -α*log(pt)      v' = -α/pt
            //
            //   = -γ*(1-pt)^(γ-1) * (-α*log(pt)) + (1-pt)^γ * (-α/pt)
            //   = α*γ*(1-pt)^(γ-1)*log(pt) - α*(1-pt)^γ/pt
            //   = α * [ γ*(1-pt)^(γ-1)*log(pt) - (1-pt)^γ/pt ]
            //
            // Note: d_pt < 0 always (loss decreases when p_t increases),
            // which correctly sends a negative gradient to p_t.
            float d_pt = alpha * (gamma * std::pow(fm1, gamma - 1.0f) * std::log(pt)
                                  - std::pow(fm1, gamma) / pt);

            // ── Backprop through softmax: dp_t/dz_j ──────────────────────────
            // For each output logit z_j, compute the full gradient chain:
            //   dFL/dz_j = dFL/dp_t * dp_t/dz_j
            //
            // Softmax Jacobian: dp_t/dz_j = p_t * (δ_{t,j} - p_j)
            //   j == target:  p_t*(1-p_t)   (positive — increasing z_t raises p_t)
            //   j != target:  p_t*(-p_j)    (negative — increasing z_j lowers p_t)
            //
            // The p_t factor appears because softmax normalization couples all logits.
            for (int64_t c = 0; c < C; ++c) {
                // δ_{t,c}: 1 if this is the target class, else 0
                float kronecker  = (c == cls ? 1.0f : 0.0f);
                float dp_dz      = p[i * C + cls] * (kronecker - p[i * C + c]);

                // Multiply all factors together:
                //   scalar_grad: dL/d(FL) = 1/N (from the mean)
                //   d_pt:        dFL/dp_t
                //   dp_dz:       dp_t/dz_c
                gl[i * C + c] = scalar_grad * d_pt * dp_dz;
            }
        }
    });

    return grad_l;
}


// ─── Python bindings ─────────────────────────────────────────────────────────
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &focal_loss_forward,  "Focal Loss forward");
    m.def("backward", &focal_loss_backward, "Focal Loss backward");
}
