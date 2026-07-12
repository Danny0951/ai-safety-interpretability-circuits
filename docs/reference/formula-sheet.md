---
title: Formula Sheet
description: Shared notation and the core equations used across the course.
---

# Formula sheet

## Shared notation

| Symbol | Meaning |
| --- | --- |
| (T) | sequence length |
| (d) | residual-stream width |
| (L) | number of transformer layers |
| (x_{ell,t}) | residual state at layer (ell), token position (t) |
| (a_{ell,t}) | attention block write |
| (m_{ell,t}) | MLP block write |
| (W_U) | unembedding matrix |
| (z_i) | activation of learned feature (i) |
| Δ | clean minus corrupt difference unless stated otherwise |

## Transformer residual update

For a simplified pre-normalization block:

$$
a_\ell = \operatorname{Attn}_\ell(\operatorname{Norm}(x_\ell)), \qquad
m_\ell = \operatorname{MLP}_\ell(\operatorname{Norm}(x_\ell+a_\ell)),
$$

$$
x_{\ell+1} = x_\ell + a_\ell + m_\ell.
$$

The residual stream is additive, but component functions are nonlinear and read the sum of earlier writes.

## Attention

For head (h):

$$
Q = XW_Q^h, \quad K=XW_K^h, \quad V=XW_V^h,
$$

$$
A^h = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + M\right),
\qquad O^h = A^h V W_O^h.
$$

The **QK circuit** determines *where to read*; the **OV circuit** determines *what to write*:

$$
W_{QK}^h = W_Q^h(W_K^h)^\top,
\qquad
W_{OV}^h = W_V^h W_O^h.
$$

## Logits and logit difference

Ignoring final normalization for the moment:

$$
\text{logits} = W_U x_L.
$$

For correct token (y^+) and contrast token (y^-):

$$
\Delta_{\text{logit}}
= \text{logit}(y^+) - \text{logit}(y^-)
= (W_U[y^+] - W_U[y^-])^\top x_L.
$$

## Direct logit attribution

If component (c) writes vector (r_c), its direct contribution is approximately:

$$
\operatorname{DLA}(c)
= (W_U[y^+] - W_U[y^-])^\top r_c,
$$

with care required for final normalization. DLA excludes the component's downstream effects.

## Activation patching effect

Let (F(x)) be a scalar behavior metric, and let (h_c(x)) be component (c)'s activation. A clean-to-corrupt patch is:

$$
E_c = F\big(x^{\text{corrupt}};\; h_c \leftarrow h_c(x^{\text{clean}})\big)
- F(x^{\text{corrupt}}).
$$

A normalized recovery score often uses:

$$
R_c = \frac{F(\text{patched})-F(\text{corrupt})}
{F(\text{clean})-F(\text{corrupt})}.
$$

The denominator can be unstable when clean and corrupt behaviors are similar.

## First-order attribution patching

For activation change (Δ h = h^{\text{clean}}-h^{\text{corrupt}}):

$$
\Delta F \approx \nabla_h F(h^{\text{corrupt}})^\top \Delta h.
$$

This fails under saturation, cancellation, or strong nonlinear interactions.

## Integrated gradients

Along path (h(\alpha)=h_0+\alpha(h_1-h_0)):

$$
\operatorname{IG}(h_0,h_1)
= (h_1-h_0) \odot
\int_0^1 \nabla_h F(h(\alpha))\,d\alpha.
$$

Numerical estimates depend on the chosen path, steps, and endpoints.

## Sparse autoencoder

Encoder and decoder:

$$
z = f\big(W_{enc}(x-b_{dec})+b_{enc}\big),
\qquad
\hat{x} = W_{dec}z+b_{dec}.
$$

A standard objective is:

$$
\mathcal{L}_{SAE}
= \|x-\hat{x}\|_2^2 + \lambda\|z\|_1.
$$

TopK variants set all but the (k) largest encoder preactivations to zero. Common reporting variables:

$$
L_0 = \mathbb{E}[\#\{i:z_i\neq0\}],
\qquad
\text{FVE}=1-\frac{\mathbb{E}\|x-\hat{x}\|_2^2}{\mathbb{E}\|x-\bar{x}\|_2^2}.
$$

Neither low (L_0) nor high FVE establishes interpretability.

## Transcoder

A transcoder predicts an MLP output from its input:

$$
z=f(W_{enc}x_{in}+b_{enc}),
\qquad
\widehat{m(x_{in})}=W_{dec}z+b_{dec}.
$$

Decoder direction (d_i) gives a fixed feature write (z_i d_i), enabling input-independent edge weights. The residual error is:

$$
\epsilon = m(x_{in})-\widehat{m(x_{in})}.
$$

## Crosscoder

For layers or models (j=1,\ldots,n):

$$
z=f\!\left(\sum_j W^{(j)}_{enc}x^{(j)}+b\right),
\qquad
\hat{x}^{(j)}=W^{(j)}_{dec}z+b^{(j)}.
$$

Shared and exclusive decoder norms are useful discovery signals, not causal proof of model differences.

## Feature steering

Add direction (v) at layer (ell) and selected positions:

$$
x_{\ell,t}' = x_{\ell,t}+\alpha v.
$$

Always examine a dose response over (α), target behavior, off-target KL divergence, and general capability.

## Jacobian lens

Average transport from layer (ell) to the final residual:

$$
J_\ell = \mathbb{E}_{x,t,t'\ge t}
\left[\frac{\partial x_{L,t'}}{\partial x_{\ell,t}}\right].
$$

Readout:

$$
\operatorname{JLens}_\ell(x)
= \operatorname{softmax}\big(W_U\operatorname{Norm}(J_\ell x)\big).
$$

It is an averaged first-order transport and inherits the vocabulary's tokenization.

## Circuit similarity

Unweighted Jaccard overlap:

$$
J(C_1,C_2)=\frac{|C_1\cap C_2|}{|C_1\cup C_2|}.
$$

Weighted overlap for nonnegative edge scores (w_1,w_2):

$$
J_w=\frac{\sum_e\min(w_{1e},w_{2e})}
{\sum_e\max(w_{1e},w_{2e})}.
$$

Low overlap does not necessarily mean different functions: distributed or redundant implementations and score variance can change selected edges.

