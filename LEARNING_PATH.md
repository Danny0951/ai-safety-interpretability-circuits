# Learning Path

This path optimizes for understanding and research judgment, not for finishing
the largest possible reading list. Each phase has an artifact that demonstrates
learning. Canonical readings will be linked to catalog IDs as screening
progresses.

## Phase 0 — Orientation and prerequisites

Learn the transformer forward pass well enough to name every value in an
activation cache: embeddings, residual stream, attention Q/K/V and patterns,
head outputs, MLP inputs/outputs, normalization, logits, and loss.

Deliverable: derive a one-layer decoder-only transformer forward pass and use
hooks to verify tensor shapes on a small open model.

## Phase 1 — The circuits viewpoint

Read the original Circuits work, the mathematical framework for transformer
circuits, induction heads, and one careful hand-built language-model circuit.
Learn composition through the residual stream, QK/OV decomposition, direct
logit attribution, activation patching, and path patching.

Deliverable: reproduce a compact result on GPT-2 Small, including a clean and a
corrupted dataset, a behavioral metric, an intervention, and an ablation.

## Phase 2 — Causality and automated circuit discovery

Compare activation patching, causal tracing, ACDC-style pruning, edge
attribution/patching, integrated-gradients variants, attribution graphs, and
learned sparse circuits. Pay close attention to the corruption distribution and
the difference between localization and explanation.

Deliverable: run at least two discovery methods on the same task and compare
sparsity, faithfulness, runtime, and stability across prompt templates.

## Phase 3 — Superposition and sparse features

Study toy models of superposition, sparse autoencoders, TopK/JumpReLU/Gated
variants, transcoders, crosscoders, feature absorption/splitting, and SAE
benchmarks. Treat feature labels as hypotheses rather than ground truth.

Deliverable: use released SAEs or transcoders on an open-weight model; validate
one feature with both dataset examples and a causal intervention, then document
one failure or ambiguity.

## Phase 4 — From features to computations

Study sparse feature circuits, cross-layer transcoders, circuit tracing, and
attention-feature interactions. Learn where surrogate error nodes enter an
attribution graph and why graph edges alone are not a proof of mechanism.

Deliverable: produce and annotate an attribution graph for a behavior not used
in the tool’s tutorial, then test at least two graph-derived predictions.

## Phase 5 — Safety case studies

Read work on refusal/jailbreaks, hidden objectives, deception and evaluation
awareness, hallucination, bias, memorization, persona drift, truthfulness, and
internal monitoring. Compare simple probes/directions against expensive feature
or circuit methods.

Deliverable: write a comparison table separating detection, causal control,
mechanistic understanding, robustness to adversaries, and collateral damage.

## Phase 6 — Current frontier methods

Study model diffing, natural-language autoencoders, Jacobian lenses/global
workspace work, automated auditing agents, circuit explainers, and recent
critiques of SAE identifiability and circuit universality.

Deliverable: replicate one frontier claim on an open model or design a strong
falsification test where full replication is computationally infeasible.

## Phase 7 — Novel project selection

A project idea graduates only when it has:

1. A precise claim that could be false.
2. A behavioral dataset and metric.
3. A causal or mechanistic measurement plan.
4. At least one cheap baseline and one current method.
5. A robustness/generalization axis.
6. A compute estimate and a one-day kill test.
7. A literature table showing the nearest prior work and the remaining gap.

The preferred first project will reuse released open-weight models and learned
features/transcoders/lenses so that compute is spent on evidence, not on
rebuilding infrastructure.
