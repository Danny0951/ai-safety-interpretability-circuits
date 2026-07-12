---
title: Reading Map
description: A sequenced path through the primary sources that define the field.
---

# Reading map

Do not read chronologically from a 2,000-paper search result. Read an **evidence chain**: each source should answer a question created by the previous one.

## The 24-source spine

### A · What is a circuit?

1. [Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/) — the feature/circuit worldview.
2. [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/) — residual paths, QK/OV, composition.
3. [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/) — mechanism formation across training.
4. [Interpretability in the Wild: IOI](https://arxiv.org/abs/2211.00593) — a full hand-built language-model circuit.

**Read for:** definitions, decomposition choices, and what authors count as causal evidence.

### B · How do we test and find circuits?

5. [Causal Scrubbing](https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing) — hypotheses as invariances.
6. [Localizing Model Behavior with Path Patching](https://arxiv.org/abs/2304.05969) — isolating routes.
7. [ACDC](https://arxiv.org/abs/2304.14997) — automated greedy pruning.
8. [Edge Attribution Patching](https://arxiv.org/abs/2310.10348) — fast gradient screening.
9. [AtP*](https://arxiv.org/abs/2403.00745) — saturation and cancellation fixes.
10. [EAP-IG](https://arxiv.org/abs/2403.17806) — faithfulness beyond matching a presumed circuit.
11. [Mechanistic Interpretability Benchmark](https://arxiv.org/abs/2504.13151) — shared comparative evaluation.

**Read for:** the baseline, computational cost, failure mode, and validation standard of each method.

### C · Why features instead of neurons?

12. [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/) — why polysemanticity can be useful.
13. [Sparse Autoencoders Find Highly Interpretable Features](https://arxiv.org/abs/2309.08600) — independent early SAE evidence.
14. [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/) — dictionary learning, splitting, feature dashboards.
15. [Scaling and Evaluating Sparse Autoencoders](https://arxiv.org/abs/2406.04093) — TopK and scaling.
16. [Gemma Scope](https://arxiv.org/abs/2408.05147) — open dictionaries across a model family.
17. [SAEBench](https://arxiv.org/abs/2503.09532) — multidimensional evaluation.
18. [Are SAE Benchmarks Reliable?](https://arxiv.org/abs/2605.18229) — measurement noise and discriminability.

**Read for:** which claims come from the objective, which come from examples, and which survive intervention.

### D · From features to computations

19. [Transcoders Find Interpretable Feature Circuits](https://arxiv.org/abs/2406.11944) — sparse MLP replacement.
20. [Sparse Feature Circuits](https://openreview.net/forum?id=I4e82CIDxv) — feature graphs with error terms.
21. [Circuit Tracing: Methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) — cross-layer transcoders and local graphs.
22. [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) — diverse mechanism case studies.
23. [Tracing Attention Computation Through Feature Interactions](https://transformer-circuits.pub/2025/attention-qk/) — feature-pair QK effects.
24. [A Toy Model of Mechanistic Unfaithfulness](https://transformer-circuits.pub/2025/faithfulness-toy-model/) — output fidelity can hide the wrong mechanism.

**Read for:** the difference between an original model, its sparse replacement, and a pruned visualization of that replacement.

## The frontier and safety extension

After the spine, add sources according to your question:

| Question | Sources |
| --- | --- |
| Can activations speak? | [Activation Oracles](https://alignment.anthropic.com/2025/activation-oracles/), [NLA](https://transformer-circuits.pub/2026/nla/), [Global Workspace/J-lens](https://transformer-circuits.pub/2026/workspace/) |
| Can directions control behavior? | [Inference-Time Intervention](https://arxiv.org/abs/2306.03341), [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681), [Evaluating Feature Steering](https://www.anthropic.com/research/evaluating-feature-steering) |
| How does refusal work? | [Single refusal direction](https://arxiv.org/abs/2406.11717), [more than one direction](https://arxiv.org/abs/2602.02132), Biology jailbreak/refusal case studies |
| Can hidden objectives be audited? | [Auditing Hidden Objectives](https://www.anthropic.com/research/auditing-hidden-objectives), [AuditBench](https://alignment.anthropic.com/2026/auditbench/), [Model Organisms for Emergent Misalignment](https://arxiv.org/abs/2506.11613) |
| How do personas drift? | [Persona Vectors](https://arxiv.org/abs/2507.21509), [Assistant Axis](https://arxiv.org/abs/2601.10387), [Emotion Concepts](https://transformer-circuits.pub/2026/emotions/) |
| How do models differ? | [Sparse Crosscoders](https://transformer-circuits.pub/2024/crosscoders/), [Cross-Architecture Diffing](https://arxiv.org/abs/2602.11729) |

## A five-question reading note

For every source, answer before moving on:

1. What exact claim is made?
2. What is the unit of analysis?
3. Which experiment is actually causal?
4. What is the strongest alternative explanation?
5. Which released artifact lets us test it ourselves?

Use the course [research templates](research-templates.md) for close reading and replication.

