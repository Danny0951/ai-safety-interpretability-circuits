---
title: Glossary
description: Precise definitions for the recurring language of circuits, sparse features, interventions, and safety auditing.
---

# Glossary

Definitions describe how terms are used in this course. Some remain contested in the research community.

## A

Ablation
: An intervention intended to remove or replace a component's information or effect. Zero, mean, resample, and learned-replacement ablations answer different counterfactual questions.

Activation
: A model-internal numerical value produced on a particular input. The term can mean a scalar neuron value, a vector at one token position, or a full tensor over a batch and sequence.

Activation addition
: Steering by adding a direction to one or more internal activations during a forward pass.

Activation cache
: Stored intermediate activations from a model run, usually indexed by component, layer, and hook point.

Activation patching
: Replacing an activation in one run with the corresponding activation from another run and measuring the behavioral effect. Also called causal tracing in some settings.

Activation Oracle
: A language model trained or prompted to answer natural-language questions about another model's activation. Its answer can combine true signal with inference or confabulation.

Attribution
: Assignment of a numerical contribution or influence score under specified assumptions. Attribution is not automatically equivalent to an intervention effect.

Attribution graph
: A prompt-local directed graph whose nodes may include tokens, sparse features, errors, and logits, with edges approximating contribution to downstream activations or outputs.

Attribution patching (AtP)
: A first-order gradient approximation to many activation-patching experiments.

AtP*
: A family of corrections to attribution patching targeting attention saturation and gradient cancellation, including a QK fix and GradDrop.

Automated interpretability
: Use of algorithms—often language models—to generate, score, test, or organize interpretations of model internals.

## B

Baseline
: The reference state used to define an intervention or attribution. Baseline choice is part of the scientific claim, not an implementation detail.

BatchTopK SAE
: A sparse autoencoder that enforces a TopK budget across a batch or token collection rather than independently at each token.

Behavioral metric
: A scalar operationalization of the behavior under study, such as a logit difference, probability, loss, answer accuracy, refusal rate, or calibrated audit success.

Black-box audit
: Investigation using inputs and outputs without direct access to weights or activations.

## C

Causal abstraction
: A formal relationship in which variables and interventions in a higher-level model correspond to variables and interventions in a lower-level neural system.

Causal scrubbing
: A method for testing whether a model's computation respects the invariances implied by an interpretability hypothesis using resampled interventions.

CEAP
: Conductance-based Edge Attribution Patching, an integrated-attribution approach designed to preserve additive ordering and reduce sampling variance in circuit discovery.

Circuit
: A subgraph of model variables and interactions proposed to implement a behavior. Circuit boundaries depend on the unit of analysis, input distribution, metric, and ablation scheme.

Circuit completeness
: The extent to which a proposed circuit accounts for all relevant causal pathways under a defined evaluation. Redundant OR-like paths make completeness difficult.

Circuit faithfulness
: How well the behavior of a selected circuit or replacement reflects the original model under a specified intervention scheme. There is no single universally sufficient faithfulness metric.

Circuit minimality
: Whether components can be removed from a circuit without materially reducing its claimed function.

Clean/corrupt pair
: Inputs constructed so that a target variable or correct answer differs while other relevant properties remain matched. Poor pairs create misleading patches.

Component
: A chosen unit such as an attention head, MLP, neuron, residual direction, feature, layer, or edge.

Composition
: Multiple features or computations represented together by adding or combining distinct constituents, contrasted with unrelated features sharing dimensions through superposition.

Contrastive activation addition (CAA)
: Steering using a direction derived from average activation differences between contrastive positive and negative examples.

Crosscoder
: A sparse dictionary model that reads from or reconstructs multiple layers, checkpoints, or models using a shared latent feature set.

Cross-layer transcoder (CLT)
: A sparse replacement model designed to expose feature computations and writes across multiple transformer layers.

## D

Decodability
: The ability of a probe or decoder to recover information from activations. Information can be decodable without being used by the model.

Decoder direction
: A column of a dictionary decoder matrix describing the activation-space write associated with one learned feature.

Dedicated Feature Crosscoder (DFC)
: A crosscoder construction that explicitly allocates capacity to shared and model-exclusive features for model diffing.

Dictionary learning
: Learning an overcomplete set of directions that reconstruct data using sparse coefficients.

Direct logit attribution (DLA)
: Algebraic projection of a component's residual write onto an output-logit or logit-difference direction, excluding downstream mediation.

Dose response
: Measurement of target and off-target effects across intervention strengths rather than reporting one favorable coefficient.

## E

EAP
: Edge Attribution Patching, a gradient method for ranking edges by an approximation to their patching effect.

EAP-IG
: Edge attribution using integrated gradients across an interpolation path to reduce some first-order approximation failures.

Error node
: A node representing residual reconstruction error introduced by replacing original computation with a sparse dictionary or transcoder. Large error influence weakens graph interpretations.

Evaluation awareness
: Internal or behavioral recognition that the model is being tested, monitored, trained, or placed in a constructed evaluation.

Explanation score
: A metric comparing a proposed natural-language explanation with feature activation behavior, often by asking another model to predict activations. It is a proxy, not semantic ground truth.

## F

Feature
: A property or latent variable useful for describing data or computation. In learned dictionaries, “feature” usually means one decoder direction and its scalar activation; it need not be uniquely identifiable or human-interpretable.

Feature absorption
: An SAE failure in which a concept's signal is partially captured by features centered on more specific or correlated contexts, obscuring the expected general feature.

Feature splitting
: Representation of one intuitive concept across multiple learned features, often increasing with dictionary width or data variation.

Feature steering
: Intervention by adding, removing, clamping, or swapping a learned feature's activation.

Fraction of variance explained (FVE)
: One minus reconstruction error divided by baseline variance. It measures reconstruction, not by itself interpretability.

## G

Gated SAE
: An SAE architecture that separates whether a feature activates from estimating its magnitude, designed partly to reduce shrinkage.

Generalization
: Persistence of a claim across held-out examples, prompt templates, contexts, seeds, models, scales, checkpoints, or tasks.

Global workspace / J-space
: A proposed sparse set of verbalizable representations that can be reported, deliberately modulated, and flexibly used in reasoning. The Jacobian Lens approximates access to this space.

GradDrop
: An AtP* strategy that reduces cancellation by separating gradients accumulated over different downstream paths or positions.

Ground-truth circuit
: A known computational graph, often compiled or injected in synthetic benchmarks. It may not resemble naturally learned mechanisms.

## H

Head
: One parallel attention operation with its own query, key, value, and output matrices.

Hidden objective
: A goal-like training-induced regularity that explains behavior but is not directly admitted or visible in ordinary outputs.

Hook point
: A named location where an instrumentation library reads or changes a model tensor.

Hydra effect / self-repair
: Compensation by other components after ablation of a component, which can hide importance or create misleading necessity conclusions.

## I

Induction head
: An attention head that implements a pattern like “find a previous occurrence of the current token and copy what followed it,” often through composition with a previous-token mechanism.

Integrated gradients
: Attribution obtained by integrating gradients along a path from a baseline to an input or activation.

Interchange intervention
: Replacing the value of a candidate neural variable with its value from a source example and comparing the result with a corresponding high-level causal intervention.

Intervention
: An externally imposed change to a model variable during computation. Causal interpretation requires specifying exactly what is changed and held fixed.

IOI
: Indirect Object Identification, a canonical GPT-2 Small task used to study a hand-built circuit and benchmark later discovery methods.

## J

Jacobian
: The matrix of partial derivatives mapping small changes in one vector to first-order changes in another.

Jacobian fidelity
: Agreement between the derivatives of a replacement model and the original model, stronger than output reconstruction but still local.

Jacobian Lens / J-lens
: A lens that transports an intermediate residual state through an average layer-to-final-state Jacobian before unembedding it into token scores.

JumpReLU SAE
: An SAE using a learned activation threshold so features activate only above feature-specific jumps.

## K–L

Knowledge neuron
: A neuron identified as associated with factual or semantic knowledge, commonly through attribution. The term does not guarantee a localized or unique memory cell.

Lens
: A map from intermediate activations to an interpretable output space, typically vocabulary logits.

Linear representation hypothesis
: The proposal that many model features correspond approximately to directions or linear subspaces in activation space.

Localization
: Identifying internal sites associated with or causally mediating behavior, without necessarily explaining their computation.

Logit difference
: Difference between logits for a target and contrast token; a signed and often more stable circuit metric than probability.

Logit lens
: Application of the model's unembedding to intermediate residual states as if they were final states.

## M

Mechanistic interpretability
: The effort to explain model behavior using internal variables, operations, and causal interactions that support testable predictions.

Mechanistic unfaithfulness
: A replacement or explanation matching observed outputs while using or depicting a different internal mechanism.

MIB
: Mechanistic Interpretability Benchmark, a shared evaluation suite for circuit and causal-variable localization across models and tasks.

Model diffing
: Comparison of internal representations or mechanisms across checkpoints, fine-tunes, seeds, or architectures.

Model organism
: A deliberately constructed or selected model exhibiting a known behavior so auditing or alignment methods can be tested against hidden ground truth.

Monosemanticity
: The degree to which a unit has one coherent human-interpretable meaning across its activation distribution. It is gradual and observer-dependent, not a guaranteed binary property.

MOLT
: Sparse Mixture of Linear Transforms, a proposed conditional low-rank alternative for sparse replacement of MLP computation.

## N–O

Natural Language Autoencoder (NLA)
: A pair of models mapping activation vectors to text descriptions and reconstructing activation directions from those descriptions.

Necessity
: A component is necessary relative to an intervention if removing it degrades the behavior. Redundancy can make genuinely used components appear unnecessary.

Neuron
: A scalar hidden unit, usually after a nonlinearity. Neurons are a privileged basis in MLP space but can be polysemantic.

Neuronpedia
: An open hosted platform for browsing model neurons and sparse features, explanations, steering, and attribution graphs.

Noising intervention
: Moving a clean run toward a corrupt state by replacing internal variables with corrupt or resampled values.

Off-manifold intervention
: A patched or steered activation unlike states the downstream model encounters naturally, potentially causing uninterpretable behavior.

OV circuit
: The value-to-output transformation of an attention head, describing what information the head writes conditional on attention.

## P–Q

Path patching
: Intervention designed to isolate the effect transmitted from specified sender components to specified receivers while preserving other routes.

Persona vector
: A direction associated with a character trait or assistant persona, often derived contrastively and used for monitoring or steering.

PLT
: A per-layer transcoder or related sparse MLP replacement used by some Circuit Tracer model releases.

Polysemanticity
: A single unit responding to multiple apparently unrelated features or functions.

Probe
: A learned predictor from activations to labels. Probe accuracy establishes decodability subject to controls, not necessarily causal use.

QK circuit
: The query–key bilinear interaction determining pre-softmax attention scores and thus where a head reads.

## R

Reconstruction error
: Difference between an activation or component output and its reconstruction by a learned dictionary or surrogate.

Representation engineering
: Measurement and control of population-level activation patterns associated with concepts or behaviors.

Resample ablation
: Replacement of a variable with a value sampled from an input distribution chosen to remove target information while keeping values in distribution.

Residual stream
: The additive vector state passed through transformer layers, read from and written to by attention and MLP blocks.

Refusal direction
: An activation direction whose addition or removal strongly modulates refusal behavior in a model. Multiple directions can affect how refusal is expressed.

Replacement model
: A model or component trained to approximate original computation in a more interpretable form, such as a transcoder.

## S

SAE
: Sparse autoencoder: an encoder-decoder trained to reconstruct activations using relatively few active latent units.

SAEBench
: A suite of reconstruction, interpretability, disentanglement, probing, unlearning, and practical metrics for comparing SAEs.

Safety case
: A structured argument and evidence that a system is acceptably safe under a defined deployment and threat model. Interpretability may supply evidence but is not a complete case by itself.

Sparse frame
: An overcomplete set of vectors in which points are represented by sparse combinations; unlike a basis, representations need not be unique.

Steering vector
: A direction added to or removed from activations to alter model behavior.

Sufficiency
: A component set is sufficient under an ablation scheme if preserving it while ablating others recovers the target behavior.

Superposition
: Encoding more sparse features than available dimensions by assigning features to non-orthogonal directions.

## T–Z

TopK SAE
: An SAE whose encoder retains only the (k) largest feature activations for each chosen sparsity group.

Transcoder
: A sparse model mapping a component's input activation to its output rather than reconstructing the same activation.

TransformerLens
: An instrumentation library exposing hooks, caches, decompositions, and patching utilities for transformer interpretability.

Tuned lens
: A lens with learned per-layer translators trained to align intermediate states with the final output distribution.

Universality
: Recurrence of a feature, component type, geometry, or mechanism across tasks, seeds, models, scales, or architectures.

White-box audit
: Investigation with access to some combination of weights, gradients, activations, training data, or internal interventions.

Zero ablation
: Setting a variable to zero. Cheap but often off-distribution and confounded with removing mean or scale.

