# Topic Taxonomy

The controlled tags below live in `data/topics.txt`. Multiple tags are expected;
the taxonomy describes methods and questions rather than mutually exclusive
paper categories.

## Foundations

- `foundations`: definitions, mathematical frameworks, and field-level agendas.
- `transformer-mechanics`: residual streams, attention, MLPs, normalization,
  tokenization, and parameter-level transformer analysis.
- `superposition`: polysemanticity, feature geometry, capacity, and toy models.
- `induction-heads`: induction circuits and in-context learning mechanisms.
- `training-dynamics`: development of mechanisms over training, phase changes,
  grokking, and checkpoint analysis.

## Finding and testing mechanisms

- `activation-patching`: activation replacement, causal tracing, interchange
  interventions, and patching baselines.
- `attribution`: direct logit attribution, integrated gradients, edge/path
  scores, and information-flow attribution.
- `circuit-discovery`: manual or automated localization of causal subgraphs.
- `circuit-evaluation`: faithfulness, completeness, minimality, specificity,
  stability, and circuit benchmark methods.
- `causal-abstraction`: mapping neural computation onto higher-level algorithms.
- `lenses`: logit, tuned, Patchscopes, Jacobian, and related readout methods.
- `probing`: classifiers or probes used to test internal representations.

## Features and learned dictionaries

- `sparse-autoencoders`: SAE architectures, training, scaling, and evaluation.
- `transcoders`: sparse MLP input-to-output replacements and feature circuits.
- `crosscoders`: cross-layer or cross-model dictionary learning and model diffing.
- `feature-explanations`: human or automated semantic labeling of features.
- `feature-steering`: causal interventions on learned features.
- `model-diffing`: internal comparison of checkpoints, fine-tunes, or unrelated
  model architectures.

## Behaviors and domains

- `reasoning`: arithmetic, multi-hop inference, planning, and chain-of-thought.
- `factual-recall`: entity recognition, knowledge retrieval, and editing.
- `linguistic-circuits`: syntax, sequence continuation, copying, and language.
- `multilingual`: shared and language-specific representations or circuits.
- `multimodal`: vision-language, image, SVG/ASCII, audio, or other modalities.
- `agents`: tool use, long-horizon behavior, and agentic auditing.

## Safety applications

- `deception`: hidden objectives, scheming, reward gaming, or concealed intent.
- `evaluation-awareness`: internal recognition of tests or oversight.
- `refusal-jailbreaks`: refusal mechanisms, jailbreaks, and adversarial prompts.
- `hallucination`: mechanisms or internal predictors of unsupported outputs.
- `bias`: social, political, demographic, or reward-model bias.
- `truthfulness`: truth representations, probes, and causal truth interventions.
- `memorization-privacy`: memorization, extraction, copyright, and privacy risks.
- `backdoors`: trojans, sleeper agents, and latent triggers.
- `persona-character`: assistant identity, persona vectors, and character drift.
- `dangerous-capabilities`: cyber, biological, chemical, nuclear, or other
  dual-use knowledge and capabilities.
- `monitoring`: internal classifiers, anomaly detection, and runtime safeguards.

## Scaling and research practice

- `automated-interpretability`: language-model-generated explanations and agents.
- `interpretability-evaluation`: benchmarks for explanation quality or utility.
- `scalability`: compute, memory, approximation, or frontier-model constraints.
- `universality`: cross-task, cross-model, and cross-scale mechanism comparison.
- `robustness`: sensitivity to prompts, baselines, seeds, or adversaries.
- `tools`: reusable software, interfaces, and infrastructure.
- `datasets`: datasets, benchmarks, and labeled evaluation sets.
- `education`: curricula, exercises, glossaries, and serious tutorials.
- `survey`: systematic or narrative reviews of the field.
