# Systematic Review Protocol

Protocol version: **0.1**
Search cutoff for the current snapshot: **2026-07-12**
Review type: **living systematic map with a narrative synthesis**

This protocol is written before the first full screening pass. Changes should
be versioned and explained so that the scope does not silently move to fit an
interesting result.

## 1. Review questions

1. Which methods currently recover candidate features, components, and causal
   circuits inside transformer language models?
2. How are circuit completeness, faithfulness, interpretability, scalability,
   stability, and generalization evaluated?
3. Which methods have produced actionable evidence about safety-relevant model
   behavior, including deception, hidden objectives, refusals, jailbreaks,
   bias, hallucination, memorization, evaluation awareness, persona drift, and
   dangerous capabilities?
4. Which claims reproduce across model families, scales, layers, prompts, and
   training checkpoints?
5. Which papers provide code, learned dictionaries/transcoders/lenses, data,
   or interactive interfaces that a small project can reuse?

## 2. Scope

### Core population

- Transformer language models, including autoregressive, multimodal, and
  diffusion-language variants when the method studies internal computation.
- Small algorithmic transformers when they establish a method or mechanistic
  result used in language-model research.
- Toy models when they isolate a phenomenon such as superposition, composition,
  phase changes, or mechanistic unfaithfulness.

### Core interventions and methods

- Activation, path, edge, node, feature, or causal patching.
- Circuit localization, discovery, pruning, validation, and comparison.
- Direct logit attribution, logit/tuned/Jacobian lenses, causal tracing, and
  related methods when used to make mechanistic claims.
- Sparse autoencoders, transcoders, crosscoders, dictionary learning, natural
  language autoencoders, and feature attribution graphs.
- Probing, steering, editing, or representation engineering when causal claims
  about internal mechanisms are tested.
- Automated explanation and interpretability agents.
- Benchmarks and critiques that test faithfulness, stability, completeness,
  identifiability, interpretability, or practical safety utility.

### Direct safety applications

Include work applying internal methods to deception, scheming, hidden goals,
alignment faking, reward-model gaming, refusal and jailbreak behavior,
hallucination, bias, truthfulness, memorization/privacy, prompt injection,
dangerous knowledge, persona/character drift, or monitoring.

### Peripheral sources

Broader explainability, model editing, probing, representation analysis, and AI
safety sources are retained only when they establish an assumption, metric,
dataset, or counterexample needed to assess core work.

## 3. Exclusion criteria

Exclude or mark peripheral sources that are solely:

- Black-box behavioral evaluation without analysis of internal computation.
- Prompt-based explanation or chain-of-thought analysis without internal
  measurements or interventions.
- Generic feature attribution/saliency with no clear connection to transformer
  mechanisms or evaluation standards used here.
- Attention visualization that treats attention weights as explanations without
  causal validation.
- A duplicate version of the same work; retain the most authoritative record
  and link alternate versions from its note.
- An uncheckable summary, marketing page, repost, or generated bibliography.

No source is excluded because its result is negative or critical of the field.

## 4. Information sources

The search uses multiple routes because terminology is inconsistent:

1. Scholarly indexes: arXiv, OpenAlex, Semantic Scholar, Crossref, ACL Anthology,
   OpenReview, PMLR, and conference proceedings.
2. Primary research collections: Transformer Circuits Thread, Distill Circuits,
   and official research pages from relevant labs.
3. Artifact discovery: official GitHub organizations/repositories, Hugging Face
   model and dataset cards, and Neuronpedia.
4. Backward and forward citation chasing from surveys, benchmark papers, and
   canonical methods.
5. Curated reading lists are candidate generators only; every included entry is
   verified against its canonical source.

## 5. Search concepts

Exact searches and hit counts belong in `data/search_log.csv`. Query families
combine terms from the following groups:

```text
("mechanistic interpretability" OR "mechanistic transparency")
(transformer OR "language model" OR LLM) AND
  (circuit OR "causal tracing" OR "activation patching" OR "path patching")
(transformer OR "language model") AND
  ("sparse autoencoder" OR transcoder OR crosscoder OR superposition)
(transformer OR "language model") AND
  ("feature steering" OR "activation steering" OR "representation engineering")
(transformer OR "language model") AND
  ("automated interpretability" OR "neuron explanation" OR "feature explanation")
```

Safety terms are added in a second pass: `deception`, `scheming`, `hidden
objective`, `alignment faking`, `backdoor`, `sleeper agent`, `refusal`,
`jailbreak`, `hallucination`, `bias`, `truthfulness`, `memorization`, `privacy`,
`prompt injection`, `evaluation awareness`, `persona`, and `dangerous
capability`.

## 6. Screening process

1. **Harvest:** store candidates and query metadata; deduplicate by DOI, arXiv
   identifier, canonical URL, and normalized title.
2. **Title/abstract screen:** apply the scope and exclusion criteria. Record
   explicit exclusions in `data/exclusions.csv`.
3. **Artifact verification:** resolve canonical URLs and check claimed code,
   model, data, and demo links.
4. **Full-text skim:** extract the claim, method, evaluation, limitations, and
   relationship to earlier work.
5. **Close read:** complete a note from `notes/TEMPLATE.md`.
6. **Replication:** rerun a central result or create a minimal falsification
   test; record environment and deviations.

Ambiguous sources remain `discovered` rather than being silently included as
established evidence.

## 7. Data extracted per source

The catalog stores bibliographic and triage metadata. Close-reading notes add:

- The exact behavioral phenomenon and model(s).
- Unit of analysis: neuron, head, MLP, residual direction, SAE feature, edge,
  subspace, or learned surrogate.
- Localization method and corruption/ablation baseline.
- Causal intervention and faithfulness/completeness metric.
- Interpretability evidence and who or what produced the explanation.
- Robustness across prompts, seeds, layers, models, and scales.
- Compute and artifact availability.
- Artifact license and maintenance status when code, weights, or data are used.
- Principal limitations, alternative explanations, and proposed falsifiers.
- Safety relevance that goes beyond an illustrative example.

## 8. Quality appraisal

Close-read sources receive separate judgments rather than a single vague score:

| Dimension | Key question |
| --- | --- |
| Construct validity | Does the metric measure the claimed mechanism or merely correlate with output? |
| Causal evidence | Do interventions change behavior as predicted, with appropriate controls? |
| Completeness | How much of the relevant behavior does the proposed circuit explain? |
| Specificity | Does the intervention preserve unrelated capabilities and representations? |
| Stability | Does the result survive seeds, paraphrases, baselines, and hyperparameters? |
| Generalization | Does it transfer across tasks, models, scales, or checkpoints? |
| Human interpretability | Are labels/explanations independently validated rather than assumed? |
| Reproducibility | Are code, data, weights/features, and sufficient details available? |
| Safety utility | Does the method help detect, predict, prevent, or audit a meaningful risk? |

## 9. Synthesis plan

The repository will provide:

- A chronological and topic-indexed evidence map.
- A method comparison matrix covering assumptions, unit of analysis, cost,
  causal status, model support, and known failure modes.
- Safety case studies separated by strength of mechanistic evidence.
- Replication notes and negative results.
- A gap analysis used to choose a small novel project.

Meta-analysis is not planned because tasks and metrics are currently too
heterogeneous. Quantitative results may be compared within benchmark families.

## 10. Known limitations

- A living review cannot guarantee literal completeness, especially for recent
  preprints and informal research notes.
- English-language and openly indexed material will be easier to find.
- “Mechanistic interpretability,” “representation engineering,” and “model
  editing” have disputed boundaries; borderline decisions will be visible.
- Repository activity and artifact availability can change after verification.
- A source being open access does not imply its code, weights, or data are open.
