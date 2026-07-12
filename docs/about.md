---
title: About the Course
description: Scope, evidence policy, maintenance, attribution, and limitations.
---

# About this course

**Inside the Model** is the teaching layer of the open [AI Safety, Interpretability, and Circuits](https://github.com/Danny0951/ai-safety-interpretability-circuits) systematic-review repository.

## Design principles

- **Visual before verbal overload:** diagrams establish structure; prose supplies precision.
- **Causal before narrative:** labels and heatmaps generate hypotheses; interventions test them.
- **Failure modes beside methods:** limitations are taught where enthusiasm is highest.
- **Open artifacts first:** labs prefer released models, SAEs, transcoders, lenses, code, and datasets.
- **Small-model validity before scale:** a cheap experiment should establish that the metric means what we think.
- **Primary sources:** modules link directly to papers, official research articles, and official repositories.

## Scope

Core coverage includes transformer computation, causal interventions, circuit discovery and evaluation, superposition, sparse features, transcoders, crosscoders, attribution graphs, lenses, automated explanations, steering, model diffing, and direct safety applications.

The course does not attempt to survey all explainable AI or all AI safety. Behavioral evaluation, chain-of-thought analysis, and general model editing appear when they clarify or test an internal mechanistic claim.

## Evidence policy

The course uses five deliberately separate labels:

1. **Observed:** an activation, output, or association was measured.
2. **Attributed:** a score assigns influence under explicit assumptions.
3. **Intervened:** changing an internal variable changes the measured behavior.
4. **Mechanistically explained:** components and interactions predict new interventions.
5. **Safety-useful:** the method improves detection, prediction, prevention, or auditing under realistic constraints.

Later labels require stronger evidence; they are not synonyms.

## Currency

The initial course snapshot covers primary work and open artifacts through **2026-07-12**. The research catalog is living, while course modules change more slowly after claims are screened and synthesized.

## Limitations

- The literature remains too heterogeneous for a clean quantitative meta-analysis.
- Important Anthropic results use closed Claude weights and unreleased dictionaries, limiting independent replication.
- “Open source,” “source available,” “open weights,” and “hosted demo” are distinct; the repository records them separately when possible.
- Course diagrams simplify. Equations, interventions, and source papers determine the precise claim.
- No course can substitute for running experiments and discovering that an apparently obvious metric was wrong.

## Attribution and license

Original course text, diagrams, and code are released under the repository's MIT License. Linked sources and artifacts retain their own copyrights and licenses. Cite original research for scientific claims; cite this repository only for its synthesis or course structure.

