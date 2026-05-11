# Sparse Autoencoders for Interpretable Medical Image Representation Learning

Reference: Philipp Wesp, Robbie Holland, Vasiliki Sideri-Lampretsa, Sergios Gatidis. *Sparse Autoencoders for Interpretable Medical Image Representation Learning*. arXiv:2603.23794v1, 24 March 2026.

Reviewed notes prepared for Yoan David.

Tags: `Sparse Autoencoders` `Medical Imaging` `Interpretability` `Foundation Models` `Mechanistic Interpretability` `DINOv3` `BiomedParse`

## Table of Contents

- [Highlights](#highlights)
- [The core idea](#the-core-idea)
- [Methodology, the architecture](#methodology-the-architecture)
  - [1. Start from frozen foundation model embeddings](#1-start-from-frozen-foundation-model-embeddings)
  - [2. Train a Sparse Autoencoder on those embeddings](#2-train-a-sparse-autoencoder-on-those-embeddings)
  - [3. Use a Matryoshka SAE instead of a single-level SAE](#3-use-a-matryoshka-sae-instead-of-a-single-level-sae)
  - [4. Force sparsity with BatchTopK during training](#4-force-sparsity-with-batchtopk-during-training)
  - [5. Replace BatchTopK with JumpReLU at inference](#5-replace-batchtopk-with-jumprelu-at-inference)
  - [6. Train only with reconstruction loss](#6-train-only-with-reconstruction-loss)
- [Datasets](#datasets)
- [Experiments](#experiments)
  - [Experiment 1: SAE quality](#experiment-1-sae-quality)
  - [Experiment 2: SAE configuration ranking](#experiment-2-sae-configuration-ranking)
  - [Experiment 3: Sparse feature interpretability](#experiment-3-sparse-feature-interpretability)
- [Metrics and quality measures](#metrics-and-quality-measures)
- [Main results](#main-results)
- [Limitations](#limitations)
- [My interpretation](#my-interpretation)
- [Cheat sheet of the important concepts](#cheat-sheet-of-the-important-concepts)

## Highlights

> - The paper studies whether Sparse Autoencoders can turn opaque medical image foundation model embeddings into sparse, interpretable, human-readable features.
> - The authors train Matryoshka SAEs on frozen embeddings from BiomedParse and DINOv3 using 909,873 CT and MRI 2D slices from TotalSegmentator.
> - The SAE is not trained to solve a clinical task directly. It is trained to reconstruct the embedding produced by a foundation model, while forcing the representation to use only a small number of active features.
> - The sparse codes preserve a surprising amount of information. With only 10 active features, BiomedParse recovers up to 87.8% of dense downstream performance.
> - Sparse fingerprints with only 5 active features preserve most dense retrieval quality, especially for BiomedParse.
> - DINOv3, even though it is not specifically biomedical, produces more monosemantic features than BiomedParse.
> - The most interesting result is not just compression. The sparse features often correspond to concepts such as modality, orientation, anatomical region, and sometimes demographics.
> - The paper uses VLM-based auto-interpretation and an independent VLM judge to evaluate whether generated feature descriptions actually match the images that activate those features.
> - The zero-shot language retrieval experiment shows a possible bridge between clinical text queries and image retrieval using automatically labeled sparse features.
> - The biggest caveat is that interpretability is evaluated mostly with proxy metrics and automated judges, not with radiologist annotation.

## The core idea

Modern vision foundation models are powerful, but their internal representations are hard to understand. A model like DINOv3 or BiomedParse can encode an image into a dense vector, but that vector is not naturally readable by a clinician.

This paper asks a simple but important question:

> Can we take the dense embedding of a medical vision foundation model and rewrite it as a small set of interpretable concepts?

The authors propose using Sparse Autoencoders, or SAEs, as an interpretability layer. The pipeline is:

1. Take a CT or MRI image slice.
2. Pass it through a frozen foundation model.
3. Obtain a dense embedding.
4. Pass that embedding through an SAE.
5. Get a sparse feature vector where only a few features are active.
6. Interpret those active features as meaningful medical concepts.

The key point is that the SAE is not replacing the foundation model. It is placed on top of it. The foundation model still produces the original embedding, but the SAE converts that embedding into a sparse representation that is easier to inspect.

In plain words:

> The foundation model says, “Here is a 1024 or 1536-dimensional vector.”
>
> The SAE tries to say, “Actually, this image is mostly feature 853, feature 1202, and feature 6969, which correspond to axial abdominal CT concepts.”

That is the practical value. Instead of trusting a dense vector that nobody can reason about, we can inspect a handful of sparse features.

## Methodology, the architecture

The methodology has a clean structure. The authors freeze the vision models, extract embeddings, train SAEs on those embeddings, then evaluate whether the sparse features preserve information and become interpretable.

### 1. Start from frozen foundation model embeddings

The paper uses three embedding sources:

1. **BiomedParse**
   - A biomedical foundation model.
   - Produces 1536-dimensional embeddings.
   - It is domain-specific, meaning it was trained with biomedical data and objectives.

2. **DINOv3**
   - A general-purpose self-supervised vision transformer.
   - Produces 1024-dimensional embeddings.
   - It is not specifically medical, but it learns rich visual structure from large-scale visual training.

3. **Random-weight BiomedParse baseline**
   - Same architecture family as BiomedParse, but with randomly initialized weights.
   - Produces 1536-dimensional embeddings.
   - Used to test whether the SAE is simply exploiting architectural properties or whether it is extracting meaningful learned structure.

The embeddings are **frozen**. That means the authors do not update BiomedParse, DINOv3, or the random model. Only the SAE parameters are trained.

This matters because the paper is not claiming to improve the original foundation model. It is claiming that meaningful structure already exists inside the embeddings and that SAEs can expose it.

### 2. Train a Sparse Autoencoder on those embeddings

A Sparse Autoencoder is an autoencoder trained with a sparsity constraint.

A normal autoencoder has two main parts:

1. **Encoder**
   - Takes an input vector.
   - Compresses or transforms it into latent activations.

2. **Decoder**
   - Takes those latent activations.
   - Reconstructs the original input.

For this paper, the input is not the original image. The input is the foundation model embedding.

So the SAE learns:

```text
Foundation model embedding -> sparse feature activations -> reconstructed embedding
```

The objective is to reconstruct the original embedding as accurately as possible, while keeping the internal representation sparse.

#### What makes it sparse?

The SAE is forced to activate only a small number of features for each image. Instead of using hundreds or thousands of latent dimensions at once, it should explain the embedding using a small subset of features.

Example:

```text
Dense embedding:
[0.13, -0.42, 0.77, ..., 0.04]

Sparse SAE representation:
feature 51 = 2.3
feature 543 = 1.1
feature 1624 = 0.8
all other features = 0
```

This is useful because a small number of active features can potentially be inspected and described.

#### Why SAEs are useful for interpretability

Dense neural representations are often **polysemantic**. One dimension may encode several unrelated things at once, or one concept may be spread across many dimensions.

SAEs aim to produce more **monosemantic** features. A monosemantic feature is a feature that corresponds to one coherent concept.

For example:

- A bad polysemantic feature might activate for abdominal CT, brain MRI, and spine anatomy all mixed together.
- A better monosemantic feature might activate mostly for axial abdominal CT slices containing liver and vasculature.

The paper is testing whether medical image embeddings contain this kind of decomposable structure.

### 3. Use a Matryoshka SAE instead of a single-level SAE

The authors use a **Matryoshka SAE**.

The name comes from Russian nesting dolls. The idea is that smaller feature dictionaries are nested inside larger ones.

Instead of training one SAE with one dictionary size, the model has multiple nested levels:

```text
[D1, D2, D3, D4]
```

In the paper, there are 4 levels:

```text
L = 4
```

Each level uses a prefix of the full feature dictionary:

```text
Level 1 uses the first D1 features.
Level 2 uses the first D2 features.
Level 3 uses the first D3 features.
Level 4 uses the first D4 features.
```

So if the dictionary sizes are:

```text
[128, 512, 2048, 8192]
```

then:

```text
Level 1 uses features 1 to 128.
Level 2 uses features 1 to 512.
Level 3 uses features 1 to 2048.
Level 4 uses features 1 to 8192.
```

This creates a hierarchy:

- Early levels should learn broad, coarse features.
- Later levels should add finer, more specific refinements.

#### Why this is useful

A single-level SAE has to choose one dictionary size. A small dictionary may be interpretable but not expressive enough. A large dictionary may reconstruct better but can become harder to analyze.

A Matryoshka SAE gives multiple levels at once:

- Small levels reveal high-level structure.
- Larger levels capture more detail.
- The same encoder and decoder are shared, so the levels are tied together instead of being completely independent.

This is conceptually similar to saying:

```text
Level 1: broad anatomy and modality.
Level 2: more specific anatomical region.
Level 3: finer anatomical structures.
Level 4: subtle details and refinements.
```

The paper does not prove that every level behaves exactly this cleanly, but this is the intended architectural motivation.

### 4. Force sparsity with BatchTopK during training

The paper uses **BatchTopK sparsification** during training.

A basic TopK sparse autoencoder would keep exactly `k` active features per sample. For example, if `k = 10`, every image gets exactly 10 active features.

BatchTopK works differently.

Instead of forcing every individual sample to use exactly `k` features, it enforces sparsity across the batch. The paper describes it as having `k` features active per sample on average across the batch.

Example:

If:

```text
batch size = 4
k = 3
```

then the batch budget is approximately:

```text
4 samples * 3 features = 12 active feature slots
```

But the distribution across samples can be flexible:

```text
sample 1: 5 active features
sample 2: 2 active features
sample 3: 4 active features
sample 4: 1 active feature
Total = 12 active features
```

This is different from fixed per-sample TopK:

```text
sample 1: exactly 3 active features
sample 2: exactly 3 active features
sample 3: exactly 3 active features
sample 4: exactly 3 active features
Total = 12 active features
```

#### Why BatchTopK matters

Some images are probably simple. Others may contain more complex anatomical content. BatchTopK lets complex images use more features and simple images use fewer, while keeping the average sparsity controlled.

That is important in medical imaging because not all slices carry the same amount of semantic content:

- A mostly empty slice may need very few features.
- A slice containing multiple organs, vessels, spine, and modality-specific structure may need more features.

So BatchTopK is more flexible than rigid TopK.

### 5. Replace BatchTopK with JumpReLU at inference

The paper uses **JumpReLU** at inference.

During training, BatchTopK decides which activations survive. But at inference time, batch-level decisions are awkward because you may want to process one image at a time. You do not want the active features for one image to depend on which other images happen to be in the same batch.

So the authors replace BatchTopK with a thresholding function called JumpReLU.

The idea is:

```text
If activation is above a learned threshold, keep it.
If activation is below the threshold, set it to zero.
```

The threshold is estimated during training as a running average of the minimum activation kept by BatchTopK.

In practical terms, JumpReLU approximates the training-time sparsification rule with a fixed feature activation threshold.

#### Why not just use ReLU?

A normal ReLU keeps all positive activations:

```text
ReLU(x) = max(0, x)
```

That means many tiny positive activations could survive, making the representation less sparse.

JumpReLU is stricter. It only keeps activations that jump over a threshold.

Conceptually:

```text
ReLU: keep anything above 0.
JumpReLU: keep only things above a meaningful threshold.
```

This helps preserve the sparse behavior learned during BatchTopK training.

### 6. Train only with reconstruction loss

The training objective is simple:

```text
mean squared error between the original embedding and the reconstructed embedding
```

The paper averages this reconstruction loss across all 4 Matryoshka levels.

Important detail:

> The authors do not add auxiliary sparsity penalties or diversity penalties.

This matters because many sparse representation methods add extra terms to force certain behavior. Here, the sparsity is mainly imposed by BatchTopK and the architecture.

So the loss is basically:

```text
For each Matryoshka level:
    reconstruct the original foundation model embedding
Average the MSE across levels
Optimize the SAE parameters
```

## Datasets

The paper uses the **TotalSegmentator** dataset.

Dataset details:

- 1,844 scans total.
- 1,228 CT scans.
- 616 MRI scans.
- 10 institutions.
- 909,873 2D image slices.
- 138 per-image metadata fields.

The metadata includes information related to:

- anatomy presence,
- imaging parameters,
- demographics.

The split is designed to test generalization across institutions:

- 3 institutions are fully withheld as the test set.
- This corresponds to 14.1% of images.
- The remaining scans are split into train and validation.
- Train: 68.6% of images.
- Validation: 17.3% of images.

The train and validation split is stratified by:

- modality,
- age group,
- sex.

This is important because the model is evaluated not only on random held-out slices, but on institutions that were not seen during training.

## Experiments

The paper evaluates 96 SAE configurations per foundation model.

The configuration search varies:

1. **Dictionary size families**
   - From smaller dictionaries such as `[16, 64, 256, 1024]`.
   - Up to larger dictionaries such as `[128, 512, 2048, 8192]`.

2. **Sparsity patterns**
   - 8 patterns total.
   - 4 fixed TopK patterns.
   - 4 progressive TopK patterns.

Training details:

- Optimizer: Adam.
- Initial learning rate: `1e-4`.
- Cosine annealing down to `1e-6`.
- 100 epochs.
- Batch size: 2048.

Baselines:

1. Dense embedding upper bound.
2. Random-weight BiomedParse baseline.

The dense baseline tells us how good the original dense embedding is. The random-weight baseline tells us whether the SAE is discovering real learned semantic structure or just fitting arbitrary architecture-induced patterns.

### Experiment 1: SAE quality

The first experiment asks:

> Does the sparse representation still preserve the original embedding structure and useful downstream information?

The paper evaluates:

- reconstruction fidelity using `R²`,
- downstream ROC-AUC,
- alive feature counts,
- performance recovery using only top-N features.

#### R² reconstruction

BiomedParse obtains higher reconstruction quality:

```text
BiomedParse R² range: 0.890 to 0.941
DINOv3 R² range:      0.649 to 0.841
```

At first glance, this might suggest BiomedParse is better. But this is not the whole story.

A high R² means the SAE can reconstruct the embedding, but it does not guarantee that the sparse features are semantically useful.

The random-weight baseline can also achieve decent reconstruction in some settings, but it performs poorly on semantic downstream tasks. This is one of the key messages of the paper:

> Reconstruction fidelity is not enough. You can reconstruct an embedding space that is not semantically meaningful.

#### Downstream ROC-AUC

Dense embedding baselines:

```text
BiomedParse dense ROC-AUC: 0.907
DINOv3 dense ROC-AUC:      0.912
```

Optimal sparse configurations recover:

```text
BiomedParse: 90.2% of dense performance
DINOv3:      93.0% of dense performance
```

The random-weight baseline only reaches:

```text
Random-weight AUC: 0.606 to 0.651
```

This supports the claim that useful sparse features depend on learned representation structure, not only on architecture.

### Experiment 2: SAE configuration ranking

The second experiment asks:

> Which SAE configuration gives the best trade-off between interpretability and performance?

The authors combine:

1. monosemanticity ranking,
2. performance recovery ranking.

The best configurations use the largest dictionary family:

```text
[128, 512, 2048, 8192]
```

#### Best BiomedParse configuration

```text
Dictionary sizes: [128, 512, 2048, 8192]
Top-K values:     [20, 40, 80, 160]
Monosemanticity rank: 2
Performance rank:     3
Combined rank:        1
```

This is a balanced configuration. It is not the absolute best in monosemanticity, but it performs well on both interpretability and performance.

#### Best DINOv3 configuration

```text
Dictionary sizes: [128, 512, 2048, 8192]
Top-K values:     [5, 10, 20, 40]
Monosemanticity rank: 1
Performance rank:     11
Combined rank:        1
```

This configuration is more aggressively sparse. It has excellent monosemanticity but lower performance recovery rank.

This illustrates an important trade-off:

> More sparsity can make features cleaner and more interpretable, but may remove some task-relevant information.

### Experiment 3: Sparse feature interpretability

The third experiment asks:

> Are the sparse features actually interpretable?

The authors test this with three demonstrations.

#### 1. Sparse fingerprint retrieval

A **sparse fingerprint** is the top-k active features for an image, including their activation values.

Example:

```text
Image fingerprint:
feature 51 = 1.8
feature 543 = 1.4
feature 1624 = 0.9
feature 853 = 0.7
feature 6969 = 0.6
```

Instead of comparing full dense embeddings, the authors compare sparse fingerprints using cosine similarity.

They evaluate whether images retrieved using sparse fingerprints are similar according to the original dense embedding space.

Results:

```text
BiomedParse retrieval quality:
k = 1:  0.929
k = 5:  0.954
k = 10: 0.964
k = 20: 0.967
Dense:  0.976

DINOv3 retrieval quality:
k = 1:  0.752
k = 5:  0.831
k = 10: 0.852
k = 20: 0.857
Dense:  0.895
```

At `k = 5`:

```text
BiomedParse preserves 97.7% of dense retrieval quality.
DINOv3 preserves 92.8% of dense retrieval quality.
```

This is strong evidence that only a few sparse features carry much of the semantic similarity information.

#### 2. Automated feature interpretation

The authors use a VLM-based pipeline to assign language descriptions to sparse features.

For each of the top-250 most monosemantic features:

1. Take the top-20 images that activate the feature most strongly.
2. Greedily select the 5 most dissimilar images among them.
3. Give those images and metadata to MedGemma 27B.
4. Ask it to generate a natural-language concept description.

The metadata includes things like:

- modality,
- orientation,
- anatomy,
- demographics.

Example concept descriptions might look like:

```text
Axial CT of the abdomen and retroperitoneum in elderly patients.
```

or:

```text
Axial MRI of the abdomen and retroperitoneum in elderly patients.
```

The idea is not just to say that a feature exists. The authors want each feature to have a human-readable meaning.

#### 3. Independent VLM-as-judge validation

The paper then uses another MedGemma 27B model as a judge.

The judge receives:

- the same images,
- five candidate descriptions,
- one true description,
- four distractor descriptions from other features.

The judge must rank which description fits best.

Results:

```text
BiomedParse:
Mean rank: 1.91
Rank 1 count: 141 / 250
Rank 2 count: 44 / 250
Rank 3 count: 28 / 250
Rank 4 count: 20 / 250
Rank 5 count: 17 / 250

DINOv3:
Mean rank: 1.60
Rank 1 count: 170 / 250
Rank 2 count: 38 / 250
Rank 3 count: 21 / 250
Rank 4 count: 13 / 250
Rank 5 count: 8 / 250
```

DINOv3 performs better. This means its generated feature descriptions are more often selected as the best match by the independent judge.

#### 4. Language-driven image retrieval

This is the most “product-like” demonstration.

Instead of starting with a reference image, the user starts with a clinical text query:

```text
Axial CT of the abdomen and retroperitoneum in an elderly patient
```

Then:

1. An LLM reads the query.
2. It selects matching feature descriptions.
3. The selected features are assembled into a sparse fingerprint.
4. The system retrieves images using cosine similarity in sparse feature space.

This is zero-shot because there is no task-specific training for that exact query.

The result:

- BiomedParse selects mixed MRI/CT concepts and retrieves thoracic images.
- DINOv3 selects CT-specific abdomen concepts and retrieves anatomically correct axial abdominal CT images.

The takeaway is that DINOv3 had a better feature vocabulary for this query.

## Metrics and quality measures

This is probably the most important section if you want to understand how the paper proves its claims.

### 1. Reconstruction MSE

**What it measures:**

How close the reconstructed embedding is to the original foundation model embedding.

Formula conceptually:

```text
MSE = average squared difference between original embedding and reconstructed embedding
```

Lower is better.

**Why it matters:**

If MSE is very high, the SAE is losing too much information. The sparse representation would not be a faithful replacement for the dense embedding.

**Limitation:**

Low MSE does not guarantee interpretability. A sparse code can reconstruct a meaningless embedding space.

### 2. R² reconstruction fidelity

**What it measures:**

How much variance in the original embedding is explained by the SAE reconstruction.

A higher `R²` means the reconstructed embeddings are closer to the original embeddings.

The paper reports:

```text
BiomedParse: R² up to 0.941
DINOv3:      R² up to 0.841
```

**Why it matters:**

It gives a normalized sense of reconstruction quality.

**Key interpretation from the paper:**

BiomedParse has higher `R²`, but DINOv3 can still have better monosemanticity and strong downstream performance. So `R²` is useful, but insufficient.

### 3. L0 sparsity

**What it measures:**

The number of non-zero active features in the sparse representation.

In this context, lower `L0` means the representation uses fewer active SAE features.

**Why it matters:**

Interpretability depends on sparsity. A representation with 200 active concepts is not that easy to inspect. A representation with 5 or 10 active concepts is much more manageable.

**How it is controlled:**

Through TopK values and BatchTopK sparsification.

### 4. Alive features

**What it measures:**

The number of SAE features that actually activate for some data.

A feature is “alive” if it is used. A feature is “dead” if it basically never activates.

**Why it matters:**

If a model has 8192 dictionary features but only 200 are alive, then the dictionary is not being fully used. Dead features indicate wasted capacity or training issues.

Alive feature count helps understand whether the SAE learned a rich feature dictionary.

### 5. Downstream ROC-AUC

**What it measures:**

How well the embeddings support anatomical classification tasks.

ROC-AUC measures the ability to rank positive examples above negative examples across classification thresholds.

Higher is better.

The dense baselines are:

```text
BiomedParse dense ROC-AUC: 0.907
DINOv3 dense ROC-AUC:      0.912
```

**Why it matters:**

The sparse representation should not only reconstruct the original embedding. It should preserve task-relevant information.

**Key result:**

Optimal sparse configurations recover:

```text
BiomedParse: 90.2% of dense performance
DINOv3:      93.0% of dense performance
```

### 6. Performance recovery

**What it measures:**

How much of the dense embedding performance is recovered by the sparse representation.

Conceptually:

```text
Performance recovery = sparse performance / dense performance
```

The paper also evaluates recovery using only top-N active features:

```text
N = 1, 3, 10, 50
```

**Why it matters:**

This tells us whether a small number of features is enough to preserve useful information.

**Key result:**

With 10 features:

```text
BiomedParse recovers 87.8% of dense ROC-AUC.
DINOv3 recovers 82.4% of dense ROC-AUC.
```

Performance gains diminish above 10 features, meaning most of the useful information is concentrated in the strongest sparse features.

### 7. Coherence score C(f)

**What it measures:**

Whether the top images activating a feature share similar organ labels.

The paper uses null-adjusted mean pairwise Jaccard similarity over organ sets from the top-10 activating samples.

In simpler terms:

1. Take the 10 images that activate a feature the most.
2. Look at which organs are present in each image.
3. Compare organ sets pairwise using Jaccard similarity.
4. Adjust relative to a null baseline.
5. Average the result.

Jaccard similarity is:

```text
Jaccard(A, B) = size of intersection / size of union
```

Example:

```text
Image A organs: liver, spleen, kidney
Image B organs: liver, kidney, aorta
Intersection: liver, kidney = 2
Union: liver, spleen, kidney, aorta = 4
Jaccard = 2 / 4 = 0.5
```

**Why it matters:**

A feature is more coherent if the images that activate it contain similar anatomical structures.

### 8. Specificity score S(f)

**What it measures:**

Whether a feature activates for a concentrated set of organ labels rather than many unrelated labels.

The paper uses normalized inverse entropy over the organ label distribution.

Entropy is high when the feature activates broadly across many labels. Entropy is low when the feature is concentrated on a small set of labels.

So inverse entropy rewards specificity.

**Why it matters:**

A feature that activates for everything is not very interpretable. A feature that activates mostly for one anatomical pattern is more useful.

### 9. Monosemanticity score M(f)

The paper defines feature-level monosemanticity as:

```text
M(f) = C(f) * S(f)
```

Where:

```text
C(f) = coherence
S(f) = specificity
```

So a feature receives a high monosemanticity score only if it is both:

1. coherent, meaning its top activating images are similar,
2. specific, meaning it does not activate broadly across unrelated concepts.

**Why multiply them?**

Multiplication penalizes features that are good on only one dimension.

Example:

```text
High coherence but low specificity -> not ideal.
High specificity but low coherence -> not ideal.
High coherence and high specificity -> strong monosemantic feature.
```

### 10. Configuration-level monosemanticity Mconfig

The paper computes configuration-level monosemanticity as the mean `M(f)` of the top-10 features in a configuration.

Conceptually:

```text
Mconfig = average monosemanticity of the top-10 most monosemantic features
```

**Why it matters:**

This gives one score per SAE configuration, making it possible to compare all 96 configurations.

**Important caveat:**

Because it uses the top-10 features, it measures whether a configuration produces some very interpretable features. It does not necessarily prove that every feature in the dictionary is interpretable.

### 11. Sparse fingerprint retrieval quality

**What it measures:**

Whether retrieval using sparse fingerprints returns images that are semantically similar according to the dense embedding space.

Procedure:

1. Pick a reference image.
2. Keep its top-k sparse features.
3. Retrieve images by cosine similarity over sparse fingerprints.
4. Measure how close the retrieved images are to the reference in the original dense embedding space.

The dense retrieval result is treated as the upper bound.

**Why it matters:**

This tests whether sparse features preserve the similarity relationships learned by the foundation model.

**Key result:**

At `k = 5`, sparse fingerprints preserve most of the dense retrieval quality.

### 12. Cosine similarity

**What it measures:**

How similar two vectors are in direction.

Cosine similarity is high when two vectors point in a similar direction, regardless of their magnitude.

It is used for:

- sparse fingerprint retrieval,
- selecting dissimilar top-activating samples for feature interpretation,
- comparing retrieved images in embedding space.

**Why it matters:**

Embedding spaces are often compared using cosine similarity because direction often captures semantic similarity better than raw Euclidean distance.

### 13. VLM-generated concept descriptions

**What it measures:**

This is not a numeric metric by itself, but it is part of the interpretability evaluation.

The VLM receives top-activating images and metadata, then generates a natural-language description of what the feature seems to represent.

**Why it matters:**

Interpretability is only useful if humans can name or understand the feature. A sparse feature index like `feature 853` is not clinically helpful by itself. A description like “axial CT of the abdomen and retroperitoneum” is much more useful.

### 14. VLM-as-judge ranking

**What it measures:**

Whether the generated description actually matches the images that activate the feature.

Procedure:

1. Show a VLM judge the feature's images.
2. Give it 5 descriptions.
3. One is the true description generated for that feature.
4. Four are distractors from other features.
5. Ask the judge to rank them.

Metric:

```text
Rank 1 = best possible result
Rank 5 = worst possible result
```

The paper reports mean rank and rank count distribution.

**Why it matters:**

It gives an automated validation signal for the generated concept descriptions.

**Caveat:**

This is still not the same as expert human validation. The judge is another model and may share biases or errors with the description generator.

### 15. Language-driven image retrieval quality

**What it measures:**

Whether a text query can be mapped to sparse feature concepts and used to retrieve relevant medical images without a reference image.

This is mostly demonstrated qualitatively with one query.

**Why it matters:**

It shows a potential interface:

```text
clinical language -> sparse feature concepts -> image retrieval
```

That is a big deal because it makes the model searchable in human terms.

**Caveat:**

The paper demonstrates this on a single query, so it is proof-of-concept rather than a fully validated retrieval system.

### 16. Random-weight baseline

**What it measures:**

Whether the observed results come from learned semantic structure or just from model architecture.

The random baseline uses an untrained BiomedParse model.

**Why it matters:**

If the random baseline had strong downstream performance and high monosemanticity, the results would be less convincing. But it does not.

Key observation:

- Random baseline can reconstruct reasonably in some cases.
- But it has poor downstream AUC and lower monosemanticity.

This supports the claim that the learned foundation model embeddings contain real semantic structure.

## Main results

### 1. SAEs reconstruct BiomedParse embeddings very well

BiomedParse reaches high reconstruction fidelity:

```text
R² up to 0.941
```

This means the sparse representation can preserve much of the dense embedding structure.

### 2. DINOv3 reconstructs less well but is more interpretable

DINOv3 has lower R²:

```text
R² up to 0.841
```

But it achieves higher monosemanticity:

```text
DINOv3 monosemanticity: 0.356 to 0.714
BiomedParse monosemanticity: 0.036 to 0.394
```

This is one of the most interesting findings.

It suggests that domain-specific biomedical pretraining does not automatically produce the most interpretable sparse features. General visual representation richness may matter more.

### 3. Sparse codes preserve downstream performance

Using only a small number of sparse features still recovers a lot of task performance.

With 10 features:

```text
BiomedParse recovers 87.8% of dense ROC-AUC.
DINOv3 recovers 82.4% of dense ROC-AUC.
```

Optimal sparse configurations recover:

```text
BiomedParse: 90.2% of dense performance.
DINOv3: 93.0% of dense performance.
```

This means sparse representations are not just interpretable toys. They still retain useful signal.

### 4. Sparse fingerprints work well for retrieval

With only 5 active features:

```text
BiomedParse: 0.954 vs 0.976 dense retrieval quality
DINOv3:      0.831 vs 0.895 dense retrieval quality
```

That corresponds to:

```text
BiomedParse: 97.7% of dense retrieval quality
DINOv3:      92.8% of dense retrieval quality
```

This is a strong compression result. A handful of sparse features can preserve most retrieval behavior.

### 5. DINOv3 feature descriptions are judged better

The independent VLM judge ranks DINOv3 concepts better than BiomedParse concepts.

```text
DINOv3 mean rank:      1.60
BiomedParse mean rank: 1.91
```

DINOv3 also has more rank-1 matches:

```text
DINOv3:      170 / 250
BiomedParse: 141 / 250
```

This supports the claim that DINOv3 produces more interpretable, language-describable sparse features.

### 6. Language-driven retrieval works better with DINOv3 in the demo

For the query:

```text
Axial CT of the abdomen and retroperitoneum in an elderly patient
```

DINOv3 retrieves anatomically correct axial abdominal CT images.

BiomedParse selects mixed MRI/CT concepts and retrieves thoracic images.

This suggests DINOv3 learned a cleaner concept vocabulary for this query.

## Limitations

### 1. The dataset excludes pathological cases

TotalSegmentator contains normal anatomy across CT and MRI, but the paper does not test pathological cases.

This is a major limitation for clinical interpretability because many clinically important concepts involve disease, lesions, abnormalities, and post-surgical changes.

A feature vocabulary that works for normal anatomy may not fully transfer to pathological imaging.

### 2. The analysis is slice-level, not volumetric

The paper works with 2D slices extracted from CT and MRI scans.

Medical imaging is often inherently 3D. Many anatomical and pathological patterns depend on volumetric context.

So a slice-level SAE may miss concepts that require spatial continuity across slices.

### 3. Monosemanticity is measured with metadata-derived organ labels

The monosemanticity score uses organ label distributions from metadata.

That is scalable, but it is still a proxy.

A feature could be clinically meaningful in a way that is not captured by organ labels. Or it could seem organ-specific without being clinically interpretable.

### 4. Automated interpretation is not expert validation

The feature descriptions are generated by a VLM and validated by another VLM.

That is useful for scale, but it is not the same as asking radiologists whether the features are correct, clinically useful, or safe.

There is a risk of model-to-model agreement that still fails under human expert review.

### 5. Language-driven retrieval is only shown as a proof-of-concept

The zero-shot language retrieval experiment is demonstrated on a single query.

That is exciting, but not enough to prove robust clinical retrieval.

A stronger evaluation would require:

- many queries,
- multiple modalities,
- different anatomical regions,
- pathological queries,
- expert relevance judgments.

### 6. Interpretability and performance trade off

The DINOv3 optimal configuration has the best monosemanticity rank but only rank 11 in performance recovery.

This shows a real trade-off:

- Very sparse features may be cleaner.
- But they may lose information useful for downstream tasks.

A clinical system would need to choose the right balance.

## My interpretation

This paper is interesting because it moves SAE interpretability from language models into medical vision foundation models.

The best part is that the authors do not only show reconstruction. They evaluate several complementary aspects:

1. Can the SAE reconstruct the original embedding?
2. Does the sparse code preserve downstream performance?
3. Do sparse fingerprints preserve retrieval behavior?
4. Are individual features coherent and specific?
5. Can a VLM describe those features?
6. Can an independent VLM identify the right description?
7. Can text queries be mapped back into sparse feature space?

That makes the paper stronger than a simple “we trained an SAE and the features look nice” study.

The most surprising result is DINOv3. You might expect BiomedParse, the biomedical model, to produce more clinically interpretable features. But DINOv3 produces stronger monosemanticity and better judged descriptions.

My read is:

> Biomedical alignment helps with domain tasks, but broad visual representation quality may be more important for clean feature factorization.

This has implications for medical AI research. Maybe the best interpretable medical systems will not always come from the most domain-specific backbone. They may come from strong general-purpose models combined with a good interpretability layer.

The biggest weakness is the lack of human clinical validation. A VLM judge is useful, but in medical imaging, the final interpretability claim should eventually be tested with radiologists.

Still, as a proof-of-concept, the paper is strong. It shows that SAEs can act as a practical bridge between dense medical image embeddings and language-describable anatomical concepts.

## Cheat sheet of the important concepts

### Foundation model embedding

A dense vector representation produced by a large pretrained model. It contains visual information, but is hard to interpret directly.

### Sparse Autoencoder, SAE

A model that reconstructs an input embedding using only a small number of active latent features.

Goal:

```text
dense embedding -> sparse interpretable features -> reconstructed dense embedding
```

### Sparse feature

One latent dimension of the SAE dictionary. Ideally, each sparse feature corresponds to one coherent concept.

### Monosemantic feature

A feature that corresponds to one clear concept rather than many unrelated concepts.

Example:

```text
Good: axial abdominal CT with liver and retroperitoneum
Bad: random mixture of abdomen, skull, MRI, CT, and demographics
```

### Polysemantic feature

A feature that activates for multiple unrelated concepts. Polysemantic features are harder to interpret.

### Matryoshka SAE

An SAE with nested dictionary levels. Smaller dictionaries are prefixes of larger dictionaries.

Purpose:

```text
coarse concepts at small levels
fine concepts at larger levels
```

### BatchTopK

Training-time sparsification method that keeps a fixed average number of active features across the batch, rather than exactly the same number for every sample.

Why useful:

```text
simple images can use fewer features
complex images can use more features
average sparsity remains controlled
```

### JumpReLU

Inference-time thresholding function that keeps activations only if they exceed a learned threshold.

Purpose:

```text
mimic BatchTopK sparsity without depending on the batch
```

### Sparse fingerprint

The top-k active sparse features and their activation values for one image.

Used for image retrieval.

### R²

Measures how well the SAE reconstruction explains the original embedding variance.

High R² means faithful reconstruction, but not necessarily semantic usefulness.

### ROC-AUC

Measures downstream classification performance. Used to test whether sparse features preserve task-relevant information.

### Performance recovery

How much dense embedding performance is recovered by the sparse representation.

### Coherence C(f)

Measures whether the top images activating a feature have similar organ sets.

### Specificity S(f)

Measures whether a feature is concentrated on a small set of organ labels rather than broadly spread across many labels.

### Monosemanticity M(f)

Defined as:

```text
M(f) = C(f) * S(f)
```

High only when the feature is both coherent and specific.

### Mconfig

Configuration-level monosemanticity score, computed as the average monosemanticity of the top-10 features for that configuration.

### VLM auto-interpretation

Using a vision-language model to generate descriptions of sparse features based on the images that activate them.

### VLM-as-judge

Using a separate vision-language model to test whether the generated description matches the feature's top-activating images better than distractor descriptions.

### Zero-shot language-driven retrieval

Retrieving images from a text query by mapping the query to sparse feature descriptions, without requiring a reference image or task-specific training.

## Final takeaway

The paper shows that Sparse Autoencoders can be used as an interpretability layer for medical vision foundation models.

They compress dense embeddings into a small number of sparse features, preserve much of the original performance and retrieval behavior, and expose features that can often be described in clinical language.

The method is not yet clinically validated, and the evaluation relies heavily on metadata and automated VLM judgments. But the direction is promising:

> Instead of asking clinicians to trust opaque dense embeddings, SAEs may allow medical AI systems to expose the concepts they are using.

That is the main contribution.
