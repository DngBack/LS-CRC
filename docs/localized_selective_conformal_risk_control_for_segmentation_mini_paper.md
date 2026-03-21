# Learning Structured Abstention for Localized Conformal Risk Control in Segmentation

## Abstract

Conformal Risk Control (CRC) provides finite-sample guarantees on expected risk for a broad family of bounded losses and has emerged as a principled tool for reliable prediction in structured tasks, including image segmentation. However, standard CRC is fundamentally **marginal**: it certifies average risk over the population but does not prevent severe heterogeneity across images, object scales, or spatial regions. In segmentation, this limitation is operationally significant because clinically or semantically critical failures often concentrate on thin boundaries, small structures, and hard examples. Recent adaptive conformal methods reduce conservativeness by using image-dependent scores or thresholds, but they do not explicitly learn **where** abstention should occur.

We propose **Localized Selective Conformal Risk Control (LS-CRC)**, a framework that combines learned structured abstention with split conformal calibration for segmentation. Our method augments a segmentation model with a pixel-wise rejector that predicts an acceptance map over spatial locations. The rejector is trained to identify unreliable regions using prediction correctness, uncertainty cues, and localized risk surrogates. A held-out calibration set is then used to select an acceptance threshold that guarantees finite-sample control of a localized selective risk, while maximizing coverage among admissible thresholds. 

We show that LS-CRC preserves the marginal risk guarantee inherited from CRC for bounded monotone losses, and we provide a subgroup deviation bound and a tail-risk improvement result under a rejector quality assumption. Empirically, LS-CRC improves worst-group risk, worst-image risk, and CVaR-style tail risk while maintaining competitive Dice and IoU, yielding substantially better risk--coverage trade-offs than standard uncertainty thresholding and global CRC baselines.

---

## 1. Introduction

Dense prediction systems are increasingly deployed in settings where failures are not equally costly. In semantic and medical image segmentation, the most consequential errors often arise on thin boundaries, small foreground objects, ambiguous interfaces, or otherwise difficult images. Average segmentation quality metrics such as Dice or IoU can mask these localized failures, and even risk-controlled methods can be insufficient if they only certify average behavior over the population.

Conformal Risk Control (CRC) offers an attractive starting point for reliable segmentation. By calibrating a parameter on a held-out exchangeable calibration set, CRC provides finite-sample control of the expected value of a bounded monotone loss. This framework naturally accommodates structured tasks and losses beyond simple miscoverage. In particular, segmentation-relevant risks such as false negative rate or weighted miss rate can be incorporated into a conformal auditing pipeline.

However, standard CRC is inherently **marginal**. It guarantees control of the population-average risk, but it does not prevent risk concentration on hard examples or critical spatial regions. This issue is especially acute in segmentation: two systems may satisfy the same average risk target while exhibiting dramatically different behavior on small lesions, thin boundaries, or images with high ambiguity. Recent work on adaptive and conditional conformal risk control addresses part of this inefficiency by using image-specific scores, weighted quantiles, or adaptive thresholds. Yet these methods remain centered on **how much** uncertainty to allocate rather than **where** the model should abstain.

Selective prediction provides the complementary viewpoint. Instead of forcing the model to commit everywhere, a selective predictor abstains on uncertain cases or regions and predicts only where it is sufficiently reliable. In segmentation, this suggests a more operational notion of reliability: the system should explicitly defer on spatially localized regions where errors are likely and where abstention is acceptable or even desirable. The challenge is that learned rejectors alone do not provide statistical guarantees, while post-hoc uncertainty thresholding is often too crude to capture the geometry of risky regions.

This paper proposes a synthesis of these ideas. We introduce **Localized Selective Conformal Risk Control (LS-CRC)**, which learns a spatially structured rejector and wraps it with split conformal calibration. The rejector produces an acceptance map over pixels or regions, allowing the model to abstain selectively on difficult spatial locations. Conformal calibration then chooses the acceptance threshold to satisfy a user-specified risk target. This design preserves the finite-sample marginal guarantee of CRC while enabling substantially more targeted abstention than global thresholds or hand-designed uncertainty heuristics.

The central technical idea is to control a **localized selective risk**: a weighted miss-type loss computed only on accepted regions, normalized by a fixed target mass. This makes the risk compatible with CRC-style calibration while reflecting realistic deployment semantics, where abstained regions are deferred to a downstream oracle, a human reviewer, or a more conservative module. By choosing spatial weight maps that emphasize boundaries, small structures, or rare regions, the method can target reliability where it matters most.

Our contributions are threefold.

1. We formalize **localized selective risk control** for segmentation and introduce LS-CRC, a framework that learns structured abstention and calibrates it with split conformal risk control.
2. We provide theory showing that LS-CRC preserves finite-sample marginal control, together with a subgroup deviation bound and a tail-risk improvement result under an explicit rejector quality condition.
3. We design an empirical protocol for binary segmentation that evaluates not only average Dice/IoU but also worst-group risk, worst-image risk, and CVaR-style tail risk, demonstrating a more meaningful notion of reliability than average performance alone.

Our overall claim is not that distribution-free conditional guarantees become possible in full generality. Rather, we show that a carefully learned spatial rejector, when combined with conformal calibration, can preserve auditable marginal guarantees while materially improving the conditional and tail behavior that matters in practice.

---

## 2. Related Work

### Conformal prediction and conformal risk control

Classical conformal prediction provides finite-sample guarantees on predictive set coverage under exchangeability. Conformal Risk Control generalizes this principle to bounded monotone losses, enabling guarantees on user-specified risks such as false negative rate, weighted miss rate, or set size--risk trade-offs. This extension is especially relevant for structured tasks because it decouples the statistical guarantee from the exact output representation and allows calibration against task-relevant losses.

### Conformal methods for segmentation and adaptive risk control

Recent segmentation-oriented conformal methods have adapted conformal calibration to dense prediction by constructing image-dependent thresholds, adaptive score functions, or conditional calibration rules. These approaches reduce conservativeness relative to global thresholds and improve efficiency on heterogeneous data. Nevertheless, they primarily operate through adaptive score calibration and do not explicitly learn a structured abstention mechanism over spatial locations. As a result, they are limited in how precisely they can target risky regions such as thin boundaries or small structures.

### Selective prediction and learning to reject

Selective prediction augments a predictor with a rejection option, allowing abstention on uncertain inputs. Most classical methods rely on scalar confidence scores or entropy-based thresholds, while more recent approaches learn rejection policies jointly with the predictor. In dense prediction, selective segmentation has shown that abstaining on uncertain regions can improve retained accuracy and robustness. However, these methods usually lack finite-sample statistical guarantees and often optimize empirical objectives that do not align with audited deployment-time risk targets.

### Tail risk, subgroup reliability, and robust deployment

Average risk control is often insufficient when failures concentrate on rare but critical subpopulations. Recent work in conformal risk training and robust evaluation has emphasized the importance of tail-aware objectives such as CVaR and subgroup-level diagnostics. Our work aligns with this perspective: we retain the auditable marginal guarantee of split CRC, but explicitly design the learned abstention mechanism to reduce subgroup disparities and tail-risk concentration.

### Positioning of the present work

Our method occupies the intersection of these lines of work. Unlike standard CRC or adaptive conformal thresholding, we learn **where** to abstain in the image. Unlike selective segmentation without guarantees, we retain a conformal calibration layer that enforces a finite-sample risk budget. Unlike purely average-case methods, we explicitly target localized, subgroup, and tail reliability. This combination yields a practically meaningful and theoretically grounded framework for reliable segmentation.

---

## 3. Problem Setup

We consider binary image segmentation. Let

$$
(X, Y) \sim P,
$$

where $X \in \mathcal{X}$ is an image and $Y \in \{0,1\}^{H \times W}$ is the corresponding binary mask. We observe a dataset of exchangeable samples split into three disjoint subsets:

- a training set $\mathcal{D}_{\mathrm{tr}}$,
- a validation set $\mathcal{D}_{\mathrm{val}}$,
- a calibration set $\mathcal{D}_{\mathrm{cal}}$,

and evaluate on a held-out test set $\mathcal{D}_{\mathrm{te}}$.

A segmentation backbone $f_\theta$ outputs a foreground probability map

$$
p_\theta(x) \in [0,1]^{H \times W},
$$

with hard prediction

$$
\hat y_\theta(x)_u = \mathbf{1}\{p_\theta(x)_u \ge 1/2\}
$$

for pixel index $u \in \{1,\dots,H\times W\}$.

We augment the backbone with a rejector head $g_\phi$ that outputs a pixel-wise **acceptance score**

$$
s_\phi(x) \in [0,1]^{H \times W}.
$$

Given a threshold $\tau \in [0,1]$, the acceptance mask is

$$
A_{\phi,\tau}(x)_u = \mathbf{1}\{s_\phi(x)_u \ge \tau\},
$$

and the abstention mask is $R_{\phi,\tau}(x)_u = 1 - A_{\phi,\tau}(x)_u$. The selective prediction is

$$
\tilde y_{\theta,\phi,\tau}(x)_u =
\begin{cases}
\hat y_\theta(x)_u, & A_{\phi,\tau}(x)_u = 1,\\
\bot, & A_{\phi,\tau}(x)_u = 0,
\end{cases}
$$

where $\bot$ denotes abstention.

The operational interpretation is that accepted pixels are committed predictions, while abstained pixels are deferred to a downstream process. The design objective is therefore not to maximize prediction everywhere, but to maximize useful accepted coverage subject to a reliable risk budget.

---

## 4. Localized Selective Risk

A key modeling choice is the loss used for conformal calibration. We require a bounded loss that reflects deployment-relevant reliability and is monotone with respect to the acceptance threshold. To this end, we define a **localized selective miss risk**.

Let $w(x,y)_u \ge 0$ be a spatial weight map. In practice, $w$ may upweight pixels near the ground-truth boundary, pixels belonging to small structures, or other critical regions. For notational brevity, write $w_u = w(x,y)_u$ and $A_u = A_{\phi,\tau}(x)_u$.

We define

$$
L_{\mathrm{loc}}(x,y;\theta,\phi,\tau)
=
\frac{\sum_u w_u\,\mathbf{1}\{y_u = 1\}\,A_u\,\mathbf{1}\{\hat y_\theta(x)_u = 0\}}
{\sum_u w_u\,\mathbf{1}\{y_u = 1\} + \varepsilon}.
$$

This loss has three important properties.

1. **Deployment semantics.** Only errors on accepted foreground pixels are penalized. Errors on abstained pixels are deferred and therefore excluded from the numerator.
2. **Localization.** The weight map $w$ emphasizes the regions of greatest practical importance.
3. **Calibration compatibility.** The denominator is fixed with respect to the threshold $\tau$, which makes the loss more amenable to monotonic threshold calibration than risks normalized by accepted mass.

We also define accepted coverage

$$
\mathrm{Cov}(x;\phi,\tau) = \frac{1}{HW}\sum_u A_{\phi,\tau}(x)_u,
$$

and population coverage

$$
\mathrm{Cov}(\phi,\tau) = \mathbb{E}[\mathrm{Cov}(X;\phi,\tau)].
$$

The calibration problem is to choose $\tau$ so that the expected localized selective risk is below a user-specified target $\alpha$, while the accepted coverage is as large as possible.

---

## 5. Method

### 5.1 Overview

LS-CRC proceeds in three stages.

1. Train a segmentation backbone $f_\theta$ on the training set.
2. Train a rejector $g_\phi$ to estimate which spatial locations are safe to accept.
3. Calibrate an acceptance threshold $\tau$ on a held-out calibration set using the localized selective risk.

The resulting procedure is summarized in Algorithm 1.

### 5.2 Rejector parameterization

The rejector head operates on segmentation features and uncertainty summaries. Let $h_\theta(x)$ denote an intermediate feature map extracted from the segmentation network. The rejector consumes a feature tensor

$$
\psi(x) = \big[h_\theta(x),\; p_\theta(x),\; u_\theta(x),\; m_\theta(x)\big],
$$

where

- $u_\theta(x)$ is an uncertainty map, e.g. entropy,
- $m_\theta(x)$ is a margin or confidence-derived map.

The rejector outputs

$$
s_\phi(x) = g_\phi(\psi(x)) \in [0,1]^{H \times W}.
$$

We use a shallow convolutional decoder or a small pixel-wise head in practice.

### 5.3 Pseudo-labels for selective supervision

There are generally no direct supervision labels for abstention. We therefore construct pseudo-labels for acceptance using prediction correctness and uncertainty. Let $r^*_u \in \{-1,0,1\}$ denote the pseudo-label at pixel $u$, where $-1$ indicates ignore, $1$ indicates safe-to-accept, and $0$ indicates unsafe / should-reject. A practical rule is:

- safe if the prediction is correct, confidence is high, and the pixel is not near a critical boundary region;
- unsafe if the prediction is incorrect, uncertainty is high, or the pixel lies in a critical boundary band.

These pseudo-labels provide a coarse signal for the rejector while allowing the final decision boundary to be determined by conformal calibration.

### 5.4 Training objective

We first train the segmentation backbone with a standard segmentation loss

$$
\mathcal{L}_{\mathrm{seg}}(\theta)
=
\mathcal{L}_{\mathrm{BCE}} + \lambda_{\mathrm{dice}}\,\mathcal{L}_{\mathrm{Dice}}.
$$

The rejector is then trained with a masked binary cross-entropy loss against pseudo-labels,

$$
\mathcal{L}_{\mathrm{rej}}(\phi)
=
\frac{1}{|\Omega_{\mathrm{sup}}|}
\sum_{u \in \Omega_{\mathrm{sup}}}
\mathrm{BCE}(s_\phi(x)_u, r_u^*),
$$

where $\Omega_{\mathrm{sup}} = \{u : r_u^* \neq -1\}$.

To encourage spatial coherence, we add a smoothness penalty

$$
\mathcal{L}_{\mathrm{smooth}}(\phi)
=
\sum_{(u,v)\in\mathcal{N}} |s_\phi(x)_u - s_\phi(x)_v|,
$$

where $\mathcal{N}$ denotes a local pixel neighborhood.

Finally, we optionally fine-tune the segmentation backbone and rejector jointly using a differentiable localized surrogate risk

$$
\mathcal{L}_{\mathrm{loc	ext{-}sur}}(\theta,\phi)
=
\frac{\sum_u w_u\,s_\phi(x)_u\,(1-p_\theta(x)_u)\,y_u}
{\sum_u w_u\,y_u + \varepsilon},
$$

which penalizes high acceptance on likely false-negative foreground pixels.

The full joint objective is

$$
\mathcal{L}(\theta,\phi)
=
\mathcal{L}_{\mathrm{seg}}
+ \lambda_1 \mathcal{L}_{\mathrm{rej}}
+ \lambda_2 \mathcal{L}_{\mathrm{smooth}}
+ \lambda_3 \mathcal{L}_{\mathrm{loc	ext{-}sur}}.
$$

This objective separates three roles:
- segmentation quality via $\mathcal{L}_{\mathrm{seg}}$,
- coarse selective supervision via $\mathcal{L}_{\mathrm{rej}}$,
- deployment-aligned reliability shaping via $\mathcal{L}_{\mathrm{loc\text{-}sur}}$.

### 5.5 Split conformal calibration

After training, we freeze $(\theta,\phi)$ and use the calibration set to choose an acceptance threshold. Let

$$
\widehat R_{\mathrm{cal}}(\tau) = \frac{1}{|\mathcal{D}_{\mathrm{cal}}|}\sum_{(x_i,y_i)\in\mathcal{D}_{\mathrm{cal}}}
L_{\mathrm{loc}}(x_i,y_i;\theta,\phi,\tau)
$$

be the empirical calibration risk, and let

$$
\widehat C_{\mathrm{cal}}(\tau) = \frac{1}{|\mathcal{D}_{\mathrm{cal}}|}\sum_{(x_i,y_i)\in\mathcal{D}_{\mathrm{cal}}}
\mathrm{Cov}(x_i;\phi,\tau)
$$

be the empirical calibration coverage.

Given target risk level $\alpha$, we choose

$$
\tau^* = \arg\max_{\tau \in \mathcal{T}} \widehat C_{\mathrm{cal}}(\tau)
\quad \text{s.t.} \quad \widehat R_{\mathrm{cal}}(\tau) \le \alpha_n,
$$

where $\alpha_n$ includes the finite-sample correction required by the chosen CRC instantiation and $\mathcal{T}$ is a threshold grid or continuous search domain.

The final predictor is $\tilde y_{\theta,\phi,\tau^*}$.

### 5.6 Algorithm

**Algorithm 1: Localized Selective Conformal Risk Control (LS-CRC)**

**Input:** training set $\mathcal{D}_{\mathrm{tr}}$, validation set $\mathcal{D}_{\mathrm{val}}$, calibration set $\mathcal{D}_{\mathrm{cal}}$, risk target $\alpha$, threshold grid $\mathcal{T}$.

1. Train segmentation backbone $f_\theta$ on $\mathcal{D}_{\mathrm{tr}}$ using $\mathcal{L}_{\mathrm{seg}}$.
2. For each $(x,y) \in \mathcal{D}_{\mathrm{tr}} \cup \mathcal{D}_{\mathrm{val}}$, compute prediction maps, uncertainty maps, and pseudo-labels $r^*$.
3. Train rejector $g_\phi$ using $\mathcal{L}_{\mathrm{rej}} + \lambda_2 \mathcal{L}_{\mathrm{smooth}}$.
4. Optionally jointly fine-tune $(\theta,\phi)$ using
   $$
   \mathcal{L}_{\mathrm{seg}} + \lambda_1 \mathcal{L}_{\mathrm{rej}} + \lambda_2 \mathcal{L}_{\mathrm{smooth}} + \lambda_3 \mathcal{L}_{\mathrm{loc\text{-}sur}}.
   $$
5. Freeze $(\theta,\phi)$.
6. For each threshold $\tau \in \mathcal{T}$, compute $\widehat R_{\mathrm{cal}}(\tau)$ and $\widehat C_{\mathrm{cal}}(\tau)$ on $\mathcal{D}_{\mathrm{cal}}$.
7. Select
   $$
   \tau^* = \arg\max_{\tau \in \mathcal{T}} \widehat C_{\mathrm{cal}}(\tau)
   \quad \text{s.t.} \widehat R_{\mathrm{cal}}(\tau) \le \alpha_n.
   $$
8. Output the selective predictor $\tilde y_{\theta,\phi,\tau^*}$.

---

## 6. Theory

We next state three theoretical results. The first preserves the standard CRC-style marginal guarantee. The second controls subgroup deviation in terms of calibration error and subgroup complexity. The third formalizes how a sufficiently accurate rejector reduces tail risk relative to a baseline acceptance rule.

### 6.1 Assumptions

We work under the following assumptions.

**Assumption A1 (Exchangeability).** Samples in the calibration and test sets are exchangeable.

**Assumption A2 (Bounded localized loss).** For every threshold $\tau \in \mathcal{T}$,
$$
0 \le L_{\mathrm{loc}}(X,Y;\theta,\phi,\tau) \le 1.
$$

**Assumption A3 (Monotonicity in threshold).** Increasing the acceptance threshold cannot increase accepted risky mass, so the localized loss is nonincreasing in $\tau$.

**Assumption A4 (Finite hypothesis search).** The calibration search is performed over a finite grid $\mathcal{T}$, or is otherwise measurable with a standard discretization argument.

These are standard or mild in split conformal risk calibration, provided the loss is constructed with a threshold-monotone numerator and a threshold-independent denominator.

### 6.2 Marginal guarantee

**Lemma 1 (Threshold monotonicity).** Suppose the segmentation model $(\theta,\phi)$ is fixed before calibration and the localized selective loss is defined as in Eq. (1). Under Assumption A3, the empirical calibration risk $\widehat R_{\mathrm{cal}}(\tau)$ is nonincreasing in $\tau$.

**Proof sketch.** If $\tau' > \tau$, then $A_{\phi,\tau'}(x)_u \le A_{\phi,\tau}(x)_u$ for every pixel. Therefore the numerator of $L_{\mathrm{loc}}$ cannot increase, while the denominator is unchanged. Averaging over the calibration set preserves monotonicity.

**Theorem 1 (Marginal localized selective risk control).** Under Assumptions A1--A4, let $\tau^*$ be selected by Algorithm 1 using a split conformal risk control rule with target $\alpha$. Then the selective predictor satisfies

$$
\mathbb{E}\big[L_{\mathrm{loc}}(X,Y;\theta,\phi,\tau^*)\big] \le \alpha + \delta_n,
$$

where $\delta_n = O(1/|\mathcal{D}_{\mathrm{cal}}|)$ is the finite-sample calibration slack induced by the particular CRC instantiation.

**Proof sketch.** Because $(\theta,\phi)$ are fixed before calibration, threshold selection is the only adaptive component. The localized loss is bounded and monotone. Standard split CRC arguments therefore apply directly to the threshold family indexed by $\tau$, yielding finite-sample control up to the usual calibration slack.

### 6.3 Subgroup deviation bound

Let $\mathcal{G}=\{G_1,\dots,G_K\}$ be a partition or collection of measurable subgroups over images, such as small-object images, boundary-heavy images, or difficulty strata. Define subgroup risk

$$
R_k(\tau) = \mathbb{E}[L_{\mathrm{loc}}(X,Y;\theta,\phi,\tau) \mid X \in G_k].
$$

Let $\widehat R_k(\tau)$ denote the corresponding empirical estimate on calibration or validation data.

**Lemma 2 (Uniform subgroup deviation).** Assume each subgroup $G_k$ contains at least $m_k$ calibration points and $L_{\mathrm{loc}} \in [0,1]$. Then for any fixed threshold $\tau$ and confidence level $1-\eta$,

$$
\max_{k \le K} |R_k(\tau) - \widehat R_k(\tau)|
\le
\sqrt{\frac{\log(2K/\eta)}{2\,m_{\min}}},
$$

where $m_{\min}=\min_k m_k$.

**Proof sketch.** Apply Hoeffding's inequality within each subgroup and union bound over $K$ groups.

**Theorem 2 (Subgroup risk gap bound).** Under the assumptions of Lemma 2 and Theorem 1, with probability at least $1-\eta$,

$$
\max_{k \le K} \big(R_k(\tau^*) - \alpha\big)
\le
\delta_n + \Gamma_{\mathrm{sel}} + \sqrt{\frac{\log(2K/\eta)}{2\,m_{\min}}},
$$

where $\Gamma_{\mathrm{sel}}$ is the subgroup mismatch term

$$
\Gamma_{\mathrm{sel}} := \max_k \big(\widehat R_k(\tau^*) - \widehat R_{\mathrm{cal}}(\tau^*)\big)_+.
$$

**Interpretation.** Exact distribution-free conditional guarantees are generally impossible. Theorem 2 instead shows that subgroup excess risk is controlled by three terms: the global conformal slack, a subgroup mismatch term induced by heterogeneity, and a sampling deviation term. This formalizes the role of the learned rejector: it should reduce $\Gamma_{\mathrm{sel}}$ by steering abstention toward subgroup-specific risky regions.

### 6.4 Tail-risk improvement

Define image-level localized risk

$$
Z_{\tau}(X,Y) := L_{\mathrm{loc}}(X,Y;\theta,\phi,\tau).
$$

For $\beta \in (0,1)$, define the upper-tail conditional value at risk

$$
\mathrm{CVaR}_{\beta}(Z_\tau) := \mathbb{E}[Z_\tau \mid Z_\tau \ge q_{\beta}(Z_\tau)],
$$

where $q_\beta$ is the $\beta$-quantile.

Let $A^{\mathrm{base}}_\tau$ denote a baseline acceptance rule, such as entropy thresholding, and let $A^{\mathrm{opt}}_\tau$ denote the ideal safe-region indicator minimizing localized miss at a given coverage. Suppose the learned rejector approximates the ideal rule in the sense that

$$
\mathbb{E}\big[\|A_{\phi,\tau}(X) - A^{\mathrm{opt}}_\tau(X)\|_1\big] \le \varepsilon,
$$

and the baseline rule is $\varepsilon_0$-far from the optimum under the same metric. Assume further that the image-level localized risk is $L$-Lipschitz with respect to the acceptance mask in normalized Hamming distance.

**Theorem 3 (Tail-risk improvement under rejector quality).** Under the above conditions,

$$
\mathrm{CVaR}_{\beta}(Z_{\tau}^{\mathrm{ours}})
\le
\mathrm{CVaR}_{\beta}(Z_{\tau}^{\mathrm{base}})
- \Delta_{\beta}
+ L(\varepsilon + \varepsilon_0),
$$

where $\Delta_{\beta} \ge 0$ is the tail advantage of the optimal acceptance rule over the baseline at the same target coverage.

**Proof sketch.** By Lipschitz continuity, replacing the optimal mask by an approximate one perturbs image-level risk by at most $L\varepsilon$ in expectation, and similarly for the baseline by $L\varepsilon_0$. Since CVaR is monotone and 1-Lipschitz with respect to additive perturbations, the tail gap between our rule and the baseline is lower bounded by the optimal tail advantage minus the approximation penalties.

**Interpretation.** Theorem 3 isolates the mechanism by which learned abstention improves reliability: if the rejector more closely matches the latent safe-region structure than hand-crafted confidence thresholds do, then tail risk improves correspondingly.

---

## 7. Discussion

LS-CRC should be interpreted as a middle ground between two extremes. On one side, pure selective segmentation methods can learn rich abstention policies but lack auditable finite-sample guarantees. On the other side, standard CRC methods provide rigorous marginal guarantees but do not explicitly learn localized abstention policies and therefore may be inefficient or unstable across difficult subgroups.

Our framework combines the strengths of both. The conformal calibration layer preserves a user-facing risk budget, while the learned rejector reshapes the accepted region so that the same risk budget is spent more intelligently. The subgroup theorem clarifies that we should not claim impossible fully distribution-free conditional guarantees. Instead, the right goal is to reduce subgroup mismatch and tail concentration while keeping a valid marginal audit trail.

The choice of localized weight map is central. A boundary-weighted risk emphasizes contour fidelity, which is attractive in medical segmentation and other dense prediction settings with critical interfaces. A small-object-weighted map would instead focus the guarantee on rare structures. This flexibility is a strength of the framework: it lets the practitioner specify *where* reliability matters most, without changing the high-level calibration pipeline.

There are also limitations. First, the theoretical guarantees depend on freezing the trained model before calibration and on using a bounded threshold-monotone loss. Second, pseudo-labels for rejection are imperfect and introduce approximation error into the learned abstention policy. Third, subgroup guarantees remain approximate and depend on finite subgroup sample sizes. Future work should address multiclass segmentation, tighter conditional guarantees under structural assumptions, and robust extensions under domain shift.

---

## 8. Experiments

### 8.1 Experimental goals

The experiments are designed to test three claims.

1. **Risk control:** LS-CRC satisfies the target localized risk budget after calibration.
2. **Conditional stability:** LS-CRC reduces image-wise and subgroup-wise risk heterogeneity relative to standard baselines.
3. **Efficiency:** LS-CRC achieves a better risk--coverage trade-off than uncertainty-threshold baselines and global CRC rules.

### 8.2 Datasets

We use binary medical segmentation benchmarks that are large enough for meaningful calibration yet small enough for rapid iteration.

**Kvasir-SEG.** A benchmark of 1000 polyp images with binary masks. We use it as the primary in-domain dataset.

**CVC-ClinicDB.** A benchmark of 612 colonoscopy frames with binary segmentation masks. We use it for external evaluation and mild cross-dataset generalization.

#### Recommended splits

For Kvasir-SEG:
- 600 training images,
- 100 validation images,
- 150 calibration images,
- 150 test images.

For CVC-ClinicDB:
- all images used as an external test set, or alternatively
- 100 calibration / 512 test if external calibration is desired.

The held-out calibration split is mandatory for split CRC.

### 8.3 Implementation details

#### Backbone

We use either:
- **U-Net** for lightweight iteration, or
- **DeepLabV3+ with ResNet-50** for a stronger default backbone.

For the main paper prototype, DeepLabV3+ is the preferred choice.

#### Input resolution

Images are resized to either:
- $256 \times 256$ for resource-constrained settings, or
- $352 \times 352$ when preserving boundary detail is more important.

#### Optimization

- optimizer: AdamW
- initial learning rate: $10^{-4}$
- weight decay: $10^{-4}$
- batch size: 8 or 16, depending on GPU memory
- training epochs:
  - 80--120 for segmentation pretraining,
  - 30--50 for rejector training,
  - 20--40 for joint fine-tuning
- early stopping based on validation Dice and validation localized risk surrogate

#### Loss weights

A practical default is:

$$
\lambda_{\mathrm{dice}} = 1,
\qquad
\lambda_1 = 1,
\qquad
\lambda_2 = 0.1,
\qquad
\lambda_3 = 0.5.
$$

These should be tuned by validation performance and stability.

### 8.4 Weight maps and localized risk instantiations

We instantiate two localized risks.

#### Unweighted accepted FNR

Set $w_u = 1$.

#### Boundary-weighted accepted FNR

Construct a thin boundary band around the foreground contour using a morphological dilation--erosion band. Then define

$$
w_u = 1 + \lambda_b \cdot \mathbf{1}\{u \in \text{boundary band}\},
$$

with $\lambda_b \in \{1,2,4\}$ tuned on validation data.

The boundary-weighted loss is the primary metric because it better reflects clinically relevant errors.

### 8.5 Baselines

We compare against the following baselines.

1. **Plain segmentation.** No abstention, no conformal calibration.
2. **Entropy threshold.** Reject pixels with entropy above a threshold.
3. **Max-softmax threshold.** Reject pixels whose maximum class probability is below a threshold.
4. **Global CRC threshold.** A single global threshold calibrated to satisfy the target risk.
5. **LS-CRC (ours).** Learned rejector + split CRC calibration.

If code availability permits, an additional comparison to a recent adaptive/conditional conformal segmentation baseline should be included.

### 8.6 Evaluation metrics

We report four groups of metrics.

#### Segmentation quality
- Dice
- IoU
- optional Boundary F-score

#### Selective efficiency
- accepted-area ratio (coverage)
- abstention ratio
- retained Dice on accepted region

#### Risk control
- expected localized selective risk
- expected unweighted accepted FNR
- expected boundary-weighted accepted FNR

#### Conditional and tail reliability
- standard deviation of image-level risk
- worst-10% image risk
- CVaR$_{0.9}$ of image-level risk
- worst-group risk across predefined subgroups

### 8.7 Subgroup definitions

To evaluate conditional stability, we define the following subgroups.

#### By object size
- small: foreground ratio < 5%
- medium: foreground ratio in [5%, 15%)
- large: foreground ratio >= 15%

#### By boundary complexity
- low complexity: low boundary-length / area ratio
- high complexity: high boundary-length / area ratio

#### By model difficulty
- low difficulty: low mean entropy
- high difficulty: high mean entropy

These subgroups are used for worst-group risk and subgroup deviation analysis.

### 8.8 Main tables

#### Table 1: Main in-domain test results (Kvasir-SEG)

| Method | Dice ↑ | IoU ↑ | Coverage ↑ | Localized Risk ↓ | Risk Std ↓ | Worst-10% Risk ↓ | CVaR$_{0.9}$ ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plain segmentation |  |  | 1.00 |  |  |  |  |
| Entropy threshold |  |  |  |  |  |  |  |
| Max-softmax threshold |  |  |  |  |  |  |  |
| Global CRC threshold |  |  |  |  |  |  |  |
| LS-CRC (ours) |  |  |  |  |  |  |  |

#### Table 2: External test results (CVC-ClinicDB)

| Method | Dice ↑ | IoU ↑ | Coverage ↑ | Localized Risk ↓ | Risk Std ↓ | Worst-10% Risk ↓ | CVaR$_{0.9}$ ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plain segmentation |  |  | 1.00 |  |  |  |  |
| Entropy threshold |  |  |  |  |  |  |  |
| Max-softmax threshold |  |  |  |  |  |  |  |
| Global CRC threshold |  |  |  |  |  |  |  |
| LS-CRC (ours) |  |  |  |  |  |  |  |

#### Table 3: Subgroup analysis on Kvasir-SEG

| Method | Small Obj Risk ↓ | Medium Obj Risk ↓ | Large Obj Risk ↓ | High Boundary Risk ↓ | High Difficulty Risk ↓ | Worst-Group Risk ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Entropy threshold |  |  |  |  |  |  |
| Global CRC threshold |  |  |  |  |  |  |
| LS-CRC (ours) |  |  |  |  |  |  |

#### Table 4: Ablation study

| Variant | Learned Rejector | Localized Weight | Smoothness | Joint Fine-tune | Coverage ↑ | Localized Risk ↓ | Worst-10% Risk ↓ |
|---|---|---|---|---|---:|---:|---:|
| A | ✗ | ✗ | ✗ | ✗ |  |  |  |
| B | ✓ | ✗ | ✗ | ✗ |  |  |  |
| C | ✓ | ✓ | ✗ | ✗ |  |  |  |
| D | ✓ | ✓ | ✓ | ✗ |  |  |  |
| E | ✓ | ✓ | ✓ | ✓ |  |  |  |

### 8.9 Figure plan

#### Figure 1: Method overview
A pipeline figure showing:
- image input,
- segmentation probability map,
- uncertainty features,
- learned rejector map,
- calibrated acceptance mask,
- final selective segmentation output.

#### Figure 2: Risk--coverage curve
Plot localized risk versus coverage for:
- entropy threshold,
- max-softmax threshold,
- global CRC threshold,
- LS-CRC.

This figure should visually demonstrate that LS-CRC dominates the baseline frontier over a substantial coverage range.

#### Figure 3: Tail-risk comparison
A bar plot or line plot for:
- mean risk,
- worst-10% image risk,
- CVaR$_{0.9}$,
across all methods.

#### Figure 4: Subgroup risk plot
A grouped bar chart comparing subgroup risks for:
- small / medium / large objects,
- low / high boundary complexity,
- low / high difficulty.

#### Figure 5: Qualitative visualization
For representative images, show:
- input image,
- ground-truth mask,
- segmentation prediction,
- entropy map,
- learned rejector map,
- accepted mask,
- final selective prediction.

At least one example should illustrate a thin boundary or small-structure case where LS-CRC rejects a localized risky region that uncertainty thresholding fails to isolate cleanly.

### 8.10 Recommended ablations

We recommend the following ablation studies.

1. **Rejector vs hand-crafted uncertainty.** Replace the learned rejector with entropy thresholding.
2. **No localized weights.** Set $w_u = 1$ everywhere.
3. **No smoothness penalty.** Remove $\mathcal{L}_{\mathrm{smooth}}$.
4. **No joint fine-tuning.** Train the rejector only after freezing the backbone.
5. **Alternative uncertainty features.** Compare entropy-only, entropy+margin, and full feature stacks.
6. **Different risk targets $\alpha$.** Evaluate at multiple target budgets.
7. **Threshold grid sensitivity.** Verify that calibration is stable with respect to threshold discretization.

### 8.11 Expected outcome narrative

The intended empirical narrative is the following.

- Plain segmentation has no abstention and therefore serves as a high-coverage but uncontrolled baseline.
- Entropy and max-softmax thresholding reduce average risk but remain poorly localized and often underperform on worst-image and subgroup metrics.
- Global CRC thresholds satisfy the global target but are typically conservative and spatially blunt.
- LS-CRC achieves comparable or better global risk control while delivering lower subgroup mismatch, lower worst-10% risk, and lower CVaR at similar coverage.

---

## 9. Conclusion

We presented LS-CRC, a framework for localized selective conformal risk control in segmentation. The method augments a segmentation network with a learned spatial rejector and calibrates the resulting acceptance threshold with split conformal risk control. This yields a predictor that preserves the core marginal guarantee of CRC while substantially improving reliability where it matters most: hard images, difficult subgroups, and spatially localized critical regions.

The framework is intentionally modular. Different spatial weight maps induce different operational notions of reliability, and different rejector architectures can be integrated without changing the calibration principle. This makes LS-CRC a promising foundation for reliable dense prediction beyond binary segmentation.

Future work should extend the framework to multiclass segmentation, domain shift, and stronger conditional guarantees under structural assumptions.

---

## 10. Appendix

This appendix expands the theory sketches, adds paper-to-code pseudocode, and provides a reproducibility section suitable for an ICML-style supplement.

### 10.1 Additional notation

For an image-mask pair (x, y) and threshold tau, define:

- A_u: acceptance indicator at pixel u
- yhat_u: hard segmentation prediction at pixel u
- M_u: foreground false-negative indicator, equal to 1 when y_u = 1 and yhat_u = 0
- W_u: local importance weight

The localized selective loss can be written as:

L_loc(x, y; theta, phi, tau)
= [sum over u of W_u * A_u * M_u] / [sum over u of W_u * y_u + eps]

Define the image-level risk random variable:

Z_tau = L_loc(X, Y; theta, phi, tau)

For a subgroup G, define subgroup risk:

R_G(tau) = E[ Z_tau | X in G ]

---

### 10.2 Expanded proof sketches

#### 10.2.1 Lemma 1: threshold monotonicity

Recall that A_u = 1 if the rejector score s_phi(x)_u is at least tau. If tau' > tau, then every accepted pixel under tau' is also accepted under tau. Therefore the accepted region can only shrink as tau increases.

Because the numerator of L_loc only counts weighted false negatives inside the accepted region, increasing tau cannot increase the numerator. The denominator does not depend on tau. Hence the loss is nonincreasing in tau for each sample, and therefore the empirical calibration risk is also nonincreasing in tau.

This monotonicity is the key property needed for split conformal threshold selection.

#### 10.2.2 Theorem 1: marginal localized selective risk control

The proof follows the standard split CRC template.

Step 1. Train the segmentation backbone and rejector using the train and validation splits. Freeze all learned parameters before calibration.

Step 2. For each threshold tau in the search set, define a bounded loss function ell_tau(x, y) = L_loc(x, y; theta, phi, tau).

Step 3. Because the calibration data are exchangeable, the loss is bounded in [0, 1], and the threshold family is monotone, split conformal risk calibration applies directly.

Step 4. Choose the most permissive threshold satisfying the corrected calibration constraint. The resulting predictor inherits the finite-sample marginal guarantee, up to the usual calibration slack delta_n.

Interpretation:
- the guarantee is exact at the marginal level,
- the only adaptive choice on calibration data is the scalar threshold,
- the user-facing risk budget remains auditable.

#### 10.2.3 Lemma 2: uniform subgroup deviation

Fix a threshold tau and a finite set of subgroups G_1, ..., G_K. Within each subgroup, the image-level losses are bounded in [0, 1]. For subgroup k with m_k examples, Hoeffding's inequality gives a concentration bound between empirical subgroup risk and population subgroup risk.

Applying a union bound over all K subgroups yields:

max over k of |R_k(tau) - Rhat_k(tau)|
<= sqrt( log(2K / eta) / (2 * m_min) )

with probability at least 1 - eta, where m_min is the smallest subgroup size.

This bound shows that subgroup evaluation is statistically meaningful once each subgroup has enough calibration or evaluation examples.

#### 10.2.4 Theorem 2: subgroup risk gap bound

For each subgroup k, decompose the excess risk as:

R_k(tau*) - alpha
= [R_k(tau*) - Rhat_k(tau*)]
+ [Rhat_k(tau*) - Rhat_cal(tau*)]
+ [Rhat_cal(tau*) - alpha]

Each term has a distinct meaning:

1. Sampling deviation term:
   bounded by the subgroup concentration result from Lemma 2.

2. Subgroup mismatch term:
   captures how much worse the subgroup is than the calibration average.
   Define:
   Gamma_sel = max over k of max( Rhat_k(tau*) - Rhat_cal(tau*), 0 )

3. Global calibration term:
   by construction, Rhat_cal(tau*) <= alpha + delta_n.

Combining the three terms gives:

max over k of [R_k(tau*) - alpha]
<= delta_n + Gamma_sel + sqrt( log(2K / eta) / (2 * m_min) )

Interpretation:
- this is not a full distribution-free conditional guarantee,
- it quantifies how subgroup excess risk depends on calibration slack, subgroup mismatch, and finite-sample estimation error,
- the learned rejector is useful precisely because it can reduce Gamma_sel.

#### 10.2.5 Theorem 3: tail-risk improvement

Let A_ours, A_base, and A_opt denote the acceptance masks from the learned rejector, a baseline uncertainty threshold, and the latent optimal acceptance rule at the same target coverage.

Assume image-level localized risk is Lipschitz with respect to the normalized Hamming distance between acceptance masks. Then if the learned rejector is closer to the optimal acceptance structure than the baseline rule is, the corresponding image-level risks are also closer to the optimum.

Because CVaR is monotone and stable under bounded perturbations, this implies an upper bound of the form:

CVaR_beta(ours)
<= CVaR_beta(base) - Delta_beta + L * (eps + eps0)

where:
- Delta_beta is the tail advantage of the optimal rule over the baseline,
- eps is the approximation error of the learned rejector relative to the optimal mask,
- eps0 is the approximation error of the baseline,
- L is the Lipschitz constant.

Interpretation:
- tail improvements should appear before mean-risk improvements if the learned rejector localizes hard regions well,
- this motivates reporting worst-10% image risk and CVaR, not only average risk.

---

### 10.3 Additional theoretical notes

#### Why normalize by total weighted foreground mass?

A denominator based on accepted foreground mass would also depend on tau, which would complicate monotonicity and calibration. Using a fixed target denominator keeps the threshold family well behaved while still measuring how many important foreground pixels are missed in accepted regions.

#### Why a subgroup theorem instead of a full conditional guarantee?

A fully distribution-free conditional guarantee is not attainable in broad generality. The subgroup formulation is therefore the right level of ambition: it is auditable, interpretable, and empirically testable.

#### When should the tail theorem hold most clearly?

The strongest gains are expected when risky regions are spatially concentrated, entropy maps are blurry, and the rejector can exploit richer feature cues than scalar uncertainty alone.

---

### 10.4 Paper-level pseudocode

Algorithm 2: Training and calibrating LS-CRC

Input:
- train set D_tr
- validation set D_val
- calibration set D_cal
- segmentation model f_theta
- rejector g_phi
- threshold grid T
- target risk alpha

Output:
- calibrated predictor (f_theta, g_phi, tau*)

1. Train f_theta on D_tr using segmentation loss.
2. For each sample in D_tr union D_val:
   - compute probability map
   - compute hard prediction
   - compute entropy and margin maps
   - compute boundary band and local weight map
   - build pseudo-labels for safe and unsafe pixels
3. Train g_phi using masked BCE plus smoothness regularization.
4. Optionally fine-tune theta and phi jointly using:
   total loss = seg loss + lambda1 * reject loss + lambda2 * smoothness loss + lambda3 * localized surrogate loss
5. Freeze theta and phi.
6. For each tau in T:
   - compute empirical calibration risk
   - compute empirical coverage
7. Choose tau* as the highest-coverage threshold satisfying the corrected risk constraint.
8. Evaluate on the test set.

---

### 10.5 Engineering bridge: implementation pseudocode

#### 10.5.1 Segmentation pretraining

```python
for epoch in range(num_epochs_seg):
    seg_model.train()
    for x, y in train_loader:
        prob, feat = seg_model(x)
        loss_seg = bce_loss(prob, y) + lambda_dice * dice_loss(prob, y)
        optimizer.zero_grad()
        loss_seg.backward()
        optimizer.step()
```

#### 10.5.2 Pseudo-label construction

```python
def build_rejector_targets(prob, pred, gt, boundary_band, e_low, e_high):
    entropy = -(prob * torch.log(prob + 1e-8) + (1 - prob) * torch.log(1 - prob + 1e-8))

    safe = (pred == gt) & (entropy < e_low) & (boundary_band == 0)
    unsafe = (pred != gt) | (entropy > e_high) | (boundary_band == 1)

    target = torch.full_like(prob, fill_value=-1)
    target[safe] = 1
    target[unsafe] = 0
    return target
```

#### 10.5.3 Rejector training

```python
for epoch in range(num_epochs_rej):
    rejector.train()
    for x, y in train_loader:
        with torch.no_grad():
            prob, feat = seg_model(x)
            pred = (prob >= 0.5).float()
            boundary_band = make_boundary_band(y)
            target = build_rejector_targets(prob, pred, y, boundary_band, e_low, e_high)

        score = rejector(feat, prob)
        loss_rej = masked_bce(score, target)
        loss_smooth = total_variation(score)
        loss = loss_rej + lambda_smooth * loss_smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

#### 10.5.4 Joint fine-tuning

```python
for epoch in range(num_epochs_joint):
    selective_model.train()
    for x, y in train_loader:
        prob, feat = seg_model(x)
        score = rejector(feat, prob)

        pred = (prob >= 0.5).float()
        boundary_band = make_boundary_band(y)
        target = build_rejector_targets(prob, pred, y, boundary_band, e_low, e_high)
        weight_map = make_boundary_weights(y, lambda_b)

        loss_seg = bce_loss(prob, y) + lambda_dice * dice_loss(prob, y)
        loss_rej = masked_bce(score, target)
        loss_smooth = total_variation(score)
        loss_loc = localized_surrogate(prob, score, y, weight_map)

        loss = loss_seg + lambda_1 * loss_rej + lambda_2 * loss_smooth + lambda_3 * loss_loc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

#### 10.5.5 Threshold calibration

```python
def compute_localized_risk(prob, score, gt, tau, weight_map):
    pred = (prob >= 0.5).float()
    accept = (score >= tau).float()
    miss = ((gt == 1) & (accept == 1) & (pred == 0)).float()
    num = (weight_map * miss).sum()
    den = (weight_map * (gt == 1).float()).sum() + 1e-8
    return (num / den).item()


def calibrate_tau(model, calib_loader, tau_grid, alpha_corr):
    best_tau = None
    best_cov = -1

    for tau in tau_grid:
        risks, covs = [], []
        for x, y in calib_loader:
            prob, feat = model.seg(x)
            score = model.rejector(feat, prob)
            weight_map = make_boundary_weights(y, lambda_b)

            risk = compute_localized_risk(prob, score, y, tau, weight_map)
            cov = (score >= tau).float().mean().item()
            risks.append(risk)
            covs.append(cov)

        mean_risk = np.mean(risks)
        mean_cov = np.mean(covs)

        if mean_risk <= alpha_corr and mean_cov > best_cov:
            best_cov = mean_cov
            best_tau = tau

    return best_tau
```

---

### 10.6 Reproducibility section (ICML-style)

#### 10.6.1 Data preprocessing

- Convert all images to RGB.
- Resize all images and masks to a fixed resolution.
- Convert all masks to binary values in {0, 1}.
- Save deterministic train, validation, calibration, and test splits.
- Never use the calibration set for training or model selection.

#### 10.6.2 Data augmentation

Recommended augmentations:
- random horizontal flip
- random vertical flip
- small-angle random rotation
- scale jitter
- mild color jitter if appropriate
- optional random crop after resize

Apply the same geometric transforms to images, masks, and pseudo-label geometry.

#### 10.6.3 Determinism and seeds

Fix:
- Python random seed
- NumPy seed
- PyTorch seed
- dataloader worker seeds

Report main metrics over at least 3 random seeds whenever feasible.

#### 10.6.4 Checkpoint selection

- Segmentation backbone: best validation Dice, tie-broken by lower validation localized surrogate risk.
- Rejector: best validation masked BCE plus smoothness objective.
- Joint model: lowest validation localized surrogate risk without meaningful Dice collapse.

This policy must be fixed before test evaluation.

#### 10.6.5 Hyperparameter tuning policy

- Tune architecture and optimizer hyperparameters using train and validation only.
- Tune boundary-weight coefficient on validation only.
- Choose threshold only on calibration data.
- Apply the same calibration protocol to every method.

#### 10.6.6 Calibration protocol

For each target alpha in {0.05, 0.10, 0.15}:
1. Define a threshold grid of 101 evenly spaced values in [0, 1].
2. Compute mean calibration risk and mean coverage for each threshold.
3. Select the highest-coverage threshold satisfying the corrected risk budget.
4. Evaluate once on the test set.

#### 10.6.7 Metrics computation

For each test image compute:
- Dice
- IoU
- coverage
- localized selective risk
- unweighted selective risk
- image-level risk value

Aggregate into:
- mean risk
- risk standard deviation
- worst-10% image risk
- CVaR_0.9
- worst-group risk

Fix subgroup definitions before inspecting test results.

#### 10.6.8 Compute budget disclosure

Report:
- GPU model and memory
- wall-clock time per training stage
- batch size
- number of epochs
- parameter counts for backbone and rejector

Recommended disclosure table:

| Component | Params | Resolution | Batch Size | Epochs | GPU | Time |
|---|---:|---:|---:|---:|---|---|
| Segmentation backbone |  |  |  |  |  |  |
| Rejector training |  |  |  |  |  |  |
| Joint fine-tuning |  |  |  |  |  |  |

#### 10.6.9 Failure analysis protocol

Save and inspect:
- top 20 highest-risk test images
- 20 random test images
- 20 images from the hardest subgroup

Visualize for each:
- input image
- ground-truth mask
- predicted mask
- uncertainty map
- rejector score map
- accepted mask
- final selective prediction

This checks whether the rejector truly localizes risk instead of merely shrinking coverage everywhere.

#### 10.6.10 Recommended result reporting format

Report:
- one in-domain main table
- one external/OOD table
- one subgroup table
- one ablation table
- one risk-coverage figure
- one qualitative figure

Where possible, report mean ± standard deviation over 3 seeds.

---

### 10.7 Practical implementation checklist

#### Phase A: basic infrastructure
- [ ] download and verify Kvasir-SEG
- [ ] download and verify CVC-ClinicDB
- [ ] create deterministic splits
- [ ] implement dataset loader
- [ ] implement resize and augmentation pipeline
- [ ] visually verify image-mask alignment

#### Phase B: segmentation backbone
- [ ] implement U-Net or DeepLabV3+
- [ ] implement BCE + Dice loss
- [ ] train and save best segmentation checkpoint
- [ ] export probability and entropy maps for debugging

#### Phase C: rejector
- [ ] implement feature hooks from the decoder
- [ ] implement entropy and margin maps
- [ ] implement boundary-band generator
- [ ] implement pseudo-label generation
- [ ] implement masked BCE and total variation loss
- [ ] train and validate the rejector

#### Phase D: localized risk and calibration
- [ ] implement boundary-weight map
- [ ] implement localized selective loss
- [ ] implement threshold grid search
- [ ] numerically verify monotonicity on calibration data
- [ ] calibrate thresholds for each risk target

#### Phase E: evaluation
- [ ] compute Dice, IoU, and coverage
- [ ] compute mean risk and risk std
- [ ] compute worst-10% risk
- [ ] compute CVaR_0.9
- [ ] compute subgroup metrics
- [ ] generate qualitative visualizations

#### Phase F: ablations
- [ ] entropy-only baseline
- [ ] max-softmax baseline
- [ ] no localized weights
- [ ] no smoothness term
- [ ] no joint fine-tuning
- [ ] multiple risk targets

---

### 10.8 Recommended experimental milestones

#### Milestone 1: reproduce baseline segmentation
Success criterion:
- stable Dice and IoU on Kvasir-SEG
- clean qualitative masks

#### Milestone 2: learned rejector beats entropy threshold on worst-case metrics
Success criterion:
- lower worst-10% risk than entropy threshold at similar coverage

#### Milestone 3: calibrated LS-CRC satisfies target risk
Success criterion:
- empirical test localized risk close to or below target across 3 seeds

#### Milestone 4: external test does not collapse
Success criterion:
- external CVC performance degrades gracefully and abstention remains spatially meaningful

#### Milestone 5: paper-ready plots and tables
Success criterion:
- main table
- subgroup table
- risk-coverage curve
- one strong qualitative figure

---

### 10.9 Suggested default hyperparameters

- backbone: DeepLabV3+ with ResNet-50
- image size: 256 x 256
- batch size: 8
- optimizer: AdamW
- learning rate: 1e-4
- weight decay: 1e-4
- segmentation epochs: 100
- rejector epochs: 40
- joint fine-tuning epochs: 30
- threshold grid: 101 points in [0, 1]
- pseudo-label entropy thresholds: e_low in [0.05, 0.15], e_high in [0.20, 0.40]
- boundary-weight coefficient lambda_b in {1, 2, 4}
- risk targets alpha in {0.05, 0.10, 0.15}

These defaults are intended for the first strong prototype.

---

### 10.10 Minimal experiment bundle for immediate execution

If you want the fastest meaningful prototype, start with this bundle:

1. Train DeepLabV3+ on Kvasir-SEG.
2. Train a rejector using entropy plus feature maps.
3. Calibrate thresholds at alpha = 0.10 and 0.15.
4. Compare against entropy threshold and max-softmax threshold.
5. Report:
   - Dice
   - coverage
   - mean localized risk
   - worst-10% risk
   - CVaR_0.9
6. Evaluate externally on CVC-ClinicDB.

This is the smallest experiment package that can still validate the main thesis of the paper.

---

## 11. Clean Code Architecture and File-by-File Implementation Guide

This section translates the method into a repository structure that can be implemented incrementally and tested stage by stage.

### 11.1 Architectural principles

The codebase should follow five principles.

1. **Stage separation.** Segmentation training, rejector training, joint fine-tuning, calibration, and evaluation should be separate executable stages.
2. **Single-responsibility modules.** Datasets, models, losses, calibration, and evaluation should not be mixed in the same file.
3. **Config-first execution.** Every experiment should be driven by YAML config files rather than hard-coded values.
4. **Paper-to-code traceability.** Every quantity in the paper should correspond to an explicit module, function, or config field.
5. **Reproducible outputs.** Every run should save config snapshots, checkpoints, metrics, figures, and calibration thresholds.

### 11.2 Recommended repository structure

```text
ls_crc_seg/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── data/
│   │   ├── kvasir.yaml
│   │   └── cvc_clinicdb.yaml
│   ├── model/
│   │   ├── unet.yaml
│   │   ├── deeplabv3p.yaml
│   │   └── rejector.yaml
│   ├── train/
│   │   ├── stage1_seg.yaml
│   │   ├── stage2_rejector.yaml
│   │   ├── stage3_joint.yaml
│   │   └── optimizer.yaml
│   ├── calib/
│   │   └── crc.yaml
│   ├── eval/
│   │   └── metrics.yaml
│   └── experiment/
│       ├── exp_kvasir_main.yaml
│       ├── exp_kvasir_cvc_transfer.yaml
│       └── exp_ablation.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── src/
│   ├── datasets/
│   │   ├── base_dataset.py
│   │   ├── polyp_dataset.py
│   │   ├── transforms.py
│   │   ├── split_loader.py
│   │   └── sample_structs.py
│   │
│   ├── models/
│   │   ├── backbones/
│   │   │   ├── unet.py
│   │   │   └── deeplabv3p.py
│   │   ├── heads/
│   │   │   ├── rejector_head.py
│   │   │   └── feature_adapter.py
│   │   ├── selective_model.py
│   │   └── build_model.py
│   │
│   ├── losses/
│   │   ├── seg_loss.py
│   │   ├── reject_loss.py
│   │   ├── smoothness_loss.py
│   │   ├── localized_surrogate.py
│   │   └── build_loss.py
│   │
│   ├── calibration/
│   │   ├── risk_functions.py
│   │   ├── weight_maps.py
│   │   ├── threshold_search.py
│   │   ├── crc_calibrator.py
│   │   └── calibration_result.py
│   │
│   ├── evaluation/
│   │   ├── metrics_seg.py
│   │   ├── metrics_selective.py
│   │   ├── metrics_risk.py
│   │   ├── subgroup_eval.py
│   │   ├── aggregators.py
│   │   └── report_builder.py
│   │
│   ├── trainers/
│   │   ├── base_trainer.py
│   │   ├── seg_trainer.py
│   │   ├── rejector_trainer.py
│   │   ├── joint_trainer.py
│   │   └── callbacks.py
│   │
│   ├── pipelines/
│   │   ├── train_seg_pipeline.py
│   │   ├── train_rejector_pipeline.py
│   │   ├── train_joint_pipeline.py
│   │   ├── calibrate_pipeline.py
│   │   └── evaluate_pipeline.py
│   │
│   ├── utils/
│   │   ├── seed.py
│   │   ├── io.py
│   │   ├── logger.py
│   │   ├── checkpoint.py
│   │   ├── config.py
│   │   ├── device.py
│   │   ├── visualization.py
│   │   └── registry.py
│   │
│   └── scripts/
│       ├── make_splits.py
│       ├── preprocess_polyp.py
│       ├── train_seg.py
│       ├── train_rejector.py
│       ├── train_joint.py
│       ├── calibrate_crc.py
│       ├── evaluate.py
│       ├── run_ablation.py
│       └── export_figures.py
│
└── outputs/
    ├── runs/
    ├── checkpoints/
    ├── metrics/
    ├── calibrations/
    └── figures/
```

### 11.3 Core design objects

The following abstractions should exist explicitly.

#### 11.3.1 Data sample object

File: `src/datasets/sample_structs.py`

Purpose:
- define a standard structure passed between dataset, trainer, and evaluator.

Recommended dataclass:

```python
@dataclass
class SegSample:
    image: torch.Tensor
    mask: torch.Tensor
    image_id: str
    meta: dict
```

Input:
- raw image and mask loaded from disk

Output:
- normalized tensors plus metadata such as original size, source dataset, and split name

Why important:
- keeps loaders, metrics, and visualization consistent

#### 11.3.2 Batch dictionary contract

Every dataloader batch should expose the same keys:

- `image`: float tensor, shape [B, C, H, W]
- `mask`: float tensor, shape [B, 1, H, W]
- `image_id`: list[str]
- `meta`: dict

This avoids trainer-specific dataset logic.

### 11.4 Dataset layer

#### File: `src/datasets/base_dataset.py`

Purpose:
- abstract dataset interface
- shared utilities for path validation and metadata loading

Main class:
- `BaseSegmentationDataset(torch.utils.data.Dataset)`

Methods:
- `__len__`
- `__getitem__`
- `_load_image`
- `_load_mask`
- `_build_meta`

#### File: `src/datasets/polyp_dataset.py`

Purpose:
- concrete dataset for Kvasir-SEG and CVC-ClinicDB

Main class:
- `PolypSegDataset(BaseSegmentationDataset)`

Inputs:
- image root
- mask root
- split file path
- transforms
- image size

Outputs:
- batch dictionary contract above

#### File: `src/datasets/transforms.py`

Purpose:
- image/mask transforms that preserve alignment

Functions/classes:
- `build_train_transforms(cfg)`
- `build_eval_transforms(cfg)`
- `ResizePair`
- `RandomFlipPair`
- `RandomRotatePair`
- `NormalizeImage`

#### File: `src/datasets/split_loader.py`

Purpose:
- read deterministic split files from JSON/YAML/TXT

Functions:
- `load_split(split_path)`
- `save_split(split_dict, path)`

### 11.5 Model layer

#### File: `src/models/backbones/unet.py`

Purpose:
- implement U-Net baseline

Main class:
- `UNetBackbone(nn.Module)`

Forward output should return:
- `prob_map`
- `logit_map`
- `feature_map`

Recommended output contract:

```python
return {
    "logits": logits,
    "prob": torch.sigmoid(logits),
    "features": decoder_features,
}
```

#### File: `src/models/backbones/deeplabv3p.py`

Purpose:
- stronger default backbone

Main class:
- `DeepLabV3PlusBackbone(nn.Module)`

Output contract should match U-Net output contract exactly.

#### File: `src/models/heads/rejector_head.py`

Purpose:
- map segmentation features and uncertainty summaries to acceptance scores

Main class:
- `RejectorHead(nn.Module)`

Inputs:
- feature map from backbone
- probability map
- optional entropy map
- optional margin map

Output:
- `score`: tensor in [0, 1], shape [B, 1, H, W]

Suggested forward signature:

```python
def forward(self, features, prob, entropy=None, margin=None):
    ...
    return score
```

#### File: `src/models/heads/feature_adapter.py`

Purpose:
- harmonize feature channels from different backbones before feeding to rejector

Main class:
- `FeatureAdapter(nn.Module)`

#### File: `src/models/selective_model.py`

Purpose:
- wrap backbone + rejector into one unified model

Main class:
- `SelectiveSegModel(nn.Module)`

Responsibilities:
- call backbone
- derive uncertainty maps
- call rejector
- return all intermediate tensors needed by losses and calibration

Recommended forward output:

```python
{
    "logits": logits,
    "prob": prob,
    "features": features,
    "entropy": entropy,
    "margin": margin,
    "score": score,
}
```

This file is central because it encodes the paper-level computation graph.

#### File: `src/models/build_model.py`

Purpose:
- construct models from config without hard-coding

Functions:
- `build_backbone(cfg)`
- `build_rejector(cfg)`
- `build_selective_model(cfg)`

### 11.6 Loss layer

#### File: `src/losses/seg_loss.py`

Purpose:
- standard segmentation losses

Functions/classes:
- `dice_loss(prob, mask)`
- `bce_dice_loss(prob, mask, cfg)`

Input:
- predicted probability map
- ground-truth mask

Output:
- scalar loss tensor

#### File: `src/losses/reject_loss.py`

Purpose:
- masked BCE for rejector pseudo-label supervision

Functions:
- `masked_bce(score, target, ignore_value=-1)`

Input:
- `score`: [B,1,H,W]
- `target`: [B,1,H,W] with values {0,1,-1}

Output:
- scalar loss tensor

#### File: `src/losses/smoothness_loss.py`

Purpose:
- total variation or edge-aware smoothness penalty on score map

Functions:
- `total_variation(score)`
- optional `edge_aware_tv(score, image)`

#### File: `src/losses/localized_surrogate.py`

Purpose:
- differentiable surrogate for localized selective risk

Functions:
- `localized_surrogate(prob, score, mask, weight_map)`

Input:
- probability map
- acceptance score map
- ground-truth mask
- local weight map

Output:
- scalar surrogate loss

#### File: `src/losses/build_loss.py`

Purpose:
- factory functions returning callables or objects from config

### 11.7 Calibration layer

This layer should be completely independent from training logic.

#### File: `src/calibration/weight_maps.py`

Purpose:
- construct spatial importance maps from masks

Functions:
- `make_boundary_band(mask, radius)`
- `make_boundary_weights(mask, lambda_b, radius)`
- optional `make_small_object_weights(mask, ...)`

Input:
- mask tensor

Output:
- weight map tensor matching mask shape

#### File: `src/calibration/risk_functions.py`

Purpose:
- exact non-differentiable risk definitions used for calibration and evaluation

Functions:
- `localized_selective_risk(prob, score, mask, tau, weight_map)`
- `coverage(score, tau)`
- `accepted_fnr(prob, score, mask, tau)`

These functions must implement the exact paper metrics, not surrogates.

#### File: `src/calibration/threshold_search.py`

Purpose:
- threshold search over tau grid

Functions:
- `search_best_tau(calib_records, tau_grid, alpha_corr)`

Input:
- list of precomputed calibration records per image
- tau grid
- corrected risk target

Output:
- best threshold and summary statistics

#### File: `src/calibration/crc_calibrator.py`

Purpose:
- orchestration object for calibration

Main class:
- `CRCCalibrator`

Methods:
- `collect_calibration_records(model, loader)`
- `fit(records)`
- `save(result_path)`
- `load(result_path)`

#### File: `src/calibration/calibration_result.py`

Purpose:
- dataclass for storing calibrated threshold and metadata

Recommended dataclass:

```python
@dataclass
class CalibrationResult:
    tau_star: float
    alpha_target: float
    alpha_corrected: float
    mean_calib_risk: float
    mean_calib_coverage: float
    grid_size: int
    risk_name: str
    metadata: dict
```

### 11.8 Evaluation layer

#### File: `src/evaluation/metrics_seg.py`

Purpose:
- segmentation metrics independent of calibration

Functions:
- `dice_score(pred, mask)`
- `iou_score(pred, mask)`
- optional `boundary_f1(pred, mask)`

#### File: `src/evaluation/metrics_selective.py`

Purpose:
- selective prediction metrics

Functions:
- `coverage_from_score(score, tau)`
- `abstention_ratio(score, tau)`
- `retained_dice(pred, mask, accept)`

#### File: `src/evaluation/metrics_risk.py`

Purpose:
- aggregate risk metrics

Functions:
- `image_level_risk(...)`
- `worst_k_percent(values, frac=0.1)`
- `cvar(values, beta=0.9)`

#### File: `src/evaluation/subgroup_eval.py`

Purpose:
- assign images to subgroups and compute group-level summaries

Functions:
- `assign_object_size_group(mask)`
- `assign_boundary_complexity_group(mask)`
- `assign_difficulty_group(entropy_map)`
- `compute_group_metrics(records)`

#### File: `src/evaluation/aggregators.py`

Purpose:
- merge per-image metrics into paper tables

Functions:
- `aggregate_main_metrics(records)`
- `aggregate_subgroup_metrics(records)`

#### File: `src/evaluation/report_builder.py`

Purpose:
- export CSV/JSON/Markdown summaries for tables and logs

### 11.9 Trainer layer

The trainer layer should be thin. Training logic belongs here; math belongs in models/losses/calibration.

#### File: `src/trainers/base_trainer.py`

Purpose:
- shared training loop utilities

Main class:
- `BaseTrainer`

Responsibilities:
- epoch loop
- metric logging
- checkpoint save/load
- validation trigger

Methods:
- `train_one_epoch`
- `validate_one_epoch`
- `save_checkpoint`
- `load_checkpoint`

#### File: `src/trainers/seg_trainer.py`

Purpose:
- stage 1 trainer

Main class:
- `SegTrainer(BaseTrainer)`

Input modules:
- backbone model
- segmentation loss
- optimizer
- scheduler
- train and val loaders

Output artifacts:
- best segmentation checkpoint
- training history

#### File: `src/trainers/rejector_trainer.py`

Purpose:
- stage 2 trainer

Main class:
- `RejectorTrainer(BaseTrainer)`

Special responsibilities:
- backbone frozen by default
- pseudo-label construction inside training/validation step
- rejector-specific logging

#### File: `src/trainers/joint_trainer.py`

Purpose:
- stage 3 trainer

Main class:
- `JointTrainer(BaseTrainer)`

Responsibilities:
- train selective model end-to-end
- combine segmentation, rejector, smoothness, and localized surrogate losses

#### File: `src/trainers/callbacks.py`

Purpose:
- reusable hooks such as early stopping, learning-rate logging, and checkpointing

### 11.10 Pipeline layer

Pipelines should orchestrate modules but not contain heavy math.

#### File: `src/pipelines/train_seg_pipeline.py`

Steps:
1. load config
2. build dataset and loaders
3. build backbone
4. build loss and optimizer
5. launch `SegTrainer`
6. save best checkpoint and metrics

#### File: `src/pipelines/train_rejector_pipeline.py`

Steps:
1. load config
2. load segmentation checkpoint
3. build selective model with frozen backbone
4. build rejector loss and optimizer
5. launch `RejectorTrainer`
6. save best rejector checkpoint

#### File: `src/pipelines/train_joint_pipeline.py`

Steps:
1. load config
2. load segmentation + rejector checkpoints
3. unfreeze required modules
4. build joint losses
5. launch `JointTrainer`
6. save final joint model checkpoint

#### File: `src/pipelines/calibrate_pipeline.py`

Steps:
1. load model checkpoint
2. load calibration split
3. build calibrator
4. compute calibration records
5. search tau*
6. save `CalibrationResult`

#### File: `src/pipelines/evaluate_pipeline.py`

Steps:
1. load model checkpoint
2. load calibration result
3. load test set
4. compute per-image metrics
5. aggregate metrics
6. export tables and figures

### 11.11 Utility layer

#### File: `src/utils/config.py`

Purpose:
- YAML loading and hierarchical config merge

Functions:
- `load_yaml(path)`
- `merge_configs(*cfgs)`
- `save_resolved_config(cfg, out_path)`

#### File: `src/utils/seed.py`

Functions:
- `set_global_seed(seed)`
- `seed_worker(worker_id)`

#### File: `src/utils/checkpoint.py`

Functions:
- `save_checkpoint(state, path)`
- `load_checkpoint(path, map_location)`

#### File: `src/utils/logger.py`

Purpose:
- consistent console/file logging

#### File: `src/utils/visualization.py`

Functions:
- `plot_prediction_panel(...)`
- `plot_risk_coverage_curve(...)`
- `plot_group_bars(...)`

#### File: `src/utils/registry.py`

Purpose:
- optional lightweight registry for models, datasets, and trainers

### 11.12 Which files to write first

Write files in the following order.

#### Phase 1: minimum runnable segmentation baseline

1. `src/datasets/polyp_dataset.py`
2. `src/datasets/transforms.py`
3. `src/models/backbones/unet.py` or `deeplabv3p.py`
4. `src/losses/seg_loss.py`
5. `src/trainers/base_trainer.py`
6. `src/trainers/seg_trainer.py`
7. `src/pipelines/train_seg_pipeline.py`
8. `src/scripts/train_seg.py`

Goal:
- get a stable segmentation checkpoint and baseline Dice/IoU

#### Phase 2: rejector-only prototype

9. `src/models/heads/rejector_head.py`
10. `src/models/selective_model.py`
11. `src/calibration/weight_maps.py`
12. `src/losses/reject_loss.py`
13. `src/losses/smoothness_loss.py`
14. `src/trainers/rejector_trainer.py`
15. `src/pipelines/train_rejector_pipeline.py`
16. `src/scripts/train_rejector.py`

Goal:
- produce reasonable acceptance maps

#### Phase 3: calibration and evaluation

17. `src/calibration/risk_functions.py`
18. `src/calibration/threshold_search.py`
19. `src/calibration/crc_calibrator.py`
20. `src/evaluation/metrics_seg.py`
21. `src/evaluation/metrics_selective.py`
22. `src/evaluation/metrics_risk.py`
23. `src/pipelines/calibrate_pipeline.py`
24. `src/pipelines/evaluate_pipeline.py`
25. `src/scripts/calibrate_crc.py`
26. `src/scripts/evaluate.py`

Goal:
- end-to-end calibrated evaluation

#### Phase 4: joint training and subgroup analysis

27. `src/losses/localized_surrogate.py`
28. `src/trainers/joint_trainer.py`
29. `src/pipelines/train_joint_pipeline.py`
30. `src/evaluation/subgroup_eval.py`
31. `src/evaluation/aggregators.py`
32. `src/scripts/train_joint.py`
33. `src/scripts/run_ablation.py`

Goal:
- full paper pipeline

### 11.13 Trainer execution order

The trainers should always run in this order:

1. **SegTrainer**
   - trains backbone only
   - output: `seg_best.pt`

2. **RejectorTrainer**
   - loads `seg_best.pt`
   - freezes backbone or most of backbone
   - trains rejector head
   - output: `rejector_best.pt`

3. **JointTrainer** (optional but recommended)
   - loads segmentation and rejector checkpoints
   - fine-tunes both using full loss
   - output: `joint_best.pt`

4. **CRCCalibrator**
   - loads either `rejector_best.pt` or `joint_best.pt`
   - computes tau*
   - output: `calibration_alpha_0.10.json`

5. **EvaluatePipeline**
   - loads model checkpoint + calibration result
   - runs test metrics and figure export

### 11.14 Input and output contract for every module

#### Dataset output

```python
{
    "image": FloatTensor[B, C, H, W],
    "mask": FloatTensor[B, 1, H, W],
    "image_id": list[str],
    "meta": dict,
}
```

#### Backbone output

```python
{
    "logits": FloatTensor[B, 1, H, W],
    "prob": FloatTensor[B, 1, H, W],
    "features": FloatTensor[B, F, H, W],
}
```

#### Selective model output

```python
{
    "logits": FloatTensor[B, 1, H, W],
    "prob": FloatTensor[B, 1, H, W],
    "features": FloatTensor[B, F, H, W],
    "entropy": FloatTensor[B, 1, H, W],
    "margin": FloatTensor[B, 1, H, W],
    "score": FloatTensor[B, 1, H, W],
}
```

#### Calibration record per image

```python
{
    "image_id": str,
    "prob": np.ndarray,
    "score": np.ndarray,
    "mask": np.ndarray,
    "weight_map": np.ndarray,
    "meta": dict,
}
```

#### Evaluation record per image

```python
{
    "image_id": str,
    "dice": float,
    "iou": float,
    "coverage": float,
    "localized_risk": float,
    "risk_group": str,
    "size_group": str,
    "difficulty_group": str,
    "meta": dict,
}
```

### 11.15 Config system: what YAML fields should exist

Use small composable YAML files and merge them at runtime.

#### `configs/data/kvasir.yaml`

Recommended fields:

```yaml
dataset_name: kvasir_seg
root_dir: data/processed/kvasir
image_dir: images
mask_dir: masks
split_dir: data/splits/kvasir
image_size: 256
num_workers: 4
pin_memory: true
augment:
  hflip: true
  vflip: true
  rotate_deg: 15
  scale_jitter: [0.9, 1.1]
  color_jitter: false
```

#### `configs/model/deeplabv3p.yaml`

```yaml
name: deeplabv3p
encoder: resnet50
in_channels: 3
num_classes: 1
pretrained: true
feature_dim: 256
```

#### `configs/model/rejector.yaml`

```yaml
name: rejector_head
in_feature_dim: 256
use_prob: true
use_entropy: true
use_margin: true
hidden_dim: 64
num_layers: 2
kernel_size: 3
use_sigmoid: true
```

#### `configs/train/stage1_seg.yaml`

```yaml
stage: seg
seed: 42
epochs: 100
batch_size: 8
optimizer:
  name: adamw
  lr: 0.0001
  weight_decay: 0.0001
scheduler:
  name: cosine
loss:
  bce_weight: 1.0
  dice_weight: 1.0
checkpoint:
  monitor: val_dice
  mode: max
```

#### `configs/train/stage2_rejector.yaml`

```yaml
stage: rejector
seed: 42
epochs: 40
batch_size: 8
freeze_backbone: true
pseudo_label:
  e_low: 0.10
  e_high: 0.30
  boundary_radius: 3
loss:
  reject_weight: 1.0
  smoothness_weight: 0.1
optimizer:
  name: adamw
  lr: 0.0001
  weight_decay: 0.0001
checkpoint:
  monitor: val_reject_loss
  mode: min
```

#### `configs/train/stage3_joint.yaml`

```yaml
stage: joint
seed: 42
epochs: 30
batch_size: 8
unfreeze_backbone: true
loss:
  seg_weight: 1.0
  reject_weight: 1.0
  smoothness_weight: 0.1
  localized_surrogate_weight: 0.5
pseudo_label:
  e_low: 0.10
  e_high: 0.30
  boundary_radius: 3
optimizer:
  name: adamw
  lr: 0.00005
  weight_decay: 0.0001
checkpoint:
  monitor: val_localized_surrogate
  mode: min
```

#### `configs/calib/crc.yaml`

```yaml
risk_name: boundary_weighted_selective_fnr
alpha_target: 0.10
alpha_correction: 0.10
tau_grid_points: 101
tau_min: 0.0
tau_max: 1.0
weight_map:
  type: boundary
  boundary_radius: 3
  lambda_b: 2.0
selection_rule: max_coverage_under_risk
```

#### `configs/eval/metrics.yaml`

```yaml
metrics:
  - dice
  - iou
  - coverage
  - localized_risk
  - risk_std
  - worst10_risk
  - cvar90
subgroups:
  object_size: true
  boundary_complexity: true
  difficulty: true
save_predictions: true
num_visualizations: 40
```

#### `configs/experiment/exp_kvasir_main.yaml`

```yaml
experiment_name: exp_kvasir_main
output_dir: outputs/runs/exp_kvasir_main
data:
  train_dataset: configs/data/kvasir.yaml
  test_dataset: configs/data/kvasir.yaml
model:
  backbone: configs/model/deeplabv3p.yaml
  rejector: configs/model/rejector.yaml
train:
  seg: configs/train/stage1_seg.yaml
  rejector: configs/train/stage2_rejector.yaml
  joint: configs/train/stage3_joint.yaml
calib:
  crc: configs/calib/crc.yaml
eval:
  metrics: configs/eval/metrics.yaml
```

### 11.16 End-to-end scripts you need

#### `src/scripts/make_splits.py`

Purpose:
- create deterministic train/val/calib/test split files

Command example:

```bash
python -m src.scripts.make_splits --dataset kvasir --seed 42
```

#### `src/scripts/preprocess_polyp.py`

Purpose:
- optional preprocessing and folder normalization

#### `src/scripts/train_seg.py`

Purpose:
- run stage 1

Command example:

```bash
python -m src.scripts.train_seg --exp configs/experiment/exp_kvasir_main.yaml
```

#### `src/scripts/train_rejector.py`

Purpose:
- run stage 2 using saved segmentation checkpoint

#### `src/scripts/train_joint.py`

Purpose:
- run stage 3 joint fine-tuning

#### `src/scripts/calibrate_crc.py`

Purpose:
- compute calibrated tau*

Command example:

```bash
python -m src.scripts.calibrate_crc --exp configs/experiment/exp_kvasir_main.yaml --alpha 0.10
```

#### `src/scripts/evaluate.py`

Purpose:
- run final evaluation with saved calibration result

#### `src/scripts/run_ablation.py`

Purpose:
- launch ablation variants by overriding config values

#### `src/scripts/export_figures.py`

Purpose:
- regenerate paper figures from saved metric files without rerunning inference

### 11.17 Exact end-to-end execution order

A full experiment should be runnable as:

```bash
python -m src.scripts.make_splits --dataset kvasir --seed 42
python -m src.scripts.train_seg --exp configs/experiment/exp_kvasir_main.yaml
python -m src.scripts.train_rejector --exp configs/experiment/exp_kvasir_main.yaml
python -m src.scripts.train_joint --exp configs/experiment/exp_kvasir_main.yaml
python -m src.scripts.calibrate_crc --exp configs/experiment/exp_kvasir_main.yaml --alpha 0.10
python -m src.scripts.evaluate --exp configs/experiment/exp_kvasir_main.yaml --alpha 0.10
python -m src.scripts.export_figures --exp configs/experiment/exp_kvasir_main.yaml --alpha 0.10
```

### 11.18 Minimal viable repo version

If you need the smallest possible first version, implement only these files first:

- `polyp_dataset.py`
- `transforms.py`
- `deeplabv3p.py`
- `rejector_head.py`
- `selective_model.py`
- `seg_loss.py`
- `reject_loss.py`
- `smoothness_loss.py`
- `weight_maps.py`
- `risk_functions.py`
- `seg_trainer.py`
- `rejector_trainer.py`
- `crc_calibrator.py`
- `train_seg.py`
- `train_rejector.py`
- `calibrate_crc.py`
- `evaluate.py`

This is enough to produce the first real result table.

### 11.19 Common mistakes to avoid

1. **Mixing calibration and validation.**  
   Keep them strictly separate.

2. **Hard-coding feature channel sizes.**  
   Use `FeatureAdapter` or config-driven channel mapping.

3. **Using surrogate risk for evaluation.**  
   Evaluation must use the exact non-differentiable risk from `risk_functions.py`.

4. **Building subgroup labels after seeing results.**  
   Define subgroup rules before test-time analysis.

5. **Saving only scalar metrics.**  
   Always save per-image metrics for worst-case, CVaR, and subgroup analysis.

6. **Entangling training loops with calibration logic.**  
   Calibration must stay independent.

### 11.20 Recommended first week coding schedule

#### Day 1
- implement dataset loader and transforms
- verify image-mask visualization

#### Day 2
- implement segmentation backbone and segmentation trainer
- run first segmentation baseline

#### Day 3
- implement rejector head and selective model wrapper
- log entropy and feature maps

#### Day 4
- implement pseudo-label generation and rejector trainer
- inspect acceptance maps visually

#### Day 5
- implement exact risk functions and threshold calibration
- run first calibrated result

#### Day 6
- implement evaluation aggregation and worst-10% / CVaR metrics
- export first draft table

#### Day 7
- clean configs, rerun with fixed seed, and save a reproducible baseline

By the end of this schedule, you should have an end-to-end system that supports the first meaningful experiment.

