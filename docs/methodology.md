# Methodology and Mathematical Derivations

**Project:** Probability of Default (PD) model on Lending Club loans (2007–2015)
**Author:** Daniel Siretanu
**Status:** v0.1 (methodology, written before model build)

This document sets out the mathematics behind every model, transformation and metric used in this repository. Each result is derived from first principles rather than stated, so a validator can check the reasoning independently of the code.

---

## Contents

1. [Problem definition](#1-problem-definition)
2. [Logistic regression](#2-logistic-regression)
3. [Weight of Evidence (WoE)](#3-weight-of-evidence-woe)
4. [Information Value (IV)](#4-information-value-iv)
5. [Scorecard scaling](#5-scorecard-scaling)
6. [Gradient boosting with LightGBM](#6-gradient-boosting-with-lightgbm)
7. [Explainable Boosting Machine (EBM)](#7-explainable-boosting-machine-ebm)
8. [SHAP values and reason codes](#8-shap-values-and-reason-codes)
9. [Calibration](#9-calibration)
10. [Discrimination metrics: ROC, AUC, Gini, KS](#10-discrimination-metrics-roc-auc-gini-ks)
11. [Stability: PSI and CSI](#11-stability-psi-and-csi)
12. [Validation design](#12-validation-design)
13. [From PD to expected loss and capital](#13-from-pd-to-expected-loss-and-capital)
14. [References](#14-references)

---

## Notation

| Symbol | Meaning |
|---|---|
| $n$ | number of loans in the train set |
| $x_i \in \mathbb{R}^d$ | feature vector of loan $i$ (known at origination) |
| $y_i \in \{0, 1\}$ | default indicator: $1$ = bad (default), $0$ = good |
| $p_i = P(y_i = 1 \mid x_i)$ | probability of default of loan $i$ |
| $\sigma(z) = 1/(1+e^{-z})$ | logistic (sigmoid) function |
| $\pi = P(y = 1)$ | portfolio default rate (prior) |
| $N_G, N_B$ | total number of goods and bads |
| $g_k, b_k$ | number of goods and bads in bin $k$ of a feature |

---

## 1. Problem definition

### 1.1 Target

A loan is labelled **bad** ($y = 1$) if its final status is *Charged Off* or *Default*, and **good** ($y = 0$) if it is *Fully Paid*. Loans still *Current*, *In Grace Period* or *Late* have no final outcome and are excluded, because including them would label unresolved loans as good and bias PD downwards.

The PD estimated here is therefore a **lifetime PD over the loan term** (36 or 60 months), conditional on information available at origination:

```math
\mathrm{PD}(x) = P\left(Y = 1 \mid X = x\right).
```

### 1.2 Leakage rule

Only variables known on the application date may enter $x$. Fields populated after origination (for example `total_pymnt`, `recoveries`, `last_pymnt_d`, `out_prncp`) are dropped, because they encode the outcome itself.

---

## 2. Logistic regression

### 2.1 Model

Each $y_i$ is a Bernoulli random variable with parameter $p_i$. Logistic regression models the **log-odds** of default as linear in the features:

```math
\ln\frac{p_i}{1 - p_i} = \beta_0 + x_i^\top \beta \quad\Longleftrightarrow\quad p_i = \sigma\left(\beta_0 + x_i^\top \beta\right).
```

The logit is not an arbitrary choice. Writing the Bernoulli mass function in exponential-family form,

```math
P(y \mid p) = p^{y}(1-p)^{1-y} = \exp\left\{ y \ln\frac{p}{1-p} + \ln(1-p) \right\},
```

shows that the natural parameter is $\theta = \ln\frac{p}{1-p}$. The logit is therefore the **canonical link** for binary data, which gives the log-likelihood the concavity derived below.

To keep notation compact, absorb the intercept into $\beta$ by setting $x_{i0} = 1$, so $p_i = \sigma(x_i^\top\beta)$.

### 2.2 Log-likelihood

Assuming independent loans, the likelihood and log-likelihood are

```math
L(\beta) = \prod_{i=1}^{n} p_i^{\,y_i}(1-p_i)^{1-y_i},
\qquad
\ell(\beta) = \sum_{i=1}^{n}\Big[ y_i \ln p_i + (1-y_i)\ln(1-p_i) \Big].
```

Maximising $\ell$ is the same as minimising the **log-loss** (binary cross-entropy) $-\ell(\beta)/n$, which is also the loss used by LightGBM in Section 6.

### 2.3 Gradient

The sigmoid satisfies $\sigma'(z) = \sigma(z)\,(1-\sigma(z))$, so

```math
\frac{\partial p_i}{\partial \beta} = p_i(1-p_i)\,x_i .
```

Differentiating each term of $\ell$:

```math
\frac{\partial \ell}{\partial \beta}
= \sum_{i=1}^{n}\left[\frac{y_i}{p_i} - \frac{1-y_i}{1-p_i}\right] p_i(1-p_i)\,x_i
= \sum_{i=1}^{n}\Big[ y_i(1-p_i) - (1-y_i)p_i \Big] x_i
= \sum_{i=1}^{n} (y_i - p_i)\,x_i .
```

In matrix form, with design matrix $X \in \mathbb{R}^{n \times (d+1)}$:

```math
\nabla \ell(\beta) = X^\top (y - p).
```

Setting this to zero gives the **score equations**. Because the intercept column is all ones, the first equation is $\sum_i (y_i - p_i) = 0$: at the MLE, the average predicted PD equals the observed default rate in the train set.

### 2.4 Hessian and concavity

Differentiating the gradient again:

```math
\nabla^2 \ell(\beta) = -\sum_{i=1}^{n} p_i(1-p_i)\,x_i x_i^\top = -X^\top W X,
\qquad W = \operatorname{diag}\big(p_i(1-p_i)\big).
```

For any vector $v$, $v^\top X^\top W X v = \sum_i p_i(1-p_i)(x_i^\top v)^2 \ge 0$, so the Hessian is negative semi-definite and $\ell$ is **concave**. If $X$ has full column rank, it is strictly concave and the maximum likelihood estimate $\hat\beta$ is **unique**. (If the classes are perfectly separable, $\hat\beta$ diverges; WoE binning and regularisation both prevent this in practice.)

### 2.5 Estimation: Newton–Raphson and IRLS

Newton's method updates

```math
\beta^{(t+1)} = \beta^{(t)} - \left[\nabla^2\ell\right]^{-1}\nabla\ell
= \beta^{(t)} + \left(X^\top W X\right)^{-1} X^\top (y - p).
```

Defining the working response $z = X\beta^{(t)} + W^{-1}(y - p)$, the update becomes

```math
\beta^{(t+1)} = \left(X^\top W X\right)^{-1} X^\top W z,
```

a weighted least squares problem. Iterating is **Iteratively Reweighted Least Squares (IRLS)**, the algorithm used by `statsmodels`.

### 2.6 Inference

Under standard regularity conditions, $\hat\beta$ is asymptotically normal with covariance equal to the inverse Fisher information:

```math
\widehat{\operatorname{Var}}(\hat\beta) = \left(X^\top \hat W X\right)^{-1},
\qquad
z_j = \frac{\hat\beta_j}{\widehat{\operatorname{se}}(\hat\beta_j)} \;\sim\; N(0,1) \text{ under } H_0:\beta_j = 0 .
```

These Wald statistics are reported for every scorecard coefficient. The coefficient $\beta_j$ has a direct reading: a one-unit increase in $x_j$ multiplies the odds of default by $e^{\beta_j}$.

---

## 3. Weight of Evidence (WoE)

### 3.1 Definition

Each feature is split into bins $k = 1,\dots,K$ (quantile bins for continuous variables, merged categories for categorical ones). Let

```math
D^{G}_k = \frac{g_k}{N_G}, \qquad D^{B}_k = \frac{b_k}{N_B}
```

be the share of all goods and all bads that fall in bin $k$. The Weight of Evidence of bin $k$ is

```math
\mathrm{WoE}_k = \ln\frac{D^{G}_k}{D^{B}_k}.
```

Positive WoE means the bin holds relatively more goods (lower risk); negative WoE means more bads.

To avoid $\ln 0$ in sparse bins, a 0.5 smoothing adjustment is used:

```math
\mathrm{WoE}_k = \ln\frac{(g_k + 0.5)/N_G}{(b_k + 0.5)/N_B}.
```

### 3.2 Why WoE: the Bayesian derivation

By Bayes' theorem, for a loan whose feature value falls in bin $k$:

```math
\frac{P(G \mid k)}{P(B \mid k)}
= \frac{P(k \mid G)}{P(k \mid B)} \cdot \frac{P(G)}{P(B)} .
```

The empirical estimates of $P(k \mid G)$ and $P(k \mid B)$ are exactly $D^G_k$ and $D^B_k$. Taking logs:

```math
\underbrace{\ln\frac{P(G \mid k)}{P(B \mid k)}}_{\text{posterior log-odds}}
= \underbrace{\mathrm{WoE}_k}_{\text{evidence from the bin}}
+ \underbrace{\ln\frac{N_G}{N_B}}_{\text{prior log-odds}} .
```

So **WoE is the amount by which observing bin $k$ shifts the log-odds of being good**. It puts every feature on the same log-odds scale as the logistic regression itself.

### 3.3 Link to logistic regression

If the $d$ features were conditionally independent given the class (the naive Bayes assumption), the same argument applied to all of them gives

```math
\ln\frac{P(G \mid x)}{P(B \mid x)} = \ln\frac{N_G}{N_B} + \sum_{j=1}^{d}\mathrm{WoE}_j(x_j).
```

Since this project models the log-odds of **bad**, $\ln\frac{p}{1-p} = -\ln\frac{P(G\mid x)}{P(B\mid x)}$, and

```math
\ln\frac{p}{1-p} = \beta_0 + \sum_{j=1}^{d} \beta_j\,\mathrm{WoE}_j(x_j)
```

recovers naive Bayes with $\beta_j = -1$. Fitting the $\beta_j$ by maximum likelihood corrects for correlation between features. A fitted coefficient that is **positive** (the wrong sign under this convention) signals multicollinearity, and that feature is reviewed or removed.

### 3.4 Properties that matter for credit risk

- **Monotonicity:** bins are merged until WoE is monotonic in the underlying variable, so the score moves in a direction a credit officer can justify (for example, higher DTI never improves the score).
- **Non-linearity:** a linear model on WoE captures any monotonic, non-linear relationship between a raw variable and the log-odds.
- **Missing values:** missing gets its own bin and WoE, so missingness is modelled rather than imputed.
- **Outliers:** extreme values sit in the end bins and cannot dominate the fit.

---

## 4. Information Value (IV)

### 4.1 Definition

```math
\mathrm{IV} = \sum_{k=1}^{K}\left(D^{G}_k - D^{B}_k\right)\mathrm{WoE}_k
= \sum_{k=1}^{K}\left(D^{G}_k - D^{B}_k\right)\ln\frac{D^{G}_k}{D^{B}_k}.
```

### 4.2 IV is a symmetric KL divergence

The Kullback–Leibler divergence between the goods' distribution $D^G$ and the bads' distribution $D^B$ over the bins is

```math
\mathrm{KL}\left(D^G \,\Vert\, D^B\right) = \sum_k D^G_k \ln\frac{D^G_k}{D^B_k},
\qquad
\mathrm{KL}\left(D^B \,\Vert\, D^G\right) = \sum_k D^B_k \ln\frac{D^B_k}{D^G_k}.
```

Adding them:

```math
\mathrm{KL}\left(D^G \Vert D^B\right) + \mathrm{KL}\left(D^B \Vert D^G\right)
= \sum_k D^G_k \ln\frac{D^G_k}{D^B_k} - \sum_k D^B_k \ln\frac{D^G_k}{D^B_k}
= \sum_k \left(D^G_k - D^B_k\right)\ln\frac{D^G_k}{D^B_k} = \mathrm{IV}.
```

So IV is the **Jeffreys divergence** between the two class distributions. It is always non-negative (each term has $(a-b)$ and $\ln(a/b)$ of the same sign), equals zero only when the feature's distribution is identical for goods and bads, and grows as the feature separates them.

### 4.3 Screening rule

The industry rule of thumb (Siddiqi, 2006) is used to shortlist features:

| IV | Predictive power |
|---|---|
| < 0.02 | not useful |
| 0.02 – 0.10 | weak |
| 0.10 – 0.30 | medium |
| 0.30 – 0.50 | strong |
| > 0.50 | suspicious: check for leakage |

---

## 5. Scorecard scaling

### 5.1 Score as a linear function of log-odds

Banks present the model as an integer score where higher means safer. The score is defined as an affine function of the log-odds of being good:

```math
\text{Score} = \text{Offset} + \text{Factor} \cdot \ln(\text{odds}),
\qquad \text{odds} = \frac{1-p}{p}.
```

Two business choices fix the constants:

1. A reference score $S_0$ corresponds to reference odds $o_0$.
2. Every **PDO** ("points to double the odds") points, the odds double.

From condition 2:

```math
\text{Factor}\cdot\ln(2\,o) - \text{Factor}\cdot\ln(o) = \mathrm{PDO}
\;\;\Longrightarrow\;\;
\text{Factor} = \frac{\mathrm{PDO}}{\ln 2}.
```

From condition 1:

```math
\text{Offset} = S_0 - \text{Factor}\cdot\ln(o_0).
```

**Worked example** (the convention used in this project): $S_0 = 600$ at odds $o_0 = 50{:}1$ with $\mathrm{PDO} = 20$ gives

```math
\text{Factor} = \frac{20}{\ln 2} \approx 28.854,
\qquad
\text{Offset} = 600 - 28.854 \cdot \ln 50 \approx 487.12 .
```

### 5.2 Points per attribute

With the WoE logistic model, $\ln(\text{odds}) = -\left(\beta_0 + \sum_j \beta_j \mathrm{WoE}_j\right)$. Substituting:

```math
\text{Score} = \text{Offset} - \text{Factor}\left(\beta_0 + \sum_{j=1}^{d}\beta_j\,\mathrm{WoE}_j\right).
```

Spreading the intercept and offset evenly over the $d$ features gives the points for bin $k$ of feature $j$:

```math
\text{Points}_{j,k} = -\left(\beta_j\,\mathrm{WoE}_{j,k} + \frac{\beta_0}{d}\right)\text{Factor} + \frac{\text{Offset}}{d}.
```

A loan's score is the sum of the points of the bins it falls in, which is what makes a scorecard auditable line by line.

---

## 6. Gradient boosting with LightGBM

### 6.1 Additive model

The challenger model builds the log-odds as a sum of $M$ regression trees:

```math
F_M(x) = F_0 + \eta \sum_{m=1}^{M} f_m(x), \qquad p(x) = \sigma\left(F_M(x)\right),
```

where $\eta \in (0,1]$ is the learning rate and $F_0 = \ln\frac{\bar y}{1-\bar y}$ is the log-odds of the base default rate.

### 6.2 Loss, gradient and Hessian

For one observation with current raw score $F$ and $p = \sigma(F)$, the log-loss is

```math
L(y, F) = -\Big[y\ln\sigma(F) + (1-y)\ln\left(1-\sigma(F)\right)\Big].
```

Using the result from Section 2.3, its first and second derivatives with respect to $F$ are

```math
g = \frac{\partial L}{\partial F} = p - y,
\qquad
h = \frac{\partial^2 L}{\partial F^2} = p(1-p).
```

### 6.3 Optimal leaf values

At round $m$, a second-order Taylor expansion of the total loss around the current prediction, with L2 penalty $\lambda$ on leaf values, gives

```math
\mathcal{L}^{(m)} \approx \sum_{i=1}^{n}\left[g_i f(x_i) + \tfrac{1}{2} h_i f(x_i)^2\right] + \tfrac{1}{2}\lambda\sum_{\ell=1}^{T} w_\ell^2 ,
```

where the tree assigns value $w_\ell$ to every observation in leaf $\ell$. Grouping by leaf, with $G_\ell = \sum_{i \in \ell} g_i$ and $H_\ell = \sum_{i \in \ell} h_i$:

```math
\mathcal{L}^{(m)} \approx \sum_{\ell=1}^{T}\left[G_\ell w_\ell + \tfrac{1}{2}\left(H_\ell + \lambda\right) w_\ell^2\right].
```

Each bracket is a quadratic in $w_\ell$, minimised at

```math
w_\ell^{*} = -\frac{G_\ell}{H_\ell + \lambda},
\qquad
\mathcal{L}^{*} = -\frac{1}{2}\sum_{\ell=1}^{T}\frac{G_\ell^2}{H_\ell + \lambda}.
```

So each leaf moves the log-odds by a damped Newton step.

### 6.4 Split gain

Splitting a node into left and right children reduces the loss by

```math
\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma ,
```

where $\gamma$ is a minimum-gain penalty (`min_gain_to_split`). LightGBM evaluates this on histogram bins and grows trees **leaf-wise** (always splitting the leaf with the largest gain), which is what makes it fast on several hundred thousand Lending Club loans.

### 6.5 Monotonic constraints

For a feature $x_j$ with an economic sign (higher income should never raise PD), a constraint is imposed:

```math
x_j \le x_j' \;\Longrightarrow\; F(x) \ge F(x') \quad \text{(decreasing PD in } x_j\text{), all other features fixed.}
```

LightGBM enforces this by rejecting any split on $x_j$ whose left and right leaf values violate the required order, and by passing bounds down to descendant nodes so that no later split can break it. Every tree is then monotonic in $x_j$, and so is their sum. This gives up a little accuracy in exchange for a model whose behaviour matches the scorecard's WoE signs, which is a common model risk requirement.

---

## 7. Explainable Boosting Machine (EBM)

The EBM is a **generalised additive model with pairwise interactions** (GA²M):

```math
\ln\frac{p}{1-p} = \beta_0 + \sum_{j=1}^{d} f_j(x_j) + \sum_{(j,k) \in \mathcal{I}} f_{jk}(x_j, x_k).
```

Each shape function $f_j$ is learned by **cyclic gradient boosting**: in each round, a very shallow tree is fit to the current residual gradients using *one feature only*, with a very small learning rate, cycling through all features many times. Because each feature gets its turn in sequence and the steps are tiny, the order in which features are visited does not matter, and collinear features share credit rather than one absorbing it all. The result is bagged over several runs for stability.

A small set $\mathcal{I}$ of pairwise interactions is then added, selected by the FAST ranking procedure (Lou et al., 2013).

The EBM sits between the two other models: as transparent as the scorecard (each $f_j$ can be plotted and read directly), with accuracy close to LightGBM.

---

## 8. SHAP values and reason codes

### 8.1 Shapley values

For a model $f$ and a loan $x$, the contribution of feature $j$ is its Shapley value from cooperative game theory:

```math
\phi_j(x) = \sum_{S \subseteq \{1,\dots,d\}\setminus\{j\}}
\frac{|S|!\,(d - |S| - 1)!}{d!}\Big[v(S \cup \{j\}) - v(S)\Big],
```

where $v(S) = \mathbb{E}\left[f(X) \mid X_S = x_S\right]$ is the expected model output when only the features in $S$ are known. The weight is the probability that, in a random ordering of features, exactly the set $S$ comes before $j$, so $\phi_j$ is the **average marginal contribution of feature $j$ over all orderings**.

### 8.2 Axioms

Shapley values are the unique attribution satisfying:

- **Efficiency (local accuracy):** the contributions add up exactly to the prediction,
  ```math
  f(x) = \phi_0 + \sum_{j=1}^{d}\phi_j(x), \qquad \phi_0 = \mathbb{E}[f(X)].
  ```
- **Symmetry:** two features that always contribute equally get equal credit.
- **Dummy:** a feature that never changes the output gets $\phi_j = 0$.
- **Additivity:** for an ensemble $f = f_1 + f_2$, $\phi_j(f) = \phi_j(f_1) + \phi_j(f_2)$. This is what lets SHAP explain a sum of trees tree by tree.

### 8.3 Linear special case

For a linear model $f(x) = \beta_0 + \sum_j \beta_j x_j$ with independent features, $v(S) = \beta_0 + \sum_{k \in S}\beta_k x_k + \sum_{k \notin S}\beta_k\,\mathbb{E}[X_k]$, so every marginal contribution of $j$ equals $\beta_j(x_j - \mathbb{E}[X_j])$ and

```math
\phi_j(x) = \beta_j\left(x_j - \mathbb{E}[X_j]\right).
```

For the WoE scorecard this means each feature's SHAP value is its points relative to the average applicant, so the scorecard and LightGBM explanations can be compared on the same footing.

### 8.4 TreeSHAP and reason codes

Exact Shapley values need $2^d$ evaluations in general. **TreeSHAP** (Lundberg et al., 2020) computes them exactly for tree ensembles in polynomial time, $O(TLD^2)$ for $T$ trees with $L$ leaves and depth $D$. All SHAP values in this project are computed in **log-odds space**, where efficiency holds exactly.

**Reason codes:** for a declined applicant, the features with the largest positive $\phi_j$ (pushing PD up) are reported as the principal reasons, as required for adverse action notices under the US Equal Credit Opportunity Act (Regulation B).

---

## 9. Calibration

Discrimination (ranking) and calibration (correct levels) are different properties. A PD feeds expected loss and capital, so its **level** must be right.

### 9.1 Measures

- **Brier score:** $\mathrm{BS} = \frac{1}{n}\sum_i (p_i - y_i)^2$, the mean squared error of the probabilities.
- **Calibration curve:** loans are grouped into deciles of predicted PD, and mean predicted PD is plotted against observed default rate in each decile. A calibrated model lies on the 45° line.
- **Hosmer–Lemeshow test:** with $G$ groups, $O_g$ observed defaults, $E_g = \sum_{i \in g} p_i$ expected defaults and $n_g$ loans,
  ```math
  \mathrm{HL} = \sum_{g=1}^{G}\frac{(O_g - E_g)^2}{E_g\left(1 - E_g/n_g\right)} \;\sim\; \chi^2_{G-2} .
  ```

### 9.2 Prior correction

If the model is trained on a sample whose default rate $\pi_s$ differs from the portfolio rate $\pi$ (for example after undersampling goods), then by Bayes' theorem only the prior odds change:

```math
\frac{p_s}{1-p_s} = \frac{p}{1-p}\cdot\frac{\pi_s/(1-\pi_s)}{\pi/(1-\pi)} .
```

Rearranging gives the correction applied to the intercept:

```math
\ln\frac{p}{1-p} = \ln\frac{p_s}{1-p_s} - \ln\frac{\pi_s}{1-\pi_s} + \ln\frac{\pi}{1-\pi}.
```

### 9.3 Recalibration

For LightGBM, whose raw probabilities can be miscalibrated, a **Platt scaling** step fits $p = \sigma(a + b\,F(x))$ on held-out validation folds of the train set (never on the test set), which is itself a one-feature logistic regression (Section 2). Isotonic regression is used as a non-parametric check.

---

## 10. Discrimination metrics: ROC, AUC, Gini, KS

Let $s(x)$ be the predicted PD (higher means riskier) and let $F_B$, $F_G$ be the cumulative distribution functions of $s$ among bads and goods.

### 10.1 ROC and AUC

At threshold $t$, the true positive rate is $\mathrm{TPR}(t) = 1 - F_B(t)$ and the false positive rate is $\mathrm{FPR}(t) = 1 - F_G(t)$. The ROC curve plots TPR against FPR as $t$ varies, and AUC is the area under it.

**AUC as a probability.** Writing $f_G$ for the density of $s$ among goods,

```math
\mathrm{AUC} = \int_0^1 \mathrm{TPR}\; d\,\mathrm{FPR}
= \int_{-\infty}^{\infty}\left(1 - F_B(t)\right) f_G(t)\,dt
= P\left(S_B > S_G\right),
```

where $S_B$ and $S_G$ are the scores of a randomly chosen bad and good. (With ties, $\mathrm{AUC} = P(S_B > S_G) + \tfrac{1}{2}P(S_B = S_G)$.) This is the Mann–Whitney $U$ statistic divided by $N_B N_G$, which is how it is computed: by ranks, in $O(n\log n)$.

### 10.2 Gini

The Gini coefficient (accuracy ratio) is Somers' $D$ between score and outcome:

```math
\mathrm{Gini} = P(S_B > S_G) - P(S_B < S_G).
```

Since $P(S_B > S_G) + P(S_B < S_G) + P(S_B = S_G) = 1$,

```math
\mathrm{Gini} = 2\left[P(S_B > S_G) + \tfrac{1}{2}P(S_B = S_G)\right] - 1 = 2\,\mathrm{AUC} - 1 .
```

A random model has Gini 0 and a perfect one has Gini 1. Geometrically, it is the area between the model's CAP (Lorenz) curve and the random line, divided by the same area for a perfect model.

### 10.3 Kolmogorov–Smirnov (KS)

```math
\mathrm{KS} = \max_t \left| F_G(t) - F_B(t) \right| .
```

KS is the largest vertical gap between the score distributions of goods and bads, equivalently $\max_t(\mathrm{TPR} - \mathrm{FPR})$, Youden's $J$. The threshold where it occurs is a natural cut-off candidate.

---

## 11. Stability: PSI and CSI

### 11.1 Population Stability Index

Let $E_k$ be the share of the train set in score band $k$ and $A_k$ the share in a later sample (the test set, or new applicants during monitoring):

```math
\mathrm{PSI} = \sum_{k=1}^{K}\left(A_k - E_k\right)\ln\frac{A_k}{E_k}.
```

By the same algebra as Section 4.2, PSI is the **Jeffreys divergence** between the train and later score distributions, so it is non-negative and zero only when they match.

| PSI | Interpretation |
|---|---|
| < 0.10 | stable |
| 0.10 – 0.25 | moderate shift: investigate |
| > 0.25 | significant shift: review or redevelop |

### 11.2 Characteristic Stability Index

The **CSI** applies the same formula to each input feature's bins, which identifies *which* variables are driving a shift in the score.

---

## 12. Validation design

### 12.1 Out-of-time (OOT) split

Loans are split by **issue date**, not at random:

| Sample | Name in code | Issue years | Use |
|---|---|---|---|
| **Train** | `train` | 2007–2012 | model fitting, WoE binning, hyperparameter tuning, calibration |
| **Test** (out-of-time) | `test` | 2013–2015 | final, untouched performance estimate |

The train set is what bank model documentation calls the *development sample*, and the test set is the *out-of-time (OOT) sample*. Within the train set, $k$-fold cross-validation splits the data into **validation folds** for tuning; the word *test* is reserved for the 2013–2015 loans, which are used once, at the end.

A random split would let the model see loans from the same economic period it is tested on. An OOT split tests what actually matters: whether a model built on past vintages ranks and calibrates **future** applicants.

### 12.2 Outcome maturity

A 60-month loan issued in 2015 may still be running at the data cut-off. Keeping only finished loans in recent vintages over-represents those that ended early, and early endings include defaults, which inflates the observed default rate. This **right-censoring** bias is handled by restricting the analysis to vintages whose terms have matured, or by reporting results separately by term.

### 12.3 What is reported

For each model on both train and test: AUC, Gini, KS, Brier score, calibration curve and Hosmer–Lemeshow, PSI between train and test, and a bootstrap 95% confidence interval for Gini. A large drop in Gini from train to test signals overfitting or population drift.

---

## 13. From PD to expected loss and capital

### 13.1 Expected loss

```math
\mathrm{EL} = \mathrm{PD} \times \mathrm{LGD} \times \mathrm{EAD},
```

where LGD is the fraction of exposure lost if default occurs and EAD is the exposure at default. This assumes PD is independent of LGD and EAD. Expected loss drives pricing and provisions.

### 13.2 Basel IRB capital (illustration)

Under the Basel internal ratings-based (IRB) approach, capital covers **unexpected** loss. It comes from the asymptotic single risk factor (Vasicek) model, which gives the default rate in a 1-in-1000 downturn. For **other retail exposures**, the category that includes unsecured consumer loans such as Lending Club's, the asset correlation is

```math
R = 0.03\cdot\frac{1 - e^{-35\,\mathrm{PD}}}{1 - e^{-35}} + 0.16\cdot\left[1 - \frac{1 - e^{-35\,\mathrm{PD}}}{1 - e^{-35}}\right],
```

and the capital requirement per unit of exposure is

```math
K = \mathrm{LGD}\cdot\left[\Phi\left(\frac{\Phi^{-1}(\mathrm{PD}) + \sqrt{R}\;\Phi^{-1}(0.999)}{\sqrt{1-R}}\right) - \mathrm{PD}\right],
\qquad
\mathrm{RWA} = 12.5 \cdot K \cdot \mathrm{EAD},
```

where $\Phi$ is the standard normal CDF. No maturity adjustment applies to retail exposures. This shows why PD calibration matters: an error in the PD level carries straight through to capital.

---

## 14. References

1. Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. Wiley.
2. Thomas, L. C., Edelman, D. B. and Crook, J. N. (2002). *Credit Scoring and Its Applications*. SIAM.
3. Hosmer, D. W., Lemeshow, S. and Sturdivant, R. X. (2013). *Applied Logistic Regression*, 3rd ed. Wiley.
4. Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
5. Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*.
6. Lou, Y., Caruana, R., Gehrke, J. and Hooker, G. (2013). Accurate Intelligible Models with Pairwise Interactions. *KDD*.
7. Nori, H., Jenkins, S., Koch, P. and Caruana, R. (2019). InterpretML: A Unified Framework for Machine Learning Interpretability. *arXiv:1909.09223*.
8. Lundberg, S. M. and Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
9. Lundberg, S. M. et al. (2020). From Local Explanations to Global Understanding with Explainable AI for Trees. *Nature Machine Intelligence*.
10. Basel Committee on Banking Supervision (2005). *An Explanatory Note on the Basel II IRB Risk Weight Functions*.
11. Board of Governors of the Federal Reserve System (2011). *SR 11-7: Guidance on Model Risk Management*.
12. Prudential Regulation Authority (2023). *SS1/23: Model Risk Management Principles for Banks*.