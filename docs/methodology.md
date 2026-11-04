# Methodology and Mathematical Derivations

**Project:** Probability of Default (PD) model on Lending Club loans (2007–2015)

**Author:** Daniel Siretanu

**Status:** v0.1 (methodology, written before model build)

This document sets out the mathematics behind every model, transformation and metric used in this repository. Each result is derived from first principles rather than stated, so a validator can check the reasoning independently of the code.

---

## Overview

### What this project does

When someone applies for a loan, the lender has to decide whether to lend and at what interest rate. The central quantity behind that decision is the borrower's **probability of default (PD)**: the chance that they will fail to repay. This project uses public data from **LendingClub**, a US peer-to-peer lending platform, covering loans issued from 2007 to 2015, with one record per loan giving the borrower's application details and how the loan ended. The models are built on loans issued in 2007–2012 (the **train set**) using only information available on the day each loan was approved, and are then tested on loans issued in 2013–2015 (the **test set**), which they have never seen. This mimics how a bank uses a model built on past loans to assess future applicants (Section 12).

### Why PD matters

Banks use PD to:

- **approve or decline** applications;
- **price** loans, charging riskier borrowers more;
- **set aside provisions** for expected losses (Section 13.1);
- **hold capital** against unexpected losses (Section 13.2).

To support these decisions, a PD model needs two properties. It must **rank** borrowers correctly, putting riskier borrowers above safer ones (*discrimination*, Section 10). Its PDs must also be **accurate**: if it says 5%, about 5 in 100 such loans should default (*calibration*, Section 9). A model can rank perfectly and still be miscalibrated, for example if every PD is twice too high, which would make loans overpriced and losses overstated.

### Approach: one transparent model, two flexible models

This project compares a **transparent model** (the most interpretable model, often called the *champion* in banks) with two **flexible models** (more complex models built to test whether the transparent one can be beaten, often called *challengers* or *benchmarks*). The comparison shows how much accuracy, if any, is given up for interpretability.

| Model | Role | Strength | Weakness |
|---|---|---|---|
| **Weight of Evidence (WoE) logistic regression scorecard** (Sections 2–5) | transparent model (champion) | every decision can be explained feature by feature, so it is easy to validate and approve | effects change only in steps between bins, and features cannot interact |
| **LightGBM with monotonic constraints** (Section 6) | flexible model (challenger) | captures non-linear effects and interactions between features | harder to explain; needs SHAP (Section 8) to break each prediction into feature contributions |
| **Explainable Boosting Machine (EBM)** (Section 7) | flexible model (challenger) | learns a flexible curve for each feature that can still be plotted and read | harder than a scorecard to validate, get approved and monitor|

**What these terms mean.**

- **Bins and points (scorecard).** Each feature is split into ranges called *bins*, for example DTI (debt-to-income ratio) 0–10, 10–18, 18–25 and 25+. Each bin is replaced by its **Weight of Evidence**, a number measuring how much more common good loans are than bad loans in that bin (Section 3), and the scorecard turns this into a fixed number of points (Section 5). An applicant's score is the sum of the points for the bins they fall into:

  | Feature | Applicant's value | Bin | Points |
  |---|---|---|---|
  | DTI | 22 | 18–25 | 0 |
  | Annual income | \$45,000 | \$40k–\$60k | +10 |
  | FICO score | 710 | 700–740 | +20 |
  | **Score** | | | **base + 30** |

  Because the points are fixed per bin, a DTI of 18.1 and a DTI of 24.9 score exactly the same, and the effect jumps only at bin edges.

- **Interactions.** An interaction is when the effect of one feature depends on another. For example, a DTI of 30 is much riskier for someone earning \$25,000 than for someone earning \$200,000. The scorecard cannot represent this: it gives the same DTI points to both, because each feature adds its points independently.

- **LightGBM** builds the prediction from hundreds of small decision trees. A tree can split on income first and then on DTI, so it learns interactions automatically, and it is not limited to fixed bins. *Monotonic constraints* force selected features to move risk in one direction only (for example, higher DTI never lowers PD), so the model stays consistent with economic common sense.

- **SHAP** (SHapley Additive exPlanations) splits a single prediction into the contribution of each feature, for example "DTI added 3 percentage points to this applicant's PD, income removed 1". It is what makes LightGBM's individual decisions explainable.

- **EBM** learns a separate smooth curve for each feature instead of fixed bins, plus a few selected pairwise interactions. Each curve can be plotted, so it is almost as readable as a scorecard and usually more accurate.
- **Monitoring** means regularly checking a model after it goes into use, to confirm it still ranks and predicts well on new applicants (Sections 9–11).

### How the pieces fit

```
Lending Club loans, 2007–2015
  │
  ├─ Define default; remove unresolved loans and leaked variables ....... Section 1
  ├─ Split by issue date: train 2007–2012, test 2013–2015 ............... Section 12
  │
  ├─ Champion: WoE binning, IV screening, logistic regression, scorecard  Sections 2–5
  ├─ Challengers: LightGBM, EBM ......................................... Sections 6–7
  ├─ Explanations and reason codes (SHAP) ............................... Section 8
  │
  ├─ Evaluate on test: calibration, discrimination, stability ........... Sections 9–11
  └─ Translate PD into expected loss and capital ........................ Section 13
```

### Why this document exists

Model risk guidance for banks (in the US, the interagency guidance SR 26-2, which in April 2026 replaced SR 11-7; in the UK, the PRA's SS1/23) expects a model's design and reasoning to be documented so that it can be independently reviewed. This document provides that: every formula used in the code is derived here, so a reviewer can verify the reasoning without reading the code.

---

## Contents

0. [Overview](#overview)
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
| PD | probability of default |
| $n$ | number of loans in the train set |
| $d$ | number of features |
| $i = 1, \dots, n$ | index of a loan (a row of the data) |
| $j = 0, 1, \dots, d$ | index of a feature (a column of the data); $j = 0$ is the intercept, with $x_{i0} = 1$ |
| $x_{ij}$ | value of feature $j$ for loan $i$ |
| $x_i \in \mathbb{R}^d$ | feature vector of loan $i$ (known on the day the loan was approved and issued, i.e. origination) |
| $y_i \in \lbrace 0, 1\rbrace$ | default indicator: $1$ = bad (default), $0$ = good |
| $p_i = P(y_i = 1 \mid x_i)$ | probability of default of loan $i$ |
| $\sigma(z) = 1/(1+e^{-z})$ | logistic (sigmoid) function |
| $\pi = P(y = 1)$ | portfolio default rate (prior) |
| $G$, $B$ | the two classes: $G$ = good loan (Fully Paid, $y = 0$), $B$ = bad loan (Charged Off or Default, $y = 1$) |
| $N_G, N_B$ | total number of good and bad loans in the train set, so $N_G + N_B = n$ |
| bin | a range of values of one feature (e.g. DTI (stands for debt-to-income ratio) from 10 to 18); each feature is split into bins $k = 1, \dots, K$ before computing WoE (Weight of Evidence), section 3 |
| $g_k, b_k$ | number of good loans and number of bad loans whose value of the feature falls in bin $k$, so $\sum_k g_k = N_G$ and $\sum_k b_k = N_B$ |
| $D^G_k, D^B_k$ | share of all good loans, and share of all bad loans, that fall in bin $k$: $D^G_k = g_k / N_G$ and $D^B_k = b_k / N_B$ |

---

## 1. Problem definition

### 1.1 Target

A loan is labelled **bad** ($y = 1$) if its final status is *Charged Off* or *Default*, and **good** ($y = 0$) if it is *Fully Paid*. Loans still *Current*, *In Grace Period* or *Late* have no final outcome yet and are excluded: some of them will default later, so counting them as good would make estimated PDs too low.

The quantity estimated is therefore a **lifetime PD over the loan term** (36 or 60 months), given the applicant's features $x$:

$$
\mathrm{PD}(x) = P\left(Y = 1 \mid X = x\right).
$$

### 1.2 Leakage rule

The model is used to decide whether to lend, so only variables known on the application date may enter $x$. Some LendingClub columns are filled in only after the loan is issued and partly record the outcome itself; using them is called **data leakage**, and they are dropped:

| Column | Meaning | Why it is excluded |
|---|---|---|
| `total_pymnt` | total amount repaid so far | defaulted loans have repaid much less |
| `recoveries` | money collected after a charge-off | non-zero only if the loan has already defaulted |
| `last_pymnt_d` | date of the most recent payment | defaulters stop paying early |
| `out_prncp` | principal still outstanding | reflects the loan's final state |

A model using these would score almost perfectly on historical data but **could not be used for a new applicant**, for whom none of these values exist yet.


---

## 2. Logistic regression

### 2.1 Model

**The outcome.** Each loan $i$ either defaults ($y_i = 1$) or is repaid ($y_i = 0$). Its probability of default is $p_i$, so $P(y_i = 1) = p_i$ and $P(y_i = 0) = 1 - p_i$. A random variable that takes only the values 0 and 1 like this is called a **Bernoulli** random variable.

**Why not model $p_i$ directly?** A probability must stay between 0 and 1, but a linear formula such as $\beta_0 + \beta_1 x_{i1}$ can produce any number, for example $-0.3$ or $1.7$. So instead of $p_i$ itself, logistic regression models a transformed version of it that can take any value.

**Odds and log-odds.** The **odds** of default are $\frac{p_i}{1 - p_i}$: a PD of 20% gives odds of $0.2/0.8 = 0.25$, i.e. 1 default for every 4 repayments. Odds range from 0 to $\infty$. Taking the natural log gives the **log-odds**, which range over all real numbers:

| PD $p_i$ | Odds $\frac{p_i}{1-p_i}$ | Log-odds $\ln\frac{p_i}{1-p_i}$ |
|---|---|---|
| 0.01 | 0.0101 | −4.60 |
| 0.20 | 0.25 | −1.39 |
| 0.50 | 1 | 0 |
| 0.80 | 4 | +1.39 |

**The model.** Logistic regression sets the log-odds equal to a linear combination of the features:

$$
\ln\frac{p_i}{1 - p_i} = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_d x_{id} .
$$

Solving for $p_i$ gives the **sigmoid** function $\sigma$, which turns any number back into a probability between 0 and 1:

$$
p_i = \sigma\left(\beta_0 + \beta_1 x_{i1} + \cdots + \beta_d x_{id}\right), \qquad \sigma(z) = \frac{1}{1 + e^{-z}} .
$$

To keep the notation short, we add a constant feature $x_{i0} = 1$ to every loan, so that the intercept $\beta_0$ is treated like any other coefficient and the model becomes $p_i = \sigma(x_i^\top\beta)$, where $x_i^\top\beta = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_d x_{id}$.

**Why log-odds?** Other transformations also keep probabilities between 0 and 1 (the *probit* model, for example, uses the normal distribution instead of the sigmoid). The log-odds is chosen because it has three practical advantages.

**1. It makes the maths simple.** Writing the Bernoulli probability as an exponential and collecting the terms in $y$:

$$
\begin{aligned}
P(y \mid p) = p^{y}(1-p)^{1-y}
&= \exp\Big\lbrace  y \ln p + (1-y)\ln(1-p) \Big\rbrace  \cr
&= \exp\Big\lbrace  y \ln p - y\ln(1-p) + \ln(1-p) \Big\rbrace  \cr
&= \exp\Big\lbrace  y \thinspace \underbrace{\ln\tfrac{p}{1-p}}_{\text{log-odds}} + \ln(1-p) \Big\rbrace .
\end{aligned}
$$

The outcome $y$ is multiplied by exactly the log-odds, so modelling the log-odds as linear in the features makes the gradient reduce to the simple prediction error $y_i - p_i$ (Section 2.3).

**2. The coefficients are easy to interpret.** Increasing feature $j$ by one unit adds $\beta_j$ to the log-odds, which multiplies the odds of default by $e^{\beta_j}$. For example, if $\beta_j = 0.10$, then $e^{0.10} \approx 1.105$, so each one-unit increase multiplies the odds of default by 1.105, i.e. increases them by about 10.5%.

**3. It matches the scorecard.** Weight of Evidence (Section 3) is also measured on the log-odds scale, so WoE values can be fed straight into the model, and scorecard points are a linear function of the log-odds (Section 5).

### 2.2 Log-likelihood

Assuming independent loans, the likelihood and log-likelihood are

$$
L(\beta) = \prod_{i=1}^{n} p_i^{\thinspace y_i}(1-p_i)^{1-y_i},
\qquad
\ell(\beta) = \sum_{i=1}^{n}\Big[ y_i \ln p_i + (1-y_i)\ln(1-p_i) \Big].
$$

Statistics maximises the log-likelihood; machine learning libraries instead minimise a loss. The two are equivalent: flipping the sign and averaging over loans gives the **log-loss**, also called **binary cross-entropy**, written $J(\beta)$:

$$
J(\beta) = -\frac{1}{n}\thinspace \ell(\beta) = -\frac{1}{n}\sum_{i=1}^{n}\Big[ y_i \ln p_i + (1-y_i)\ln(1-p_i) \Big].
$$

Multiplying by $-\frac{1}{n}$ turns a maximum into a minimum without moving it, so the $\beta$ that maximises $\ell$ is the same $\beta$ that minimises $J$. Because $y_i$ is either 0 or 1, only one term in the bracket applies to each loan. A loan that **defaulted** ($y_i = 1$) adds a penalty of $-\ln p_i$; a loan that was **repaid** ($y_i = 0$) adds $-\ln(1 - p_i)$. Either way, the penalty is minus the log of the probability the model gave to what actually happened:

| PD given $p_i$ | Penalty if defaulted ($y_i = 1$) | Penalty if repaid ($y_i = 0$) |
|---|---|---|
| 0.01 | 4.61 | 0.01 |
| 0.10 | 2.30 | 0.11 |
| 0.50 | 0.69 | 0.69 |
| 0.90 | 0.11 | 2.30 |

Correct predictions cost little. Wrong predictions cost more the more confident they are, without limit. Minimising the average penalty therefore makes avoiding confident mistakes, such as calling a loan safe when it defaults, the model's highest priority.

LightGBM (Section 6) minimises this same loss, so the two models are optimising the same objective and their log-loss on the test set can be compared directly.

### 2.3 Gradient

**Step 1: derivative of the sigmoid.** With $\sigma(z) = (1+e^{-z})^{-1}$,

$$
\sigma'(z) = \frac{e^{-z}}{(1+e^{-z})^2}
= \frac{1}{1+e^{-z}}\cdot\frac{e^{-z}}{1+e^{-z}}
= \sigma(z)\thinspace \big(1-\sigma(z)\big).
$$

**Step 2: derivative of $p_i$.** Since $p_i = \sigma(x_i^\top\beta)$ and $\partial (x_i^\top\beta)/\partial\beta = x_i$, the chain rule gives

$$
\frac{\partial p_i}{\partial \beta} = p_i(1-p_i)\thinspace x_i .
$$

**Step 3: derivative of the log-likelihood.** Using $\frac{d}{dp}\ln p = \frac{1}{p}$ and $\frac{d}{dp}\ln(1-p) = -\frac{1}{1-p}$, and then Step 2:

$$
\frac{\partial \ell}{\partial \beta}
= \sum_{i=1}^{n}\left[\frac{y_i}{p_i} - \frac{1-y_i}{1-p_i}\right] p_i(1-p_i)\thinspace x_i
= \sum_{i=1}^{n}\Big[ y_i(1-p_i) - (1-y_i)p_i \Big] x_i
= \sum_{i=1}^{n} (y_i - p_i)\thinspace x_i .
$$

The $p_i(1-p_i)$ from the sigmoid cancels the denominators, which is the simplification the canonical link buys (Section 2.1).

**Step 4: derivative of the log-loss.** Since $J(\beta) = -\frac{1}{n}\ell(\beta)$,

$$
\frac{\partial J}{\partial \beta} = -\frac{1}{n}\frac{\partial \ell}{\partial \beta} = \frac{1}{n}\sum_{i=1}^{n} (p_i - y_i)\thinspace x_i .
$$

Each loan contributes its **prediction error** $p_i - y_i$, weighted by its features. A defaulted loan ($y_i = 1$) given a low PD has a large negative error, which pushes $\beta$ towards giving it a higher PD; a repaid loan given a high PD is pushed the other way. The same quantity $p_i - y_i$ is the gradient LightGBM uses in Section 6.2.

**Matrix form.** Stack the loans into the **feature matrix** $X \in \mathbb{R}^{n \times (d+1)}$: one row per loan, one column per feature, plus a first column of ones for the intercept, so that row $i$ is $x_i^\top$ and $X\beta$ is the vector of log-odds for all loans. With $y$ and $p$ the vectors of outcomes and predicted PDs,

$$
X =
\begin{pmatrix}
1 & x_{11} & \cdots & x_{1d} \cr
\vdots & \vdots & \ddots & \vdots \cr
1 & x_{n1} & \cdots & x_{nd}
\end{pmatrix},
\qquad
\nabla \ell(\beta) = X^\top (y - p),
\qquad
\nabla J(\beta) = \frac{1}{n}\thinspace X^\top (p - y).
$$

Multiplying by $X^\top$ is the same as the sum $\sum_i (p_i - y_i)x_i$, written compactly.

### 2.4 Score equations

**What they look like.** The minimum of $J$ (equivalently the maximum of $\ell$) is where the gradient is zero:

$$
X^\top (y - p) = 0 .
$$

These are the **score equations**: $d + 1$ equations, one per coefficient $\beta_j$. Row $j$ of $X^\top$ is feature $j$ across all loans $i = 1, \dots, n$, so writing out each PD in full,

$$
p_i = \sigma\left(x_i^\top\beta\right) = \frac{1}{1 + e^{-\left(\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_d x_{id}\right)}},
$$

equation $j$ is

$$
\sum_{i=1}^{n}\left[\thinspace  y_i - \frac{1}{1 + e^{-\left(\beta_0 + \beta_1 x_{i1} + \cdots + \beta_d x_{id}\right)}} \right] x_{ij} = 0, \qquad j = 0, 1, \dots, d .
$$

Every $\beta_j$ sits inside an exponential, in every equation. Unlike linear regression, where $X^\top X\beta = X^\top y$ can be solved directly, $\beta$ cannot be isolated algebraically, so the equations are solved iteratively (Section 2.6).

**What they mean.** Rearranged, equation $j$ says that a *predicted sum* equals an *observed sum*:

$$
\underbrace{\sum_{i=1}^{n} p_i\thinspace x_{ij}}_{\text{predicted sum}} \thickspace =\thickspace  \underbrace{\sum_{i=1}^{n} y_i\thinspace x_{ij}}_{\text{observed sum}}, \qquad j = 0, 1, \dots, d .
$$

**Intercept ($j = 0$, $x_{i0} = 1$).** The equation becomes $\sum_i p_i = \sum_i y_i$: the predicted number of defaults equals the actual number. Dividing by $n$,

$$
\boxed{\thickspace \frac{1}{n}\sum_{i=1}^{n} p_i \thickspace =\thickspace  \frac{1}{n}\sum_{i=1}^{n} y_i\thickspace }
\qquad\text{average predicted PD} = \text{observed default rate.}
$$

This holds exactly at the fitted $\hat\beta$ for any logistic regression with an intercept, so on the train set the model's average PD always reproduces the observed default rate. It is a useful check on the code: if the two differ, the model has not converged or has no intercept.

**Each feature ($j = 1, \dots, d$).** Because $y_i = 1$ only for loans that defaulted, the observed sum is the total of feature $j$ over the defaulted loans; it is fixed by the data. The predicted sum weights each loan by its PD instead, so it is the total the model expects among defaulters, and it changes with $\beta$ through $p_i$.

**Effect of $\beta_j$ on the predicted sum.** Changing $\beta_j$ alone, with the other coefficients fixed, and using $\partial p_i / \partial \beta_j = p_i(1-p_i)x_{ij}$ from Step 2:

$$
\frac{\partial}{\partial \beta_j}\sum_{i} p_i\thinspace x_{ij}
= \sum_{i} x_{ij}\thinspace \frac{\partial p_i}{\partial \beta_j}
= \sum_{i} p_i(1-p_i)\thinspace x_{ij}^2 \thickspace >\thickspace  0 .
$$

Every term is positive, since $0 < p_i < 1$ and $x_{ij}^2 \ge 0$, so the predicted sum always increases with $\beta_j$, regardless of the sign of the feature values.

**Under- and over-prediction.** If the predicted sum is below the observed sum, the model gives too little default risk to loans with high values of feature $j$, so $\beta_j$ is increased; if it is above, $\beta_j$ is decreased. At the fitted $\hat\beta$, the predicted and observed sums are equal for every feature at once, so the model attributes exactly as much default risk to each feature as the train data shows.

### 2.5 Hessian and concavity

**Deriving the Hessian.** The Hessian is the matrix of second derivatives, $\nabla^2 \ell(\beta)$, with entry $(j, k)$ equal to $\partial^2 \ell / \partial \beta_j \partial \beta_k$. Start from the gradient in Section 2.3,

$$
\nabla \ell(\beta) = \sum_{i=1}^{n} (y_i - p_i)\thinspace x_i ,
$$

and differentiate again with respect to $\beta$. The outcomes $y_i$ and features $x_i$ are fixed data, so only $p_i$ changes, and by Step 2 of Section 2.3, $\partial p_i / \partial \beta^\top = p_i(1-p_i)x_i^\top$. Hence

$$
\nabla^2 \ell(\beta) = -\sum_{i=1}^{n} p_i(1-p_i)\thinspace x_i\thinspace x_i^\top .
$$

Each loan contributes the $(d+1) \times (d+1)$ matrix $x_i x_i^\top$ (entry $(j,k)$ equal to $x_{ij}x_{ik}$), scaled by the weight $w_i = p_i(1-p_i)$.

**The weight matrix $W$.** Collecting the weights into an $n \times n$ diagonal matrix, one entry per loan,

$$
W = \begin{pmatrix}
p_1(1-p_1) & 0 & \cdots & 0 \cr
0 & p_2(1-p_2) & \cdots & 0 \cr
\vdots & \vdots & \ddots & \vdots \cr
0 & 0 & \cdots & p_n(1-p_n)
\end{pmatrix},
$$

the sum becomes a single matrix product:

$$
\nabla^2 \ell(\beta) = -X^\top W X .
$$

where $X$ is the feature matrix from Section 2.3, with one row per loan and a first column of ones for the intercept:

$$
X =
\begin{pmatrix}
1 & x_{11} & x_{12} & \cdots & x_{1d} \cr
1 & x_{21} & x_{22} & \cdots & x_{2d} \cr
\vdots & \vdots & \vdots & \ddots & \vdots \cr
1 & x_{n1} & x_{n2} & \cdots & x_{nd}
\end{pmatrix}
\in \mathbb{R}^{n \times (d+1)} .
$$

The sizes match: $X^\top$ is $(d+1) \times n$, $W$ is $n \times n$ and $X$ is $n \times (d+1)$, so $X^\top W X$ is $(d+1) \times (d+1)$, one row and one column per coefficient $\beta_j$.

Each weight $w_i = p_i(1-p_i)$ is the variance of the Bernoulli outcome $y_i$. It is largest, $0.25$, when $p_i = 0.5$, and close to zero when the model is nearly certain ($p_i$ near 0 or 1), so the loans the model is least sure about carry the most weight. $W$ reappears in the fitting algorithm (Section 2.6) and the standard errors (Section 2.7).

**Concavity.** A function is concave when it curves downwards in every direction. For a function of a vector, the curvature in direction $v$ is $v^\top \nabla^2 \ell v$, and here

$$
v^\top \nabla^2 \ell(\beta)\thinspace  v
= -\sum_{i=1}^{n} p_i(1-p_i)\left(x_i^\top v\right)^2 \thickspace \le\thickspace  0
\qquad \text{for every vector } v ,
$$

because each term is a positive weight times a square. The Hessian is therefore **negative semi-definite**, and $\ell$ is **concave**: it has no local maxima other than the global one, so any point where the gradient is zero is the best possible $\beta$.

**Uniqueness.** If no feature is an exact linear combination of the others ($X$ has *full column rank*), then $Xv \neq 0$ for every $v \neq 0$, so the curvature is strictly negative in every direction and $\ell$ is **strictly concave**. Equivalently, the log-loss $J$, with Hessian $\frac{1}{n}X^\top W X$, is **strictly convex**: a bowl with a single lowest point. If a minimum exists, it is therefore **unique**, and refitting on the same data always gives the same $\hat\beta$.

**When no maximum exists: separation.** Uniqueness says there is *at most* one maximum, not that one exists. A maximum fails to exist when the features **perfectly separate** the two outcomes: there is a vector $v$ with $x_i^\top v > 0$ for every defaulted loan and $x_i^\top v < 0$ for every repaid loan, i.e. a hyperplane with all defaults on one side and all repayments on the other. (This is a relationship between the features and the *outcome*, unlike collinearity, which is a relationship among the features.) For example, if every loan with DTI (debt-to-income ratio) above 30 defaulted and every other loan was repaid, $v = (-30, 1)$ separates them. Setting $\beta = cv$ and letting $c \to \infty$ sends every defaulted loan's PD to 1 and every repaid loan's PD to 0, so $\ell(\beta)$ rises towards its upper bound of 0 without ever reaching it: the fit always improves by making the coefficients larger, and $\hat\beta$ diverges. A milder form, *quasi-complete separation*, occurs when, for example, one category of a feature contains no defaults at all: its score equation requires the PDs in that category to sum to 0, which no finite $\beta$ achieves, so the coefficient $\beta_j$ on that category's indicator diverges to $-\infty$.

In credit data, separation is almost always a symptom of **leakage** (Section 1.2), since genuine application data never predicts default perfectly. It is prevented here in three ways:

1. **Leaked variables are removed**, which rules out complete separation.
2. **WoE (Weight of Evidence) binning** rules out the quasi-complete form. Each feature enters as a single WoE column with one shared coefficient $\beta_j$, rather than one indicator per category, so a category with no defaults has no coefficient of its own to diverge. Bins are also merged until each contains both goods and bads, and the 0.5 adjustment of Section 3.1 keeps every WoE finite (without it, a bin with no bads would have $\mathrm{WoE}_k = +\infty$).
3. **L2 regularisation**, where used, adds a penalty $\lambda \lVert\beta\rVert^2$ that grows as coefficients grow, so the penalised objective always has a finite optimum.

### 2.6 Estimation: Newton–Raphson and IRLS

Newton's method updates

$$
\beta^{(t+1)} = \beta^{(t)} - \left[\nabla^2\ell\right]^{-1}\nabla\ell
= \beta^{(t)} + \left(X^\top W X\right)^{-1} X^\top (y - p).
$$

Defining the working response $z = X\beta^{(t)} + W^{-1}(y - p)$, the update becomes

$$
\beta^{(t+1)} = \left(X^\top W X\right)^{-1} X^\top W z,
$$

a weighted least squares problem. Iterating is **Iteratively Reweighted Least Squares (IRLS)**, the algorithm used by `statsmodels`.

### 2.7 Inference

Under standard regularity conditions, $\hat\beta$ is asymptotically normal with covariance equal to the inverse Fisher information:

$$
\widehat{\operatorname{Var}}(\hat\beta) = \left(X^\top \hat W X\right)^{-1},
\qquad
z_j = \frac{\hat\beta_j}{\widehat{\operatorname{se}}(\hat\beta_j)} \thickspace \sim\thickspace  N(0,1) \text{ under } H_0:\beta_j = 0 .
$$

These Wald statistics are reported for every scorecard coefficient. The coefficient $\beta_j$ has a direct reading: a one-unit increase in $x_j$ multiplies the odds of default by $e^{\beta_j}$.

---

## 3. Weight of Evidence (WoE)

### 3.1 Definition

Each feature is split into bins $k = 1,\dots,K$ (quantile bins for continuous variables, merged categories for categorical ones). Let

$$
D^{G}_k = \frac{g_k}{N_G}, \qquad D^{B}_k = \frac{b_k}{N_B}
$$

be the share of all goods and all bads that fall in bin $k$. The Weight of Evidence of bin $k$ is

$$
\mathrm{WoE}_k = \ln\frac{D^{G}_k}{D^{B}_k}.
$$

Positive WoE means the bin holds relatively more goods (lower risk); negative WoE means more bads.

To avoid $\ln 0$ in sparse bins, a 0.5 smoothing adjustment is used:

$$
\mathrm{WoE}_k = \ln\frac{(g_k + 0.5)/N_G}{(b_k + 0.5)/N_B}.
$$

### 3.2 Why WoE: the Bayesian derivation

By Bayes' theorem, for a loan whose feature value falls in bin $k$:

$$
\frac{P(G \mid k)}{P(B \mid k)}
= \frac{P(k \mid G)}{P(k \mid B)} \cdot \frac{P(G)}{P(B)} .
$$

The empirical estimates of $P(k \mid G)$ and $P(k \mid B)$ are exactly $D^G_k$ and $D^B_k$. Taking logs:

$$
\underbrace{\ln\frac{P(G \mid k)}{P(B \mid k)}}_{\text{posterior log-odds}}
= \underbrace{\mathrm{WoE}_k}_{\text{evidence from the bin}}
+ \underbrace{\ln\frac{N_G}{N_B}}_{\text{prior log-odds}} .
$$

So **WoE is the amount by which observing bin $k$ shifts the log-odds of being good**. It puts every feature on the same log-odds scale as the logistic regression itself.

### 3.3 Link to logistic regression

If the $d$ features were conditionally independent given the class (the naive Bayes assumption), the same argument applied to all of them gives

$$
\ln\frac{P(G \mid x)}{P(B \mid x)} = \ln\frac{N_G}{N_B} + \sum_{j=1}^{d}\mathrm{WoE}_j(x_j).
$$

Since this project models the log-odds of **bad**, $\ln\frac{p}{1-p} = -\ln\frac{P(G\mid x)}{P(B\mid x)}$, and

$$
\ln\frac{p}{1-p} = \beta_0 + \sum_{j=1}^{d} \beta_j\thinspace \mathrm{WoE}_j(x_j)
$$

recovers naive Bayes with $\beta_j = -1$. Fitting the $\beta_j$ by maximum likelihood corrects for correlation between features. A fitted coefficient that is **positive** (the wrong sign under this convention) signals multicollinearity, and that feature is reviewed or removed.

### 3.4 Properties that matter for credit risk

- **Monotonicity:** bins are merged until WoE is monotonic in the underlying variable, so the score moves in a direction a credit officer can justify (for example, higher DTI never improves the score).
- **Non-linearity:** a linear model on WoE captures any monotonic, non-linear relationship between a raw variable and the log-odds.
- **Missing values:** missing gets its own bin and WoE, so missingness is modelled rather than imputed.
- **Outliers:** extreme values sit in the end bins and cannot dominate the fit.

---

## 4. Information Value (IV)

### 4.1 Definition

$$
\mathrm{IV} = \sum_{k=1}^{K}\left(D^{G}_k - D^{B}_k\right)\mathrm{WoE}_k
= \sum_{k=1}^{K}\left(D^{G}_k - D^{B}_k\right)\ln\frac{D^{G}_k}{D^{B}_k}.
$$

### 4.2 IV is a symmetric KL divergence

The Kullback–Leibler divergence between the goods' distribution $D^G$ and the bads' distribution $D^B$ over the bins is

$$
\mathrm{KL}\left(D^G \thinspace \Vert\thinspace  D^B\right) = \sum_k D^G_k \ln\frac{D^G_k}{D^B_k},
\qquad
\mathrm{KL}\left(D^B \thinspace \Vert\thinspace  D^G\right) = \sum_k D^B_k \ln\frac{D^B_k}{D^G_k}.
$$

Adding them:

$$
\mathrm{KL}\left(D^G \Vert D^B\right) + \mathrm{KL}\left(D^B \Vert D^G\right)
= \sum_k D^G_k \ln\frac{D^G_k}{D^B_k} - \sum_k D^B_k \ln\frac{D^G_k}{D^B_k}
= \sum_k \left(D^G_k - D^B_k\right)\ln\frac{D^G_k}{D^B_k} = \mathrm{IV}.
$$

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

$$
\text{Score} = \text{Offset} + \text{Factor} \cdot \ln(\text{odds}),
\qquad \text{odds} = \frac{1-p}{p}.
$$

Two business choices fix the constants:

1. A reference score $S_0$ corresponds to reference odds $o_0$.
2. Every **PDO** ("points to double the odds") points, the odds double.

From condition 2:

$$
\text{Factor}\cdot\ln(2\thinspace o) - \text{Factor}\cdot\ln(o) = \mathrm{PDO}
\thickspace \thickspace \Longrightarrow\thickspace \thickspace 
\text{Factor} = \frac{\mathrm{PDO}}{\ln 2}.
$$

From condition 1:

$$
\text{Offset} = S_0 - \text{Factor}\cdot\ln(o_0).
$$

**Worked example** (the convention used in this project): $S_0 = 600$ at odds $o_0 = 50{:}1$ with $\mathrm{PDO} = 20$ gives

$$
\text{Factor} = \frac{20}{\ln 2} \approx 28.854,
\qquad
\text{Offset} = 600 - 28.854 \cdot \ln 50 \approx 487.12 .
$$

### 5.2 Points per attribute

With the WoE logistic model, $\ln(\text{odds}) = -\left(\beta_0 + \sum_j \beta_j \mathrm{WoE}_j\right)$. Substituting:

$$
\text{Score} = \text{Offset} - \text{Factor}\left(\beta_0 + \sum_{j=1}^{d}\beta_j\thinspace \mathrm{WoE}_j\right).
$$

Spreading the intercept and offset evenly over the $d$ features gives the points for bin $k$ of feature $j$:

$$
\text{Points}_{j,k} = -\left(\beta_j\thinspace \mathrm{WoE}_{j,k} + \frac{\beta_0}{d}\right)\text{Factor} + \frac{\text{Offset}}{d}.
$$

A loan's score is the sum of the points of the bins it falls in, which is what makes a scorecard auditable line by line.

---

## 6. Gradient boosting with LightGBM

### 6.1 Additive model

The challenger model builds the log-odds as a sum of $M$ regression trees:

$$
F_M(x) = F_0 + \eta \sum_{m=1}^{M} f_m(x), \qquad p(x) = \sigma\left(F_M(x)\right),
$$

where $\eta \in (0,1]$ is the learning rate and $F_0 = \ln\frac{\bar y}{1-\bar y}$ is the log-odds of the base default rate.

### 6.2 Loss, gradient and Hessian

For one observation with current raw score $F$ and $p = \sigma(F)$, the log-loss is

$$
L(y, F) = -\Big[y\ln\sigma(F) + (1-y)\ln\left(1-\sigma(F)\right)\Big].
$$

Using the result from Section 2.3, its first and second derivatives with respect to $F$ are

$$
g = \frac{\partial L}{\partial F} = p - y,
\qquad
h = \frac{\partial^2 L}{\partial F^2} = p(1-p).
$$

### 6.3 Optimal leaf values

At round $m$, a second-order Taylor expansion of the total loss around the current prediction, with L2 penalty $\lambda$ on leaf values, gives

$$
\mathcal{L}^{(m)} \approx \sum_{i=1}^{n}\left[g_i f(x_i) + \tfrac{1}{2} h_i f(x_i)^2\right] + \tfrac{1}{2}\lambda\sum_{\ell=1}^{T} w_\ell^2 ,
$$

where the tree assigns value $w_\ell$ to every observation in leaf $\ell$. Grouping by leaf, with $G_\ell = \sum_{i \in \ell} g_i$ and $H_\ell = \sum_{i \in \ell} h_i$:

$$
\mathcal{L}^{(m)} \approx \sum_{\ell=1}^{T}\left[G_\ell w_\ell + \tfrac{1}{2}\left(H_\ell + \lambda\right) w_\ell^2\right].
$$

Each bracket is a quadratic in $w_\ell$, minimised at

$$
w_\ell^{*} = -\frac{G_\ell}{H_\ell + \lambda},
\qquad
\mathcal{L}^{*} = -\frac{1}{2}\sum_{\ell=1}^{T}\frac{G_\ell^2}{H_\ell + \lambda}.
$$

So each leaf moves the log-odds by a damped Newton step.

### 6.4 Split gain

Splitting a node into left and right children reduces the loss by

$$
\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma ,
$$

where $\gamma$ is a minimum-gain penalty (`min_gain_to_split`). LightGBM evaluates this on histogram bins and grows trees **leaf-wise** (always splitting the leaf with the largest gain), which is what makes it fast on several hundred thousand Lending Club loans.

### 6.5 Monotonic constraints

For a feature $x_j$ with an economic sign (higher income should never raise PD), a constraint is imposed:

$$
x_j \le x_j' \thickspace \Longrightarrow\thickspace  F(x) \ge F(x') \quad \text{(decreasing PD in } x_j\text{), all other features fixed.}
$$

LightGBM enforces this by rejecting any split on $x_j$ whose left and right leaf values violate the required order, and by passing bounds down to descendant nodes so that no later split can break it. Every tree is then monotonic in $x_j$, and so is their sum. This gives up a little accuracy in exchange for a model whose behaviour matches the scorecard's WoE signs, which is a common model risk requirement.

---

## 7. Explainable Boosting Machine (EBM)

The EBM is a **generalised additive model with pairwise interactions** (GA²M):

$$
\ln\frac{p}{1-p} = \beta_0 + \sum_{j=1}^{d} f_j(x_j) + \sum_{(j,k) \in \mathcal{I}} f_{jk}(x_j, x_k).
$$

Each shape function $f_j$ is learned by **cyclic gradient boosting**: in each round, a very shallow tree is fit to the current residual gradients using *one feature only*, with a very small learning rate, cycling through all features many times. Because each feature gets its turn in sequence and the steps are tiny, the order in which features are visited does not matter, and collinear features share credit rather than one absorbing it all. The result is bagged over several runs for stability.

A small set $\mathcal{I}$ of pairwise interactions is then added, selected by the FAST ranking procedure (Lou et al., 2013).

The EBM sits between the two other models: as transparent as the scorecard (each $f_j$ can be plotted and read directly), with accuracy close to LightGBM.

---

## 8. SHAP values and reason codes

### 8.1 Shapley values

For a model $f$ and a loan $x$, the contribution of feature $j$ is its Shapley value from cooperative game theory:

$$
\phi_j(x) = \sum_{S \subseteq \lbrace 1,\dots,d\rbrace \setminus\lbrace j\rbrace }
\frac{|S|!\thinspace (d - |S| - 1)!}{d!}\Big[v(S \cup \lbrace j\rbrace ) - v(S)\Big],
$$

where $v(S) = \mathbb{E}\left[f(X) \mid X_S = x_S\right]$ is the expected model output when only the features in $S$ are known. The weight is the probability that, in a random ordering of features, exactly the set $S$ comes before $j$, so $\phi_j$ is the **average marginal contribution of feature $j$ over all orderings**.

### 8.2 Axioms

Shapley values are the unique attribution satisfying:

- **Efficiency (local accuracy):** the contributions add up exactly to the prediction (equation below).
- **Symmetry:** two features that always contribute equally get equal credit.
- **Dummy:** a feature that never changes the output gets $\phi_j = 0$.
- **Additivity:** for an ensemble $f = f_1 + f_2$, $\phi_j(f) = \phi_j(f_1) + \phi_j(f_2)$. This is what lets SHAP explain a sum of trees tree by tree.

The efficiency property, written out, is

$$
f(x) = \phi_0 + \sum_{j=1}^{d}\phi_j(x), \qquad \phi_0 = \mathbb{E}[f(X)].
$$

### 8.3 Linear special case

For a linear model $f(x) = \beta_0 + \sum_j \beta_j x_j$ with independent features, $v(S) = \beta_0 + \sum_{k \in S}\beta_k x_k + \sum_{k \notin S}\beta_k\mathbb{E}[X_k]$, so every marginal contribution of $j$ equals $\beta_j(x_j - \mathbb{E}[X_j])$ and

$$
\phi_j(x) = \beta_j\left(x_j - \mathbb{E}[X_j]\right).
$$

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
- **Hosmer–Lemeshow test:** a formal test of calibration, defined below.

With $G$ groups, $O_g$ observed defaults, $E_g = \sum_{i \in g} p_i$ expected defaults and $n_g$ loans in group $g$, the Hosmer–Lemeshow statistic is

$$
\mathrm{HL} = \sum_{g=1}^{G}\frac{(O_g - E_g)^2}{E_g\left(1 - E_g/n_g\right)} \thickspace \sim\thickspace  \chi^2_{G-2} .
$$

### 9.2 Prior correction

If the model is trained on a sample whose default rate $\pi_s$ differs from the portfolio rate $\pi$ (for example after undersampling goods), then by Bayes' theorem only the prior odds change:

$$
\frac{p_s}{1-p_s} = \frac{p}{1-p}\cdot\frac{\pi_s/(1-\pi_s)}{\pi/(1-\pi)} .
$$

Rearranging gives the correction applied to the intercept:

$$
\ln\frac{p}{1-p} = \ln\frac{p_s}{1-p_s} - \ln\frac{\pi_s}{1-\pi_s} + \ln\frac{\pi}{1-\pi}.
$$

### 9.3 Recalibration

For LightGBM, whose raw probabilities can be miscalibrated, a **Platt scaling** step fits $p = \sigma(a + bF(x))$ on held-out validation folds of the train set (never on the test set), which is itself a one-feature logistic regression (Section 2). Isotonic regression is used as a non-parametric check.

---

## 10. Discrimination metrics: ROC, AUC, Gini, KS

Let $s(x)$ be the predicted PD (higher means riskier) and let $F_B$, $F_G$ be the cumulative distribution functions of $s$ among bads and goods.

### 10.1 ROC and AUC

At threshold $t$, the true positive rate is $\mathrm{TPR}(t) = 1 - F_B(t)$ and the false positive rate is $\mathrm{FPR}(t) = 1 - F_G(t)$. The ROC curve plots TPR against FPR as $t$ varies, and AUC is the area under it.

**AUC as a probability.** Writing $f_G$ for the density of $s$ among goods,

$$
\mathrm{AUC} = \int_0^1 \mathrm{TPR}\thickspace  d\thinspace \mathrm{FPR}
= \int_{-\infty}^{\infty}\left(1 - F_B(t)\right) f_G(t)\thinspace dt
= P\left(S_B > S_G\right),
$$

where $S_B$ and $S_G$ are the scores of a randomly chosen bad and good. (With ties, $\mathrm{AUC} = P(S_B > S_G) + \tfrac{1}{2}P(S_B = S_G)$.) This is the Mann–Whitney $U$ statistic divided by $N_B N_G$, which is how it is computed: by ranks, in $O(n\log n)$.

### 10.2 Gini

The Gini coefficient (accuracy ratio) is Somers' $D$ between score and outcome:

$$
\mathrm{Gini} = P(S_B > S_G) - P(S_B < S_G).
$$

Since $P(S_B > S_G) + P(S_B < S_G) + P(S_B = S_G) = 1$,

$$
\mathrm{Gini} = 2\left[P(S_B > S_G) + \tfrac{1}{2}P(S_B = S_G)\right] - 1 = 2\thinspace \mathrm{AUC} - 1 .
$$

A random model has Gini 0 and a perfect one has Gini 1. Geometrically, it is the area between the model's CAP (Lorenz) curve and the random line, divided by the same area for a perfect model.

### 10.3 Kolmogorov–Smirnov (KS)

$$
\mathrm{KS} = \max_t \left| F_G(t) - F_B(t) \right| .
$$

KS is the largest vertical gap between the score distributions of goods and bads, equivalently $\max_t(\mathrm{TPR} - \mathrm{FPR})$, Youden's $J$. The threshold where it occurs is a natural cut-off candidate.

---

## 11. Stability: PSI and CSI

### 11.1 Population Stability Index

Let $E_k$ be the share of the train set in score band $k$ and $A_k$ the share in a later sample (the test set, or new applicants during monitoring):

$$
\mathrm{PSI} = \sum_{k=1}^{K}\left(A_k - E_k\right)\ln\frac{A_k}{E_k}.
$$

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

$$
\mathrm{EL} = \mathrm{PD} \times \mathrm{LGD} \times \mathrm{EAD},
$$

where LGD is the fraction of exposure lost if default occurs and EAD is the exposure at default. This assumes PD is independent of LGD and EAD. Expected loss drives pricing and provisions.

### 13.2 Basel IRB capital (illustration)

Under the Basel internal ratings-based (IRB) approach, capital covers **unexpected** loss. It comes from the asymptotic single risk factor (Vasicek) model, which gives the default rate in a 1-in-1000 downturn. For **other retail exposures**, the category that includes unsecured consumer loans such as Lending Club's, the asset correlation is

$$
R = 0.03\cdot\frac{1 - e^{-35\thinspace \mathrm{PD}}}{1 - e^{-35}} + 0.16\cdot\left[1 - \frac{1 - e^{-35\thinspace \mathrm{PD}}}{1 - e^{-35}}\right],
$$

and the capital requirement per unit of exposure is

$$
K = \mathrm{LGD}\cdot\left[\Phi\left(\frac{\Phi^{-1}(\mathrm{PD}) + \sqrt{R}\thickspace \Phi^{-1}(0.999)}{\sqrt{1-R}}\right) - \mathrm{PD}\right],
\qquad
\mathrm{RWA} = 12.5 \cdot K \cdot \mathrm{EAD},
$$

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
11. Board of Governors of the Federal Reserve System, Office of the Comptroller of the Currency and Federal Deposit Insurance Corporation (2026). *SR 26-2: Revised Guidance on Model Risk Management* (supersedes SR 11-7, 2011).
12. Prudential Regulation Authority (2023). *SS1/23: Model Risk Management Principles for Banks*.