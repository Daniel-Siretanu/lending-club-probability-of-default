# Methodology

Project: Probability of Default (PD) model on LendingClub loans

Author: Daniel Siretanu

This document explains how the model is built.

---

## 1. What this project does

When someone applies for a loan, the lender has to decide whether to lend and at what interest rate. The key number behind that decision is the probability of default (PD), the chance that the borrower will not repay.

This project builds a model that estimates the PD of each loan, using only what was known on the day the loan was approved. A good PD model does two things. It ranks borrowers correctly, which means riskier borrowers come above safer ones (*discrimination*) and its PDs are right as numbers, so if it says 5%, about 5 in 100 such loans default (*calibration*).

### 1.1 How the work is organised

| Notebook | What it does | Sections |
|---|---|---|
| [1_explore_data](../notebooks/1_explore_data.ipynb) | picks the loans and columns, builds the target and features, splits train and test | 2–6 |
| [2_binning_woe](../notebooks/2_binning_woe.ipynb) | finds the bins and WoE values | 7 |
| [3_model](../notebooks/3_model.ipynb) | chooses the features and fits the models | 8 |
| [4_test](../notebooks/4_test.ipynb) | tests the models and sets the lending decision | 9–10 |
| [5_scorecard](../notebooks/5_scorecard.ipynb) | writes the models as scorecards and explains decisions | 11 |
| [6_challenger](../notebooks/6_challenger.ipynb) | compares the scorecard with LightGBM | 12 |

---

## 2. The data

Sections 2 to 6 are done in [1_explore_data](../notebooks/1_explore_data.ipynb), in this order.

The data is LendingClub's loan file `accepted_2007_to_2018Q4.csv.gz` from Kaggle: 2,260,701 rows and 151 columns, one row per loan. A few rows at the end are totals, not loans. They have no issue date, so they drop out when the dates are read.

### 2.1 Two loan terms

Loans last 36 or 60 months. A 60-month loan has more time to default, so its PD is naturally higher. We model the two terms separately, and every step below is done for each one.

### 2.2 Which years we can use

Statuses are recorded in December 2018. We only know how a loan ended if it was due to finish by then.

| Term | Offered from | Issue years used | Loans |
|---|---|---|---|
| 36 months | 2007 | 2007–2015 | 621,169 |
| 60 months | 2010 | 2010–2013 | 55,290 |

---

## 3. Which columns we use

We load only a few of the 151 columns. A column can be a feature only if:

1. **It was known when the loan was approved.** Payment columns, for example, are filled in later and partly record the outcome. Using them would be *data leakage*.
2. **It was recorded in every year.** If LendingClub started recording it later, an empty value just means "old loan". We check this in Section 5.2.
3. **It holds a usable value.** No free text, links or ids.
4. **It isn't LendingClub's own risk rating.** `grade`, `sub_grade`, `int_rate` and `installment` are left out, so the model judges borrowers itself instead of copying LendingClub.
5. **It isn't location.** `zip_code` and `addr_state` are left out because location is closely linked to race and national origin, which fair-lending law protects.

### 3.1 Loan columns

Used to pick loans, build the target and split the data. Never features.

| Column | Meaning | Used for |
|---|---|---|
| `id` | loan number | identifying each loan |
| `issue_d` | month of issue | choosing years (2.2), train/test split (6) |
| `loan_status` | status in December 2018 | the target (4) |
| `term` | 36 or 60 months | the two groups (2.1) |

### 3.2 Candidate features

These 19 columns pass checks 1, 3, 4 and 5. Check 2 removes two of them in Section 5.2.

| Group | Column | Meaning |
|---|---|---|
| The loan | `loan_amnt` | amount requested |
| | `purpose` | what the loan is for |
| The borrower | `annual_inc` | yearly income |
| | `verification_status` | whether LendingClub checked the income |
| | `emp_length` | years in current job |
| | `home_ownership` | rents, owns or has a mortgage |
| | `dti` | monthly debt payments (excluding mortgage and this loan) / monthly income |
| Credit history | `fico_range_low`, `fico_range_high` | FICO score range |
| | `earliest_cr_line` | date of first credit account |
| | `delinq_2yrs` | payments 30+ days late in the last two years |
| | `inq_last_6mths` | credit applications in the last six months |
| | `open_acc`, `total_acc` | open accounts, and all accounts ever opened |
| | `pub_rec` | public records, such as court judgments and bankruptcies |
| | `pub_rec_bankruptcies` | bankruptcies |
| | `revol_bal`, `revol_util` | credit card balance, and share of the limit used |
| | `mort_acc` | number of mortgage accounts |

### 3.3 Payment columns

`funded_amnt` (amount lent) and `total_pymnt` (total paid back) are only known after the loan ends, so they're never features. We keep them aside for the profit calculation in Section 10.

---

## 4. The target

The target comes from `loan_status`. We keep only loans with a final outcome.

| Status | 36-month | 60-month | Target |
|---|---|---|---|
| Fully Paid | 532,629 | 41,139 | 0 (good) |
| Charged Off | 85,955 | 13,830 | 1 (bad) |
| Does not meet the credit policy (Fully Paid or Charged Off) | 2,438 | 311 | removed: old, looser rules |
| Current, Late, In Grace Period | 147 | 10 | removed: not finished |
| **Loans used** | **618,584** | **54,969** | |

"Default" would also count as 1, but no loan in these years has that status. By December 2018 they had all been charged off.

The default rate is 13.9% for 36-month loans and 25.2% for 60-month loans.

---

## 5. Preparing the features

### 5.1 New features

Each one uses only its own loan, so it's safe to build them before the train/test split.

| Feature | From | How |
|---|---|---|
| `fico` | `fico_range_low`, `fico_range_high` | middle of the range |
| `credit_history_years` | `earliest_cr_line`, `issue_d` | years from first credit account to the loan |
| `loan_to_income` | `loan_amnt`, `annual_inc` | loan / income (income of 0 counts as missing) |
| `emp_length_years` | `emp_length` | 0 (under a year) to 10 (ten or more) |

The replaced columns are dropped. `annual_inc` stays as a feature.

Why `loan_to_income` instead of `loan_amnt`? A loan is risky when it's large *for that income*. A scorecard scores each feature separately, so it can't work out the ratio itself. Nothing is lost: loan = ratio × income.

### 5.2 Missing values

Counting missing values by year shows two columns that fail check 2:

- **`mort_acc`**: empty for every loan from 2007 to 2011, partly empty in 2012, never after. The test loans are never empty, so the model would just learn "empty = old".
- **`pub_rec_bankruptcies`**: empty for 244 of 251 loans in 2007 and 450 of 1,562 in 2008. Bankruptcies are already in `pub_rec`, so nothing is lost. It's dropped from both groups so they start with the same features.

Both are dropped. The other missing values are few and show up in recent years too, so "empty" doesn't mean "old". They look like unanswered questions. We keep them, and they get their own bin in Section 7.

| Feature | Missing, 36-month | Missing, 60-month |
|---|---|---|
| `emp_length_years` | 36,139 (5.8%) | 1,432 (2.6%) |
| `revol_util` | 347 | 31 |
| `dti`, `loan_to_income` | 2 | 0 |

### 5.3 The 16 candidate features

| Kind | Features |
|---|---|
| Category | `purpose`, `verification_status`, `home_ownership` |
| Number | `annual_inc`, `dti`, `delinq_2yrs`, `inq_last_6mths`, `open_acc`, `total_acc`, `pub_rec`, `revol_bal`, `revol_util`, `fico`, `credit_history_years`, `loan_to_income`, `emp_length_years` |

Section 8 picks the final features from these.

---

## 6. Train and test sets

We train on older loans and test on newer ones, the way a bank uses past loans to judge new applicants.

| Group | Train | Test |
|---|---|---|
| 36-month | 2007–2014: 335,558 loans, 13.1% default | 2015: 283,026 loans, 14.9% default |
| 60-month | Jan 2010 – Jun 2013: 32,837 loans, 25.4% default | Jul–Dec 2013: 22,132 loans, 24.8% default |

Why split 60-month loans mid-2013? Most of them (34,382 of 54,969) were issued in 2013. Testing on the whole year would leave too few loans to train on.

The model learns only from the train set. The test set is used once, at the end.

The train and test sets are saved with the 16 features and the target. The payment columns are saved separately, in the same row order, for Section 10.

---

## 7. Binning and WoE

The steps in this section are done in [2_binning_woe](../notebooks/2_binning_woe.ipynb).

Each feature is grouped into a few bins, and each bin gets a value that says how risky its loans are. This is done separately for each loan term, using its own train set.

### 7.1 Weight of evidence

WoE compares one bin with the whole train set. Let $g_i$ and $b_i$ be the number of good and bad loans in bin $i$, and $G$ and $B$ the totals. Then

$$
\mathrm{WoE}_i = \ln \left( \frac{g_i / G}{b_i / B} \right)
$$

The top part is the share of all good loans that fall in the bin, and the bottom part is the share of all bad loans. If a bin holds more than its share of good loans, it is safer than average and its WoE is positive. If it holds more than its share of bad loans, it is riskier and its WoE is negative. A WoE of zero means the bin is about average.

The model uses these WoE values instead of the raw values. This lets risk change in steps rather than in a straight line, gives missing values a value of their own, and stops extreme values from distorting the model.

### 7.2 Finding the bins

The bins are found with optbinning (Navas-Palencia, 2020). It first cuts each feature into small groups, then searches for the best way to join them into bins that follow three rules:

1. Each bin holds at least 5% of the loans, so that its WoE is not driven by chance.
2. Each bin has both good and bad loans, since WoE cannot be worked out if either count is zero.
3. WoE goes in one direction from bin to bin, so that the effect of a feature is simple to explain. The direction, up or down, is chosen from the data.

Categories are sorted by default rate before being joined, so their WoE also goes in one direction. Missing values always get their own bin. If a feature had no missing values in the train set, a missing value later gets a WoE of zero, meaning average risk.

### 7.3 What binning gains and loses

Two loans whose values fall in the same bin get the same WoE for that feature, so the model sees the bin and not the exact value. This gives up some detail, and brings several things in return.

What is lost:

- Detail inside a bin. Two borrowers with FICO scores of 710 and 720 are treated the same if they share a bin, although the second is probably slightly safer.
- Smoothness at the edges. Two borrowers just either side of a bin edge get different WoE values, although they are almost identical.

What is gained:

- Risk does not have to follow a straight line. Without bins, the model assumes each extra FICO point lowers risk by the same amount everywhere. With bins, each range gets its own level of risk.
- The estimates are steady. The default rate of a bin rests on at least 5% of the loans.
- Extreme values do no harm. A very high income, or a typing error, just falls in the top bin.
- Missing values need no guessing. They have a bin and a WoE of their own.
- The model is easy to explain. Each bin later becomes a number of points, so anyone can see why a borrower got their score.

---

---

## 8. The PD model

The steps in this section are done in [3_model](../notebooks/3_model.ipynb).

The model takes the features of a loan and gives back its PD, the chance between 0% and 100% that the loan will not be repaid. It is a logistic regression on the WoE values.

### 8.1 From WoE values to a PD

Each loan's value for a feature is first replaced by the WoE of the bin it falls in (Section 7). A loan is then described by $k$ numbers $w_1, \dots, w_k$, one WoE per feature. The model gives each feature a weight and adds everything up into a score

$$
z = \beta_0 + \beta_1 w_1 + \beta_2 w_2 + \dots + \beta_k w_k
$$

The score can be any number, positive or negative. It is turned into a probability with the logistic function

$$
p = \frac{1}{1 + e^{-z}}
$$

which is always between 0 and 1. A very negative score gives a PD close to 0, a score of zero gives 50%, and a very positive score gives a PD close to 1.

Solving the second formula for $z$ shows what the score means.

$$
z = \ln \left( \frac{p}{1 - p} \right)
$$

The ratio $p / (1 - p)$ is the odds of default. A PD of 20% means odds of 0.25, one default for every four repaid loans. So the score is the log of the odds, and each weight says how much the log odds change when that feature's WoE goes up by one.

### 8.2 Why WoE fits this model

WoE is already measured in log odds, which is why it works so well as the input. Take one bin $i$ with $g_i$ good and $b_i$ bad loans, in a train set with $G$ good and $B$ bad loans. The log odds of default in that bin are

$$
\ln \left( \frac{b_i}{g_i} \right) = \ln \left( \frac{B}{G} \right) - \ln \left( \frac{g_i / G}{b_i / B} \right) = \ln \left( \frac{B}{G} \right) - \mathrm{WoE}_i
$$

Therefore, with a single feature, the model that matches the data exactly has $\beta_0 = \ln (B / G)$, the log odds of default in the whole train set, and $\beta_1 = -1$.

With several features the weights are no longer exactly $-1$, because the features share information. Each weight should still be negative. A positive weight is a warning that the feature repeats what another feature already says.

### 8.3 Finding the weights

The weights are chosen so that the model agrees as closely as possible with what happened to the loans in the train set. Let $y_i$ be 1 if loan $i$ defaulted and 0 if it was repaid, and let $p_i$ be its PD from the model. The probability the model gives to the outcome that really happened is $p_i$ for a defaulted loan and $1 - p_i$ for a repaid one. Both cases fit in one expression.

$$
p_i^{y_i} (1 - p_i)^{1 - y_i}
$$

Multiplying this over all $n$ loans gives the likelihood, the probability the model gives to everything that happened in the train set.

$$
L(\beta) = \prod_{i=1}^{n} p_i^{y_i} (1 - p_i)^{1 - y_i}
$$

A product of many small numbers is hard to work with, so we take the log, which turns it into a sum. The result is called the log-likelihood.

$$
\ell(\beta) = \sum_{i=1}^{n} \left[ y_i \ln p_i + (1 - y_i) \ln (1 - p_i) \right]
$$

The weights are the values of $\beta_0, \dots, \beta_k$ that make the log-likelihood as large as possible. This is called maximum likelihood. There is no formula that gives them directly, so the computer starts from a guess and improves it step by step until it stops changing.

At the best weights, the slope of the log-likelihood is zero in every direction. To find the slope, differentiate the log-likelihood with respect to each weight, using the chain rule in three small steps.

First, the logistic function has a simple derivative with respect to the score.

$$
\frac{\partial p_i}{\partial z_i} = p_i (1 - p_i)
$$

Second, differentiate one loan's term of the log-likelihood with respect to its score.

$$
\frac{\partial}{\partial z_i} \left[ y_i \ln p_i + (1 - y_i) \ln (1 - p_i) \right] = \left( \frac{y_i}{p_i} - \frac{1 - y_i}{1 - p_i} \right) p_i (1 - p_i) = y_i - p_i
$$

Third, the score is $z_i = \beta_0 + \beta_1 w_{i1} + \dots + \beta_k w_{ik}$, where $w_{ij}$ is the WoE of feature $j$ for loan $i$. So its derivative is 1 with respect to $\beta_0$ and $w_{ij}$ with respect to $\beta_j$.

Putting the steps together and adding up over all loans gives the slope for each weight.

$$
\frac{\partial \ell}{\partial \beta_0} = \sum_{i=1}^{n} (y_i - p_i)
$$

$$
\frac{\partial \ell}{\partial \beta_j} = \sum_{i=1}^{n} (y_i - p_i) w_{ij}
$$

Setting each slope to zero gives one equation per weight.

$$
\sum_{i=1}^{n} (y_i - p_i) = 0
$$

$$
\sum_{i=1}^{n} (y_i - p_i) w_{ij} = 0 \quad \mathrm{for\ each\ feature\ } j
$$

The first equation says that the PDs add up to the number of defaults, so the average PD in the train set equals its default rate. The model is right on average by construction. Whether it is also right on new loans is checked on the test set.

### 8.4 Measuring how good the predictions are

The log-loss is the log-likelihood with the sign changed, divided by the number of loans.

$$
\mathrm{log\ loss} = - \frac{\ell(\beta)}{n} = - \frac{1}{n} \sum_{i=1}^{n} \left[ y_i \ln p_i + (1 - y_i) \ln (1 - p_i) \right]
$$

The log-likelihood is maximised and the log-loss is minimised, and both happen at the same weights. Changing the sign turns a maximum into a minimum, and dividing by $n$ does not move it. Therefore, setting the slope of the log-loss to zero gives exactly the equations of Section 8.3.

The principle: each loan adds a penalty of $- \ln p_i$ if it defaulted and $- \ln (1 - p_i)$ if it was repaid. The penalty is small when the PD was close to what happened and large when it was far off. A lower log-loss means better predictions.

### 8.5 Choosing the features

Not every feature helps, and some repeat each other. The features are chosen by forward selection. The model starts with no features and adds one at a time, each round taking the feature that improves the predictions the most.

Banks traditionally keep a feature if its weight passes a significance test. With hundreds of thousands of loans almost every weight passes, so here a feature is judged by whether it improves predictions on loans the model was not fitted on. This is done with cross-validation.

The train set is split at random into 5 parts of equal size. One part is set aside for test, and the weights are found from the other four, as in Section 8.3. Those weights are then used to work out a PD for each loan in the part that was set aside, and the log-loss of these PDs is the score of that part.

$$
\mathrm{CV}_m(S) = - \frac{1}{n_m} \sum_{i \in \mathrm{part\ } m} \left[ y_i \ln p_i + (1 - y_i) \ln (1 - p_i) \right]
$$

Here $S$ is the set of features in the model, part $m$ is the part set aside, and $n_m$ is its number of loans, about a fifth of the train set. Each PD $p_i$ comes from weights found without part $m$, so every loan is scored by a model that never saw it.

This is repeated five times, setting aside a different part each time. The overall score is the average of the five.

$$
\mathrm{CV}(S) = \frac{1}{5} \sum_{m=1}^{5} \mathrm{CV}_m(S)
$$

The selection then works as follows.

1. Start with no features, so $S$ is empty and every loan gets the same PD.
2. For each feature $j$ not yet in $S$, work out the score with $j$ added.
3. Keep only the features that improve the score in all five parts.
4. If there are any, add the one with the lowest average score to $S$ and go back to step 2.
5. If there are none, stop.

The rule in step 3 can be written as

$$
\mathrm{CV}_m(S \cup \lbrace j \rbrace) < \mathrm{CV}_m(S) \quad \mathrm{for\ every\ part\ } m
$$

Asking for an improvement in every part, and not only on average, keeps out features whose gain is so small that it could be chance.

### 8.6 Checking that the selection is stable

The selection of Section 8.5 is run once, on one train set. A feature with a large effect would be chosen whatever loans were used. A feature with a very small effect might be chosen only because of the particular loans in the sample.

To tell these apart, the selection is repeated 5 times, each on a random 80% of the train set drawn without replacement. Each run is a complete selection as in Section 8.5, so within a run a feature is still added only if it improves the log-loss in all five parts. Writing $S_b$ for the set of features chosen in run $b$, the stability of feature $j$ is the share of runs in which it is chosen.

$$
\mathrm{stability}_j = \frac{1}{5} \sum_{b=1}^{5} \mathbf{1} \lbrace j \in S_b \rbrace
$$

The symbol $\mathbf{1} \lbrace j \in S_b \rbrace$ is 1 if feature $j$ is chosen in run $b$ and 0 if not.

A feature is kept in the final model if it is chosen in at least 4 of the 5 runs, a stability of 0.8 or more. This allows for one unlucky run, while leaving out the features that are chosen only some of the time. The selection on the whole train set shows how the method works and how much each feature adds, and the 5 runs decide which features are kept.

A feature just under the threshold can also be added by judgement, when there is a clear reason to expect it to matter. Any feature added this way is named, with the reason, and must pass the same checks as the others.

### 8.7 The features chosen

The table shows the features of each final model, with the number of runs in which each was chosen.

| Feature | 36-month model | 60-month model |
|---|---|---|
| `fico` | 5 of 5 | 5 of 5 |
| `inq_last_6mths` | 5 of 5 | 5 of 5 |
| `loan_to_income` | 5 of 5 | 5 of 5 |
| `annual_inc` | 5 of 5 | 4 of 5 |
| `home_ownership` | 5 of 5 | 3 of 5, added by judgement |
| `purpose` | 5 of 5 | not kept |
| `dti` | 5 of 5 | not kept |
| `total_acc` | 5 of 5 | not kept |
| `verification_status` | 4 of 5 | not kept |
| `open_acc` | 4 of 5 | not kept |
| `pub_rec` | not kept | 3 of 5, added by judgement |
| Number of features | 10 | 6 |

The 36-month model has 10 features, all chosen in at least 4 of the 5 runs.

The 60-month model has 6. Four are chosen in at least 4 of the 5 runs. The other two, `home_ownership` and `pub_rec`, are chosen in 3 runs and are added by judgement. Both are well-known signs of credit risk. Home ownership says how settled a borrower is, and public records, such as court judgments and bankruptcies, show past trouble with debts. With about a tenth as many loans as the 36-month group, a real effect can easily miss the threshold here.

The 60-month model is simpler because there is less data. With fewer loans, fewer features can be identified reliably. Four features are shared by both models, which are `fico`, `inq_last_6mths`, `loan_to_income` and `annual_inc`, or five counting `home_ownership`.

### 8.8 Reading the fitted weights

Once the final features are settled, the model is fitted on the whole train set. Three numbers are reported for each feature.

- The weight. It should be negative, because a higher WoE means a safer bin. A feature with a positive weight is removed and the model is fitted again.
- The standard error. A different sample of loans would give a slightly different weight, and the standard error is the typical size of that difference.
- The p-value. It is the chance of getting a weight this far from zero if the feature really had no effect. It comes from the weight divided by its standard error, and a value below 0.05 is the usual sign that the effect is real.

### 8.9 Checking for features that repeat each other

Two features can carry nearly the same information. The model then cannot tell which of them deserves the weight, and both weights become unreliable. This is measured with the variance inflation factor (VIF).

For each feature $j$ in the final model, its WoE values are predicted from the WoE values of all the other features with a linear regression.

$$
w_{ij} = \alpha_0 + \sum_{l \neq j} \alpha_l w_{il} + \varepsilon_i
$$

Here $w_{ij}$ is the WoE of feature $j$ for loan $i$, the $\alpha$ are the weights of this regression, and $\varepsilon_i$ is the part of $w_{ij}$ that the other features cannot explain. Writing $\hat{w}_{ij}$ for the predicted value and $\bar{w}_j$ for the average of feature $j$, the share of its variation that the others explain is

$$
R_j^2 = 1 - \frac{\sum_{i=1}^{n} (w_{ij} - \hat{w}_{ij})^2}{\sum_{i=1}^{n} (w_{ij} - \bar{w}_j)^2}
$$

The top of the fraction is the variation left over after the prediction, and the bottom is the total variation. So $R_j^2$ is 0 when the other features say nothing about feature $j$, and 1 when they predict it perfectly. The VIF is then

$$
\mathrm{VIF}_j = \frac{1}{1 - R_j^2}
$$

It is 1 when $R_j^2 = 0$, and grows without limit as $R_j^2$ gets close to 1.

The name comes from its effect on the weight. Only the part of feature $j$ that the others cannot explain helps to pin down its weight, and that part is a share $1 - R_j^2$ of its variation. So the more the others explain, the more uncertain the weight becomes, by about a factor of the VIF.

A VIF above 5 is the usual sign that a feature repeats the others. It corresponds to $R_j^2 = 0.8$, since

$$
\frac{1}{1 - 0.8} = 5
$$

so the other features explain more than 80% of feature $j$. Such a feature is removed and the model is fitted again. This does not happen in our case. The highest VIF is 1.64 in the 36-month model and 1.39 in the 60-month model, so the other features explain at most 39% of any feature. No feature is removed.

### 8.10 How the models do on the train sets

Before the models are tested on new loans, three things are checked on the loans they were built on.

The first is the average PD. By the first equation of Section 8.3, it equals the default rate of the train set exactly, 13.1% for the 36-month loans and 25.4% for the 60-month loans.

The second is how well the model ranks borrowers. Take one loan that defaulted and one that was repaid. The model ranks the pair correctly if it gave the higher PD to the one that defaulted. The AUC is the share of all such pairs that are ranked correctly, with a pair counted as half when the two PDs are equal. Writing $D$ for the set of defaulted loans and $R$ for the set of repaid loans, with $n_D$ and $n_R$ loans in them,

$$
\mathrm{AUC} = \frac{1}{n_D n_R} \sum_{i \in D} \sum_{k \in R} \left[ \mathbf{1} \lbrace p_i > p_k \rbrace + \frac{1}{2} \mathbf{1} \lbrace p_i = p_k \rbrace \right]
$$

A model that knows nothing ranks half the pairs correctly, so its AUC is 0.5, and a perfect model has an AUC of 1. The Gini rescales this to run from 0 to 1.

$$
\mathrm{Gini} = 2 \times \mathrm{AUC} - 1
$$

The Gini depends only on the order of the PDs and not on their size. On the train sets, the 36-month model has an AUC of 0.653 and a Gini of 0.306, and the 60-month model has an AUC of 0.627 and a Gini of 0.253. So the models rank about 65% and 63% of the pairs correctly.

The third is calibration, whether the PDs match what happened at every level of risk. The loans are sorted by PD and cut into 10 groups with the same number of loans in each. For each group $g$ with $n_g$ loans, the average PD is compared with the share of loans that defaulted.

$$
\bar{p}_g = \frac{1}{n_g} \sum_{i \in g} p_i \qquad d_g = \frac{1}{n_g} \sum_{i \in g} y_i
$$

If the PDs are accurate, $\bar{p}_g$ and $d_g$ are close in every group. They are. For the 36-month loans the default rate rises from 4.0% in the safest group to 25.1% in the riskiest, against average PDs of 4.3% and 25.8%. For the 60-month loans it rises from 11.4% to 40.6%, against average PDs of 11.9% and 41.0%.

**Remark**: a single loan cannot be checked this way. A loan with a PD of 20% that is repaid is not a wrong prediction. The PD is right if, among many loans given about 20%, about 20% default.

---

## 9. Testing the models

The steps in this section are done in [4_test](../notebooks/4_test.ipynb).

The two models are now tried on the test sets, the newer loans set aside in Section 6 and not used since. Each test loan gets a PD from the bins and weights found on the train set.

### 9.1 The level of the PDs

| | Average PD | Actual default rate |
|---|---|---|
| 36-month loans, train set | 13.1% | 13.1% |
| 36-month loans, test set | 13.4% | 14.9% |
| 60-month loans, train set | 25.4% | 25.4% |
| 60-month loans, test set | 26.5% | 24.8% |

On the test sets the models are within two points of what happened. The 36-month model is a little low and the 60-month model a little high.

### 9.2 Ranking

| | AUC | Gini |
|---|---|---|
| 36-month loans, train set | 0.653 | 0.306 |
| 36-month loans, test set | 0.654 | 0.307 |
| 60-month loans, train set | 0.627 | 0.253 |
| 60-month loans, test set | 0.607 | 0.214 |

The 36-month model ranks new borrowers as well as the ones it was built on. The 60-month model loses some of its power, with the Gini falling from 0.253 to 0.214. It was built on about a tenth as many loans, so it is less reliable.

These Ginis are modest. Every borrower in the data was already approved by LendingClub, so the clearly risky applicants are missing and the rest are harder to tell apart.

### 9.3 Calibration

A model is calibrated when its PDs match what happens. Loans given a PD of 10% should default about 10% of the time. To check this, the test loans are sorted by PD and cut into 10 groups with the same number of loans in each, and in each group the average PD is compared with the share of loans that defaulted, as in Section 8.10.

| Group | 36-month, average PD | 36-month, default rate | 60-month, average PD | 60-month, default rate |
|---|---|---|---|---|
| 1, safest | 4.4% | 4.2% | 13.5% | 13.1% |
| 2 | 6.9% | 7.6% | 17.9% | 16.4% |
| 3 | 8.6% | 9.7% | 20.6% | 19.6% |
| 4 | 10.2% | 11.7% | 22.9% | 23.2% |
| 5 | 11.7% | 13.5% | 25.0% | 24.3% |
| 6 | 13.3% | 14.9% | 27.1% | 24.6% |
| 7 | 15.1% | 17.4% | 29.3% | 27.4% |
| 8 | 17.2% | 19.8% | 32.0% | 29.6% |
| 9 | 20.2% | 22.4% | 35.2% | 32.5% |
| 10, riskiest | 26.5% | 27.7% | 41.1% | 37.4% |

For the 36-month loans, the default rate rises in every group, so the order is right. But the PDs are a little low in almost every group, by 1 to 2.6 points.

For the 60-month loans, the default rate also rises from group to group. The PDs are accurate for the safer groups and 2 to 4 points too high for the riskier ones.

### 9.4 What the tests show

Both models rank borrowers on loans they have never seen, and the 36-month model does so as well as on its own train set. Their PDs are close to the default rates that followed, but not exact, because default rates moved over time. The 60-month model is the weaker of the two, as expected from its smaller train set.

---

## 10. The lending decision

The steps in this section are done in [4_test](../notebooks/4_test.ipynb).

The model gives each applicant a PD. It does not decide anything. The lender turns the PD into a decision with a cut-off $c$.

$$
\mathrm{approve\ loan\ } i \mathrm{\ if\ } p_i < c
$$

A cut-off of 0.5 would be useless here. Only a minority of loans default, so almost no applicant has a PD of 0.5 or more, and every loan would be approved. The cut-off has to be set from what the lender wants to achieve.

### 10.1 What a cut-off does

Each cut-off is judged by what it would have done on loans whose outcome is known. Five quantities are worked out, over the $n$ loans of a data set.

The approval rate is the share of applicants who get a loan.

$$
A(c) = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1} \lbrace p_i < c \rbrace
$$

The default rate of the approved loans is the share of them that default.

$$
D(c) = \frac{\sum_{i=1}^{n} y_i \mathbf{1} \lbrace p_i < c \rbrace}{\sum_{i=1}^{n} \mathbf{1} \lbrace p_i < c \rbrace}
$$

The share of defaults avoided is the share of all defaulted loans that the cut-off declines.

$$
V(c) = \frac{\sum_{i=1}^{n} y_i \mathbf{1} \lbrace p_i \geq c \rbrace}{\sum_{i=1}^{n} y_i}
$$

The profit is the total gain or loss on the approved loans. The profit of one loan is what the borrower paid back in the end, including interest and anything recovered after a default, minus the amount lent. Writing $r_i$ for the amount paid back and $a_i$ for the amount lent,

$$
\Pi(c) = \sum_{i=1}^{n} (r_i - a_i) \mathbf{1} \lbrace p_i < c \rbrace
$$

The return on the money lent is the profit divided by the total amount lent on the approved loans.

$$
R(c) = \frac{\Pi(c)}{\sum_{i=1}^{n} a_i \mathbf{1} \lbrace p_i < c \rbrace}
$$

It says how much each dollar lent earns, over the whole life of the loans.

The amounts paid back are known only after the loan has ended, so they are never used as features. They are used here only to judge a decision afterwards, in the same way as the target.

Raising the cut-off approves more applicants and lets more defaults through. Lowering it does the opposite. No cut-off is best on every count, so the choice depends on the aim.

### 10.2 Three ways to choose the cut-off

By profit. The cut-off is the one with the highest profit.

$$
c_{\mathrm{profit}} = \arg\max_{c} \Pi(c)
$$

By risk. The lender sets a limit $d$ on the default rate of its approved loans and takes the highest cut-off that respects it. This approves as many applicants as the limit allows.

$$
c_{\mathrm{risk}} = \max \lbrace c : D(c) \leq d \rbrace
$$

The limit is 10% for the 36-month loans and 20% for the 60-month loans. The 60-month loans default about twice as often, so the same limit would decline almost all of them.

By volume. The lender wants to approve a set share $v$ of applicants, here 85%, and takes the cut-off that comes closest.

$$
c_{\mathrm{volume}} = \arg\min_{c} \lvert A(c) - v \rvert
$$

The cut-offs tried go up in steps of 0.01, from 0.04 to 0.41 for the 36-month loans and from 0.08 to 0.57 for the 60-month loans, whose PDs are higher.

### 10.3 Chosen on the train set, tried on the test set

A cut-off is a decision, so it is chosen without looking at the test set. The three cut-offs are found on the train set. Each is then applied to the test loans, and its approval rate, default rate and profit there are reported.

### 10.4 What this leaves out

The profit used here is simple. It does not count the time value of money, or the lender's own costs such as funding and servicing.

The profit also rests on the interest rates LendingClub charged, which it set from its own assessment of each borrower's risk. These do shape the profit figures. Riskier borrowers paid more, which is why even risky loans made money on average and why the most profitable cut-off is so high. The profit results therefore say what each cut-off would have earned at LendingClub's prices. The cut-offs chosen by risk and by volume do not depend on those prices.

In practice the PD is only one part of the decision. A lender applies fixed policy rules before the score, and checks separately that the borrower can afford the repayments.

### 10.5 Results

The cut-offs are chosen on the train sets. The figures below are from the test sets.

**36-month loans**

| Way of choosing | Cut-off | Approved | Default rate of approved loans | Defaults avoided | Profit | Return on money lent |
|---|---|---|---|---|---|---|
| By profit | 0.38 | 99.9% | 14.9% | 0.3% | $237.4 million | 6.6% |
| By risk | 0.16 | 69.7% | 11.3% | 47.3% | $190.6 million | 7.2% |
| By volume | 0.20 | 84.7% | 12.9% | 26.9% | $218.9 million | 6.9% |

**60-month loans**

| Way of choosing | Cut-off | Approved | Default rate of approved loans | Defaults avoided | Profit | Return on money lent |
|---|---|---|---|---|---|---|
| By profit | 0.52 | 100.0% | 24.8% | 0.0% | $87.6 million | 19.9% |
| By risk | 0.27 | 54.0% | 19.6% | 57.2% | $50.9 million | 21.3% |
| By volume | 0.35 | 84.7% | 22.8% | 22.2% | $76.9 million | 20.6% |

Each way of choosing gives a different cut-off. By profit, almost everyone is approved, because riskier borrowers paid more interest and even risky loans made money on average. By risk, the cut-off is the strictest, and the applicants it declines hold about half of all the defaults. By volume, it sits in between.

So there is a trade-off. A stricter cut-off means fewer defaults and a better return on each dollar lent, but less profit in total, because the declined loans still made money as a group.

The 36-month risk cut-off was chosen to keep the default rate at 10% or less, and the test set gives 11.3%, because loans issued in 2015 defaulted more than earlier ones. The 60-month cut-off stays under its 20% limit. A rule chosen on past loans has to be watched and adjusted over time.

---

## 11. The scorecard

The steps in this section are done in [5_scorecard](../notebooks/5_scorecard.ipynb).

The model is written as a scorecard. Each bin of each feature gets a number of points, and a borrower's score is the sum of their points. A higher score means a safer borrower. Nothing new is fitted. The points come straight from the weights of Section 8 and the WoE values of Section 7.

### 11.1 The scale

The model works in log odds, which are hard to read. The score is a straight-line stretch of the log odds of repaying. With $p$ the PD, the odds of repaying are $(1 - p) / p$, and

$$
\mathrm{score} = \mathrm{offset} + \mathrm{factor} \times \ln \left( \frac{1 - p}{p} \right)
$$

The two numbers are fixed by two choices.

- A score of 600 means odds of 50 to 1, that is 50 borrowers who repay for each one who defaults.
- Every 20 points more doubles the odds.

The second choice gives the factor. Doubling the odds adds $\ln 2$ to the log odds, and this must add 20 points.

$$
\mathrm{factor} = \frac{20}{\ln 2} = 28.85
$$

The first choice then gives the offset.

$$
\mathrm{offset} = 600 - \mathrm{factor} \times \ln 50 = 487.12
$$

### 11.2 Points for each bin

The log odds of repaying are the model's score $z$ of Section 8.1 with the sign changed, so

$$
\mathrm{score} = \mathrm{offset} - \mathrm{factor} \times (\beta_0 + \beta_1 w_1 + \dots + \beta_k w_k)
$$

This splits into one part that is the same for everyone and one part for each feature.

$$
\mathrm{score} = \underbrace{\mathrm{offset} - \mathrm{factor} \times \beta_0}_{\mathrm{base\ points}} + \sum_{j=1}^{k} \underbrace{\left( - \mathrm{factor} \times \beta_j w_j \right)}_{\mathrm{points\ of\ feature\ } j}
$$

Every borrower starts with the base points. Each feature then adds the points of the bin the borrower falls in, which are the bin's WoE times the feature's weight times the factor, with the sign changed. The weights are negative, so a bin with a positive WoE, a safer bin, gets positive points. A bin with a WoE of zero gets no points, so the base points are the score of a borrower who is average on every feature. The points are rounded to whole numbers.

The base points are 542 for the 36-month loans and 518 for the 60-month loans, which default more often. The two scorecards are saved in the `reports` folder.

A score can be turned back into a PD.

$$
p = \frac{1}{1 + e^{(\mathrm{score} - \mathrm{offset}) / \mathrm{factor}}}
$$

### 11.3 Checking the scores

The scores are the PDs on another scale, so they should rank borrowers equally well. On the test sets the Gini of the scores is 0.308 for the 36-month loans and 0.213 for the 60-month loans, against 0.307 and 0.214 for the PDs (Section 9.2). The small differences come from rounding the points.

The default rate should also fall as the score rises. The test loans are grouped into bands of 20 points.

| Score | 36-month loans | 36-month default rate | 60-month loans | 60-month default rate |
|---|---|---|---|---|
| 480 to 499 | 238 | 36.1% | 1,557 | 37.6% |
| 500 to 519 | 16,989 | 29.2% | 11,435 | 28.3% |
| 520 to 539 | 95,656 | 20.5% | 8,128 | 19.1% |
| 540 to 559 | 115,569 | 12.4% | 995 | 10.7% |
| 560 to 579 | 45,019 | 6.4% | 17 | 5.9% |
| 580 to 599 | 9,264 | 2.5% | 0 | |
| 600 to 619 | 291 | 1.4% | 0 | |

The default rate falls in every step in both groups. A given score also means about the same risk for both loan terms. The two groups differ in where their borrowers sit. Most 36-month loans score between 520 and 559, and most 60-month loans between 500 and 539.

### 11.4 Explaining a decision

The scorecard shows why a borrower got their score. For each feature $j$, the points the borrower got are compared with the most points that feature can give.

$$
\mathrm{gap}_j = \mathrm{points}_j - \max_{\mathrm{bins\ of\ } j} \mathrm{points}
$$

The gap is zero when the borrower is in the best bin and negative otherwise. The features with the largest gaps are the main reasons the score is not higher, and the features with no gap are the borrower's strengths. A lender can give the top few as the reasons for a decision. Each one is a fact about the applicant that can be checked against the scorecard.

Two borrowers from the test sets show how this reads.

A 36-month borrower. The base is 542 points, the features take away 13, and the score is 529.

| Feature | Value | Points | Best possible | Gap |
|---|---|---|---|---|
| `fico` | 692 | 0 | 28 | −28 |
| `loan_to_income` | 0.35 | −8 | 6 | −14 |
| `annual_inc` | 34,000 | −5 | 7 | −12 |
| `total_acc` | 6 | −6 | 3 | −9 |
| `purpose` | debt_consolidation | −1 | 5 | −6 |
| `home_ownership` | RENT | −3 | 3 | −6 |
| `verification_status` | Source Verified | −1 | 2 | −3 |
| `dti` | 10.2 | 4 | 6 | −2 |
| `inq_last_6mths` | 0 | 5 | 5 | 0 |
| `open_acc` | 5 | 2 | 2 | 0 |

The main reasons the score is not higher are an average FICO score, a loan worth 35% of the borrower's yearly income, and a low income. The FICO score costs no points against the average borrower, but it is 28 points short of the best, which makes it the largest gap. In the borrower's favour, they made no credit applications in the last six months and have few open accounts.

A 60-month borrower. The base is 518 points, the features add 5, and the score is 523.

| Feature | Value | Points | Best possible | Gap |
|---|---|---|---|---|
| `fico` | 677 | −7 | 22 | −29 |
| `home_ownership` | RENT | −3 | 2 | −5 |
| `loan_to_income` | 0.10 | 8 | 11 | −3 |
| `annual_inc` | 105,680 | 2 | 5 | −3 |
| `inq_last_6mths` | 0 | 5 | 5 | 0 |
| `pub_rec` | 0 | 0 | 0 | 0 |

Here one feature stands out. The FICO score is 29 points short of the best, far more than anything else. The borrower is otherwise in good shape, with a small loan for their income, no recent credit applications and no public records.

---

## 12. A challenger model

The steps in this section are done in [6_challenger](../notebooks/6_challenger.ipynb).

The scorecard is simple on purpose. It bins every feature and adds up points. To see what that simplicity costs, a second model is fitted on the same train sets and tested on the same test sets. It is a machine-learning model called LightGBM. If it is only a little better than the scorecard, the scorecard loses little by being simple.

### 12.1 Decision trees

LightGBM is built from decision trees. A tree asks a series of yes or no questions about a loan, such as "is the FICO score below 690?" and then "is the dti above 20?". Each answer sends the loan down a branch, until it reaches an end point called a leaf. Every leaf holds a number, and that number is the tree's output for all the loans that end up there.

The questions are chosen to separate the defaulted loans from the repaid ones. A group of loans is pure if they all had the same outcome, and mixed if it holds both. How mixed it is, is measured by the entropy. For a group in which a share $q$ of the loans defaulted,

$$
H(q) = - q \ln q - (1 - q) \ln (1 - q)
$$

The entropy is 0 when the group is pure and highest when $q = 0.5$, an even mix. A basic tree tries every feature and every place to split it, and takes the question that lowers the entropy the most, so that the two new groups are purer than the old one. It then does the same inside each new group, until the groups are too small to split.

A tree can do two things the scorecard cannot. It splits a feature wherever it helps, without fixed bins. And its questions follow one another, so it can treat a high dti differently for a borrower with a low FICO score than for one with a high score. This is called an interaction between features.


### 12.2 Adding up many trees

One tree is a crude model. LightGBM adds up many small ones. Like the logistic regression, it works in log odds. For loan $i$ with features $x_i$, the score after $M$ trees is

$$
z_i = F_0 + \eta \sum_{m=1}^{M} f_m(x_i)
$$

Here $F_0$ is the starting value, the log odds of default in the whole train set, $f_m(x_i)$ is the number that tree $m$ gives to loan $i$, and $\eta$ is a small number called the learning rate. The score is turned into a PD with the same logistic function as in Section 8.1.

$$
p_i = \frac{1}{1 + e^{-z_i}}
$$

The trees are built one after another, and each one corrects the mistakes of those before it. This is called boosting. Before a new tree is built, every loan has a PD from the trees so far, and an error

$$
e_i = y_i - p_i
$$

which is positive for a defaulted loan whose PD is too low and negative for a repaid loan whose PD is too high. This is the same quantity as in Section 8.3, the slope of the log-likelihood with respect to the loan's score.

The old trees affect the new one only through these errors. A question is good if it separates loans whose PDs are too low from loans whose PDs are too high. Where the old trees already predict well, the errors on both sides of a question cancel out, so the new tree gains nothing by asking it. Where they have missed a pattern, the errors lean one way in some group of loans, and a question that picks out that group is chosen. In this way each tree works on what the earlier ones left unexplained.

The finished tree is added to the model, multiplied by the learning rate so that it makes only a small correction, and the next tree works on the errors that remain.

So the logistic regression and LightGBM aim at the same thing, the highest log-likelihood. The first does it with one weight per feature. The second does it with hundreds of small trees.

### 12.3 Keeping the model from memorising

A model this flexible can fit the train set almost perfectly and still do badly on new loans, by learning patterns that are only chance. The settings are chosen to prevent this.

| Setting | Value | What it does |
|---|---|---|
| Number of trees | 300 | how many corrections are made |
| Learning rate | 0.05 | each tree makes only a small correction |
| Leaves per tree | 15 | each tree stays small |
| Loans per leaf | at least 200 | no leaf rests on a handful of loans |
| Share of loans per tree | 80%, drawn at random | each tree sees a different sample |
| Share of features per tree | 80%, drawn at random | no tree can lean on one feature alone |


### 12.4 What the challenger is given

The challenger gets all 16 features of Section 5.3 and decides for itself which to use. The number features go in with their exact values. Missing values are left as they are, and at each question the tree learns which branch to send them down. The three category features are marked as categories, and a tree can split them into any two groups.

| | Scorecard | Challenger |
|---|---|---|
| Features | 10 or 6, chosen in Section 8 | all 16 |
| Values | binned, then WoE | exact |
| Effect of a feature | fixed points per bin | depends on the other features |
| Size | one weight per feature | 300 trees |
| Can be read by a person | yes | no |

### 12.5 Results

Both models are tested on the same test loans.

| Loans | Model | AUC | Gini | Average PD | Default rate |
|---|---|---|---|---|---|
| 36-month | Scorecard | 0.654 | 0.307 | 13.4% | 14.9% |
| 36-month | LightGBM | 0.670 | 0.341 | 13.5% | 14.9% |
| 60-month | Scorecard | 0.607 | 0.214 | 26.5% | 24.8% |
| 60-month | LightGBM | 0.612 | 0.224 | 26.3% | 24.8% |

LightGBM ranks borrowers a little better, so the simplicity of the scorecard costs little. In return, every decision can be explained, the model can be checked by hand, and it always gives the same points for the same answers. This is why we can keep the scorecard as the main model, with LightGBM as a check on how much more the data could give.

---

## 13. Limitations

The models in this project work, but they have limits that anyone using them should know.

### 13.1 The data

Only approved loans. Every loan in the data was approved by LendingClub. The applicants it turned down are missing, so the models never learned from the riskiest people. One lender, one country, one period. The loans are from a single US platform, issued between 2007 and 2015. Borrowers, lending rules and the economy have all changed since then. The models describe that market at that time.

Few features. Only what LendingClub recorded at application could be used. A bank would also have its own history with the customer and a much fuller credit bureau report.

### 13.2 The models

Several choices were made by judgement. These include the 5% minimum bin size, the threshold of 4 out of 5 runs in the stability check, the two features added by hand for the 60-month period loans, and the risk limits and volume target in the lending decision. Other reasonable choices would give different models.

### 13.3 The lending decision

The cut-off was chosen in three ways. By risk and by volume, it depends only on the PDs and on which loans defaulted. By profit, it also depends on the interest rates LendingClub charged, and on a simple measure of profit that leaves out the time value of money and the lender's own costs. So the profit cut-off is the least reliable of the three.

### 13.5 Fairness

Location was left out because it can stand in for race and national origin. Beyond that, the models were not tested for fairness. Some features, such as income and home ownership, also differ between groups of people. A lender would need to check whether the models treat protected groups differently before using them.