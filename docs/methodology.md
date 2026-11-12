# Methodology

Project: Probability of Default (PD) model on LendingClub loans

Author: Daniel Siretanu

This document explains how the model is built and why. Each section is written when that step of the work is done.

---

## 1. What this project does

When someone applies for a loan, the lender has to decide whether to lend and at what interest rate. The key number behind that decision is the probability of default (PD), the chance that the borrower will not repay.

This project builds a model that estimates the PD of each loan, using only what was known on the day the loan was approved. A good PD model does two things. It ranks borrowers correctly, so riskier borrowers come above safer ones (*discrimination*). And its PDs are right as numbers, so if it says 5%, about 5 in 100 such loans default (*calibration*).

---

## 2. The data

The steps in Sections 2 to 6 are done in [1_explore_data](../notebooks/1_explore_data.ipynb).

The data is LendingClub's public loan file `accepted_2007_to_2018Q4.csv.gz` from Kaggle. It has 2,260,701 rows and 151 columns, with one row per loan describing the borrower, the loan and what happened to it. A few rows are summary lines rather than loans, and they are removed.

### 2.1 Loan terms

Loans run for either 36 months (offered since 2007) or 60 months (since 2010). The PD here is the chance of default at any point before the loan ends, so a 60-month loan has more time to default than a 36-month one. The two terms are modelled as separate groups.

### 2.2 Which years can be used

Loan statuses are recorded as of December 2018. We only know how a loan ended if it was due to finish by then, so each term keeps only the issue years where that is true.

| Term | Loans finish | Issue years used | Loans |
|---|---|---|---|
| 36 months | 3 years after issue | 2007–2015 | 621,169 |
| 60 months | 5 years after issue | 2010–2013 | 55,290 |

---

## 3. Which columns are used

Most of the 151 columns cannot be used. A column becomes a candidate feature only if it passes five checks.

1. Known at application. Columns filled in after the loan was issued, such as payments received, partly record the outcome. Using them would be *data leakage*.
2. Recorded in every year. Columns that LendingClub only started recording later are empty for older loans, so an empty value would just mean "old loan" (Section 5.2).
3. A usable value. Free text, web links and identifiers are left out.
4. Not LendingClub's own risk assessment. LendingClub's grade (`grade`, `sub_grade`) and the interest rate and payment set from it (`int_rate`, `installment`) are left out, so the model judges the borrower on its own instead of copying LendingClub.
5. Not location. `zip_code` and `addr_state` are left out because where someone lives is closely linked to race and national origin, which fair-lending law protects, so location can act as a proxy for them.

### 3.1 Columns kept

Four columns describe the loan itself and are not features.

| Column | Meaning | Use |
|---|---|---|
| `id` | loan number | identifies each loan |
| `issue_d` | month the loan was issued | splits train and test (Section 6) |
| `loan_status` | status in December 2018 | builds the target (Section 4) |
| `term` | 36 or 60 months | splits the two groups (Section 2.1) |

These are the candidate features. Two are dropped later because they were not recorded in every year (Section 5.2), and the final features are picked from the rest.

| Group | Column | Meaning |
|---|---|---|
| The loan | `loan_amnt` | amount requested, used only relative to income (Section 5.1) |
| | `purpose` | what the loan is for, such as debt consolidation |
| The borrower | `annual_inc` | declared yearly income |
| | `verification_status` | whether LendingClub checked that income |
| | `emp_length` | years in the current job |
| | `home_ownership` | rents, owns, or has a mortgage |
| | `dti` | monthly debt payments (excluding mortgage and this loan) divided by monthly income |
| Credit history, from the credit bureau | `fico_range_low`, `fico_range_high` | FICO credit score range |
| | `earliest_cr_line` | date of the first credit account |
| | `delinq_2yrs` | payments 30+ days late in the last two years |
| | `inq_last_6mths` | credit applications in the last six months |
| | `open_acc`, `total_acc` | open credit accounts, and all accounts ever opened |
| | `pub_rec` | public records, such as court judgments and bankruptcies |
| | `pub_rec_bankruptcies` | bankruptcies (dropped, Section 5.2) |
| | `revol_bal`, `revol_util` | amount owed on credit cards, and share of the limit used |
| | `mort_acc` | number of mortgage accounts (dropped, Section 5.2) |

---

## 4. The target

The target $y$ comes from `loan_status`, the loan's status in December 2018.

| Status | Target |
|---|---|
| Fully Paid | $y = 0$ (good) |
| Charged Off, Default | $y = 1$ (bad) |
| Current, In Grace Period, Late | not used, the loan had not finished |
| Does not meet the credit policy … | not used, issued under older and looser rules |

Because only finished years are used (Section 2.2), very few loans are removed.

| | 36-month loans | 60-month loans |
|---|---|---|
| Loans in the period | 621,169 | 55,290 |
| Removed, old credit policy | 2,438 | 311 |
| Removed, still running | 147 | 10 |
| Loans used | 618,584 | 54,969 |

---

## 5. Preparing the features

### 5.1 New features

Some columns are turned into a form the model can use. Each new value depends only on the loan itself, so this is done before the train and test split.

| Feature | Made from | How |
|---|---|---|
| `fico` | `fico_range_low`, `fico_range_high` | average of the two ends of the range |
| `credit_history_years` | `earliest_cr_line`, `issue_d` | years from the first credit account to the loan |
| `loan_to_income` | `loan_amnt`, `annual_inc` | loan amount divided by annual income |
| `emp_length_years` | `emp_length` | years in the job as a number, from 0 (under a year) to 10 (ten or more) |

The loan amount is only used through `loan_to_income`. A loan is risky when it is large compared with income, and a scorecard gives points to each feature separately, so it cannot work out that ratio by itself.

### 5.2 Missing values

Counting missing values year by year showed two columns that LendingClub did not record for its early loans.

- `mort_acc` is empty for every loan before 2012. The model would learn that empty means old, and test loans are never empty.
- `pub_rec_bankruptcies` is empty for almost all 2007 loans and about a third of 2008 loans. Bankruptcies are already counted in `pub_rec`, so nothing is lost.

Both are dropped. The other gaps are small and spread across all years, which looks like borrowers not answering, so those features are kept with their gaps.

| Feature | Missing, 36-month loans | Missing, 60-month loans |
|---|---|---|
| `emp_length_years` | 36,139 (5.8%) | 1,432 (2.6%) |
| `revol_util` | 347 | 31 |
| `dti`, `loan_to_income` | 2 | 0 |

### 5.3 Features used

After these changes, 16 features are left. The final features for the model are chosen from these.

| Kind | Features |
|---|---|
| Category | `purpose`, `verification_status`, `home_ownership` |
| Number | `annual_inc`, `dti`, `delinq_2yrs`, `inq_last_6mths`, `open_acc`, `total_acc`, `pub_rec`, `revol_bal`, `revol_util`, `fico`, `credit_history_years`, `loan_to_income`, `emp_length_years` |

---

## 6. Train and test sets

Within each group, the model is built on older loans (the train set) and tested on newer loans it has never seen (the test set). This is how a bank works, using past loans to judge new applicants. For 60-month loans the test set is only the second half of 2013, because few loans were issued before then.

| Group | Train set | Test set |
|---|---|---|
| 36-month loans | 2007–2014, 335,558 loans, default rate 13.1% | 2015, 283,026 loans, default rate 14.9% |
| 60-month loans | Jan 2010 – Jun 2013, 32,837 loans, default rate 25.4% | Jul–Dec 2013, 22,132 loans, default rate 24.8% |

Everything the model learns comes from the train set. The test set is only used at the end, to check the model on new loans.

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

The ratio $p / (1 - p)$ is the odds of default. A PD of 20% means odds of 0.25, one default for every four repaid loans. So the score is the log of the odds, and each weight says how much the log odds change when that feature's WoE change.

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
3. Take the feature with the lowest score.
4. Add it to $S$ if it improves the score in all five parts, and go back to step 2.
5. Otherwise stop.

The rule in step 4 can be written as

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

The name comes from its effect on the weight. Only the part of feature $j$ that the others cannot explain helps to pin down its weight, and that part is a share $1 - R_j^2$ of its variation. 

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

The third is whether the PDs match what happened at every level of risk. The loans are sorted by PD and cut into 10 groups of equal size. For each group $g$ with $n_g$ loans, the average PD is compared with the share of loans that defaulted.

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

To check this, the test loans are sorted by PD and cut into 10 groups with the same number of loans in each, and in each group the average PD is compared with the share of loans that defaulted, as in Section 8.10.

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