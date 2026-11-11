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