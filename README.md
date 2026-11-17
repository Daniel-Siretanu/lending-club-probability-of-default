# LendingClub probability of default

When someone applies for a loan, the lender needs to know how likely they are to pay it back. This project builds a model that answers that, using real loans from LendingClub, a US lending platform. It then shows how a lender could use the model to decide who gets a loan.

Everything is explained step by step, with the maths, in [docs/methodology.md](docs/methodology.md).

## What's inside

The model is a credit scorecard, the kind banks have used for decades. Each borrower gets points for things like their credit score, income and recent credit applications, and the points add up to a score. A higher score means a safer borrower.

Along the way, the project:

- uses only what was known on the day each loan was approved, so nothing from the future leaks in;
- lets the data decide which features matter, and checks that the choice holds up on different samples;
- tests the model on newer loans it has never seen;
- shows the trade-off a lender faces when choosing who to approve;
- explains, for any borrower, why they got their score;
- compares the scorecard with a machine-learning model, to see how much the simple approach gives up.

## How well it works

| Loans | Scorecard | Machine learning |
|---|---|---|
| 36-month | 0.307 | 0.341 |
| 60-month | 0.214 | 0.224 |

These numbers (the Gini) measure how well each model ranks borrowers from safest to riskiest, from 0 for guessing to 1 for perfect. The scorecard does almost as well as the machine-learning model, and unlike it, every decision can be explained.

In practice, turning away the riskiest 30% of applicants for 36-month loans would have avoided nearly half of all defaults.

## The notebooks

| Notebook | What happens there |
|---|---|
| [1_explore_data](notebooks/1_explore_data.ipynb) | choosing the loans and columns, and splitting old loans from new ones |
| [2_binning_woe](notebooks/2_binning_woe.ipynb) | grouping each feature into bins |
| [3_model](notebooks/3_model.ipynb) | choosing the features and building the model |
| [4_test](notebooks/4_test.ipynb) | testing on new loans and deciding who to lend to |
| [5_scorecard](notebooks/5_scorecard.ipynb) | turning the model into points and explaining scores |
| [6_challenger](notebooks/6_challenger.ipynb) | the comparison with machine learning |

The finished scorecards are in the [reports](reports) folder.

## Running it yourself

1. Download `accepted_2007_to_2018Q4.csv.gz` from the [LendingClub dataset on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club) and put it in `data/raw/`.
2. Set up the environment:
   ```bash
   conda create -n pd-model python=3.11
   conda activate pd-model
   pip install -r requirements.txt
   ```
3. Run the notebooks in order.

## What to keep in mind

The data only includes loans LendingClub approved, all from one US lender between 2007 and 2015. Default rates also change over time, so a model like this needs regular updating. The full list of limits is in Section 13 of the methodology.