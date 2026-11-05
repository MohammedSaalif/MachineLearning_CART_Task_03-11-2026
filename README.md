# CART Classification – Supermarket Promo Purchase Prediction

## What this project is about
A supermarket puts up a promo poster for a product. I built a **CART (Classification and Regression Tree)** model to predict whether a customer **buys the promoted product** (Yes / No).

## Dataset (created by me)
- 30 customers, 5 columns
- **Input features:** `Customer_Type` (Regular / New / Occasional), `Discount_Level` (Low / Medium / High), `Poster_Noticed` (Yes / No), `Visit_Day` (Weekday / Weekend)
- **Target:** `Buys_Product` (18 Yes, 12 No)

## What I did, step by step
1. **Explored the data** – checked structure, unique values, Yes/No counts, total records, target and features.
2. **Calculated the parent Gini impurity** – the impurity of the data before any split.
3. **Checked every possible binary split** – for each one I calculated the left and right Gini, the weighted Gini and the Gini reduction. The split with the highest reduction (`Visit_Day`) became the root.
4. **Encoded the data** – used one-hot encoding so scikit-learn can use the categories.
5. **Split into train and test** – 70% training, 30% testing, stratified.
6. **Built the full tree** – no limits, so it fits the training data fully and may overfit. Displayed as a plot and as text rules.
7. **Pruned the tree in two ways**
   - *Pre-pruning:* limited growth with `max_depth=3` and `min_samples_leaf=2`.
   - *Post-pruning:* grew the full tree, then cut weak branches with cost-complexity pruning, picking the `ccp_alpha` by cross-validation.
8. **Compared the three trees** – depth, leaves, train accuracy, test accuracy, cross-validation accuracy, confusion matrix and classification report.
9. **Checked feature importance** and **predicted a new customer**.

## Key results
| Model | Depth | Leaves |
|---|---|---|
| Full tree | 5 | 8 |
| Pre-pruned tree | 3 | 5 |
| Post-pruned tree | 2 | 3 |

- The pruned trees are smaller and easier to read.
- The post-pruned tree had the best cross-validation accuracy (about 0.83), against about 0.70 for the full tree.
- With only 30 rows, accuracy values change a lot between splits, so this is a learning demonstration, not a production benchmark.

## How to run
1. Open `CART_Supermarket_Promo_Assignment.ipynb` in Jupyter or Google Colab.
2. Click **Run All**.
3. Needs: `pandas`, `numpy`, `matplotlib`, `scikit-learn`.
