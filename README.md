# Customer Churn Prediction 🛒

A supervised machine learning project that predicts customer churn for an e-commerce platform, with a focus on data preprocessing, model evaluation, and optimizing for business-relevant metrics rather than accuracy alone.

---

## Project Overview

This project analyzes ~50,000 e-commerce customer records to predict whether a customer is likely to churn (leave the platform). Beyond just building a model, the focus here is on understanding *why a metric like accuracy can be misleading* in a churn context, and how to deliberately tune a model toward the metric that actually matters for the business problem — Recall.

---

## Data Cleaning

The raw dataset contained missing values across most numerical columns. Each column was handled based on what made the most sense for that specific feature, rather than applying one blanket strategy:

- **Age** — filled with the column mean
- **Customer_Service_Calls, Wishlist_Items, Product_Reviews_Written** — filled with `0`, since a missing value here most likely means the customer had no activity in that category
- **Session_Duration_Avg, Pages_Per_Session, Days_Since_Last_Purchase, Discount_Usage_Rate, Returns_Rate, Email_Open_Rate, Social_Media_Engagement_Score, Mobile_App_Usage** — filled with the column median, to avoid the influence of outliers on the average
- **Payment_Method_Diversity** — filled with `1`, a safe assumption since these customers already had recorded purchases
- **Credit_Balance** — filled with `0`, assuming a missing value means no balance on file

The `Country` column was dropped, as it added high cardinality without a clear, consistent relationship to churn.

---

## Label Encoding

Categorical columns were converted into numerical form so they could be used by the models:

- **Gender** and **Signup_Quarter** — converted using `LabelEncoder`
- **City** — converted using one-hot encoding (`pd.get_dummies`), since it had many unique values with no natural order, making `LabelEncoder` unsuitable

---

## Feature Scaling

All numerical features (20 columns, including engagement metrics, purchase behavior, and financial values) were standardized using `StandardScaler`.

This step was necessary because features were on very different scales — for example, `Lifetime_Value` ranged into the thousands, while `Returns_Rate` ranged from 0–100. Without scaling, models that rely on distance or gradient-based optimization would have been disproportionately influenced by larger-magnitude features.

---

## Model Training & Comparison

Two supervised models were trained on the cleaned, encoded, and scaled data: **Logistic Regression** and **Random Forest**.

Logistic Regression struggled to identify churned customers, achieving a recall of only **42%** for the churned class — meaning it missed the majority of customers who actually left, despite a respectable overall accuracy of **78%**.

Random Forest performed considerably better across the board, even before further tuning:

| Metric | Logistic Regression (Churned) | Random Forest (Churned) |
|---|---|---|
| Precision | 68% | 76% |
| Recall | 42% | 85% |
| F1-Score | 52% | 80% |
| **Accuracy (overall)** | **78%** | **88%** |

This comparison made it clear that Random Forest was the better choice for this problem, particularly on the metric that matters most for churn — Recall.

---

## Custom Confusion Matrix

Rather than relying on `confusion_matrix`'s default class ordering, a custom matrix layout was built so that the **Churned class is shown first**, in a more intuitive format:

|               | Predicted: Churned | Predicted: Not Churned |
|---------------|:------------------:|:-----------------------:|
| **Actual: Churned**     | TP | FN |
| **Actual: Not Churned** | FP | TN |

This was done by unpacking sklearn's default `.ravel()` output (`TN, FP, FN, TP`) and remapping it into this clearer layout for both models, making it immediately obvious how many actual churners were being missed.

---

## Focusing on False Negatives

Despite Random Forest's strong baseline performance, a closer look at the confusion matrix revealed the real issue: a meaningful share of actual churners were still being misclassified as "Not Churned." These are **False Negatives (FN)**, and in this context, they are the most costly type of error.

**Why is a False Negative bad here?**
- The customer is actually about to leave.
- The model says everything is fine.
- The company takes no retention action.
- The customer leaves, and revenue is lost permanently.

This means that in churn prediction, **Recall for the Churned class is often more important than overall accuracy.** A model that looks accurate on paper but quietly misses real churners gives a business false confidence — the cost of a missed churner (lost customer, lost revenue) is significantly higher than the cost of a false alarm (an unnecessary retention offer sent to a loyal customer).

---

## Improving Recall Through Model Tuning

To address this, the Random Forest model was re-tuned with the following configuration:

```python
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
```

This combination of class balancing and depth/split tuning further improved the model's ability to catch churned customers — recall for the churned class reached **85%**, a substantial reduction in missed churn cases compared to the untuned baseline. This came with a trade-off: precision for the churned class settled at **76%**, meaning the model now flags more loyal customers as "at risk" than a stricter model would. This trade-off was accepted deliberately — in a churn-prevention context, a false alarm (an unnecessary discount sent to a loyal customer) is a low-cost mistake, while a missed churner is a high-cost, often irreversible one.

It's also worth noting that `class_weight='balanced'` and the depth/split tuning were tested individually as well — neither alone produced this level of improvement. The gain only appeared when both were applied together, suggesting that unconstrained, deeper trees were initially overfitting toward the majority class.

---

## Tech Stack

- **Python**
- **Pandas** – data loading and preprocessing
- **Scikit-learn** – encoding, scaling, model training, and evaluation
- **Matplotlib & Seaborn** – confusion matrix visualization

---

## How to Run Locally

```bash
git clone https://github.com/shriya-gudkaa/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
python model.py
```

---

## Project Structure

```
customer-churn-prediction/
├── data.csv             # Dataset
├── model.py             # Preprocessing, model training, evaluation & tuning
├── requirements.txt     # Project dependencies
└── README.md
```

---

## What I Learned

- How to make deliberate, justified decisions when handling missing data, rather than applying one blanket method to every column
- Why accuracy alone can be a misleading metric, especially with an imbalanced target variable
- The real-world cost asymmetry between False Positives and False Negatives, and how that should guide which metric to optimize for
- How to isolate the effect of multiple hyperparameter changes by testing them individually before combining them — rather than assuming which change caused an improvement

---

## Author
**Shriya Gudka**
