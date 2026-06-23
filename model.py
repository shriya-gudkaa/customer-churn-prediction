import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load Data
df = pd.read_csv('data.csv')

print(df.describe())
print(df.info())
print(df.head())
print(df.shape)
print(df.isnull().sum())

# 2. Drop unnecessary columns
df.drop(['Country'], axis=1, inplace=True)

#3. Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Customer_Service_Calls'] = df['Customer_Service_Calls'].fillna(0)
df['Wishlist_Items'] = df['Wishlist_Items'].fillna(0)
df['Product_Reviews_Written'] = df['Product_Reviews_Written'].fillna(0)

remaining_fills = {
    'Session_Duration_Avg': df['Session_Duration_Avg'].median(),
    'Pages_Per_Session': df['Pages_Per_Session'].median(),
    'Days_Since_Last_Purchase': df['Days_Since_Last_Purchase'].median(),
    'Discount_Usage_Rate': df['Discount_Usage_Rate'].median(),
    'Returns_Rate': df['Returns_Rate'].median(),
    'Email_Open_Rate': df['Email_Open_Rate'].median(),
    'Social_Media_Engagement_Score': df['Social_Media_Engagement_Score'].median(),
    'Mobile_App_Usage': df['Mobile_App_Usage'].median(),
    'Payment_Method_Diversity': 1,
    'Credit_Balance': 0
}
df.fillna(value=remaining_fills, inplace=True)
print(df.isnull().sum())

#4. Encode categorical columns
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Signup_Quarter'] = le.fit_transform(df['Signup_Quarter'])
df = pd.get_dummies(df, columns=['City'], drop_first=True, dtype=int)

print(df.dtypes)
print(df.head())

#5. Feature scaling
features = [
    'Age', 'Membership_Years', 'Login_Frequency', 'Session_Duration_Avg',
    'Pages_Per_Session', 'Cart_Abandonment_Rate', 'Wishlist_Items',
    'Total_Purchases', 'Average_Order_Value', 'Days_Since_Last_Purchase',
    'Discount_Usage_Rate', 'Returns_Rate', 'Email_Open_Rate',
    'Customer_Service_Calls', 'Product_Reviews_Written',
    'Social_Media_Engagement_Score', 'Mobile_App_Usage',
    'Payment_Method_Diversity', 'Lifetime_Value', 'Credit_Balance'
]

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])
print(df_scaled[features].describe())

# 6. Train-test split
X = df_scaled.drop(columns=['Churned'])
y = df_scaled['Churned']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Logistic Regression Report")
print(classification_report(y_test, y_pred))

# 8. Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Report")
print(classification_report(y_test, y_pred_rf))

# 9. Confusion Matrix - Logistic Regression
conf_matrix = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = conf_matrix.ravel()

custom_cm = [
    [TP, FN],
    [FP, TN]
]

plt.figure(figsize=(6, 5))
sns.heatmap(
    custom_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Churned", "Not Churned"],
    yticklabels=["Churned", "Not Churned"]
)

plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 10. Confusion Matrix - Random Forest 
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

TN, FP, FN, TP = conf_matrix_rf.ravel()

custom_cm_rf = [
    [TP, FN],
    [FP, TN]
]

plt.figure(figsize=(6, 5))
sns.heatmap(
    custom_cm_rf,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Churned", "Not Churned"],
    yticklabels=["Churned", "Not Churned"]
)

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
