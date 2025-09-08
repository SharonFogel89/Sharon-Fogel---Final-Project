# Sharon-Fogel---Final-Project
Customer Analytics for Growth, Retention, and Risk

This dataset has 5,000 customers with attributes like demographics, behavior, loyalty, churn risk, and even fraud flags.


Problem Statement

The company has a large customer base across different countries, but it lacks insights into customer behavior, loyalty, and risks. Without these insights, it’s difficult to:

improve retention,

detect potential churn,

and optimize marketing efforts.

This project aims to analyze customer data to identify key behavioral drivers, predict churn, and uncover opportunities for increasing customer lifetime value.

Project Objectives

Customer Profiling: Segment customers based on demographics, purchase patterns, and preferences.

Retention & Churn: Identify factors contributing to churn risk and loyalty scores.

Fraud Detection: Explore fraud patterns and potential indicators.

Engagement Analysis: Evaluate customer responsiveness (e.g., email open rate) by segment.

Actionable Insights: Provide recommendations for targeted marketing, retention strategies, and fraud prevention.

5 Key Business Questions

Customer Segmentation:

What are the main customer segments by demographics, loyalty, and preferred categories?

Revenue Drivers:

Which factors (e.g., age, loyalty score, order behavior) have the strongest impact on average order value and total orders?

Churn Risk:

What customer characteristics are most associated with a high churn risk?

Can we predict which customers are likely to churn soon?

Fraudulent Behavior:

What patterns differentiate fraudulent from non-fraudulent customers?

Are fraud cases concentrated in specific regions, categories, or loyalty levels?

Customer Engagement:

How does email open rate vary by segment, and does it correlate with loyalty or churn risk?

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from google.colab import drive

# Load dataset
import pandas as pd
file_path = '/content/drive/MyDrive/Colab Notebooks/Final Project/customer_analytics_dataset.csv'
df = pd.read_csv(file_path)

#Preview
df.head()

#Data Understanding
print("\n📌 Column Names:")
print(df.columns.tolist())

#Data types - numeric vs categorical
print("\n📌 Data Types:")
print(df.dtypes)

#Missing values - important for data cleaning
print("\n📌 Missing Values:")
print(df.isnull().sum())

#Unique values for categorical variables (like gender, country, preferred_category)
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"{col}: {df[col].nunique()} unique values → {df[col].unique()[:10]}")

#Quick statistical summary (numeric) - min, max, mean, std, quartiles
print("\n📌 Statistical Summary (Numeric Variables):")
display(df.describe().T)

#Statistical summary (categorical) - most frequent value, counts, etc
print("\n📌 Statistical Summary (Categorical Variables):")
display(df.describe(include=['object']).T)

#Data Cleaning & Preparation
import pandas as pd
import numpy as np

#Convert data types IN PLACE
#Dates
df['customer_since'] = pd.to_datetime(df['customer_since'], errors='coerce')

#Categoricals
for col in ['gender', 'country', 'preferred_category']:
    if col in df.columns:
        df[col] = df[col].astype('category')

#IDs as string
if 'customer_id' in df.columns:
    df['customer_id'] = df['customer_id'].astype('string')

#Numeric safe coercions
num_cols = ['age', 'avg_order_value', 'total_orders', 'email_open_rate',
            'loyalty_score', 'churn_risk', 'last_purchase']
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

#Booleans from 0/1
if 'is_fraudulent' in df.columns:
    df['is_fraudulent'] = (
        pd.to_numeric(df['is_fraudulent'], errors='coerce')
          .fillna(0).astype(int).astype(bool)
    )

#Keep churn in [0, 1]
if 'churn_risk' in df.columns:
    df['churn_risk'] = df['churn_risk'].clip(lower=0, upper=1)

#Remove duplicates IN PLACE
if 'customer_id' in df.columns:
    df.drop_duplicates(subset=['customer_id'], keep='last', inplace=True)
else:
    df.drop_duplicates(keep='last', inplace=True)

#Handle missing values IN PLACE
#Categorical → "Unknown"
for col in ['gender', 'country', 'preferred_category']:
    if col in df.columns:
        # ensure the category exists before filling
        if not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype('category')
        df[col] = df[col].cat.add_categories(['Unknown']).fillna('Unknown')

def fill_with_group_median_inplace(df_, target, by):
    """Fill NA in target using group-median by 'by'; fallback to global median."""
    if target not in df_.columns or by not in df_.columns:
        return
    med = df_.groupby(by, dropna=False)[target].transform('median')
    df_.loc[df_[target].isna(), target] = med[df_[target].isna()]
    df_[target].fillna(df_[target].median(), inplace=True)

#Age by gender
if 'age' in df.columns and 'gender' in df.columns:
    fill_with_group_median_inplace(df, 'age', 'gender')

#AOV / orders / open rate by preferred_category
if 'avg_order_value' in df.columns and 'preferred_category' in df.columns:
    fill_with_group_median_inplace(df, 'avg_order_value', 'preferred_category')

if 'total_orders' in df.columns and 'preferred_category' in df.columns:
    fill_with_group_median_inplace(df, 'total_orders', 'preferred_category')
    df['total_orders'] = df['total_orders'].round().astype('Int64')

if 'email_open_rate' in df.columns and 'preferred_category' in df.columns:
    fill_with_group_median_inplace(df, 'email_open_rate', 'preferred_category')

#Global medians for the rest
if 'loyalty_score' in df.columns:
    df['loyalty_score'].fillna(df['loyalty_score'].median(), inplace=True)

if 'churn_risk' in df.columns:
    df['churn_risk'].fillna(df['churn_risk'].median(), inplace=True)

if 'customer_since' in df.columns:
    median_date = df['customer_since'].dropna().median()
    df['customer_since'].fillna(median_date, inplace=True)

if 'last_purchase' in df.columns:
    df['last_purchase'].fillna(df['last_purchase'].median(), inplace=True)

#Feature engineering IN PLACE
#Total spend
if set(['avg_order_value', 'total_orders']).issubset(df.columns):
    df['total_spend'] = df['avg_order_value'] * df['total_orders'].astype(float)

#Tenure (days since sign-up)
if 'customer_since' in df.columns:
    today = pd.Timestamp('today').normalize()
    df['tenure_days'] = (today - df['customer_since']).dt.days

#Last purchase date from "days since last purchase"
if 'last_purchase' in df.columns:
    today = pd.Timestamp('today').normalize()
    df['last_purchase_date'] = today - pd.to_timedelta(df['last_purchase'], unit='D')

#Sales quarter (prefer last_purchase_date; fallback to customer_since)
if 'last_purchase_date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['last_purchase_date']):
    df['sales_quarter'] = (
        df['last_purchase_date'].dt.year.astype('Int64').astype('string') +
        'Q' + df['last_purchase_date'].dt.quarter.astype('Int64').astype('string')
    )
elif 'customer_since' in df.columns:
    df['sales_quarter'] = (
        df['customer_since'].dt.year.astype('Int64').astype('string') +
        'Q' + df['customer_since'].dt.quarter.astype('Int64').astype('string')

#Age groups
if 'age' in df.columns:
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 24, 34, 44, 54, 64, 120],
        labels=['<25', '25-34', '35-44', '45-54', '55-64', '65+'],
        right=True, include_lowest=True
    ).astype('category')

#Engagement bands
if 'email_open_rate' in df.columns:
    df['engagement_band'] = pd.cut(
        df['email_open_rate'],
        bins=[-0.001, 10, 20, 40, 60, 100],
        labels=['0–10%', '10–20%', '20–40%', '40–60%', '60–100%']
    ).astype('category')

#Active in last 90 days
if 'last_purchase' in df.columns:
    df['is_active_90d'] = (df['last_purchase'] <= 90)

#Value segment (quartiles; fallback to binary)
    try:
        df['value_segment'] = pd.qcut(
            df['total_spend'].rank(method='first'),
            q=4, labels=['Low', 'Mid', 'High', 'Top']
        ).astype('category')
    except ValueError:
        df['value_segment'] = pd.cut(
            df['total_spend'],
            bins=[-np.inf, df['total_spend'].median(), np.inf],
            labels=['Low','High']
        ).astype('category')

#RFM-like scores
if 'last_purchase' in df.columns:
    df['r_score'] = pd.qcut(df['last_purchase'].rank(method='first'), 5, labels=[5,4,3,2,1]).astype(int)
if 'total_orders' in df.columns:
    df['f_score'] = pd.qcut(df['total_orders'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
if 'avg_order_value' in df.columns:
    df['m_score'] = pd.qcut(df['avg_order_value'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
if set(['r_score','f_score','m_score']).issubset(df.columns):
    df['rfm_score'] = df['r_score'] + df['f_score'] + df['m_score']

#Targeting flags
if 'churn_risk' in df.columns:
    df['is_high_churn_risk'] = df['churn_risk'] >= 0.5
if 'total_spend' in df.columns:
    df['is_high_value'] = df['total_spend'] >= df['total_spend'].median()

#Ensure engineered categoricals have the right dtype
for col in ['age_group', 'engagement_band', 'value_segment', 'sales_quarter']:
    if col in df.columns:
        df[col] = df[col].astype('category')

#Quick checks
print("✅ Final shape:", df.shape)
print("✅ Top missing after cleaning:\n", df.isna().sum().sort_values(ascending=False).head(15))
print("✅ Dtypes:\n", df.dtypes)

#Imports, style, and light prep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")  # clean default look

#Ensure date columns are parsed (safe even if already parsed)
if 'customer_since' in df.columns:
    df['customer_since'] = pd.to_datetime(df['customer_since'], errors='coerce')

#If you haven't engineered last_purchase_date yet, we can infer it from "days since last purchase" (proxy; per-customer not transactional)
if 'last_purchase' in df.columns and 'last_purchase_date' not in df.columns:
    today = pd.Timestamp('today').normalize()
    # Use a temporary series to avoid permanently modifying df if you prefer
    df['last_purchase_date'] = today - pd.to_timedelta(pd.to_numeric(df['last_purchase'], errors='coerce'), unit='D')

#Helpful proxy for monetary value if present
if {'avg_order_value', 'total_orders'}.issubset(df.columns) and 'total_spend' not in df.columns:
    df['total_spend'] = pd.to_numeric(df['avg_order_value'], errors='coerce') * pd.to_numeric(df['total_orders'], errors='coerce')

#Distribution plots for key numeric variables

#Choose common numeric columns automatically (filters out ids)
candidate_numeric = ['age','avg_order_value','total_orders','email_open_rate',
                     'loyalty_score','churn_risk','last_purchase','total_spend']
numeric_cols = [c for c in candidate_numeric if c in df.columns]

for col in numeric_cols:
    plt.figure(figsize=(7,4))
    sns.histplot(df[col].dropna(), kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

#Quick boxplots to spot outliers
for col in numeric_cols:
    plt.figure(figsize=(7,2.8))
    sns.boxplot(x=df[col].dropna())
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

#Bar charts for top categories

#Pick likely categorical fields (only those that exist)
cat_cols = [c for c in ['country','preferred_category','gender'] if c in df.columns]

TOP_N = 10  # adjust as needed

for col in cat_cols:
    counts = df[col].astype('object').fillna('Unknown').value_counts().head(TOP_N)
    plt.figure(figsize=(8,4))
    sns.barplot(x=counts.values, y=counts.index)
    plt.title(f"Top {TOP_N} {col} categories")
    plt.xlabel("Count")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

#Time series trends (if applicable)
#These are customer-level proxies (not transaction-level).
#New customers per month uses customer_since.
#New customers per month (signups)
if 'customer_since' in df.columns:
    monthly_signups = (
        df.dropna(subset=['customer_since'])
          .assign(month=df['customer_since'].dt.to_period('M').astype(str))
          .groupby('month', as_index=False)
          .size()
    )

#sort by month
    monthly_signups['month'] = pd.to_datetime(monthly_signups['month'])
    monthly_signups = monthly_signups.sort_values('month')

    plt.figure(figsize=(9,4))
    plt.plot(monthly_signups['month'], monthly_signups['size'], marker='o')
    plt.title("New customers per month")
    plt.xlabel("Month")
    plt.ylabel("Number of new customers")
    plt.tight_layout()
    plt.show()

#Spend proxy by last_purchase month
if {'last_purchase_date','total_spend'}.issubset(df.columns):
    monthly_spend_proxy = (
        df.dropna(subset=['last_purchase_date'])
          .assign(month=df['last_purchase_date'].dt.to_period('M').astype(str))
          .groupby('month', as_index=False)['total_spend'].sum()
    )
    monthly_spend_proxy['month'] = pd.to_datetime(monthly_spend_proxy['month'])
    monthly_spend_proxy = monthly_spend_proxy.sort_values('month')

    plt.figure(figsize=(9,4))
    plt.plot(monthly_spend_proxy['month'], monthly_spend_proxy['total_spend'], marker='o')
    plt.title("Total spend proxy by last purchase month")
    plt.xlabel("Month")
    plt.ylabel("Total spend (proxy)")
    plt.tight_layout()
    plt.show()

#Active customers proxy by month (last purchase within that month)
if 'last_purchase_date' in df.columns:
    monthly_active = (
        df.dropna(subset=['last_purchase_date'])
          .assign(month=df['last_purchase_date'].dt.to_period('M').astype(str))
          .groupby('month', as_index=False)
          .size()
    )
    monthly_active['month'] = pd.to_datetime(monthly_active['month'])
    monthly_active = monthly_active.sort_values('month')

    plt.figure(figsize=(9,4))
    plt.plot(monthly_active['month'], monthly_active['size'], marker='o')
    plt.title("Customers with a last purchase in each month (activity proxy)")
    plt.xlabel("Month")
    plt.ylabel("Customers active in month")
    plt.tight_layout()
    plt.show()

#Correlation heatmap for numeric variables
#Select numeric columns (keeps bool as 0/1)
num_df = df.select_dtypes(include=[np.number]).copy()

#Optional: drop id-like or degenerate columns if they slipped in
for drop_candidate in ['customer_id']:
    if drop_candidate in num_df.columns:
        num_df = num_df.drop(columns=[drop_candidate])

#Compute correlation (Pearson)
corr = num_df.corr(numeric_only=True)

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", square=True)
plt.title("Correlation Heatmap (numeric features)")
plt.tight_layout()
plt.show()

#Step 5 – Analysis / Modeling

#Prereqs & lightweight feature prep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.power import TTestIndPower

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.inspection import permutation_importance

#Gentle feature prep (safe if already done)

if 'customer_since' in df.columns:
    df['customer_since'] = pd.to_datetime(df['customer_since'], errors='coerce')

if {'avg_order_value','total_orders'}.issubset(df.columns) and 'total_spend' not in df.columns:
    df['total_spend'] = pd.to_numeric(df['avg_order_value'], errors='coerce') * \
                        pd.to_numeric(df['total_orders'], errors='coerce')

if 'last_purchase' in df.columns and 'last_purchase_date' not in df.columns:
    today = pd.Timestamp('today').normalize()
    df['last_purchase_date'] = today - pd.to_timedelta(pd.to_numeric(df['last_purchase'], errors='coerce'), unit='D')

#Optional binary “high churn” label for classification demos
if 'churn_risk' in df.columns and 'is_high_churn_risk' not in df.columns:
    df['is_high_churn_risk'] = (df['churn_risk'] >= 0.5).astype(int)

#Make sure booleans are ints for sklearn
for b in ['is_fraudulent', 'is_high_churn_risk']:
    if b in df.columns:
        df[b] = df[b].astype(int)

#A) Statistical tests (answer why/what’s different?) A1) Association between loyalty_score and churn_risk (Spearman)

#Good for monotonic relationships, robust to non-normal data.

if {'loyalty_score','churn_risk'}.issubset(df.columns):
    x = pd.to_numeric(df['loyalty_score'], errors='coerce')
    y = pd.to_numeric(df['churn_risk'], errors='coerce')
    mask = x.notna() & y.notna()
    rho, p = stats.spearmanr(x[mask], y[mask])
    print(f"Spearman rho(loyalty_score, churn_risk) = {rho:.3f}, p = {p:.3g}")
    print("Interpretation: negative rho → higher loyalty associates with lower churn risk (good).")

#A2) Do categories differ in avg_order_value? (ANOVA + Tukey)
if {'avg_order_value','preferred_category'}.issubset(df.columns):
    tmp = df[['avg_order_value','preferred_category']].dropna()
    # One-way ANOVA
    groups = [g['avg_order_value'].values for _, g in tmp.groupby('preferred_category')]
    F, p = stats.f_oneway(*groups)
    print(f"ANOVA AOV ~ preferred_category: F={F:.2f}, p={p:.3g}")
    print("If p<0.05: at least one category’s mean AOV differs; see Tukey below.")

#Post-hoc Tukey HSD
tukey = pairwise_tukeyhsd(endog=tmp['avg_order_value'], groups=tmp['preferred_category'], alpha=0.05)
print(tukey.summary())

#A3) Does avg_order_value differ by gender? (t-test + Mann–Whitney)
if {'avg_order_value','gender'}.issubset(df.columns):
    sub = df[['avg_order_value','gender']].dropna()
    # Keep two groups (e.g., Female vs Male); adapt if you want to compare Other separately
    g_keep = sub['gender'].isin(['Female','Male'])
    sub = sub[g_keep]
    g1 = sub.loc[sub['gender']=='Female','avg_order_value'].values
    g2 = sub.loc[sub['gender']=='Male','avg_order_value'].values

    t,p = stats.ttest_ind(g1, g2, equal_var=False)  # Welch’s t-test
    print(f"Welch t-test AOV Female vs Male: t={t:.2f}, p={p:.3g}")

    u,p_u = stats.mannwhitneyu(g1, g2, alternative='two-sided')
    print(f"Mann–Whitney U (robust): U={u:.0f}, p={p_u:.3g}")
    print("Interpretation: p<0.05 → significant difference in median/mean AOV between genders.")

#A4) Are fraud rates different across countries? (Chi-square)
if {'is_fraudulent','country'}.issubset(df.columns):
    top_c = df['country'].value_counts().head(10).index
    sub = df[df['country'].isin(top_c)].copy()
    ct = pd.crosstab(sub['country'], sub['is_fraudulent'])
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    print(f"Chi-square(country vs fraud): chi2={chi2:.2f}, dof={dof}, p={p:.3g}")
    print("Interpretation: p<0.05 → fraud rate depends on country. Inspect rows with higher fraud proportion.")

#A5) Explanatory OLS: avg_order_value ~ age + total_orders + C(country) + C(preferred_category)
cols_needed = {'avg_order_value','age','total_orders','country','preferred_category'}
if cols_needed.issubset(df.columns):
    sub = df[list(cols_needed)].dropna()
    model = smf.ols("avg_order_value ~ age + total_orders + C(country) + C(preferred_category)", data=sub).fit()
    print(model.summary())
    print("Interpretation: significant positive coefficients indicate drivers of higher AOV (hold others constant).")

#B) Machine learning models (predict who/what/when)

#B1) Churn classification: who is high churn risk?

#Target: is_high_churn_risk (derived from churn_risk >= 0.5)

#Features: demographics, behavior, engagement, preference

#Metrics: ROC-AUC (ranking), PR-AUC (imbalanced), confusion matrix at 0.5 threshold

target = 'is_high_churn_risk'
num_feats = [c for c in ['age','avg_order_value','total_orders','email_open_rate','loyalty_score','last_purchase'] if c in df.columns]
cat_feats = [c for c in ['gender','country','preferred_category'] if c in df.columns]
feats = num_feats + cat_feats

if target in df.columns and len(feats) > 0:
    data = df[feats + [target]].dropna()
    X = data[feats]
    y = data[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                   ('scaler', StandardScaler())]), num_feats),
            ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                   ('oh', OneHotEncoder(handle_unknown='ignore'))]), cat_feats)
        ]
    )

    clf = LogisticRegression(max_iter=200, class_weight='balanced')  # simple, explainable baseline
    pipe = Pipeline(steps=[('pre', pre), ('model', clf)])
    pipe.fit(X_train, y_train)

#Evaluation
proba = pipe.predict_proba(X_test)[:,1]
preds = (proba >= 0.5).astype(int)
auc = roc_auc_score(y_test, proba)
ap = average_precision_score(y_test, proba)
print(f"Churn ROC-AUC: {auc:.3f} | PR-AUC: {ap:.3f}")
print(classification_report(y_test, preds, digits=3))

#Curves
fpr, tpr, _ = roc_curve(y_test, proba)
prec, rec, _ = precision_recall_curve(y_test, proba)
fig, ax = plt.subplots(1,2, figsize=(10,4))
ax[0].plot(fpr, tpr); ax[0].set_title("ROC"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
ax[1].plot(rec, prec); ax[1].set_title("Precision–Recall"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
plt.tight_layout(); plt.show()

#Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, preds)
plt.title("Churn: Confusion Matrix (thr=0.5)"); plt.show()

#B2) Fraud detection (often imbalanced) Target: is_fraudulent

#Model: RandomForestClassifier with class_weight='balanced'

#Metrics: ROC-AUC, PR-AUC, confusion matrix

target = 'is_fraudulent'
num_feats = [c for c in ['age','avg_order_value','total_orders','email_open_rate','loyalty_score','last_purchase','churn_risk'] if c in df.columns]
cat_feats = [c for c in ['gender','country','preferred_category'] if c in df.columns]
feats = num_feats + cat_feats

if target in df.columns and len(feats) > 0:
    data = df[feats + [target]].dropna()
    X = data[feats]
    y = data[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))]), num_feats),
            ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                   ('oh', OneHotEncoder(handle_unknown='ignore'))]), cat_feats)
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=42, class_weight='balanced'
    )

    pipe = Pipeline(steps=[('pre', pre), ('model', clf)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:,1]
    preds = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    print(f"Fraud ROC-AUC: {auc:.3f} | PR-AUC: {ap:.3f}")
    print(classification_report(y_test, preds, digits=3))
    ConfusionMatrixDisplay.from_predictions(y_test, preds)
    plt.title("Fraud: Confusion Matrix (thr=0.5)"); plt.show()

    # Which features matter?
    X_test_transformed = pipe.named_steps['pre'].transform(X_test)
    r = permutation_importance(pipe.named_steps['model'], X_test_transformed, y_test, n_repeats=10, random_state=42)
    # Get feature names after preprocessing
    feature_names_out = pipe.named_steps['pre'].get_feature_names_out(feats)
    imp = pd.Series(r.importances_mean, index=feature_names_out)
    print("\nTop 15 features by permutation importance:")
    print(imp.sort_values(ascending=False).head(15))

#B3) Customer value regression (predict total_spend)

#Target: total_spend

#Model: Gradient Boosting (strong baseline)

#Metrics: MAE, RMSE, R²; permutation importance for drivers

target = 'total_spend'
num_feats = [c for c in ['age','avg_order_value','total_orders','email_open_rate','loyalty_score','last_purchase','churn_risk'] if c in df.columns]
cat_feats = [c for c in ['gender','country','preferred_category'] if c in df.columns]
feats = num_feats + cat_feats

if target in df.columns and len(feats) > 0:
    data = df[feats + [target]].dropna()
    X = data[feats]
    y = data[target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pre = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                   ('scaler', StandardScaler())]), num_feats),
            ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                   ('oh', OneHotEncoder(handle_unknown='ignore'))]), cat_feats)
        ]
    )

    reg = GradientBoostingRegressor(random_state=42)
    pipe = Pipeline(steps=[('pre', pre), ('model', reg)])
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)
    print(f"Value regression → MAE: {mae:,.2f} | RMSE: {rmse:,.2f} | R²: {r2:.3f}")

    # Cross-val R² for stability
    cv_r2 = cross_val_score(pipe, X, y, cv=5, scoring='r2').mean()
    print(f"5-fold CV R²: {cv_r2:.3f}")

    # Drivers via permutation importance
    X_test_transformed = pipe.named_steps['pre'].transform(X_test)
    r = permutation_importance(pipe.named_steps['model'], X_test_transformed, y_test, n_repeats=10, random_state=42)
    imp = pd.Series(r.importances_mean, index=pipe.named_steps['pre'].get_feature_names_out())
    print("\nTop 15 drivers of total_spend:")
    print(imp.sort_values(ascending=False).head(15))


#Conclusion:
Quick “turn insights into action” examples

Churn: sort customers by predicted probability and trigger save offers for the top N.

Fraud: route high-probability cases for manual review before fulfillment.

Value: focus campaigns on features driving spend (e.g., categories/regions with positive importances).

If you want, I can package this into a single run_analysis(df) function that executes all selected tests/models and returns tidy result tables and charts.

Insights & Interpretation

Executive Summary Customer & Revenue Snapshot

Scale: 5,000 customers across 10 countries and 5 categories.

Revenue proxy (AOV × Orders):  5.19Mtotal;AOV≈ 108.44, orders ≈ 10.0/customer.

Top revenue contributors:

Categories: Electronics ( 1.08M),Home( 1.04M), Beauty ( 1.03M),Sports( 1.02M), Fashion ($1.02M) — fairly balanced, with Electronics slightly ahead.

Countries: Brazil ( 568k)leadsdespiteonly504customers;USA( 553k), India ( 547k),Germany( 519k), Australia ($517k).

High-value orders: Top 5% of AOV >  239;maxAOV 555 (about 4.8% of customers above 95th percentile).

Engagement & Acquisition

Email engagement: Overall open rate ≈ 50.7% (5% missing). Highest by country: India (53%), Japan (52.7%); lowest: Germany (48.5%). By category: Beauty/Fashion (51.6%) slightly higher; Electronics (50.4%) slightly lower.

New customer trend: Signups down ~4.1% in the last 12 months vs the prior 12 months → mild headwind in acquisition.

Retention & Churn

Recent activity: Only 25.5% purchased within the last 90 days → sizable re-activation opportunity.

Loyalty vs churn: No meaningful relationship (Spearman ρ ≈ 0.00).

Open rate vs churn: Near zero (ρ ≈ 0.03).

Value vs churn: Average churn risk ~0.28–0.29 across value quintiles (flat). Implication: Current churn scoring isn’t reflecting behavioral signals; revisit features/definition.

Fraud Risk

Overall fraud rate: 2.58%.

By country: Highest in Japan (3.8%) and Brazil (3.8%); lowest in France (1.4%).

By category: Electronics/Home/Fashion (~3.2%) higher than Beauty (1.5%) and Sports (1.8%).

By value: Elevated in mid-value segments (Q2–Q4) vs lowest/highest.

What Drives Spend (fast wins)

Total spend correlates very strongly with AOV (ρ ≈ 0.885) and moderately with order count (ρ ≈ 0.443). Translation: Lifting AOV (bundles, cross-sell, premium add-ons) moves revenue more than nudging frequency by a small margin.

Anomalies & Patterns Worth Attention

AOV tail: A small but meaningful high-AOV tail (> $239). Review margin/profitability and potential fraud signals in this tail (esp. Electronics/Home/Fashion and Brazil/Japan).

Engagement variation: Countries with solid spend but lower open rates (e.g., Germany) look primed for creative/subject-line testing.

Acquisition softness: Slight 12-month decline suggests reallocating spend toward top-converting geos/categories.

Recommended Actions Revenue Growth

AOV playbook:

Introduce bundled offers/threshold discounts (“free shipping over $X”).

Cross-sell complementary items in Electronics and Home, where revenue is highest.

Geo targeting:

Double down in Brazil/USA/India; test engagement lifts in Germany to unlock underperformance.

Retention

Re-activation program: Target the ~74.5% inactive (no purchase in 90d) with:

Win-back series (tiered incentives), category-specific recommendations, and reminders tied to preferred_category.

Churn model refresh: Current score isn’t predictive. Rebuild with features like recency, frequency, value, category mix, email engagement, and country; validate on ROC-AUC & PR-AUC and calibrate thresholds to budget.

Fraud

Tiered review rules:

Enhanced checks for Japan/Brazil and Electronics/Home/Fashion at mid-value ranges.

Keep customer friction low for Beauty/Sports and France where rates are lower.

Marketing Efficiency

Creative tests by geo:

Optimize subject lines/send times for Germany; replicate India/Japan tactics across similar cohorts.

Fill data gaps: Reduce email_open_rate missingness (5%) to improve targeting reliability.
