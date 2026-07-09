import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# Load dataset
df = pd.read_csv('mock_dataset.csv')
numerical_cols = ['sales', 'profit', 'shipping_cost', 'unitprice']

# Basic clean of numerical columns (fill NAs) for outlier detection
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

# 1. IQR Method
def get_iqr_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

# 2. Z-Score Method
def get_zscore_outliers(data, column, threshold=3):
    z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
    return data[z_scores > threshold]

# 3. Isolation Forest
def get_iso_forest_outliers(data, columns):
    iso = IsolationForest(contamination=0.05, random_state=42)
    preds = iso.fit_predict(data[columns])
    return data[preds == -1]

# Comparison and Automatic Decision Logic
# Define heuristic:
# If outliers < 1%, Keep (do not disrupt data).
# If 1% < outliers < 5%, Cap (limit extremes).
# If outliers > 5%, Winsorize (reduce sensitivity).

results = {}
for col in numerical_cols:
    iqr_out = get_iqr_outliers(df, col)
    z_out = get_zscore_outliers(df, col)
    
    # Store outlier counts
    results[col] = {
        'IQR': len(iqr_out),
        'Z-Score': len(z_out),
        'Percentage': (len(iqr_out) / len(df)) * 100
    }
    
    # Decision logic
    pct = results[col]['Percentage']
    if pct < 1:
        results[col]['Decision'] = 'Keep'
    elif pct < 5:
        results[col]['Decision'] = 'Cap'
    else:
        results[col]['Decision'] = 'Winsorize'

print("Outlier Analysis Results:")
print(pd.DataFrame(results).T)

# Visualizations
plt.figure(figsize=(15, 10))

# Boxplot
plt.subplot(2, 2, 1)
sns.boxplot(data=df[numerical_cols])
plt.title("Boxplot of Numerical Features")

# Histograms
plt.subplot(2, 2, 2)
sns.histplot(df['sales'], kde=True)
plt.title("Distribution of Sales")

# Scatterplot
plt.subplot(2, 2, 3)
sns.scatterplot(x='sales', y='profit', data=df)
plt.title("Sales vs Profit")

plt.tight_layout()
plt.savefig('outlier_analysis.png')