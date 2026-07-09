import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Page Config
st.set_page_config(page_title="Sales Dashboard", layout="wide")
st.title("📊 Retail Sales Intelligence Dashboard")

# 2. Data Loading & Cleaning
@st.cache_data
def load_data():
    df = pd.read_csv('mock_dataset.csv')
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('&', 'and').str.replace('-', '_')
    df['order_date'] = pd.to_datetime(df['order_date'])
    return df

df = load_data()

# 3. Sidebar Filters
st.sidebar.header("Dashboard Filters")
category = st.sidebar.multiselect("Select Product Category", options=df['product_category'].unique(), default=df['product_category'].unique())
filtered_df = df[df['product_category'].isin(category)]

# 4. Visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader("Monthly Sales Trend")
    monthly_sales = filtered_df.groupby(filtered_df['order_date'].dt.to_period('M'))['sales'].sum().astype(float)
    st.line_chart(monthly_sales)

with col2:
    st.subheader("Profit Distribution by Category")
    fig, ax = plt.subplots()
    sns.boxplot(x='product_category', y='profit', data=filtered_df, ax=ax)
    st.pyplot(fig)

# 5. Raw Data View
if st.checkbox("Show Raw Data"):
    st.write(filtered_df)