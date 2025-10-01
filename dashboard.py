import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="GIC Internal Audit - Anomaly Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6 0%, #e8eaf0 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
    }
    /* Professional icons */
    .stMarkdown h2, .stMarkdown h3 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================
@st.cache_data
def load_and_prepare_data(file_path):
    """Load and prepare transaction data with anomaly detection"""
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Remove negative amounts only (keep zeros and valid values)
    df = df[df['amount'] >= 0].copy()
    
    # Clean and prepare dates - strip time component since all times are the same
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['date'] = pd.to_datetime(df['date'])  # Convert back to datetime for processing
    
    # Derived features
    df['day_of_week'] = df['date'].dt.day_name()
    df['day_of_week_num'] = df['date'].dt.dayofweek  
    df['month'] = df['date'].dt.to_period('M').astype(str)
    df['month_name'] = df['date'].dt.strftime('%b %Y')
    
    # Isolation Forest for anomaly detection
    features = pd.get_dummies(df[['amount', 'category', 'payment_method', 'account_type', 'transaction_type']], 
                                     columns=['category', 'payment_method', 'account_type', 'transaction_type'])
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    iso_forest = IsolationForest(contamination=0.15, random_state=42, n_estimators=100)
    df['anomaly_score'] = iso_forest.fit_predict(features_scaled)
    df['is_anomaly'] = (df['anomaly_score'] == -1).astype(int)
    
    # Get anomaly decision scores (more negative = more anomalous)
    df['anomaly_decision_score'] = iso_forest.decision_function(features_scaled)
    
    # Calculate risk score (0-100) based on anomaly decision score
    # Normalize decision scores to 0-100 range
    min_score = df['anomaly_decision_score'].min()
    max_score = df['anomaly_decision_score'].max()
    df['risk_score'] = ((df['anomaly_decision_score'] - min_score) / (max_score - min_score) * -100 + 100).clip(0, 100)
    
    # Risk level categorization
    df['risk_level'] = pd.cut(df['risk_score'], 
                               bins=[0, 40, 70, 100], 
                               labels=['Low', 'Medium', 'High'],
                               include_lowest=True)
    
    return df

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Header
st.markdown('<div class="main-header">GIC Internal Audit - Anomalous Transactions Dashboard</div>', 
            unsafe_allow_html=True)

# Load data
try:
    df = load_and_prepare_data('financial_transactions.csv')
    
except FileNotFoundError:
    st.error("⚠️ Please upload 'financial_transactions.csv' to proceed.")
    st.stop()

# ============================================================================
# SIDEBAR FILTERS - LEFT PANEL
# ============================================================================
st.sidebar.header("Dashboard Controls")

# Date range filter
min_date = df['date'].min().date()
max_date = df['date'].max().date()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Multi-select filters
categories = st.sidebar.multiselect(
    "Category",
    options=sorted(df['category'].unique()),
    default=sorted(df['category'].unique())
)

payment_methods = st.sidebar.multiselect(
    "Payment Method",
    options=sorted(df['payment_method'].unique()),
    default=sorted(df['payment_method'].unique())
)

account_types = st.sidebar.multiselect(
    "Account Type",
    options=sorted(df['account_type'].unique()),
    default=sorted(df['account_type'].unique())
)

transaction_types = st.sidebar.multiselect(
    "Transaction Type",
    options=sorted(df['transaction_type'].unique()),
    default=sorted(df['transaction_type'].unique())
)

# Risk level filter
risk_levels = st.sidebar.multiselect(
    "Risk Level",
    options=['Low', 'Medium', 'High'],
    default=['Low', 'Medium', 'High']
)

# Amount range filter
amount_min = float(df['amount'].min())
amount_max = float(df['amount'].max())
amount_range = st.sidebar.slider(
    "Amount Range ($)",
    min_value=amount_min,
    max_value=amount_max,
    value=(amount_min, amount_max)
)

# Anomaly detection method selector
st.sidebar.subheader("Anomaly Detection Settings")
st.sidebar.info("Using **Isolation Forest** for anomaly detection")

# Show only anomalies toggle
show_only_anomalies = st.sidebar.checkbox("Show Only Anomalies", value=False)

st.sidebar.markdown("---")
st.sidebar.info("**Data Analytics Team**\n\nInternal Audit Department\n\nGIC - Government of Singapore Investment Corporation")

# ============================================================================
# APPLY FILTERS
# ============================================================================
if len(date_range) == 2:
    filtered_df = df[
        (df['date'].dt.date >= date_range[0]) &
        (df['date'].dt.date <= date_range[1]) &
        (df['category'].isin(categories)) &
        (df['payment_method'].isin(payment_methods)) &
        (df['account_type'].isin(account_types)) &
        (df['transaction_type'].isin(transaction_types)) &
        (df['risk_level'].isin(risk_levels)) &
        (df['amount'] >= amount_range[0]) &
        (df['amount'] <= amount_range[1])
    ].copy()
else:
    filtered_df = df.copy()

# Apply anomaly filter
if show_only_anomalies:
    filtered_df = filtered_df[filtered_df['is_anomaly'] == 1]

# ============================================================================
# EXECUTIVE SUMMARY SECTION 
# ============================================================================
st.subheader("Executive Summary")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Total Transactions",
        value=f"{len(filtered_df):,}",
        delta=f"{len(filtered_df)/len(df)*100:.1f}% of total"
    )

with col2:
    total_anomalies = filtered_df['is_anomaly'].sum()
    st.metric(
        label="Anomalies Detected",
        value=f"{total_anomalies:,}",
        delta=f"{total_anomalies/len(filtered_df)*100:.1f}%",
        delta_color="inverse"
    )

with col3:
    anomaly_amount = filtered_df[filtered_df['is_anomaly'] == 1]['amount'].sum()
    st.metric(
        label="Anomaly Amount",
        value=f"${anomaly_amount:,.0f}",
        delta=f"{anomaly_amount/filtered_df['amount'].sum()*100:.1f}% of total",
        delta_color="inverse"
    )

with col4:
    high_risk_count = (filtered_df['risk_level'] == 'High').sum()
    st.metric(
        label="High Risk Transactions",
        value=f"{high_risk_count:,}",
        delta=f"{high_risk_count/len(filtered_df)*100:.1f}%",
        delta_color="inverse"
    )

with col5:
    total_amount = filtered_df['amount'].sum()
    st.metric(
        label="Total Amount",
        value=f"${total_amount:,.0f}",
        delta=f"Avg: ${filtered_df['amount'].mean():.2f}"
    )

st.markdown("---")

# ============================================================================
# TRENDS OVER TIME SECTION
# ============================================================================
st.markdown("<h3 style='text-align: center;'>Trends over Time</h3>", unsafe_allow_html=True)

col_monthly, col_dow = st.columns([1.2, 1])

with col_monthly:
    # Prepare monthly data
    monthly_data = filtered_df.groupby('month').agg({
        'transaction_id': 'count',
        'is_anomaly': 'sum',
        'amount': 'sum'
    }).reset_index()
    monthly_data.columns = ['Month', 'Total_Count', 'Anomalous_Count', 'Total_Amount']
    monthly_data['Normal_Count'] = monthly_data['Total_Count'] - monthly_data['Anomalous_Count']
    
    # Calculate amounts for normal vs anomalous
    monthly_amount = filtered_df.groupby(['month', 'is_anomaly'])['amount'].sum().reset_index()
    monthly_normal_amount = monthly_amount[monthly_amount['is_anomaly'] == 0].set_index('month')['amount']
    monthly_anomalous_amount = monthly_amount[monthly_amount['is_anomaly'] == 1].set_index('month')['amount']
    
    monthly_data = monthly_data.set_index('Month')
    monthly_data['Normal_Amount'] = monthly_normal_amount
    monthly_data['Anomalous_Amount'] = monthly_anomalous_amount
    monthly_data = monthly_data.fillna(0).reset_index()
    
    # Sort by month chronologically
    monthly_data = monthly_data.sort_values('Month')
    
    # Create subplot with 2 charts stacked vertically
    fig_monthly = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Transaction Amount vs Month', 
                       'Number of Transactions vs Month'),
        vertical_spacing=0.25
    )
    
    # Top chart: Amount trends
    fig_monthly.add_trace(go.Scatter(
        x=monthly_data['Month'],
        y=monthly_data['Normal_Amount'],
        name='Normal',
        mode='lines+markers',
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=8),
        hovertemplate='<b>%{x}</b><br>Normal: $%{y:,.0f}<extra></extra>',
        legendgroup='amount'
    ), row=1, col=1)
    
    fig_monthly.add_trace(go.Scatter(
        x=monthly_data['Month'],
        y=monthly_data['Anomalous_Amount'],
        name='Anomalous',
        mode='lines+markers',
        line=dict(color='#d62728', width=3),
        marker=dict(size=8, symbol='diamond'),
        hovertemplate='<b>%{x}</b><br>Anomalous: $%{y:,.0f}<extra></extra>',
        legendgroup='amount'
    ), row=1, col=1)
    
    # Bottom chart: Transaction count trends
    fig_monthly.add_trace(go.Scatter(
        x=monthly_data['Month'],
        y=monthly_data['Normal_Count'],
        name='Normal',
        mode='lines+markers',
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=8),
        hovertemplate='<b>%{x}</b><br>Normal: %{y}<extra></extra>',
        showlegend=False,
        legendgroup='count'
    ), row=2, col=1)
    
    fig_monthly.add_trace(go.Scatter(
        x=monthly_data['Month'],
        y=monthly_data['Anomalous_Count'],
        name='Anomalous',
        mode='lines+markers',
        line=dict(color='#d62728', width=3),
        marker=dict(size=8, symbol='diamond'),
        hovertemplate='<b>%{x}</b><br>Anomalous: %{y}<extra></extra>',
        showlegend=False,
        legendgroup='count'
    ), row=2, col=1)
    
    fig_monthly.update_xaxes(title_text="Month", row=1, col=1)
    fig_monthly.update_xaxes(title_text="Month", row=2, col=1)
    fig_monthly.update_yaxes(title_text="Amount ($)", row=1, col=1)
    fig_monthly.update_yaxes(title_text="Number of Transactions", row=2, col=1)
    
    fig_monthly.update_layout(
        height=750,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='white'
    )
    
    fig_monthly.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig_monthly.update_yaxes(showgrid=True, gridcolor='lightgray')
    
    st.plotly_chart(fig_monthly, use_container_width=True)

with col_dow:
    # Prepare day of week data
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    dow_data = filtered_df.groupby('day_of_week').agg({
        'transaction_id': 'count',
        'is_anomaly': 'sum',
        'amount': 'sum'
    }).reset_index()
    dow_data.columns = ['Day', 'Total_Count', 'Anomalous_Count', 'Total_Amount']
    dow_data['Normal_Count'] = dow_data['Total_Count'] - dow_data['Anomalous_Count']
    
    # Calculate amounts for normal vs anomalous
    dow_amount = filtered_df.groupby(['day_of_week', 'is_anomaly'])['amount'].sum().reset_index()
    dow_normal_amount = dow_amount[dow_amount['is_anomaly'] == 0].set_index('day_of_week')['amount']
    dow_anomalous_amount = dow_amount[dow_amount['is_anomaly'] == 1].set_index('day_of_week')['amount']
    
    dow_data = dow_data.set_index('Day')
    dow_data['Normal_Amount'] = dow_normal_amount
    dow_data['Anomalous_Amount'] = dow_anomalous_amount
    dow_data = dow_data.fillna(0).reset_index()
    
    # Calculate number of occurrences of each day in the filtered data
    day_counts = filtered_df.groupby('day_of_week')['date'].apply(lambda x: x.dt.date.nunique()).to_dict()
    
    # Calculate averages per occurrence of that day
    dow_data['Num_Occurrences'] = dow_data['Day'].map(day_counts)
    dow_data['Avg_Normal_Count'] = (dow_data['Normal_Count'] / dow_data['Num_Occurrences']).round(1)
    dow_data['Avg_Anomalous_Count'] = (dow_data['Anomalous_Count'] / dow_data['Num_Occurrences']).round(1)
    dow_data['Avg_Normal_Amount'] = (dow_data['Normal_Amount'] / dow_data['Num_Occurrences']).round(2)
    dow_data['Avg_Anomalous_Amount'] = (dow_data['Anomalous_Amount'] / dow_data['Num_Occurrences']).round(2)
    
    # Sort by day of week
    dow_data['day_order'] = dow_data['Day'].map({day: i for i, day in enumerate(day_order)})
    dow_data = dow_data.sort_values('day_order')
    
    # Create subplot with 2 charts stacked vertically
    fig_dow = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Average Transaction Amount vs Day of the Week',
                       'Average Number of Transactions vs Day of the Week'),
        vertical_spacing=0.25
    )
    
    # Top chart: Average amount
    fig_dow.add_trace(go.Bar(
        x=dow_data['Day'],
        y=dow_data['Avg_Normal_Amount'],
        name='Normal',
        marker_color='#2ca02c',
        text=dow_data['Avg_Normal_Amount'],
        textposition='outside',
        texttemplate='$%{text:,.0f}',
        hovertemplate='<b>%{x}</b><br>Avg Normal Amount: $%{y:,.2f}<extra></extra>',
        legendgroup='amount'
    ), row=1, col=1)
    
    fig_dow.add_trace(go.Bar(
        x=dow_data['Day'],
        y=dow_data['Avg_Anomalous_Amount'],
        name='Anomalous',
        marker_color='#d62728',
        text=dow_data['Avg_Anomalous_Amount'],
        textposition='outside',
        texttemplate='$%{text:,.0f}',
        hovertemplate='<b>%{x}</b><br>Avg Anomalous Amount: $%{y:,.2f}<extra></extra>',
        legendgroup='amount'
    ), row=1, col=1)
    
    # Bottom chart: Average transaction count
    fig_dow.add_trace(go.Bar(
        x=dow_data['Day'],
        y=dow_data['Avg_Normal_Count'],
        name='Normal',
        marker_color='#2ca02c',
        text=dow_data['Avg_Normal_Count'],
        textposition='outside',
        texttemplate='%{text:.1f}',
        hovertemplate='<b>%{x}</b><br>Avg Normal: %{y:.1f}<extra></extra>',
        showlegend=False,
        legendgroup='count'
    ), row=2, col=1)
    
    fig_dow.add_trace(go.Bar(
        x=dow_data['Day'],
        y=dow_data['Avg_Anomalous_Count'],
        name='Anomalous',
        marker_color='#d62728',
        text=dow_data['Avg_Anomalous_Count'],
        textposition='outside',
        texttemplate='%{text:.1f}',
        hovertemplate='<b>%{x}</b><br>Avg Anomalous: %{y:.1f}<extra></extra>',
        showlegend=False,
        legendgroup='count'
    ), row=2, col=1)
    
    fig_dow.update_xaxes(title_text="Day of the Week", row=1, col=1)
    fig_dow.update_xaxes(title_text="Day of the Week", row=2, col=1)
    fig_dow.update_yaxes(title_text="Average Amount ($)", row=1, col=1)
    fig_dow.update_yaxes(title_text="Average Number of Transactions", row=2, col=1)
    
    fig_dow.update_layout(
        height=750,
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='white'
    )
    
    fig_dow.update_xaxes(showgrid=False)
    fig_dow.update_yaxes(showgrid=True, gridcolor='lightgray', rangemode='tozero')
    
    # Set specific y-axis ranges with dtick to show gridlines at regular intervals
    fig_dow.update_yaxes(range=[0, 1550], dtick=300, row=1, col=1)
    fig_dow.update_yaxes(range=[0, 4.1], dtick=1, row=2, col=1)
    
    st.plotly_chart(fig_dow, use_container_width=True)

st.markdown("---")

# ============================================================================
# ANOMALIES BY CATEGORY AND MERCHANT
# ============================================================================

# Two columns - Anomaly by Category and Anomaly by Merchant side-by-side
col_category, col_merchant = st.columns([1, 1])

with col_category:
    st.markdown("<h3 style='margin-bottom: 1rem;'>Number of Anomalous Transactions by Category</h3>", unsafe_allow_html=True)
    
    # Calculate anomalies by category
    anomaly_by_category = filtered_df.groupby('category').agg({
        'is_anomaly': 'sum',
        'transaction_id': 'count',
        'amount': 'sum'
    }).reset_index()
    anomaly_by_category.columns = ['Category', 'Anomalies', 'Total_Txns', 'Total_Amount']
    anomaly_by_category['Anomaly_Rate'] = (anomaly_by_category['Anomalies'] / anomaly_by_category['Total_Txns'] * 100).round(1)
    anomaly_by_category = anomaly_by_category.sort_values('Anomalies', ascending=True)
    
    # Create horizontal bar chart with red gradient
    fig_category = px.bar(
        anomaly_by_category,
        y='Category',
        x='Anomalies',
        orientation='h',
        color='Anomalies',
        color_continuous_scale='Reds',
        text='Anomalies',
        labels={'Anomalies': 'Number of Anomalies'},
        hover_data={'Anomaly_Rate': ':.1f'}
    )
    
    fig_category.update_traces(
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Anomalies: %{x}<br>Rate: %{customdata[0]:.1f}%<extra></extra>'
    )
    
    fig_category.update_layout(
        height=550,
        xaxis_title="Number of Anomalies",
        yaxis_title="Category",
        showlegend=False,
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=False)
    )
    
    st.plotly_chart(fig_category, use_container_width=True)

with col_merchant:
    st.markdown("<h3 style='margin-bottom: 1rem;'>Top 15 Merchants with the Most Anomalous Transactions</h3>", unsafe_allow_html=True)
    
    merchant_analysis = filtered_df[filtered_df['is_anomaly'] == 1].groupby('merchant').agg({
        'transaction_id': 'count',
        'amount': ['sum', 'mean', 'max'],
        'risk_score': 'mean'
    }).reset_index()
    
    merchant_analysis.columns = ['Merchant', 'Anomaly Count', 'Total Amount', 'Avg Amount', 'Max Amount', 'Avg Risk Score']
    merchant_analysis = merchant_analysis.sort_values('Anomaly Count', ascending=False).head(15)
    merchant_analysis = merchant_analysis.sort_values('Anomaly Count', ascending=True)
    
    fig_merchant = px.bar(
        merchant_analysis,
        x='Anomaly Count',
        y='Merchant',
        orientation='h',
        color='Anomaly Count',
        color_continuous_scale='Reds',
        text='Anomaly Count',
        hover_data=['Total Amount', 'Avg Amount', 'Max Amount', 'Avg Risk Score'],
        labels={'Anomaly Count': 'Number of Anomalies'}
    )
    fig_merchant.update_traces(textposition='outside')
    fig_merchant.update_layout(
        height=550,
        xaxis_title="Number of Anomalies",
        yaxis_title="Merchant",
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(categoryorder='total ascending', showgrid=False)
    )
    st.plotly_chart(fig_merchant, use_container_width=True)

# Risk Heatmap 
st.markdown("<h3 style='margin-top: 1.5rem; margin-bottom: 0.3rem;'>Risk Heatmap: Category vs Payment Method</h3>", unsafe_allow_html=True)

# Create pivot table for heatmap
heatmap_data = filtered_df.groupby(['category', 'payment_method'])['risk_score'].mean().reset_index()
heatmap_pivot = heatmap_data.pivot(index='category', columns='payment_method', values='risk_score')

# Sort categories alphabetically (descending for top-to-bottom display)
heatmap_pivot = heatmap_pivot.sort_index(ascending=False)

fig_heatmap = go.Figure(data=go.Heatmap(
    z=heatmap_pivot.values,
    x=heatmap_pivot.columns,
    y=heatmap_pivot.index,
    colorscale='Reds',
    text=np.round(heatmap_pivot.values, 1),
    texttemplate='%{text}',
    textfont={"size": 14},
    colorbar=dict(title="Avg Risk<br>Score", outlinewidth=0),
    hovertemplate='Category: %{y}<br>Payment: %{x}<br>Risk Score: %{z:.1f}<extra></extra>'
))

fig_heatmap.update_layout(
    height=600,
    xaxis_title="Payment Method",
    yaxis_title="Category",
    plot_bgcolor='white'
)

st.plotly_chart(fig_heatmap, use_container_width=True)

st.markdown("---")

# ============================================================================
# PIE CHARTS 
# ============================================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Payment Methods for Anomalous Transactions")
    
    payment_anomalies = filtered_df.groupby('payment_method').agg({
        'is_anomaly': 'sum',
        'transaction_id': 'count',
        'amount': 'mean'
    }).reset_index()
    payment_anomalies.columns = ['Payment Method', 'Anomalies', 'Total', 'Avg Amount']
    
    fig_payment = px.pie(
        payment_anomalies,
        values='Anomalies',
        names='Payment Method',
        color_discrete_sequence=px.colors.sequential.RdBu,
        hole=0.4
    )
    fig_payment.update_traces(textposition='inside', textinfo='percent+label')
    fig_payment.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig_payment, use_container_width=True)

with col2:
    st.subheader("Transaction Type for Anomalous Transactions")
    
    txn_type_data = filtered_df[filtered_df['is_anomaly'] == 1].groupby('transaction_type').agg({
        'transaction_id': 'count'
    }).reset_index()
    txn_type_data.columns = ['Type', 'Anomalies']
    
    fig_txn_type = px.pie(
        txn_type_data,
        values='Anomalies',
        names='Type',
        color_discrete_sequence=px.colors.sequential.RdBu,
        hole=0.4
    )
    fig_txn_type.update_traces(textposition='inside', textinfo='percent+label')
    fig_txn_type.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig_txn_type, use_container_width=True)

with col3:
    st.subheader("Account Type for Anomalous Transactions")
    
    account_data = filtered_df[filtered_df['is_anomaly'] == 1].groupby('account_type').agg({
        'transaction_id': 'count'
    }).reset_index()
    account_data.columns = ['Account Type', 'Anomalies']
    
    fig_account = px.pie(
        account_data,
        values='Anomalies',
        names='Account Type',
        color_discrete_sequence=px.colors.sequential.RdBu,
        hole=0.4
    )
    fig_account.update_traces(textposition='inside', textinfo='percent+label')
    fig_account.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig_account, use_container_width=True)

st.markdown("---")

# ============================================================================
# SCATTER PLOT 
# ============================================================================

st.subheader("Risk Score vs Transaction Amount Scatter Plot")

# Color by risk level
color_map = {'Low': '#2ca02c', 'Medium': '#ff7f0e', 'High': '#d62728'}

fig_scatter = px.scatter(
    filtered_df,
    x='amount',
    y='risk_score',
    color='risk_level',
    size='amount',
    hover_data=['category', 'merchant', 'transaction_type', 'payment_method', 'day_of_week'],
    color_discrete_map=color_map,
    labels={'amount': 'Transaction Amount ($)', 'risk_score': 'Risk Score'}
)

fig_scatter.add_hline(y=70, line_dash="dash", line_color="orange", 
                      annotation_text="High Risk Threshold", annotation_position="right")

fig_scatter.update_layout(height=500)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# ============================================================================
# DATA TABLE WITH ANOMALIES 
# ============================================================================

st.subheader("Flagged Anomalies")

# Prepare display dataframe
display_df = filtered_df[filtered_df['is_anomaly'] == 1].copy()
display_df = display_df.sort_values('risk_score', ascending=False)

# Format columns
display_cols = ['transaction_id', 'date', 'amount', 'category', 'merchant', 
                'payment_method', 'account_type', 'transaction_type', 
                'day_of_week', 'month_name', 'risk_score', 'risk_level']

display_df_formatted = display_df[display_cols].copy()
display_df_formatted['date'] = display_df_formatted['date'].dt.strftime('%Y-%m-%d %H:%M')
display_df_formatted['amount'] = display_df_formatted['amount'].apply(lambda x: f"${x:,.2f}")
display_df_formatted['risk_score'] = display_df_formatted['risk_score'].apply(lambda x: f"{x:.1f}")

# Add download button
csv = display_df_formatted.to_csv(index=False)
st.download_button(
    label="Download Anomalies as CSV",
    data=csv,
    file_name=f"anomalies_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)

# Display interactive table
st.dataframe(
    display_df_formatted.head(100),
    use_container_width=True,
    height=400
)

st.caption(f"Showing top 100 of {len(display_df)} anomalous transactions. Download full dataset using button above.")

st.markdown("---")

# ============================================================================
# CHATBOT MOCK-UP
# ============================================================================

st.subheader("AI Risk Assistant (Mock-up)")

user_question = st.text_input("Ask about the data:", placeholder="e.g., Which merchant has the highest risk?")

if user_question:
    st.markdown("**AI Assistant Response:**")
    st.info(f"""
    Based on the current filtered data (using Isolation Forest ML):
    
    - **Total Anomalies**: {filtered_df['is_anomaly'].sum()} transactions flagged
    - **Highest Risk Category**: {filtered_df.groupby('category')['risk_score'].mean().idxmax()}
    - **Top Risk Merchant**: {filtered_df.groupby('merchant')['risk_score'].mean().idxmax()}
    - **Anomaly Rate**: {filtered_df['is_anomaly'].sum() / len(filtered_df) * 100:.1f}%
    
    *This is a mock-up. In production, this would use LLM integration for natural language queries.*
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <strong>GIC Internal Audit - Data Analytics Team</strong><br>
    Anomalous Transactions Dashboard | Powered by Python and Streamlit<br>
    For internal use only | Confidential
</div>
""", unsafe_allow_html=True)