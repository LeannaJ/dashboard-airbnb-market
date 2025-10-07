import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import glob
import numpy as np

@st.cache_data(show_spinner=False)
def load_city_frames(data_dir: str):
    files = sorted(glob.glob(str(Path(data_dir) / "airbnb_*.csv")))
    city_frames = {}
    for fp in files:
        city = Path(fp).stem.replace("airbnb_", "")
        try:
            df = pd.read_csv(fp)
        except Exception:
            df = pd.read_csv(fp, engine="python", encoding="utf-8")
        # Standardize date
        if "Scraped Date" in df.columns:
            df["Scraped Date"] = pd.to_datetime(df["Scraped Date"], errors="coerce")
        # Coerce helpful numeric columns
        for c in ["superhost_ratio","booked_days_period_city","revenue_period_city",
                  "booked_days_period_tract","revenue_period_tract",
                  "rating_ave_pastYear","numReviews_pastYear"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        city_frames[city] = df
    return city_frames

def kpi_card(label: str, value, help_txt: str = ""):
    st.metric(label, value if value is not None else "—", help=help_txt)

def try_mean(df, col):
    return float(df[col].mean()) if col in df.columns and df[col].notna().any() else None

def try_sum(df, col):
    return float(df[col].sum()) if col in df.columns and df[col].notna().any() else None

def format_currency(value, decimals=1):
    """Format large numbers with appropriate units (K, M, B)"""
    if value is None:
        return "—"
    
    abs_value = abs(value)
    if abs_value >= 1e9:
        return f"${value/1e9:.{decimals}f}B"
    elif abs_value >= 1e6:
        return f"${value/1e6:.{decimals}f}M"
    elif abs_value >= 1e3:
        return f"${value/1e3:.{decimals}f}K"
    else:
        return f"${value:.{decimals}f}"

def calculate_profitability_score(df):
    """Calculate profitability score based on the formula provided"""
    df = df.copy()
    
    # Required columns for profitability score calculation
    required_cols = ['Nightly Rate', 'occupancy_rate', 'numReviews_pastYear', 'rating_ave_pastYear']
    
    # Check if all required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return df
    
    # Calculate profitability score components
    df['profitability_score'] = 0.0
    
    # Component 1: Nightly Rate (normalized to 0-100)
    if 'Nightly Rate' in df.columns:
        df['nightly_rate_norm'] = (df['Nightly Rate'] - df['Nightly Rate'].min()) / (df['Nightly Rate'].max() - df['Nightly Rate'].min()) * 100
        df['profitability_score'] += df['nightly_rate_norm'] * 0.3  # 30% weight
    
    # Component 2: Occupancy Rate (normalized to 0-100)
    if 'occupancy_rate' in df.columns:
        df['occupancy_norm'] = df['occupancy_rate'] * 100
        df['profitability_score'] += df['occupancy_norm'] * 0.4  # 40% weight
    
    # Component 3: Review Count (normalized to 0-100)
    if 'numReviews_pastYear' in df.columns:
        df['reviews_norm'] = (df['numReviews_pastYear'] - df['numReviews_pastYear'].min()) / (df['numReviews_pastYear'].max() - df['numReviews_pastYear'].min()) * 100
        df['profitability_score'] += df['reviews_norm'] * 0.2  # 20% weight
    
    # Component 4: Rating (normalized to 0-100)
    if 'rating_ave_pastYear' in df.columns:
        df['rating_norm'] = (df['rating_ave_pastYear'] - df['rating_ave_pastYear'].min()) / (df['rating_ave_pastYear'].max() - df['rating_ave_pastYear'].min()) * 100
        df['profitability_score'] += df['rating_norm'] * 0.1  # 10% weight
    
    # Clean up temporary columns
    df = df.drop(['nightly_rate_norm', 'occupancy_norm', 'reviews_norm', 'rating_norm'], axis=1, errors='ignore')
    
    return df

def create_quarterly_data(df):
    """Create quarterly aggregated data from date column"""
    if "Scraped Date" not in df.columns:
        return pd.DataFrame()
    
    # Convert to datetime and create quarter column
    df_quarterly = df.copy()
    df_quarterly["Scraped Date"] = pd.to_datetime(df_quarterly["Scraped Date"])
    df_quarterly["Year"] = df_quarterly["Scraped Date"].dt.year
    df_quarterly["Quarter"] = df_quarterly["Scraped Date"].dt.quarter
    df_quarterly["Quarter_Label"] = df_quarterly["Year"].astype(str) + " Q" + df_quarterly["Quarter"].astype(str)
    
    return df_quarterly

def aggregate_quarterly_metrics(df_quarterly):
    """Aggregate metrics by quarter"""
    if df_quarterly.empty:
        return pd.DataFrame()
    
    # Group by quarter and calculate metrics
    quarterly_data = df_quarterly.groupby("Quarter_Label").agg({
        "Airbnb Property ID": "count",  # Count of listings
        "revenue_period_city": "sum",  # Total revenue
    }).reset_index()
    
    quarterly_data.columns = ["Quarter", "Listing_Count", "Total_Revenue"]
    
    return quarterly_data

def aggregate_yearly_metrics(df_quarterly):
    """Aggregate metrics by year"""
    if df_quarterly.empty:
        return pd.DataFrame()
    
    # Group by year and calculate metrics
    yearly_data = df_quarterly.groupby("Year").agg({
        "Nightly Rate": "mean",  # Average nightly rate
        "occupancy_rate": "mean",  # Average occupancy rate
    }).reset_index()
    
    yearly_data.columns = ["Year", "Avg_Nightly_Rate", "Avg_Occupancy_Rate"]
    
    return yearly_data

def try_unique(df, col):
    return int(df[col].nunique()) if col in df.columns else None

def optional_chart(title, chart):
    """Display a plotly chart with title if chart is not None"""
    if chart is not None:
        st.plotly_chart(chart, use_container_width=True)

def create_line_chart(df, x_col, y_col, title, color="#FF5A5F"):
    """Create a line chart using plotly express"""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    
    fig = px.line(
        df, 
        x=x_col, 
        y=y_col,
        title=title,
        color_discrete_sequence=[color],
        markers=True
    )
    fig.update_layout(
        height=400,
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title=y_col.replace("_", " ").title(),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    fig.update_traces(line=dict(width=3))
    return fig

def create_bar_chart(df, x_col, y_col, title, color="#FF8E53", orientation='v'):
    """Create a bar chart using plotly express"""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    
    if orientation == 'v':
        fig = px.bar(df, x=x_col, y=y_col, title=title, color_discrete_sequence=[color])
        x_title = x_col.replace("_", " ").title()
        y_title = y_col.replace("_", " ").title()
    else:
        fig = px.bar(df, x=x_col, y=y_col, title=title, color_discrete_sequence=[color], orientation='h')
        x_title = y_col.replace("_", " ").title()
        y_title = x_col.replace("_", " ").title()
    
    fig.update_layout(
        height=400,
        xaxis_title=x_title,
        yaxis_title=y_title,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    return fig

def create_histogram(df, column, title, color="#FF5A5F", nbins=30):
    """Create a histogram using plotly express"""
    if df.empty or column not in df.columns:
        return None
    
    fig = px.histogram(
        df, 
        x=column,
        title=title,
        color_discrete_sequence=[color],
        nbins=nbins
    )
    fig.update_layout(
        height=400,
        xaxis_title=column.replace("_", " ").title(),
        yaxis_title="Count",
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    return fig

def create_scatter_plot(df, x_col, y_col, title, color="#FF8E53"):
    """Create a scatter plot using plotly express"""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    
    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col,
        title=title,
        color_discrete_sequence=[color],
        opacity=0.6
    )
    fig.update_layout(
        height=400,
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title=y_col.replace("_", " ").title(),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    return fig

def create_pie_chart(df, names_col, values_col, title, color_palette=None):
    """Create a pie chart using plotly express"""
    if df.empty or names_col not in df.columns or values_col not in df.columns:
        return None
    
    if color_palette is None:
        color_palette = px.colors.qualitative.Set3
    
    fig = px.pie(
        df, 
        names=names_col, 
        values=values_col,
        title=title,
        color_discrete_sequence=color_palette
    )
    fig.update_layout(
        height=400,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    return fig

def create_box_plot(df, x_col, y_col, title, color="#FF5A5F"):
    """Create a box plot using plotly express"""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    
    fig = px.box(
        df, 
        x=x_col, 
        y=y_col,
        title=title,
        color_discrete_sequence=[color]
    )
    fig.update_layout(
        height=400,
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title=y_col.replace("_", " ").title(),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    return fig

def create_scatter_map(df, lat_col, lon_col, title, color_col=None, color_scale=None):
    """Create a scatter map using plotly express"""
    if df.empty or lat_col not in df.columns or lon_col not in df.columns:
        return None
    
    fig = px.scatter_mapbox(
        df, 
        lat=lat_col, 
        lon=lon_col,
        title=title,
        color=color_col if color_col and color_col in df.columns else None,
        color_continuous_scale=color_scale or "Viridis",
        mapbox_style="open-street-map",
        zoom=10,
        opacity=0.6
    )
    fig.update_layout(
        height=400,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    return fig

def create_multi_line_chart(df, x_col, y_cols, title, color_palette=None):
    """Create a multi-line chart using plotly express"""
    if df.empty or x_col not in df.columns:
        return None
    
    if color_palette is None:
        color_palette = ["#FF5A5F", "#FF8E53", "#4ECDC4", "#45B7D1"]
    
    fig = go.Figure()
    
    for i, y_col in enumerate(y_cols):
        if y_col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode='lines+markers',
                name=y_col.replace("_", " ").title(),
                line=dict(color=color_palette[i % len(color_palette)], width=3),
                marker=dict(size=6)
            ))
    
    fig.update_layout(
        title=title,
        height=400,
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title="Value",
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    return fig

def create_correlation_heatmap(df, title):
    """Create a correlation heatmap using plotly"""
    if df.empty:
        return None
    
    # Select numeric columns for correlation
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter for specific columns we want to include
    desired_columns = [
        'Nightly Rate', 'Bedrooms', 'Bathrooms', 'Max Guests', 
        'occupancy_rate', 'Minimum Stay', 'Number of Photos', 'profitability_score'
    ]
    
    # Only include columns that exist in the data
    correlation_columns = [col for col in desired_columns if col in numeric_columns]
    
    if len(correlation_columns) < 2:
        return None
    
    # Calculate correlation matrix
    correlation_data = df[correlation_columns].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_data.values,
        x=correlation_data.columns,
        y=correlation_data.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlation_data.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=0)
    )
    
    return fig

# ---------- App ----------
st.set_page_config(
    page_title="Airbnb Market Analytics Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏠"
)

# ---------- CSS Styles ----------
st.markdown("""
<style>
/* Expander Header Styling */
.streamlit-expanderHeader,
[data-testid="streamlit-expanderHeader"],
.stExpander > div[data-testid="streamlit-expanderHeader"],
div[data-testid="streamlit-expanderHeader"] {
    font-size: 2rem !important;
    font-weight: bold !important;
    color: #FF5A5F !important;
}

/* Alternative approach - target expander content */
.streamlit-expander .streamlit-expanderHeader,
.stExpander .streamlit-expanderHeader {
    font-size: 2rem !important;
    font-weight: bold !important;
    color: #FF5A5F !important;
}

/* Metric Cards Styling */
[data-testid="metric-container"] {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    padding: 1rem;
    border-radius: 0.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Sidebar Styling */
.css-1d391kg {
    background-color: #f8f9fa;
}

/* Chart Container Styling */
.stPlotlyChart {
    border-radius: 0.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Portfolio Header
st.markdown("""
<div style="padding: 2rem; margin-bottom: 2rem;">
    <h1 style="text-align: center; font-size: 3rem; margin: 0;">
        <span style="font-size: 3rem;">🏠</span>
        <span style="color: #FF5A5F; display: inline-block; background: linear-gradient(90deg, #FF5A5F 0%, #FF8E53 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; color: #FF5A5F;"> Airbnb Market Analytics Dashboard</span>
    </h1>
    <p style="color: #FF5A5F; background: linear-gradient(90deg, #FF5A5F 0%, #FF8E53 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; color: #FF5A5F; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.1rem;">
        Comprehensive Analysis of Airbnb Market Data Across Major US Cities
    </p>
</div>
""", unsafe_allow_html=True)

# Project Info
st.markdown("### 📋 Project Overview")
with st.expander("**Project Goal:** Analyze Airbnb market the trends and the proper locations to invest in Airbnb across major US cities", expanded=False):
    st.markdown("""
    **Data Sources:** 
    - Synthetic Tabular Data of Airbnb in major 4 US cities: Chicago, Houston, Philadelphia, and Washington DC
    - Census tract demographics and economic indicators
    
    **Technologies Used:**
    - Python (Pandas, Streamlit, Plotly)
    - Data Visualization and Interactive Dashboards
    - Statistical Analysis and Market Research
    
    **Key Insights:**
    - Airbnb market trends over time
    - Property type distribution
    - Performance metrics
    - Geographic performance patterns
    """)
    
st.markdown("")


# ---------- Sidebar Configuration ----------
st.sidebar.markdown("### 🎛️ Dashboard Controls")

# Load data (try multiple possible paths)
possible_paths = [".", "./airbnb-market-analytics", "airbnb-market-analytics"]
city_frames = {}

for path in possible_paths:
    try:
        city_frames = load_city_frames(path)
        if city_frames:
            break
    except Exception as e:
        continue

if not city_frames:
    st.error("❌ No CSV files found! Please ensure airbnb_*.csv files are in the specified folder.")
    st.stop()

# Sidebar Info Section
st.sidebar.metric("Cities Analyzed", "4", "Major US Cities")
st.sidebar.metric("Data Points", "120,217", "Listings & Bookings")
st.sidebar.metric("Time Period", "2016-2020", "Historical Analysis")

st.sidebar.markdown("---")

# Filters Section
st.sidebar.markdown("#### 🔍 Filters")

cities = list(city_frames.keys())
city = st.sidebar.selectbox("Selected City", options=cities, index=0, help="Choose a city to analyze")

df = city_frames[city].copy()

# Date Range Filter
if "Scraped Date" in df.columns and df["Scraped Date"].notna().any():
    min_d, max_d = pd.to_datetime(df["Scraped Date"]).min(), pd.to_datetime(df["Scraped Date"]).max()
    date_range = st.sidebar.date_input("Data Range", value=(min_d, max_d), help="Filter by scraping date")
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df = df[(df["Scraped Date"] >= start) & (df["Scraped Date"] <= end)]
    
# Property Type Filter
if "Property Type" in df.columns:
    # First, create "Others" group in the data (same logic as pie chart)
    total_count = len(df)
    prop_counts = df["Property Type"].value_counts()
    prop_percentages = (prop_counts / total_count) * 100
    
    # Create a copy of the dataframe and group small property types as "Others"
    df_filtered = df.copy()
    minor_property_types = prop_percentages[prop_percentages < 1.0].index.tolist()
    
    if minor_property_types:
        df_filtered.loc[df_filtered["Property Type"].isin(minor_property_types), "Property Type"] = "Others"
    
    # Now get the updated property type counts
    updated_prop_counts = df_filtered["Property Type"].value_counts()
    
    # Define the property types from the image (excluding Others for now)
    image_property_types = [
        "Apartment", "Condominium", "House", "Loft", 
        "Townhouse", "Serviced apartment", "Guest suite", "Entire apartment"
    ]
    
    # Filter to only include property types that exist in the updated data
    available_types = [pt for pt in image_property_types if pt in updated_prop_counts.index]
    
    # Sort by count (largest first)
    available_types_sorted = sorted(available_types, key=lambda x: updated_prop_counts[x], reverse=True)
    
    # Add "Others" at the end if it exists in updated data
    if "Others" in updated_prop_counts.index:
        available_types_sorted.append("Others")
    
    # Create final list with "All" at the beginning
    property_types = ["All"] + available_types_sorted
    
    selected_type = st.sidebar.selectbox("Property Type", property_types)
    if selected_type != "All":
        df = df_filtered[df_filtered["Property Type"] == selected_type]
    else:
        df = df_filtered

# Price Range Filter
if "Nightly Rate" in df.columns:
    price_data = df["Nightly Rate"].dropna()
    if not price_data.empty:
        min_price, max_price = price_data.min(), price_data.max()
        price_range = st.sidebar.slider("Price Range($)", 
                                  min_value=float(min_price), 
                                  max_value=float(max_price), 
                                  value=(float(min_price), float(max_price)),
                                  step=10.0)
    df = df[(df["Nightly Rate"] >= price_range[0]) & (df["Nightly Rate"] <= price_range[1])]

# Superhost Filter
if "Superhost" in df.columns:
    sh_filter = st.sidebar.checkbox("Superhost Only", value=False, help="Show only superhost listings")
    if sh_filter:
        df = df[df["Superhost"].fillna(0) == 1]



# ---------- KPIs ----------
st.markdown("### 📊 Key Performance Indicators")

# Main KPI Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    listings_count = try_unique(df, "Airbnb Property ID")
    st.metric(
        label="Total Listings", 
        value=f"{listings_count:,}" if listings_count else "—",
        help="Unique properties in current filter"
    )

with col2:
    hosts_count = try_unique(df, "Airbnb Host ID")
    st.metric(
        label="Unique Hosts", 
        value=f"{hosts_count:,}" if hosts_count else "—",
        help="Unique hosts in current filter"
    )

with col3:
    avg_superhost_ratio = try_mean(df, "superhost_ratio")
    st.metric(
        label="Superhost Ratio", 
        value=f"{avg_superhost_ratio:.1%}" if avg_superhost_ratio else "—",
        help="Average superhost percentage"
    )

with col4:
    avg_rating = try_mean(df, "rating_ave_pastYear")
    st.metric(
        label="Avg Rating", 
        value=f"{avg_rating:.2f}" if avg_rating else "—",
        help="Average rating over past year"
    )


# Secondary metrics
col5, col6, col7, col8 = st.columns(4)

with col5:
    total_rev_city = try_sum(df, "revenue_period_city") or try_sum(df, "revenue_period_tract")
    st.metric(
        label="Total Revenue", 
        value=format_currency(total_rev_city),
        help="Estimated total revenue"
    )

with col6:
    avg_price = try_mean(df, "Nightly Rate")
    st.metric(
        label="Avg Nightly Rate", 
        value=f"${avg_price:.0f}" if avg_price else "—",
        help="Average nightly price"
    )

with col7:
    occupancy_rate = try_mean(df, "occupancy_rate")
    st.metric(
        label="Avg Occupancy", 
        value=f"{occupancy_rate:.1%}" if occupancy_rate else "—",
        help="Average occupancy rate"
    )

with col8:
    avg_reviews = try_mean(df, "numReviews_pastYear")
    st.metric(
        label="Avg Reviews", 
        value=f"{avg_reviews:.0f}" if avg_reviews else "—",
        help="Average reviews per year"
    )

st.markdown("")


# ---------- Charts ----------
st.markdown("### 📈 Data Visualizations")

# Create tabs for different chart categories
tab1, tab2, tab3, tab4 = st.tabs(["◽ Market Trends", "◽ Property Type", "◽ Performance", "◽ Geographic"])

with tab1:
    st.markdown("##### (1) Market Trends Over Time")
    
    # Create quarterly and yearly data
    df_quarterly = create_quarterly_data(df)
    quarterly_metrics = aggregate_quarterly_metrics(df_quarterly)
    yearly_metrics = aggregate_yearly_metrics(df_quarterly)
    
    if not quarterly_metrics.empty:
        # 1. Quarterly Listing Count Trend
        listing_chart = create_line_chart(
            quarterly_metrics,
            "Quarter",
            "Listing_Count",
            f"{city} - Quarterly Listing Count Trend"
        )
        optional_chart("", listing_chart)
        
        # 2. Annual Nightly Rate & Occupancy Rate Trends (side by side)
        if not yearly_metrics.empty:
            col_price, col_occupancy = st.columns(2)
            
            with col_price:
                if "Avg_Nightly_Rate" in yearly_metrics.columns:
                    price_chart = create_line_chart(
                        yearly_metrics,
                        "Year",
                        "Avg_Nightly_Rate",
                        f"{city} - Annual Average Nightly Rate Trend"
                    )
                    optional_chart("", price_chart)
            
            with col_occupancy:
                if "Avg_Occupancy_Rate" in yearly_metrics.columns:
                    occupancy_chart = create_line_chart(
                        yearly_metrics,
                        "Year",
                        "Avg_Occupancy_Rate",
                        f"{city} - Annual Average Occupancy Rate Trend"
                    )
                    optional_chart("", occupancy_chart)
        
        # 3. Quarterly Revenue Trend
        if "Total_Revenue" in quarterly_metrics.columns:
            revenue_chart = create_line_chart(
                quarterly_metrics,
                "Quarter",
                "Total_Revenue",
                f"{city} - Quarterly Revenue Trend"
            )
            optional_chart("", revenue_chart)

with tab2:
    st.markdown("##### (2) Property Type Analysis")
    
    # 1. Property Type Distribution (Pie Chart)
    if "Property Type" in df.columns:
        # Since "Others" group is already created in the sidebar filter, just use the data as is
        prop_counts = df["Property Type"].value_counts().reset_index()
        prop_counts.columns = ["Property Type", "Count"]
        
        if not prop_counts.empty:
            # Display as pie chart (Others group already included from sidebar filter)
            pie_chart = create_pie_chart(
                prop_counts,
                "Property Type",
                "Count",
                f"{city} - Property Type Distribution"
            )
            optional_chart("", pie_chart)
    
    # 2. Price Distribution by Bedrooms/Bathrooms (Box Plot)
    if "Bedrooms" in df.columns and "Nightly Rate" in df.columns:
        # Filter for realistic bedroom counts (1-8)
        bedroom_price_data = df[["Bedrooms", "Nightly Rate"]].dropna()
        bedroom_price_data = bedroom_price_data[(bedroom_price_data["Bedrooms"] >= 1) & (bedroom_price_data["Bedrooms"] <= 8)]
        if not bedroom_price_data.empty:
            bedroom_box = create_box_plot(
                bedroom_price_data,
                "Bedrooms",
                "Nightly Rate",
                f"{city} - Price Distribution by Bedrooms (1-8)"
            )
            optional_chart("", bedroom_box)
    
    if "Bathrooms" in df.columns and "Nightly Rate" in df.columns:
        # Filter for realistic bathroom counts (1-4)
        bathroom_price_data = df[["Bathrooms", "Nightly Rate"]].dropna()
        bathroom_price_data = bathroom_price_data[(bathroom_price_data["Bathrooms"] >= 1) & (bathroom_price_data["Bathrooms"] <= 4)]
        if not bathroom_price_data.empty:
            bathroom_box = create_box_plot(
                bathroom_price_data,
                "Bathrooms",
                "Nightly Rate",
                f"{city} - Price Distribution by Bathrooms (1-4)"
            )
            optional_chart("", bathroom_box)
    
    # 3. Correlation Heatmap - Property Characteristics vs Profitability
    # Calculate profitability score first
    df_with_score = calculate_profitability_score(df)
    
    if not df_with_score.empty and "profitability_score" in df_with_score.columns:
        correlation_heatmap = create_correlation_heatmap(
            df_with_score,
            f"{city} - Property Characteristics Correlation Matrix"
        )
        optional_chart("", correlation_heatmap)

with tab3:
    st.markdown("##### (3) Performance Metrics and Analysis")
    
    # Calculate profitability score
    df_with_score = calculate_profitability_score(df)
    
    # 1. Top 10 Profitability Score Properties + Summary Stats
    if "profitability_score" in df_with_score.columns and "Airbnb Property ID" in df_with_score.columns:
        top_profitability = df_with_score[["Airbnb Property ID", "profitability_score"]].dropna()
        if not top_profitability.empty:
            # Get top 10
            top_10_properties = top_profitability.nlargest(10, "profitability_score")
            
            # Create two columns layout
            col_chart, col_stats = st.columns([2, 1])
            
            with col_chart:
                # Bar chart for top 10
                top_10_properties = top_10_properties.copy()
                top_10_properties["Property_Index"] = range(1, len(top_10_properties) + 1)
                top_10_properties["Property_Label"] = "Property " + top_10_properties["Property_Index"].astype(str)
                
                profitability_bar = create_bar_chart(
                    top_10_properties,
                    "Property_Label",
                    "profitability_score",
                    f"{city} - Top 10 Profitability Score",
                    orientation='v'
                )
                optional_chart("", profitability_bar)
            
            with col_stats:
                # Summary statistics for top 10 properties
                st.markdown("##### Top 10 Characteristics")
                st.markdown("---")
                # Get the actual property IDs for top 10
                top_10_ids = top_10_properties["Airbnb Property ID"].tolist()
                top_10_data = df_with_score[df_with_score["Airbnb Property ID"].isin(top_10_ids)]
                
                if not top_10_data.empty:
                    # Calculate summary statistics
                    avg_price = top_10_data["Nightly Rate"].mean() if "Nightly Rate" in top_10_data.columns else None
                    avg_reviews = top_10_data["numReviews_pastYear"].mean() if "numReviews_pastYear" in top_10_data.columns else None
                    avg_rating = top_10_data["rating_ave_pastYear"].mean() if "rating_ave_pastYear" in top_10_data.columns else None
                    avg_occupancy = top_10_data["occupancy_rate"].mean() if "occupancy_rate" in top_10_data.columns else None
                    # Calculate superhost ratio - simple approach
                    if "Superhost" in top_10_data.columns:
                        # Check superhost status for each property ID (use existing top_10_ids)
                        superhost_count = 0
                        for prop_id in top_10_ids:
                            prop_data = df[df["Airbnb Property ID"] == prop_id]
                            if not prop_data.empty and prop_data["Superhost"].iloc[0] == 1.0:
                                superhost_count += 1
                        superhost_ratio = superhost_count / 10  # Simple: n/10 (0.5 = 50%)
                else:
                    superhost_ratio = 0
                    
                # Display metrics
                st.markdown(f"✔️ Avg Nightly Rate: **${avg_price:.0f}**" if avg_price else "Avg Nightly Rate: N/A")
                st.markdown("")
                st.markdown(f"✔️ Avg Reviews: **{avg_reviews:.0f}**" if avg_reviews else "Avg Reviews: N/A")
                st.markdown("")
                st.markdown(f"✔️ Avg Rating: **{avg_rating:.2f}**" if avg_rating else "Avg Rating: N/A")
                st.markdown("")
                st.markdown(f"✔️ Avg Occupancy: **{avg_occupancy:.1%}**" if avg_occupancy else "Avg Occupancy: N/A")
                st.markdown("")
                st.markdown(f"✔️ Superhost Ratio: **{superhost_ratio:.1%}**")
    
    # 2. Price Distribution (Histogram) - Limited to 0-1500
    if "Nightly Rate" in df.columns:
        price_data = df["Nightly Rate"].dropna()
        if not price_data.empty:
            # Filter to 0-1500 range
            price_data = price_data[(price_data >= 0) & (price_data <= 1500)]
            if not price_data.empty:
                price_hist = create_histogram(
                    price_data.to_frame("Nightly Rate"), 
                    "Nightly Rate", 
                    f"{city} - Price Distribution (0-1500)"
                )
                # Customize histogram bins to 50-unit intervals
                if price_hist is not None:
                    price_hist.update_layout(
                        xaxis=dict(
                            tickmode='linear',
                            tick0=0,
                            dtick=50
                        )
                    )
                optional_chart("", price_hist)
    
    # 3. Two Scatter Plots: Reviews vs Profitability & Rating vs Profitability
    col_scatter1, col_scatter2 = st.columns(2)
    
    with col_scatter1:
        # Reviews vs Profitability Score
        if "numReviews_pastYear" in df_with_score.columns and "profitability_score" in df_with_score.columns:
            review_profit_data = df_with_score[["numReviews_pastYear", "profitability_score"]].dropna()
            if not review_profit_data.empty:
                review_profit_chart = create_scatter_plot(
                    review_profit_data,
                    "numReviews_pastYear",
                    "profitability_score",
                    f"{city} - Reviews vs Profitability"
                )
                optional_chart("", review_profit_chart)
    
    with col_scatter2:
        # Rating vs Profitability Score
        if "rating_ave_pastYear" in df_with_score.columns and "profitability_score" in df_with_score.columns:
            rating_profit_data = df_with_score[["rating_ave_pastYear", "profitability_score"]].dropna()
            if not rating_profit_data.empty:
                rating_profit_chart = create_scatter_plot(
                    rating_profit_data,
                    "rating_ave_pastYear",
                    "profitability_score",
                    f"{city} - Rating vs Profitability"
                )
                optional_chart("", rating_profit_chart)

with tab4:
    st.markdown("##### (4) Geographic Analysis")
    
    # 1. Top 10 Revenue Neighborhoods
    if "Neighborhood" in df.columns and "revenue_period_city" in df.columns:
        neighborhood_revenue = df.groupby("Neighborhood", as_index=False)["revenue_period_city"].sum()
        if not neighborhood_revenue.empty:
            neighborhood_revenue = neighborhood_revenue.nlargest(10, "revenue_period_city")
            neighborhood_chart = create_bar_chart(
                neighborhood_revenue,
                "Neighborhood",
                "revenue_period_city",
                f"{city} - Top Revenue Neighborhoods",
                orientation='v'
            )
            optional_chart("", neighborhood_chart)
    
    # 2. Map Visualization (Property Distribution)
    if "Latitude" in df.columns and "Longitude" in df.columns:
        map_data = df[["Latitude", "Longitude", "Nightly Rate"]].dropna()
        if not map_data.empty:
            # Sampling for performance improvement (if too many points)
            if len(map_data) > 1000:
                map_data = map_data.sample(1000)
            
            map_chart = create_scatter_map(
                map_data,
                "Latitude",
                "Longitude",
                f"{city} - Property Distribution Map",
                color_col="Nightly Rate",
                color_scale="Viridis"
            )
            optional_chart("", map_chart)
    
    # 3. Superhost Ratio by Neighborhood
    if "Neighborhood" in df.columns and "superhost_ratio" in df.columns:
        superhost_neighborhood = df.groupby("Neighborhood", as_index=False)["superhost_ratio"].mean()
        if not superhost_neighborhood.empty:
            superhost_neighborhood = superhost_neighborhood.nlargest(10, "superhost_ratio")
            superhost_chart = create_bar_chart(
                superhost_neighborhood,
                "Neighborhood",
                "superhost_ratio",
                f"{city} - Superhost Ratio by Neighborhood",
                orientation='v'
            )
            optional_chart("", superhost_chart)

st.markdown("---")


# ---------- Key Insights Section ----------
st.markdown("### 💡 Key Insights & Analysis")

# City Comparison Table
st.markdown("##### **City Performance Comparison**")

# Create comparison data
comparison_data = {
    "City": ["Chicago", "Houston", "Philadelphia", "Washington"],
    "Total Revenue": ["$1,800B", "$920.1B", "$1,040.0B", "$2,178.4B"],
    "Avg Nightly Rate": ["$120", "$307", "$327", "$213"],
    "Avg Occupancy": ["16.5%", "17.1%", "17.8%", "20.7%"],
    "Avg Rating": ["4.73", "4.73", "4.71", "4.74"],
    "Superhost Ratio": ["19.9%", "19.9%", "19.5%", "26.9%"],
    "Top 10 Avg Rate": ["$419", "$526", "$378", "$654"],
    "Top 10 Avg Rating": ["4.69", "4.81", "4.78", "4.87"],
    "Top 10 Occupancy": ["42.2%", "34.9%", "32.5%", "49.5%"]
}

comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, use_container_width=True)


# Generate common insights based on all city data
insights_col1, insights_col2 = st.columns(2)

with insights_col1:
    st.markdown("##### (1) Market Insights")
    
    # Market Trends Insights (Common across cities) - Yellow background
    st.markdown("""
    <div style="background-color: #fff8e1; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <strong>📊 Universal Market Volatility</strong><br>
    All four cities show significant quarterly fluctuations in listing counts and revenue, indicating seasonal market dynamics across major US markets.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #fff8e1; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <strong>💰 Price Pressure Pattern</strong><br>
    All cities experienced price declines from 2017-2018 peaks to 2020 lows, suggesting increased competition and market saturation across the industry. The COVID-19 pandemic in 2020 further accelerated these declines across all markets.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #fff8e1; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <strong>🏠 Apartment Dominance</strong><br>
    Apartments consistently dominate all markets (38-58%), followed by houses (21-33%), showing universal property type preferences in major US cities.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #fff8e1; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <strong>🗺️ Geographic Concentration</strong><br>
    All cities show strong geographic concentration with 2-3 neighborhoods generating significantly higher revenue than others, indicating location premium is universal.
    </div>
    """, unsafe_allow_html=True)

with insights_col2:
    st.markdown("##### (2) Strategic Recommendations")
    
    # Strategic Recommendations - Light blue background
    st.markdown("""
    <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <strong>🎯 Focus on High-Performing Areas</strong><br>
    Target top 2-3 revenue-generating neighborhoods in each city, as they consistently outperform others by significant margins.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <strong>📈 Optimize for Occupancy</strong><br>
    Given the universal strong correlation (0.91-0.93) between occupancy rate and profitability score across all cities, prioritize strategies that maximize occupancy over pure price increases.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <strong>⭐ Quality Over Quantity</strong><br>
    Invest in property quality and guest experience universally - top performers across all cities show 4.7+ ratings and 1,000+ reviews.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <strong>🔍 Geographic Diversification</strong><br>
    Consider emerging neighborhoods with lower Superhost ratios for less competitive market entry across all cities.
    </div>
    """, unsafe_allow_html=True)

st.markdown("")
st.markdown("")

# Key Insights Summary
st.markdown("""
<div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #FF5A5F;">
<h4 style="color: #FF5A5F; margin-top: 0;">Multi-City Airbnb Market Analysis Summary</h4>

<p><strong>[1] Universal Market Patterns:</strong> All four major US cities (Chicago, Houston, Philadelphia, Washington) show consistent patterns: apartment dominance (38-58%), geographic concentration in top neighborhoods, and strong correlation between occupancy rate and profitability (0.91-0.93).</p>

<p><strong>[2] Performance Drivers:</strong> Quality metrics are universal success factors - top-performing properties achieve 4.7+ ratings, 30%+ occupancy rates, and 1,000+ reviews across all cities, indicating guest satisfaction is the primary profitability driver.</p>

<p><strong>[3] Market Challenges:</strong> All cities face similar challenges - declining prices from 2017-2018 peaks, reduced occupancy rates post-2018, and increased competition, suggesting industry-wide market saturation and changing demand patterns.</p>

<p><strong>[4] Strategic Opportunities:</strong> Focus on occupancy optimization over price increases, leverage apartment market dominance, target high-performing geographic areas, and explore emerging neighborhoods with lower Superhost competition for market entry across all cities.</p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>Airbnb Market Analytics Dashboard</strong> | Built with Python, Streamlit & Plotly</p>
    <p>Data-driven insights for Airbnb market analysis and investment decisions</p>
</div>
""", unsafe_allow_html=True)
