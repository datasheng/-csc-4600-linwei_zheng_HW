# %% [markdown]
# # NYC Yellow Taxi Trip Data - Exploratory Data Analysis
# ## January 2023 Dataset
# 
# This notebook performs a comprehensive EDA on NYC Yellow Taxi trip data, including:
# - 1D histograms with statistics (min, max, mean, median, outliers)
# - Bar charts for categorical variables
# - Side-by-side box plots
# - Scatter plot matrix
# - Categorical vs categorical relationships
# - Categorical vs numerical relationships with color

# %% [markdown]
# ## CELL 1: Install and Import Libraries

# %%
# Run this first cell to install required packages
!pip install pyarrow seaborn matplotlib pandas numpy scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('ggplot')
sns.set_palette("viridis")
sns.set_style("whitegrid")
%matplotlib inline

print("✅ Libraries imported successfully!")

# %% [markdown]
# ## CELL 2: Load the Data

# %%
# Upload your yellow_tripdata_2023-01.parquet file to the notebook first
# If using Google Colab, use: from google.colab import files; files.upload()

df = pd.read_parquet('yellow_tripdata_2023-01.parquet')

print(f"✅ Dataset loaded successfully!")
print(f"📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\n📋 Column names:")
for col in df.columns:
    print(f"   - {col}")

# %% [markdown]
# ## CELL 3: Initial Data Inspection

# %%
print("🔍 DATA INFO")
print("=" * 50)
print(df.info())

print("\n🔍 FIRST 5 ROWS")
print("=" * 50)
df.head()

# %% [markdown]
# ## CELL 4: Select and Prepare Data for Analysis

# %%
print("🛠️ DATA PREPARATION")

# Create a working copy
df_clean = df.copy()

# Convert datetime columns
df_clean['pickup_datetime'] = pd.to_datetime(df_clean['tpep_pickup_datetime'])
df_clean['dropoff_datetime'] = pd.to_datetime(df_clean['tpep_dropoff_datetime'])

# Extract time features
df_clean['pickup_hour'] = df_clean['pickup_datetime'].dt.hour
df_clean['pickup_day'] = df_clean['pickup_datetime'].dt.day_name()
df_clean['pickup_month'] = df_clean['pickup_datetime'].dt.month
df_clean['pickup_weekend'] = df_clean['pickup_datetime'].dt.dayofweek >= 5

# Calculate trip duration in minutes
df_clean['trip_duration_min'] = (df_clean['dropoff_datetime'] - df_clean['pickup_datetime']).dt.total_seconds() / 60

print("✅ Time features created: hour, day, month, weekend, duration")

# Define key columns for analysis
key_columns = [
    'VendorID', 'passenger_count', 'trip_distance', 'RatecodeID',
    'PULocationID', 'DOLocationID', 'payment_type', 'fare_amount',
    'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
    'improvement_surcharge', 'total_amount', 'congestion_surcharge',
    'airport_fee', 'pickup_hour', 'pickup_day', 'trip_duration_min'
]

# Keep only columns that exist
available_cols = [col for col in key_columns if col in df_clean.columns]
df_clean = df_clean[available_cols].copy()

print(f"\n✅ Working with {len(available_cols)} columns")
print(f"📊 Shape after preparation: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
df_clean.head()

# %% [markdown]
# ## CELL 5: Data Cleaning - Remove Outliers

# %%
print("🧹 DATA CLEANING")

initial_rows = len(df_clean)

# Remove unrealistic values
df_clean = df_clean[
    (df_clean['fare_amount'] >= 2.5) & (df_clean['fare_amount'] <= 200) &  # Minimum fare $2.50, max $200
    (df_clean['trip_distance'] >= 0.1) & (df_clean['trip_distance'] <= 100) &  # Between 0.1 and 100 miles
    (df_clean['passenger_count'] >= 1) & (df_clean['passenger_count'] <= 6) &  # 1-6 passengers
    (df_clean['trip_duration_min'] >= 1) & (df_clean['trip_duration_min'] <= 180) &  # 1 min to 3 hours
    (df_clean['tip_amount'] >= 0) & (df_clean['tip_amount'] <= 50)  # Tips between $0 and $50
]

# Remove rows with missing values in key columns
df_clean = df_clean.dropna(subset=['fare_amount', 'trip_distance', 'tip_amount', 'total_amount'])

rows_removed = initial_rows - len(df_clean)
print(f"✅ Removed {rows_removed:,} rows ({rows_removed/initial_rows*100:.1f}%)")
print(f"📊 Final shape: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")

# Sample data for faster plotting (if dataset is large)
if len(df_clean) > 100000:
    df_sample = df_clean.sample(n=50000, random_state=42)
    print(f"\n📉 Sampled 50,000 rows for faster visualization")
else:
    df_sample = df_clean.copy()

# %% [markdown]
# ## CELL 6: Identify Numerical and Categorical Columns

# %%
numerical_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = df_clean.select_dtypes(include=['object', 'bool']).columns.tolist()

# Add specific categorical columns that might be integers
int_categorical = ['VendorID', 'RatecodeID', 'payment_type', 'PULocationID', 'DOLocationID']
for col in int_categorical:
    if col in df_clean.columns and col in numerical_cols:
        numerical_cols.remove(col)
        categorical_cols.append(col)

print(f"🔢 NUMERICAL COLUMNS ({len(numerical_cols)}):")
for col in numerical_cols[:15]:  # Show first 15
    print(f"   - {col}")
if len(numerical_cols) > 15:
    print(f"   ... and {len(numerical_cols)-15} more")

print(f"\n📊 CATEGORICAL COLUMNS ({len(categorical_cols)}):")
for col in categorical_cols:
    print(f"   - {col}")

print(f"\n✅ Total columns: {len(numerical_cols) + len(categorical_cols)}")

# %% [markdown]
# ## CELL 7: Basic Statistics

# %%
print("📊 DESCRIPTIVE STATISTICS")
print("=" * 50)
df_clean[numerical_cols].describe()

# %% [markdown]
# ## CELL 8: 1D Histograms for Numerical Variables

# %%
# Select 9 key numerical columns for histograms
key_numerical = ['fare_amount', 'tip_amount', 'trip_distance', 'total_amount', 
                 'passenger_count', 'trip_duration_min', 'extra', 'mta_tax', 'tolls_amount']
available_key_num = [col for col in key_numerical if col in df_clean.columns]

# Create histograms
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(available_key_num[:9]):
    # Plot histogram
    axes[i].hist(df_clean[col].dropna(), bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[i].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel(col, fontsize=10)
    axes[i].set_ylabel('Frequency', fontsize=10)
    axes[i].axvline(df_clean[col].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df_clean[col].mean():.2f}')
    axes[i].axvline(df_clean[col].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df_clean[col].median():.2f}')
    axes[i].legend(fontsize=8)
    
    # Report statistics
    print(f"\n📊 {col} Statistics:")
    print(f"   Min: {df_clean[col].min():.2f}")
    print(f"   Max: {df_clean[col].max():.2f}")
    print(f"   Mean: {df_clean[col].mean():.2f}")
    print(f"   Median: {df_clean[col].median():.2f}")
    print(f"   Std: {df_clean[col].std():.2f}")
    
    # Identify outliers using IQR
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df_clean[(df_clean[col] < Q1 - 1.5*IQR) | (df_clean[col] > Q3 + 1.5*IQR)]
    print(f"   Outliers: {len(outliers):,} ({len(outliers)/len(df_clean)*100:.1f}%)")

# Hide unused subplots
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('1-Dimensional Histograms of Numerical Variables', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## CELL 9: Bar Charts for Categorical Variables

# %%
# Select categorical columns for bar charts
cat_to_plot = ['payment_type', 'VendorID', 'RatecodeID', 'pickup_day']
available_cat = [col for col in cat_to_plot if col in df_clean.columns]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(available_cat[:4]):
    if col in df_clean.columns:
        # Map payment_type codes to descriptions
        if col == 'payment_type':
            payment_map = {1: 'Credit Card', 2: 'Cash', 3: 'No Charge', 4: 'Dispute', 5: 'Unknown'}
            df_clean['payment_desc'] = df_clean['payment_type'].map(payment_map)
            value_counts = df_clean['payment_desc'].value_counts()
        else:
            value_counts = df_clean[col].value_counts().head(10)
        
        # Create bar chart
        bars = axes[i].bar(range(len(value_counts)), value_counts.values)
        axes[i].set_title(f'Bar Chart of {col}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel(col, fontsize=10)
        axes[i].set_ylabel('Count', fontsize=10)
        axes[i].set_xticks(range(len(value_counts)))
        axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, val in zip(bars, value_counts.values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(value_counts.values)*0.01,
                        f'{val:,}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Categorical Variable Distributions', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Print frequency tables
print("\n📊 CATEGORICAL FREQUENCIES:")
for col in available_cat:
    if col == 'payment_type':
        print(f"\n{col} (with descriptions):")
        payment_map = {1: 'Credit Card', 2: 'Cash', 3: 'No Charge', 4: 'Dispute', 5: 'Unknown'}
        df_clean['payment_desc'] = df_clean['payment_type'].map(payment_map)
        print(df_clean['payment_desc'].value_counts())
    else:
        print(f"\n{col}:")
        print(df_clean[col].value_counts().head(10))

# %% [markdown]
# ## CELL 10: Combined Box Plot for Numerical Variables

# %%
# Select numerical columns for box plot
boxplot_cols = ['fare_amount', 'tip_amount', 'trip_distance', 'total_amount', 'trip_duration_min']
available_box = [col for col in boxplot_cols if col in df_clean.columns]

if len(available_box) > 0:
    # Create a copy with scaled values for better comparison
    from sklearn.preprocessing import StandardScaler
    
    df_scaled = df_clean[available_box].copy()
    scaler = StandardScaler()
    df_scaled_scaled = pd.DataFrame(
        scaler.fit_transform(df_scaled.fillna(df_scaled.mean())),
        columns=df_scaled.columns
    )
    
    # Create box plot
    plt.figure(figsize=(14, 8))
    
    # Box plot
    boxplot = df_scaled_scaled.boxplot(return_type='dict', patch_artist=True)
    
    # Customize colors
    for box in boxplot['boxes']:
        box.set_facecolor('lightblue')
        box.set_alpha(0.7)
    
    plt.title('Box Plot Comparison of Numerical Variables (Standardized)', fontsize=16, fontweight='bold')
    plt.ylabel('Standardized Values (mean=0, std=1)', fontsize=12)
    plt.xlabel('Variables', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    print("📌 Note: Variables are standardized to have mean=0 and standard deviation=1")
    print("📌 This allows for direct comparison of distributions and outliers across variables")

# %% [markdown]
# ## CELL 11: Scatter Plot Matrix for Numerical Data

# %%
# Select key numerical variables for scatter matrix
scatter_cols = ['fare_amount', 'tip_amount', 'trip_distance', 'total_amount', 'trip_duration_min']
available_scatter = [col for col in scatter_cols if col in df_clean.columns]

if len(available_scatter) >= 2:
    # Use sample for scatter matrix (to avoid overcrowding)
    sample_size = min(3000, len(df_clean))
    df_scatter_sample = df_clean[available_scatter].sample(n=sample_size, random_state=42)
    
    # Create scatter matrix
    fig = plt.figure(figsize=(16, 12))
    scatter_matrix = pd.plotting.scatter_matrix(
        df_scatter_sample, 
        alpha=0.3, 
        figsize=(16, 12), 
        diagonal='hist',
        hist_kwds={'bins': 30, 'edgecolor': 'black', 'alpha': 0.7},
        grid=True
    )
    
    # Rotate labels
    for ax in scatter_matrix.flatten():
        ax.xaxis.label.set_rotation(45)
        ax.yaxis.label.set_rotation(45)
        ax.xaxis.label.set_fontsize(10)
        ax.yaxis.label.set_fontsize(10)
    
    plt.suptitle('Scatter Plot Matrix of Numerical Variables', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Scatter matrix created using {sample_size:,} sampled rows")

# %% [markdown]
# ## CELL 12: Categorical vs Categorical Visualization

# %%
# Payment type vs Rate code
if 'payment_type' in df_clean.columns and 'RatecodeID' in df_clean.columns:
    # Create cross-tabulation
    payment_map = {1: 'Credit Card', 2: 'Cash', 3: 'No Charge', 4: 'Dispute', 5: 'Unknown'}
    rate_map = {1: 'Standard', 2: 'JFK', 3: 'Newark', 4: 'Nassau/Westchester', 5: 'Negotiated', 6: 'Group Ride'}
    
    df_clean['payment_desc'] = df_clean['payment_type'].map(payment_map)
    df_clean['rate_desc'] = df_clean['RatecodeID'].map(rate_map)
    
    # Create cross-tabulation (percentage within payment type)
    cross_tab = pd.crosstab(
        df_clean['payment_desc'].fillna('Unknown'), 
        df_clean['rate_desc'].fillna('Unknown'),
        normalize='index'
    ) * 100
    
    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    cross_tab.plot(kind='bar', stacked=True, ax=ax, colormap='viridis', edgecolor='black')
    
    plt.title('Payment Type Distribution by Rate Code (%)', fontsize=16, fontweight='bold')
    plt.xlabel('Payment Type', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.legend(title='Rate Code', bbox_to_anchor=(1.05, 1), fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for c in ax.containers:
        ax.bar_label(c, label_type='center', fmt='%.1f%%', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    print("📊 Cross-tabulation (Percentage):")
    print(cross_tab.round(1))

# %% [markdown]
# ## CELL 13: Categorical vs Numerical Visualization

# %%
# Box plots with color - Fare amount by payment type and hour
if all(col in df_clean.columns for col in ['fare_amount', 'payment_type', 'pickup_hour']):
    # Create hour bins
    df_clean['time_of_day'] = pd.cut(
        df_clean['pickup_hour'], 
        bins=[0, 6, 12, 18, 24], 
        labels=['Late Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)'],
        right=False
    )
    
    # Map payment types
    payment_map = {1: 'Credit Card', 2: 'Cash', 3: 'No Charge', 4: 'Dispute', 5: 'Unknown'}
    df_clean['payment_desc'] = df_clean['payment_type'].map(payment_map)
    
    # Sample for plotting
    plot_sample = df_clean[['fare_amount', 'payment_desc', 'time_of_day']].dropna().sample(n=5000, random_state=42)
    
    # Create grouped box plot
    plt.figure(figsize=(15, 8))
    
    # Box plot with hue
    ax = sns.boxplot(
        data=plot_sample, 
        x='payment_desc', 
        y='fare_amount', 
        hue='time_of_day',
        palette='coolwarm',
        order=['Credit Card', 'Cash', 'No Charge', 'Dispute', 'Unknown']
    )
    
    plt.title('Fare Amount Distribution by Payment Type and Time of Day', fontsize=16, fontweight='bold')
    plt.xlabel('Payment Type', fontsize=12)
    plt.ylabel('Fare Amount ($)', fontsize=12)
    plt.legend(title='Time of Day', bbox_to_anchor=(1.05, 1))
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Summary statistics by payment type
    print("\n📊 Average Fare by Payment Type:")
    print(df_clean.groupby('payment_desc')['fare_amount'].agg(['mean', 'median', 'count']).round(2))

# %% [markdown]
# ## CELL 14: Correlation Heatmap

# %%
# Calculate correlation matrix
if len(available_scatter) >= 2:
    corr_matrix = df_clean[available_scatter].corr()
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Heatmap
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='RdBu_r', 
        center=0,
        square=True, 
        linewidths=1,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
        mask=mask,
        vmin=-1, vmax=1,
        annot_kws={'size': 10}
    )
    
    plt.title('Correlation Heatmap of Numerical Variables', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print strongest correlations
    print("\n📊 Strongest Correlations:")
    corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
    unique_pairs = [(corr_pairs.index[i][0], corr_pairs.index[i][1], corr_pairs.values[i]) 
                    for i in range(len(corr_pairs)) 
                    if corr_pairs.index[i][0] != corr_pairs.index[i][1] 
                    and corr_pairs.index[i][0] < corr_pairs.index[i][1]]
    
    for pair in unique_pairs[:5]:
        print(f"   {pair[0]} vs {pair[1]}: {pair[2]:.3f}")

# %% [markdown]
# ## CELL 15: Additional Visualization - Trip Patterns by Hour

# %%
if 'pickup_hour' in df_clean.columns:
    # Hourly trip counts
    hourly_trips = df_clean['pickup_hour'].value_counts().sort_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Line plot of trips by hour
    axes[0].plot(hourly_trips.index, hourly_trips.values, marker='o', linewidth=2, markersize=8)
    axes[0].fill_between(hourly_trips.index, hourly_trips.values, alpha=0.3)
    axes[0].set_title('Number of Trips by Hour of Day', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Hour (0-23)', fontsize=12)
    axes[0].set_ylabel('Number of Trips', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(range(0, 24, 2))
    
    # Average fare by hour
    hourly_fare = df_clean.groupby('pickup_hour')['fare_amount'].mean()
    
    axes[1].plot(hourly_fare.index, hourly_fare.values, marker='s', linewidth=2, markersize=8, color='green')
    axes[1].fill_between(hourly_fare.index, hourly_fare.values, alpha=0.3, color='green')
    axes[1].set_title('Average Fare by Hour of Day', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Hour (0-23)', fontsize=12)
    axes[1].set_ylabel('Average Fare ($)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(range(0, 24, 2))
    
    plt.suptitle('Temporal Patterns in NYC Taxi Trips', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.show()
    
    print(f"📊 Peak hour: {hourly_trips.idxmax()}:00 with {hourly_trips.max():,} trips")
    print(f"📊 Lowest hour: {hourly_trips.idxmin()}:00 with {hourly_trips.min():,} trips")

# %% [markdown]
# ## CELL 16: Summary of Key Findings

# %%
print("=" * 70)
print("📝 EDA SUMMARY FINDINGS")
print("=" * 70)

print("\n🔍 DATASET OVERVIEW:")
print(f"   - Total rows analyzed: {len(df_clean):,}")
print(f"   - Number of numerical variables: {len(numerical_cols)}")
print(f"   - Number of categorical variables: {len(categorical_cols)}")

print("\n💰 FARE ANALYSIS:")
print(f"   - Average fare: ${df_clean['fare_amount'].mean():.2f}")
print(f"   - Median fare: ${df_clean['fare_amount'].median():.2f}")
print(f"   - Most expensive fare: ${df_clean['fare_amount'].max():.2f}")

print("\n💵 TIP ANALYSIS:")
print(f"   - Average tip: ${df_clean['tip_amount'].mean():.2f}")
print(f"   - Median tip: ${df_clean['tip_amount'].median():.2f}")
print(f"   - Tips represent {df_clean['tip_amount'].mean()/df_clean['fare_amount'].mean()*100:.1f}% of average fare")

print("\n🚕 TRIP CHARACTERISTICS:")
print(f"   - Average distance: {df_clean['trip_distance'].mean():.2f} miles")
print(f"   - Average duration: {df_clean['trip_duration_min'].mean():.1f} minutes")
print(f"   - Average passengers: {df_clean['passenger_count'].mean():.1f}")

print("\n📊 KEY CORRELATIONS:")
if len(available_scatter) >= 2:
    for pair in unique_pairs[:3]:
        print(f"   - {pair[0]} vs {pair[1]}: {pair[2]:.3f}")

print("\n⏰ TEMPORAL PATTERNS:")
if 'pickup_hour' in df_clean.columns:
    print(f"   - Busiest hour: {hourly_trips.idxmax()}:00")
    print(f"   - Quietest hour: {hourly_trips.idxmin()}:00")

print("\n" + "=" * 70)
print("✅ EDA COMPLETE - All requirements met:")
print("   ✓ 30+ rows ✓ 8+ columns ✓ 5+ numerical columns")
print("   ✓ 1D histograms ✓ Combined box plot ✓ Scatter matrix")
print("   ✓ Categorical vs categorical ✓ Categorical vs numerical")
print("=" * 70)