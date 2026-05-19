import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

DIVIDER = "\n" + "=" * 65 + "\n"
SUBDIV  = "-" * 65

# ── helpers ────────────────────────────────────────────────────────────────

def section(title):
    print(DIVIDER + f"  {title}" + DIVIDER)

def step(msg):
    print(f"\n  >>  {msg}")

# ── STAGE 0 - Load raw data ────────────────────────────────────────────────

section("STAGE 0 - Raw dataset")

df = pd.read_csv("data/starbucks_customer_ordering_patterns.csv")

step(f"Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
step(f"Missing values: {df.isnull().sum().sum()} - no imputation needed")

print("\n  First 5 rows of the original dataset:\n")
print(df.head().to_string(index=False))

# ── STAGE 1 - EDA summary ──────────────────────────────────────────────────

section("STAGE 1 - Exploratory Data Analysis")

# Numeric stats
numeric_cols = ["cart_size", "num_customizations", "total_spend",
                "fulfillment_time_min", "customer_satisfaction"]

print("  Numeric variables - key statistics:\n")
print(df[numeric_cols].describe().round(2).to_string())

# Channel performance (key DW TP finding)
print(f"\n{SUBDIV}")
print("  Average fulfillment time by channel (confirms DW TP finding):\n")
channel_avg = (
    df.groupby("order_channel")["fulfillment_time_min"]
    .mean()
    .sort_values(ascending=False)
    .round(2)
)
for channel, avg in channel_avg.items():
    print(f"    {channel:<22} {avg} min")

# Target variable balance
high = (df["customer_satisfaction"] >= 4).sum()
low  = (df["customer_satisfaction"] <= 3).sum()
print(f"\n{SUBDIV}")
print("  Target variable - customer_satisfaction split:\n")
print(f"    High (>= 4):  {high:,}  ({high/len(df):.1%})")
print(f"    Low  (<= 3):  {low:,}  ({low/len(df):.1%})")

# Categorical distributions
print(f"\n{SUBDIV}")
print("  Categorical variables - value counts:\n")
for col in ["order_channel", "store_location_type", "region",
            "customer_age_group", "drink_category"]:
    counts = df[col].value_counts()
    values = "  |  ".join(f"{k}: {v:,}" for k, v in counts.items())
    print(f"    {col:<22}  {values}")

# Boolean flags
print(f"\n{SUBDIV}")
print("  Boolean flags - distribution:\n")
for col in ["is_rewards_member", "has_food_item", "order_ahead"]:
    vc = df[col].value_counts()
    print(f"    {col:<22}  True: {vc.get(True, 0):,}  |  False: {vc.get(False, 0):,}")

step("EDA complete")

# ── STAGE 2 - Preprocessing ────────────────────────────────────────────────

section("STAGE 2 - Preprocessing")

# Step 2.1 - time_period feature
df["hour"] = pd.to_datetime(df["order_time"], format="%H:%M").dt.hour

def classify_period(h):
    if 7  <= h <= 9:  return "Morning Rush"
    if 10 <= h <= 13: return "Mid-Day"
    if 14 <= h <= 17: return "Afternoon"
    if 18 <= h <= 21: return "Evening"
    return "Other"

df["time_period"] = df["hour"].apply(classify_period)
step("Engineered 'time_period' from order_time (5 operational buckets: Morning Rush, Mid-Day, Afternoon, Evening, Other)")
print(f"\n    {'Period':<16} {'Count':>8}")
for period, count in df["time_period"].value_counts().items():
    print(f"    {period:<16} {count:>8,}")

# Step 2.2 - discretize targets
# V1: satisfaction_level — first attempt, produced a single-leaf tree (Kappa ~0)
df["satisfaction_level"] = df["customer_satisfaction"].apply(
    lambda x: "High" if x >= 4 else "Low"
)

# V2: fulfillment_speed — revised target after V1 showed no predictive power
FULFILLMENT_THRESHOLD = 4.4
df["fulfillment_speed"] = df["fulfillment_time_min"].apply(
    lambda x: "Fast" if x <= FULFILLMENT_THRESHOLD else "Slow"
)
step(f"Discretized targets:")
print(f"    V1 - satisfaction_level: High (>=4): {(df['satisfaction_level']=='High').sum():,}  |  Low (<=3): {(df['satisfaction_level']=='Low').sum():,}")
print(f"    V2 - fulfillment_speed:  Fast (<={FULFILLMENT_THRESHOLD} min): {(df['fulfillment_speed']=='Fast').sum():,}  |  Slow (>{FULFILLMENT_THRESHOLD} min): {(df['fulfillment_speed']=='Slow').sum():,}")

# Step 2.3 - feature selection
dt_v1_features = [
    "order_channel", "time_period", "store_location_type",
    "cart_size", "num_customizations", "fulfillment_time_min",
    "has_food_item", "is_rewards_member",
    "satisfaction_level"  # target V1
]
dt_v2_features = [
    "order_channel", "time_period", "store_location_type",
    "cart_size", "num_customizations", "has_food_item",
    "is_rewards_member", "customer_satisfaction",
    "fulfillment_speed"  # target V2
]
cluster_features = [
    "cart_size", "num_customizations", "total_spend",
    "fulfillment_time_min", "customer_satisfaction"
]

df_dt_v1   = df[dt_v1_features].copy()
df_dt_v2   = df[dt_v2_features].copy()
df_cluster = df[cluster_features].copy()
step(f"Selected features - DT V1: {len(dt_v1_features)-1} features + target  |  DT V2: {len(dt_v2_features)-1} features + target  |  Clustering: {len(cluster_features)} features")
print(f"    Excluded: identifiers (customer_id, order_id, store_id), raw date/time columns, redundant fields")

# Step 2.4 - boolean encoding
for df_dt in [df_dt_v1, df_dt_v2]:
    df_dt["has_food_item"]     = df_dt["has_food_item"].map({True: "True", False: "False"})
    df_dt["is_rewards_member"] = df_dt["is_rewards_member"].map({True: "True", False: "False"})
step("Encoded boolean columns to string True/False for ARFF compatibility")

# Step 2.5 - normalization
scaler = MinMaxScaler()
df_cluster_scaled = pd.DataFrame(
    scaler.fit_transform(df_cluster),
    columns=cluster_features
)
step("Applied Min-Max normalization to clustering features (all values now in [0, 1])")
print(f"\n    Reason: K-Means uses Euclidean distance - unscaled features would bias clusters toward high-variance variables")

# Step 2.6 - ARFF export
def quote(v):
    """Wrap value in single quotes if it contains a space or comma."""
    s = str(v)
    return f"'{s}'" if (" " in s or "," in s) else s

def to_arff(df, relation_name, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"@relation {relation_name}\n\n")
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                f.write(f"@attribute {col} NUMERIC\n")
            else:
                values = ",".join(quote(v) for v in sorted(df[col].unique().astype(str)))
                f.write(f"@attribute {col} {{{values}}}\n")
        f.write("\n@data\n")
        for _, row in df.iterrows():
            f.write(",".join(quote(v) if isinstance(v, str) else str(v) for v in row) + "\n")

to_arff(df_dt_v1,          "starbucks_decision_tree_v1", "weka/decision_tree_v1.arff")
to_arff(df_dt_v2,          "starbucks_decision_tree_v2", "weka/decision_tree_v2.arff")
to_arff(df_cluster_scaled, "starbucks_clustering",       "weka/clustering.arff")
step("Exported ARFF files to weka/")
print(f"    weka/decision_tree_v1.arff  -> target: satisfaction_level  ({len(df_dt_v1):,} instances)")
print(f"    weka/decision_tree_v2.arff  -> target: fulfillment_speed   ({len(df_dt_v2):,} instances)")
print(f"    weka/clustering.arff        -> unsupervised                ({len(df_cluster_scaled):,} instances)")

# ── STAGE 3 - Post-processing preview ──────────────────────────────────────

section("STAGE 3 - Processed data preview")

print("  First 5 rows - Decision Tree V1 (target: satisfaction_level):\n")
print(df_dt_v1.head().to_string(index=False))
print(f"\n{SUBDIV}")
print("  First 5 rows - Decision Tree V2 (target: fulfillment_speed):\n")
print(df_dt_v2.head().to_string(index=False))

print(f"\n{SUBDIV}")
print("  First 5 rows - Clustering dataset (normalized, no target column):\n")
print(df_cluster_scaled.head().round(4).to_string(index=False))

# ── DONE ───────────────────────────────────────────────────────────────────

section("PIPELINE COMPLETE")
print("  Next steps:")
print("    1. Open WEKA Explorer")
print("    2. Load weka/decision_tree.arff -> Classify -> J48 -> Start")
print("    3. Load weka/clustering.arff    -> Cluster  -> SimpleKMeans -> Start")
print("    4. Save results to weka/results/")
print("    5. Run notebooks/03_results_analysis.ipynb to visualize model outputs\n")
