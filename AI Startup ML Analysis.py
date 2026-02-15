
# =============================================================================
# 🚀 AI STARTUP FUNDING - MACHINE LEARNING PROJECT
# =============================================================================
# This script walks you through a complete Machine Learning pipeline
# from loading raw data to evaluating a trained model.
# No prior ML knowledge needed — every step is explained!
# =============================================================================


# =============================================================================
# 1️⃣  IMPORT LIBRARIES
# =============================================================================
# Think of libraries as "toolboxes" — each one gives us special powers.

import zipfile                            # To extract ZIP files
import os                                 # To work with files and folders
import pandas as pd                       # pandas = our data table manager
import numpy as np                        # numpy = math and number crunching
import matplotlib.pyplot as plt           # matplotlib = draw charts
import seaborn as sns                     # seaborn = prettier, easier charts
import warnings                           # To suppress unimportant warnings
warnings.filterwarnings('ignore')         # Hide clutter in output

# sklearn = scikit-learn, the most popular ML library in Python
from sklearn.model_selection import train_test_split          # Split data
from sklearn.preprocessing import LabelEncoder                # Convert text → numbers
from sklearn.linear_model import LogisticRegression           # Simple ML model
from sklearn.ensemble import RandomForestClassifier           # Powerful ML model
from sklearn.metrics import (accuracy_score,                  # How often we're right
                              confusion_matrix,                # Detailed error table
                              classification_report,           # Full score card
                              ConfusionMatrixDisplay)          # Visualise errors

print("✅ All libraries imported successfully!\n")


# =============================================================================
# 2️⃣  LOAD THE DATASET
# =============================================================================
# We already have the CSV ready. Let's load it into a "DataFrame" (a table).
# A DataFrame is just like an Excel spreadsheet, but in Python.

csv_file = '/home/claude/ai_startup_funding_clean.csv'

# Load the CSV into a DataFrame
df = pd.read_csv(csv_file)

print("=" * 60)
print("📂 DATASET LOADED SUCCESSFULLY")
print("=" * 60)
print(f"\n🔢 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\n👀 First 5 rows of the dataset:")
print(df.head().to_string())
print("\n📋 Column names:", df.columns.tolist())


# =============================================================================
# 3️⃣  BASIC DATA EXPLORATION
# =============================================================================
# Before we do anything, we need to UNDERSTAND our data.
# This is like reading a book's table of contents before diving in.

print("\n" + "=" * 60)
print("🔍 BASIC DATA EXPLORATION")
print("=" * 60)

# --- 3a. Data types and non-null counts ---
print("\n📊 Dataset Info (column types & non-null counts):")
print(df.info())

# --- 3b. Statistical summary ---
print("\n📈 Statistical Summary of Numerical Columns:")
print(df.describe().round(2))

# --- 3c. Check for missing values ---
missing = df.isnull().sum()
print("\n❓ Missing Values Per Column:")
print(missing[missing >= 0].to_string())  # Show all columns

# --- 3d. Check for duplicate rows ---
num_duplicates = df.duplicated().sum()
print(f"\n🔁 Number of Duplicate Rows: {num_duplicates}")

# --- 3e. Value counts for our target column ---
print("\n🎯 Distribution of 'Profitability' (our target column):")
print(df['Profitability'].value_counts())

print("""
📝 EXPLORATION FINDINGS:
   • The dataset has 60 AI startup records with 11 features each.
   • 'Profitability' is our target — it tells us if a startup is profitable (TRUE/FALSE).
   • This is a CLASSIFICATION problem (predicting one of two categories).
   • Most columns look clean with proper data types.
""")


# =============================================================================
# 4️⃣  DATA CLEANING
# =============================================================================
# Real-world data is messy. We need to fix problems before training any model.

print("=" * 60)
print("🧹 DATA CLEANING")
print("=" * 60)

# --- Step 4a: Make a copy so we don't damage the original ---
# Always work on a copy — good habit!
df_clean = df.copy()

# --- Step 4b: Strip whitespace from column names ---
# Sometimes column names have hidden spaces like " Name " — let's remove them
df_clean.columns = df_clean.columns.str.strip()
print("✅ Column names stripped of whitespace.")

# --- Step 4c: Strip whitespace from text columns ---
# Text cells may also have leading/trailing spaces
text_columns = df_clean.select_dtypes(include='object').columns
for col in text_columns:
    df_clean[col] = df_clean[col].str.strip()
print("✅ Whitespace removed from all text columns.")

# --- Step 4d: Remove duplicate rows ---
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
after = len(df_clean)
print(f"✅ Duplicates removed: {before - after} rows dropped. Rows remaining: {after}")

# --- Step 4e: Handle missing values ---
# Check again after cleaning
missing_after = df_clean.isnull().sum().sum()
print(f"✅ Total missing values: {missing_after}")

if missing_after > 0:
    # Fill numerical columns with the median (middle value — robust to outliers)
    num_cols = df_clean.select_dtypes(include='number').columns
    df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())
    # Fill text columns with the most common value
    cat_cols = df_clean.select_dtypes(include='object').columns
    for col in cat_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    print("✅ Missing values filled (numbers → median, text → most common).")
else:
    print("✅ No missing values found — dataset is already clean!")

# --- Step 4f: Fix the Profitability column ---
# It's stored as 'TRUE'/'FALSE' strings or booleans — convert to 1/0
# 1 = Profitable, 0 = Not Profitable
df_clean['Profitability'] = df_clean['Profitability'].astype(str).str.upper().str.strip()
df_clean['Profitability'] = df_clean['Profitability'].map({'TRUE': 1, 'FALSE': 0})
print("✅ 'Profitability' converted: TRUE → 1, FALSE → 0")

# --- Step 4g: Ensure numeric columns are correct type ---
df_clean['Valuation ($M)'] = pd.to_numeric(df_clean['Valuation ($M)'], errors='coerce')
df_clean['Funding_Amount ($M)'] = pd.to_numeric(df_clean['Funding_Amount ($M)'], errors='coerce')
df_clean['Employee_Count'] = pd.to_numeric(df_clean['Employee_Count'], errors='coerce')
df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce')
print("✅ Numeric columns verified and converted.")

print(f"\n🟢 Clean dataset shape: {df_clean.shape}")


# =============================================================================
# 5️⃣  FEATURE SELECTION
# =============================================================================
# Features (X) = the input columns the model learns FROM
# Target  (y) = the output column the model tries to PREDICT
#
# 🎯 TARGET: 'Profitability'
# WHY? We want to predict whether a startup will be profitable
# based on things we know upfront — like funding amount, country, stage, etc.
#
# FEATURES: We drop columns that either ARE the target, or are just
# identifiers (like company name) that don't help with prediction.

print("\n" + "=" * 60)
print("🎯 FEATURE SELECTION")
print("=" * 60)

# Columns to drop from features:
# - 'Startup_Name'  → Just a label, no predictive value
# - 'Investors'     → Free-text with many unique values, hard to use directly
# - 'Profitability' → This IS our target, so it can't be a feature too

drop_cols = ['Startup_Name', 'Investors', 'Profitability']

# X = our features (everything except what we're predicting)
X = df_clean.drop(columns=drop_cols)

# y = our target (what we want the model to predict)
y = df_clean['Profitability']

print(f"\n📌 Features (X) — {X.shape[1]} columns:")
for col in X.columns:
    print(f"   • {col}")

print(f"\n🎯 Target (y): 'Profitability'")
print(f"   Distribution: {y.value_counts().to_dict()}")
print(f"   (1 = Profitable ✅, 0 = Not Profitable ❌)")


# =============================================================================
# 6️⃣  ENCODE CATEGORICAL VARIABLES
# =============================================================================
# Machine Learning models only understand NUMBERS.
# But many of our columns contain TEXT (e.g., "USA", "Series B", "Healthcare").
# We need to convert those text values into numbers — this is called ENCODING.
#
# We'll use LabelEncoder: it assigns a unique number to each unique text value.
# Example: "USA"→0, "UK"→1, "Canada"→2, "Germany"→3, "France"→4 ...

print("\n" + "=" * 60)
print("🔢 ENCODING CATEGORICAL VARIABLES")
print("=" * 60)

# Find all text (object) columns in our feature set
cat_features = X.select_dtypes(include='object').columns.tolist()
print(f"\nText columns to encode: {cat_features}")

# Create a LabelEncoder object — it's like a translation dictionary
encoder = LabelEncoder()

# Loop through each text column and encode it
for col in cat_features:
    # Fit the encoder on the column's values, then transform them to numbers
    X[col] = encoder.fit_transform(X[col].astype(str))
    print(f"   ✅ '{col}' encoded — {X[col].nunique()} unique categories → numbers 0 to {X[col].nunique()-1}")

print("\n🔍 First 5 rows AFTER encoding (all numbers now):")
print(X.head().to_string())

print("""
📝 WHAT IS ENCODING?
   Imagine teaching a math student about colours using only numbers.
   You might say: Red=0, Blue=1, Green=2.
   LabelEncoder does the same thing for our text columns!
""")


# =============================================================================
# 7️⃣  TRAIN-TEST SPLIT
# =============================================================================
# We can't train AND test on the same data — that would be like
# giving a student the exam answers before the test. Not fair!
#
# We split our data into:
# → Training set (80%) : The model LEARNS from this data
# → Testing set  (20%) : We EVALUATE the model on data it has NEVER seen
#
# random_state=42 means we get the same split every time we run (reproducible)

print("\n" + "=" * 60)
print("✂️  TRAIN-TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% goes to testing
    random_state=42,     # Seed for reproducibility
    stratify=y           # Keep the same TRUE/FALSE ratio in both splits
)

print(f"\n📚 Training set:  {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"🧪 Testing set:   {X_test.shape[0]}  samples ({X_test.shape[0]/len(X)*100:.0f}%)")

print(f"\n🎯 Target distribution in Training set:  {y_train.value_counts().to_dict()}")
print(f"🎯 Target distribution in Testing set:   {y_test.value_counts().to_dict()}")

print("""
📝 WHY SPLIT THE DATA?
   If you studied only the exam questions (training data) and were then
   tested on the SAME questions, you'd get 100% — but that's cheating!
   
   The test set simulates "real world" data the model has NEVER seen before.
   A model that does well on the test set is truly learning patterns,
   not just memorising the training data.
""")


# =============================================================================
# 8️⃣  MODEL TRAINING
# =============================================================================
# We'll train TWO models and compare them:
#
# MODEL 1: Logistic Regression
#   → Simple, fast, interpretable
#   → Despite its name, it's used for CLASSIFICATION (not regression)
#   → Works by finding a "boundary line" that separates the two classes
#
# MODEL 2: Random Forest Classifier
#   → More powerful, handles complex patterns
#   → Builds many decision trees and combines their answers (like asking 100 experts)
#   → Usually more accurate than Logistic Regression

print("\n" + "=" * 60)
print("🤖 MODEL TRAINING")
print("=" * 60)

# -----------------------------------------------
# MODEL 1: Logistic Regression
# -----------------------------------------------
print("\n--- Model 1: Logistic Regression ---")

# Create the model object (like setting up a blank notebook to learn in)
lr_model = LogisticRegression(
    max_iter=1000,    # How many times it can adjust its learning (more = more thorough)
    random_state=42   # For reproducibility
)

# Train the model on the training data
# .fit() is where the "learning" happens!
lr_model.fit(X_train, y_train)

print("✅ Logistic Regression trained!")
print("   The model found the best 'boundary' to separate profitable vs non-profitable startups.")

# -----------------------------------------------
# MODEL 2: Random Forest Classifier
# -----------------------------------------------
print("\n--- Model 2: Random Forest Classifier ---")

# Create the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,   # Build 100 decision trees
    max_depth=5,        # Each tree can have at most 5 levels (prevents over-memorising)
    random_state=42     # For reproducibility
)

# Train the model
rf_model.fit(X_train, y_train)

print("✅ Random Forest trained!")
print(f"   Built 100 decision trees, each learning slightly different patterns.")


# =============================================================================
# 9️⃣  MODEL EVALUATION
# =============================================================================
# Now we test each model on the TEST data (data it has NEVER seen).
# We'll use several metrics to understand how well each model performs.

print("\n" + "=" * 60)
print("📊 MODEL EVALUATION")
print("=" * 60)

# Helper function to print evaluation results neatly
def evaluate_model(model, X_test, y_test, model_name):
    print(f"\n{'='*50}")
    print(f"📌 {model_name}")
    print(f"{'='*50}")
    
    # Make predictions on the TEST set
    # .predict() = the model makes its best guess for each row
    y_pred = model.predict(X_test)
    
    # --- ACCURACY ---
    # What % of predictions were correct?
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Accuracy: {acc*100:.1f}%")
    print(f"   (The model predicted correctly {acc*100:.1f}% of the time)")
    
    # --- CLASSIFICATION REPORT ---
    # Precision: Of all the times we said "Profitable", how often were we right?
    # Recall:    Of all ACTUAL profitable startups, how many did we catch?
    # F1-Score:  Balance between Precision and Recall (higher = better)
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred,
                                  target_names=['Not Profitable (0)', 'Profitable (1)']))
    
    # --- CONFUSION MATRIX ---
    # A table showing:
    #   True Negatives  (correctly said NOT profitable)
    #   True Positives  (correctly said profitable)
    #   False Positives (wrongly said profitable)
    #   False Negatives (missed a profitable startup)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔲 Confusion Matrix:")
    print(f"   [[True  Neg  |  False Pos]]")
    print(f"   [[False Neg  |  True  Pos]]")
    print(f"   {cm}")
    
    return y_pred, cm, acc

# Evaluate both models
lr_pred, lr_cm, lr_acc = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
rf_pred, rf_cm, rf_acc = evaluate_model(rf_model, X_test, y_test, "Random Forest")

# --- Quick comparison ---
print(f"\n{'='*50}")
print("⚖️  MODEL COMPARISON SUMMARY")
print(f"{'='*50}")
print(f"   Logistic Regression : {lr_acc*100:.1f}% accuracy")
print(f"   Random Forest       : {rf_acc*100:.1f}% accuracy")
winner = "Random Forest" if rf_acc >= lr_acc else "Logistic Regression"
print(f"\n   🏆 Winner: {winner}")


# =============================================================================
# 🔟  VISUALISATIONS (BONUS)
# =============================================================================
# Charts make it much easier to understand results.
# We'll create 4 visualisations:
#   1. Distribution of Profitability (class balance)
#   2. Funding Amount vs Valuation coloured by Profitability
#   3. Confusion Matrices for both models
#   4. Feature Importance from Random Forest

print("\n" + "=" * 60)
print("📈 CREATING VISUALISATIONS")
print("=" * 60)

# Set a nice visual style
sns.set_style("whitegrid")
sns.set_palette("husl")

fig = plt.figure(figsize=(20, 18))
fig.suptitle("🚀 AI Startup Funding — ML Analysis Dashboard", 
             fontsize=18, fontweight='bold', y=0.98)

# -----------------------------------------------
# CHART 1: Profitability Distribution (Bar Chart)
# -----------------------------------------------
ax1 = fig.add_subplot(3, 3, 1)
profit_counts = df_clean['Profitability'].map({1: 'Profitable', 0: 'Not Profitable'}).value_counts()
colors = ['#2ecc71', '#e74c3c']
bars = ax1.bar(profit_counts.index, profit_counts.values, color=colors, 
               edgecolor='white', linewidth=1.5, width=0.5)
ax1.set_title('Target Variable Distribution', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Startups')
ax1.set_xlabel('Profitability')
for bar, val in zip(bars, profit_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
             str(val), ha='center', va='bottom', fontweight='bold')
ax1.set_ylim(0, max(profit_counts.values) + 3)

# -----------------------------------------------
# CHART 2: Funding Amount Distribution (Histogram)
# -----------------------------------------------
ax2 = fig.add_subplot(3, 3, 2)
profitable = df_clean[df_clean['Profitability'] == 1]['Funding_Amount ($M)']
not_profitable = df_clean[df_clean['Profitability'] == 0]['Funding_Amount ($M)']
ax2.hist(not_profitable, bins=15, alpha=0.6, color='#e74c3c', label='Not Profitable', edgecolor='white')
ax2.hist(profitable, bins=15, alpha=0.6, color='#2ecc71', label='Profitable', edgecolor='white')
ax2.set_title('Funding Amount Distribution\nby Profitability', fontsize=12, fontweight='bold')
ax2.set_xlabel('Funding Amount ($M)')
ax2.set_ylabel('Count')
ax2.legend()

# -----------------------------------------------
# CHART 3: Employee Count vs Valuation (Scatter)
# -----------------------------------------------
ax3 = fig.add_subplot(3, 3, 3)
scatter_colors = df_clean['Profitability'].map({1: '#2ecc71', 0: '#e74c3c'})
scatter = ax3.scatter(df_clean['Employee_Count'], df_clean['Valuation ($M)'],
                       c=scatter_colors, alpha=0.7, s=60, edgecolor='white', linewidth=0.5)
ax3.set_title('Employees vs Valuation\n(Green=Profitable, Red=Not)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Employee Count')
ax3.set_ylabel('Valuation ($M)')
ax3.set_yscale('log')  # Log scale helps show spread better

# -----------------------------------------------
# CHART 4: Funding Stage Distribution
# -----------------------------------------------
ax4 = fig.add_subplot(3, 3, 4)
stage_order = df_clean['Funding_Stage'].value_counts()
ax4.barh(stage_order.index, stage_order.values, 
          color=sns.color_palette("husl", len(stage_order)), edgecolor='white')
ax4.set_title('Funding Stage Distribution', fontsize=12, fontweight='bold')
ax4.set_xlabel('Number of Startups')
ax4.set_ylabel('Funding Stage')
for i, v in enumerate(stage_order.values):
    ax4.text(v + 0.1, i, str(v), va='center', fontweight='bold')

# -----------------------------------------------
# CHART 5: Confusion Matrix — Logistic Regression
# -----------------------------------------------
ax5 = fig.add_subplot(3, 3, 5)
labels = ['Not Profitable', 'Profitable']
sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels, ax=ax5,
            linewidths=2, linecolor='white', cbar=False,
            annot_kws={'size': 14, 'weight': 'bold'})
ax5.set_title(f'Confusion Matrix\nLogistic Regression ({lr_acc*100:.1f}% acc)', 
               fontsize=12, fontweight='bold')
ax5.set_ylabel('Actual Label')
ax5.set_xlabel('Predicted Label')

# -----------------------------------------------
# CHART 6: Confusion Matrix — Random Forest
# -----------------------------------------------
ax6 = fig.add_subplot(3, 3, 6)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=labels, yticklabels=labels, ax=ax6,
            linewidths=2, linecolor='white', cbar=False,
            annot_kws={'size': 14, 'weight': 'bold'})
ax6.set_title(f'Confusion Matrix\nRandom Forest ({rf_acc*100:.1f}% acc)', 
               fontsize=12, fontweight='bold')
ax6.set_ylabel('Actual Label')
ax6.set_xlabel('Predicted Label')

# -----------------------------------------------
# CHART 7: Feature Importance (Random Forest)
# -----------------------------------------------
ax7 = fig.add_subplot(3, 1, 3)
# Get feature importance scores from the trained Random Forest
importance_scores = rf_model.feature_importances_
feature_names = X.columns.tolist()

# Sort features by importance (highest first)
sorted_idx = np.argsort(importance_scores)[::-1]
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_scores = importance_scores[sorted_idx]

# Colour the bars: top 3 features get a special colour
bar_colors = ['#e74c3c' if i < 3 else '#3498db' for i in range(len(sorted_features))]

bars = ax7.bar(range(len(sorted_features)), sorted_scores, color=bar_colors, 
               edgecolor='white', linewidth=1.5)
ax7.set_xticks(range(len(sorted_features)))
ax7.set_xticklabels(sorted_features, rotation=35, ha='right', fontsize=10)
ax7.set_title('🌲 Random Forest — Feature Importance\n(How much each feature helps predict profitability)', 
               fontsize=12, fontweight='bold')
ax7.set_ylabel('Importance Score')
ax7.set_xlabel('Feature')

# Add value labels on top of each bar
for bar, score in zip(bars, sorted_scores):
    ax7.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
             f'{score:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add a legend for colours
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', label='Top 3 Most Important'),
                   Patch(facecolor='#3498db', label='Other Features')]
ax7.legend(handles=legend_elements, loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/mnt/user-data/outputs/ai_startup_ml_dashboard.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Dashboard saved: ai_startup_ml_dashboard.png")


# =============================================================================
# 📊 FEATURE IMPORTANCE — DETAILED PRINTOUT
# =============================================================================
print("\n" + "=" * 60)
print("🌲 RANDOM FOREST FEATURE IMPORTANCE (Detailed)")
print("=" * 60)
print("\nFeature importance tells us WHICH inputs the model relies on most.")
print("Higher score = more influential in predicting profitability.\n")

for rank, (feat, score) in enumerate(zip(sorted_features, sorted_scores), 1):
    bar = "█" * int(score * 200)
    print(f"  {rank:2}. {feat:<25} {bar:<20} {score:.4f}")


# =============================================================================
# 📝 FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("📝 FINAL PROJECT SUMMARY")
print("=" * 70)

print(f"""
🗂️  DATASET:
   • 60 AI startup funding records from 2020–2023
   • 11 features: Name, Industry, Country, Stage, Valuation, Funding,
     Investors, Year, Profitability, Employee Count, AI Model Type
   • Target variable: Profitability (1 = Profitable, 0 = Not Profitable)

🧹 CLEANING STEPS:
   • Stripped whitespace from column names and text values
   • Removed duplicate rows (none found)
   • Converted 'Profitability' from TRUE/FALSE text → 1/0 numbers
   • Verified numeric column types

🤖 MODELS TRAINED:
   • Logistic Regression  → Accuracy: {lr_acc*100:.1f}%
   • Random Forest        → Accuracy: {rf_acc*100:.1f}%
   • Best performer: {winner}

🏆 TOP FEATURES (what matters most for predicting profitability):
   1. {sorted_features[0]} (score: {sorted_scores[0]:.4f})
   2. {sorted_features[1]} (score: {sorted_scores[1]:.4f})
   3. {sorted_features[2]} (score: {sorted_scores[2]:.4f})

💡 WHAT COULD BE IMPROVED:
   ① More data  — 60 rows is small for ML. More data = better learning.
   ② Feature engineering — e.g. "Funding per Employee" ratio
   ③ Hyperparameter tuning — test different model settings automatically
   ④ Try more models — e.g. XGBoost, SVM, or Gradient Boosting
   ⑤ Cross-validation — a more robust evaluation technique
   ⑥ Handling class imbalance — if one class dominates, use SMOTE or weights

📊 VISUALISATIONS SAVED:
   • ai_startup_ml_dashboard.png — All charts in one dashboard
""")

print("🎉 Project complete! Scroll up to read through all the steps.\n")
