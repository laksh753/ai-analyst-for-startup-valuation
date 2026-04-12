# =============================================================================
# 🚀 STARTUP SUCCESS PREDICTION - MACHINE LEARNING PROJECT
# =============================================================================
# This script predicts whether a startup will succeed (IPO/Acquisition) or Fail
# using funding data, team info, and market characteristics.
# Perfect for ML beginners - every step is explained!
# =============================================================================


# =============================================================================
# 1️⃣  IMPORT LIBRARIES
# =============================================================================
# Libraries are like toolboxes - each one gives us special abilities.

import pandas as pd                       # pandas = manage data tables (like Excel)
import numpy as np                        # numpy = math and number operations
import matplotlib.pyplot as plt           # matplotlib = create charts and graphs
import seaborn as sns                     # seaborn = beautiful statistical charts
import warnings                           # to hide unnecessary warning messages
warnings.filterwarnings('ignore')         # keeps our output clean

# sklearn = scikit-learn, the most popular Machine Learning library
from sklearn.model_selection import train_test_split          # split data into training/testing
from sklearn.preprocessing import LabelEncoder                # convert text → numbers
from sklearn.linear_model import LogisticRegression           # simple ML classifier
from sklearn.ensemble import RandomForestClassifier           # powerful tree-based classifier
from sklearn.metrics import (accuracy_score,                  # how often we're correct
                              confusion_matrix,                # detailed error breakdown
                              classification_report,           # full performance report
                              ConfusionMatrixDisplay)          # visualize confusion matrix

print("✅ All libraries imported successfully!\n")


# =============================================================================
# 2️⃣  LOAD THE DATASET
# =============================================================================
# We have a CSV file with 100,000 startup records. Let's load it!

csv_file = '/home/claude/startup_success_dataset.csv'

# Read the CSV into a DataFrame (a fancy spreadsheet in Python)
df = pd.read_csv(csv_file)

print("=" * 70)
print("📂 DATASET LOADED SUCCESSFULLY")
print("=" * 70)
print(f"\n🔢 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print("\n👀 First 5 rows of the dataset:")
print(df.head().to_string())
print("\n📋 Column names:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2}. {col}")


# =============================================================================
# 3️⃣  BASIC DATA EXPLORATION
# =============================================================================
# Before building models, we must UNDERSTAND our data.
# This is like studying a map before going on a road trip.

print("\n" + "=" * 70)
print("🔍 BASIC DATA EXPLORATION")
print("=" * 70)

# --- 3a. Data types and non-null counts ---
print("\n📊 Dataset Info (column types & non-null counts):")
print(df.info())

# --- 3b. Statistical summary of numerical columns ---
print("\n📈 Statistical Summary (Numerical Columns):")
print(df.describe().round(2))

# --- 3c. Check for missing values ---
missing_count = df.isnull().sum()
print("\n❓ Missing Values Per Column:")
print(missing_count)
print(f"\n   Total missing values: {missing_count.sum()}")

# --- 3d. Check for duplicate rows ---
num_duplicates = df.duplicated().sum()
print(f"\n🔁 Number of Duplicate Rows: {num_duplicates}")

# --- 3e. Check the target variable distribution ---
print("\n🎯 Distribution of 'outcome' (our target variable):")
outcome_counts = df['outcome'].value_counts()
print(outcome_counts)
print(f"\n   Percentages:")
for outcome, count in outcome_counts.items():
    pct = (count / len(df)) * 100
    print(f"   {outcome:15} : {pct:5.1f}%")

print("""
📝 EXPLORATION FINDINGS:
   • We have 100,000 startup records with 11 features each
   • Target variable 'outcome' has 3 classes: Acquisition, Failure, IPO
   • This is a MULTI-CLASS CLASSIFICATION problem
   • We'll predict which of the 3 outcomes a startup will achieve
   • All columns have complete data (no missing values!)
""")


# =============================================================================
# 4️⃣  DATA CLEANING
# =============================================================================
# Real data often needs cleaning. Let's fix any issues we find.

print("=" * 70)
print("🧹 DATA CLEANING")
print("=" * 70)

# --- Step 4a: Make a copy so we don't damage the original ---
df_clean = df.copy()

# --- Step 4b: Strip whitespace from column names ---
# Sometimes column names have hidden spaces like " Name " — remove them
df_clean.columns = df_clean.columns.str.strip()
print("✅ Column names stripped of whitespace.")

# --- Step 4c: Strip whitespace from text columns ---
text_columns = df_clean.select_dtypes(include='object').columns
for col in text_columns:
    df_clean[col] = df_clean[col].str.strip()
print("✅ Whitespace removed from all text columns.")

# --- Step 4d: Remove duplicate rows ---
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
after = len(df_clean)
print(f"✅ Duplicates removed: {before - after} rows dropped. Rows remaining: {after:,}")

# --- Step 4e: Handle missing values (if any) ---
missing_after = df_clean.isnull().sum().sum()
print(f"✅ Total missing values after cleaning: {missing_after}")

if missing_after > 0:
    # For numerical columns: fill with median (middle value, robust to outliers)
    num_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())
    
    # For text columns: fill with mode (most common value)
    cat_cols = df_clean.select_dtypes(include='object').columns
    for col in cat_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    print("✅ Missing values filled.")
else:
    print("✅ No missing values found — dataset is already clean!")

# --- Step 4f: Check data types are correct ---
print("\n✅ Data types verified:")
print(df_clean.dtypes)

print(f"\n🟢 Clean dataset shape: {df_clean.shape}")


# =============================================================================
# 5️⃣  FEATURE SELECTION
# =============================================================================
# Features (X) = input columns the model learns FROM
# Target  (y) = output column the model tries to PREDICT
#
# 🎯 TARGET: 'outcome'
# WHY? We want to predict if a startup will succeed (IPO/Acquisition) or Fail
# based on early-stage signals like funding, team size, market size, etc.

print("\n" + "=" * 70)
print("🎯 FEATURE SELECTION")
print("=" * 70)

# The 'outcome' column is what we're trying to predict
# All other columns are features we'll use to make the prediction
X = df_clean.drop(columns=['outcome'])
y = df_clean['outcome']

print(f"\n📌 Features (X) — {X.shape[1]} columns:")
for i, col in enumerate(X.columns, 1):
    print(f"   {i:2}. {col}")

print(f"\n🎯 Target (y): 'outcome'")
print(f"   Classes: {y.unique().tolist()}")
print(f"   Distribution:")
for outcome, count in y.value_counts().items():
    print(f"   • {outcome:15} : {count:7,} ({count/len(y)*100:5.1f}%)")


# =============================================================================
# 6️⃣  ENCODE CATEGORICAL VARIABLES
# =============================================================================
# Machine Learning models only understand NUMBERS, not text.
# We need to convert text values into numbers — this is called ENCODING.
#
# We'll use LabelEncoder: assigns a unique number to each text value.
# Example: "angel" → 0, "tier1_vc" → 1, "tier2_vc" → 2, etc.

print("\n" + "=" * 70)
print("🔢 ENCODING CATEGORICAL VARIABLES")
print("=" * 70)

# Find all text columns in our features
cat_features = X.select_dtypes(include='object').columns.tolist()
print(f"\nText columns to encode: {cat_features}")

# Create a dictionary to store our encoders (so we can reverse them later if needed)
encoders = {}

# Encode each text column
for col in cat_features:
    encoders[col] = LabelEncoder()
    X[col] = encoders[col].fit_transform(X[col].astype(str))
    unique_count = X[col].nunique()
    print(f"   ✅ '{col}' encoded — {unique_count} unique values → numbers 0 to {unique_count-1}")

# Also encode the TARGET variable (outcome)
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

print(f"\n   ✅ Target 'outcome' encoded:")
for i, label in enumerate(target_encoder.classes_):
    print(f"      {label:15} → {i}")

print("\n🔍 First 5 rows AFTER encoding:")
print(X.head().to_string())

print("""
📝 WHAT IS ENCODING?
   Think of it like translating languages for someone who only speaks numbers.
   "angel" becomes 0, "tier1_vc" becomes 1, etc.
   The model can now do math with these values!
""")


# =============================================================================
# 7️⃣  TRAIN-TEST SPLIT
# =============================================================================
# We NEVER train and test on the same data — that would be like
# giving a student the exact exam questions before the test!
#
# We split our data into:
# → Training set (80%) : Model LEARNS patterns from this
# → Testing set  (20%) : We EVALUATE on data the model has NEVER seen
#
# This tests if the model truly learned patterns vs. just memorizing answers.

print("\n" + "=" * 70)
print("✂️  TRAIN-TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,       # 20% for testing
    random_state=42,     # seed for reproducibility (same split every time)
    stratify=y_encoded   # keep same class distribution in train & test
)

print(f"\n📚 Training set:  {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"🧪 Testing set:   {X_test.shape[0]:,}  samples ({X_test.shape[0]/len(X)*100:.0f}%)")

# Show class distribution in both sets
print(f"\n🎯 Target distribution in Training set:")
train_dist = pd.Series(y_train).value_counts()
for class_num, count in train_dist.items():
    class_name = target_encoder.classes_[class_num]
    print(f"   {class_name:15} : {count:7,} ({count/len(y_train)*100:5.1f}%)")

print(f"\n🎯 Target distribution in Testing set:")
test_dist = pd.Series(y_test).value_counts()
for class_num, count in test_dist.items():
    class_name = target_encoder.classes_[class_num]
    print(f"   {class_name:15} : {count:7,} ({count/len(y_test)*100:5.1f}%)")

print("""
📝 WHY SPLIT THE DATA?
   Imagine studying from a practice exam, then being tested on THE SAME QUESTIONS.
   You'd score 100%, but that doesn't prove you understand the material!
   
   The test set simulates "real world" data the model has NEVER seen.
   Only if the model performs well on unseen data do we know it truly learned.
""")


# =============================================================================
# 8️⃣  MODEL TRAINING
# =============================================================================
# We'll train TWO models and compare them:
#
# MODEL 1: Logistic Regression
#   → Simple, fast, interpretable
#   → Good baseline for classification
#   → Despite the name, it's for CLASSIFICATION not regression
#
# MODEL 2: Random Forest Classifier
#   → More powerful, handles complex patterns
#   → Builds many decision trees and votes on the answer
#   → Usually more accurate than Logistic Regression

print("\n" + "=" * 70)
print("🤖 MODEL TRAINING")
print("=" * 70)

# -----------------------------------------------
# MODEL 1: Logistic Regression
# -----------------------------------------------
print("\n--- Model 1: Logistic Regression ---")

lr_model = LogisticRegression(
    max_iter=1000,        # number of iterations to find best parameters
    random_state=42       # for reproducibility
)

# .fit() is where the LEARNING happens!
lr_model.fit(X_train, y_train)

print("✅ Logistic Regression trained!")
print("   The model learned patterns to separate Acquisition/Failure/IPO startups.")

# -----------------------------------------------
# MODEL 2: Random Forest Classifier
# -----------------------------------------------
print("\n--- Model 2: Random Forest Classifier ---")

rf_model = RandomForestClassifier(
    n_estimators=100,     # build 100 decision trees
    max_depth=15,         # each tree can have at most 15 levels
    random_state=42,      # for reproducibility
    n_jobs=-1             # use all CPU cores (faster!)
)

# Train the Random Forest
rf_model.fit(X_train, y_train)

print("✅ Random Forest trained!")
print(f"   Built {rf_model.n_estimators} decision trees that vote on predictions.")


# =============================================================================
# 9️⃣  MODEL EVALUATION
# =============================================================================
# Now we test both models on the TEST data (data they've NEVER seen).
# We'll use several metrics to understand performance.

print("\n" + "=" * 70)
print("📊 MODEL EVALUATION")
print("=" * 70)

# Helper function to evaluate and print results
def evaluate_model(model, X_test, y_test, model_name, target_encoder):
    print(f"\n{'='*60}")
    print(f"📌 {model_name}")
    print(f"{'='*60}")
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # --- ACCURACY ---
    # What % of predictions were correct?
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Accuracy: {acc*100:.2f}%")
    print(f"   (The model predicted correctly {acc*100:.1f}% of the time)")
    
    # --- CLASSIFICATION REPORT ---
    # Precision: Of all times we said "X", how often were we right?
    # Recall:    Of all actual "X" cases, how many did we catch?
    # F1-Score:  Harmonic mean of Precision & Recall (1 is perfect)
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred,
                                  target_names=target_encoder.classes_))
    
    # --- CONFUSION MATRIX ---
    # Shows actual vs predicted for each class
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔲 Confusion Matrix:")
    print(f"   Rows = Actual class | Columns = Predicted class")
    print(f"   Classes: {target_encoder.classes_.tolist()}")
    print(cm)
    
    return y_pred, cm, acc

# Evaluate both models
lr_pred, lr_cm, lr_acc = evaluate_model(lr_model, X_test, y_test, 
                                         "Logistic Regression", target_encoder)
rf_pred, rf_cm, rf_acc = evaluate_model(rf_model, X_test, y_test, 
                                         "Random Forest", target_encoder)

# --- Quick comparison ---
print(f"\n{'='*60}")
print("⚖️  MODEL COMPARISON SUMMARY")
print(f"{'='*60}")
print(f"   Logistic Regression : {lr_acc*100:.2f}% accuracy")
print(f"   Random Forest       : {rf_acc*100:.2f}% accuracy")
winner = "Random Forest" if rf_acc >= lr_acc else "Logistic Regression"
print(f"\n   🏆 Winner: {winner}")


# =============================================================================
# 🔟  VISUALIZATIONS (BONUS)
# =============================================================================
# Charts make results much easier to understand!

print("\n" + "=" * 70)
print("📈 CREATING VISUALIZATIONS")
print("=" * 70)

# Set visual style
sns.set_style("whitegrid")
sns.set_palette("husl")

fig = plt.figure(figsize=(20, 14))
fig.suptitle("🚀 Startup Success Prediction — ML Analysis Dashboard", 
             fontsize=20, fontweight='bold', y=0.98)

# -----------------------------------------------
# CHART 1: Target Distribution (Bar Chart)
# -----------------------------------------------
ax1 = fig.add_subplot(2, 3, 1)
outcome_counts = df_clean['outcome'].value_counts()
colors = ['#2ecc71', '#e74c3c', '#f39c12']
bars = ax1.bar(outcome_counts.index, outcome_counts.values, color=colors, 
               edgecolor='white', linewidth=2, width=0.6)
ax1.set_title('Outcome Distribution (Full Dataset)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Number of Startups')
ax1.set_xlabel('Outcome')
for bar, val in zip(bars, outcome_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 500,
             f'{val:,}', ha='center', va='bottom', fontweight='bold')
ax1.set_ylim(0, max(outcome_counts.values) * 1.1)

# -----------------------------------------------
# CHART 2: Funding Rounds Distribution
# -----------------------------------------------
ax2 = fig.add_subplot(2, 3, 2)
funding_dist = df_clean['funding_rounds'].value_counts().sort_index()
ax2.bar(funding_dist.index, funding_dist.values, color='#3498db', 
        edgecolor='white', linewidth=1.5)
ax2.set_title('Funding Rounds Distribution', fontsize=13, fontweight='bold')
ax2.set_xlabel('Number of Funding Rounds')
ax2.set_ylabel('Count')

# -----------------------------------------------
# CHART 3: Revenue vs Product Traction (Scatter)
# -----------------------------------------------
ax3 = fig.add_subplot(2, 3, 3)
outcome_colors = df_clean['outcome'].map({
    'Acquisition': '#2ecc71',
    'Failure': '#e74c3c',
    'IPO': '#f39c12'
})
scatter = ax3.scatter(df_clean['product_traction_users'], 
                       df_clean['revenue_million'],
                       c=outcome_colors, alpha=0.4, s=5, edgecolor='none')
ax3.set_title('Revenue vs Product Traction\nby Outcome', fontsize=13, fontweight='bold')
ax3.set_xlabel('Product Traction (Users)')
ax3.set_ylabel('Revenue ($M)')
ax3.set_xscale('log')
ax3.set_yscale('log')
# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', label='Acquisition'),
    Patch(facecolor='#e74c3c', label='Failure'),
    Patch(facecolor='#f39c12', label='IPO')
]
ax3.legend(handles=legend_elements, loc='upper left', fontsize=9)

# -----------------------------------------------
# CHART 4: Model Accuracy Comparison
# -----------------------------------------------
ax4 = fig.add_subplot(2, 3, 4)
models = ['Logistic\nRegression', 'Random\nForest']
accuracies = [lr_acc * 100, rf_acc * 100]
bars = ax4.bar(models, accuracies, color=['#3498db', '#9b59b6'], 
               edgecolor='white', linewidth=2, width=0.5)
ax4.set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
ax4.set_ylabel('Accuracy (%)')
ax4.set_ylim(0, 100)
for bar, val in zip(bars, accuracies):
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
             f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

# -----------------------------------------------
# CHART 5: Confusion Matrix — Logistic Regression
# -----------------------------------------------
ax5 = fig.add_subplot(2, 3, 5)
disp_lr = ConfusionMatrixDisplay(confusion_matrix=lr_cm, 
                                   display_labels=target_encoder.classes_)
disp_lr.plot(ax=ax5, cmap='Blues', colorbar=False)
ax5.set_title(f'Confusion Matrix\nLogistic Regression ({lr_acc*100:.2f}% acc)', 
               fontsize=13, fontweight='bold')

# -----------------------------------------------
# CHART 6: Confusion Matrix — Random Forest
# -----------------------------------------------
ax6 = fig.add_subplot(2, 3, 6)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=rf_cm, 
                                   display_labels=target_encoder.classes_)
disp_rf.plot(ax=ax6, cmap='Greens', colorbar=False)
ax6.set_title(f'Confusion Matrix\nRandom Forest ({rf_acc*100:.2f}% acc)', 
               fontsize=13, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/mnt/user-data/outputs/startup_success_dashboard.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Dashboard saved: startup_success_dashboard.png")


# =============================================================================
# 🌲 FEATURE IMPORTANCE (Random Forest Only)
# =============================================================================
print("\n" + "=" * 70)
print("🌲 RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)
print("\nFeature importance tells us WHICH inputs matter most for predictions.")
print("Higher score = more influential in predicting outcome.\n")

# Get feature importance scores
importance_scores = rf_model.feature_importances_
feature_names = X.columns.tolist()

# Sort by importance (highest first)
sorted_idx = np.argsort(importance_scores)[::-1]
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_scores = importance_scores[sorted_idx]

# Print ranked list
for rank, (feat, score) in enumerate(zip(sorted_features, sorted_scores), 1):
    bar = "█" * int(score * 300)
    print(f"  {rank:2}. {feat:<30} {bar:<30} {score:.4f}")


# =============================================================================
# 📝 FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("📝 FINAL PROJECT SUMMARY")
print("=" * 80)

print(f"""
🗂️  DATASET:
   • {len(df_clean):,} startup records from various sectors (AI, Health, Fintech, SaaS, etc.)
   • 10 input features + 1 target variable (outcome)
   • Target has 3 classes: Acquisition, Failure, IPO
   • Class distribution:
     - Acquisition: {outcome_counts.get('Acquisition', 0):,} ({outcome_counts.get('Acquisition', 0)/len(df_clean)*100:.1f}%)
     - Failure:     {outcome_counts.get('Failure', 0):,} ({outcome_counts.get('Failure', 0)/len(df_clean)*100:.1f}%)
     - IPO:         {outcome_counts.get('IPO', 0):,} ({outcome_counts.get('IPO', 0)/len(df_clean)*100:.1f}%)

🧹 CLEANING STEPS:
   • Stripped whitespace from column names and text values
   • Removed {before - after} duplicate rows
   • Encoded categorical variables (investor_type, sector, founder_background)
   • No missing values found — clean dataset!

🤖 MODELS TRAINED:
   • Logistic Regression  → Accuracy: {lr_acc*100:.2f}%
   • Random Forest        → Accuracy: {rf_acc*100:.2f}%
   • Best performer: {winner}

🏆 TOP 3 FEATURES (what matters most for predicting outcome):
   1. {sorted_features[0]:<30} (importance: {sorted_scores[0]:.4f})
   2. {sorted_features[1]:<30} (importance: {sorted_scores[1]:.4f})
   3. {sorted_features[2]:<30} (importance: {sorted_scores[2]:.4f})

💡 KEY INSIGHTS:
   • With 100K records, Random Forest likely performs better due to data sufficiency
   • The model can predict startup outcomes with {max(lr_acc, rf_acc)*100:.1f}% accuracy
   • {sorted_features[0]} is the strongest predictor — it matters most!
   • Multi-class classification is harder than binary (2-class) problems

💡 WHAT COULD BE IMPROVED:
   ① Feature engineering — create ratios like "revenue per employee"
   ② Hyperparameter tuning — use GridSearchCV to find optimal settings
   ③ Try more models — XGBoost, LightGBM, or Neural Networks
   ④ Cross-validation — more robust evaluation than single train/test split
   ⑤ Address class imbalance — IPO class is rare, consider SMOTE or class weights
   ⑥ Collect more features — founder education, previous startup exits, etc.

📊 VISUALIZATIONS SAVED:
   • startup_success_dashboard.png — All charts in one dashboard
""")

print("🎉 Project complete! You now know how to build and evaluate ML classifiers.\n")
