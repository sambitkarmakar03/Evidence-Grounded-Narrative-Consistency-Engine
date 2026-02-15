import json
import pandas as pd
import numpy as np
import torch
import joblib
import warnings
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# --- 0. Setup ---
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Using device: {DEVICE}")

# --- 1. Load the JSON files you generated in the previous step ---
print("🔄 Loading pre-computed claims...")
try:
    with open("train_with_claims_final.json", "r") as f:
        train_data = json.load(f)
    with open("test_with_claims_final.json", "r") as f:
        test_data = json.load(f)
except FileNotFoundError as e:
    print(f"❌ Error: Could not find the JSON files. Ensure you ran the RAG script first! {e}")
    exit()

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# --- 2. Feature Engineering ---
print("🤖 Loading models for feature extraction...")
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=DEVICE)

def extract_ml_features(row):
    """Calculates scores based on the top 3 claims retrieved via RAG"""
    caption = str(row.get('caption', ''))
    content = str(row.get('content', ''))
    claims = [str(row.get(f'claim{i}', '')) for i in range(1, 4)]
    input_text = f"{caption} {content}".strip()
    
    valid_claims = [c for c in claims if c and "No context found" not in c]
    
    # Semantic Similarity (Bi-Encoder)
    if valid_claims and input_text:
        input_emb = bi_encoder.encode([input_text], convert_to_tensor=True, show_progress_bar=False)
        claim_embs = bi_encoder.encode(valid_claims, convert_to_tensor=True, show_progress_bar=False)
        sim_scores = util.cos_sim(input_emb, claim_embs)[0].cpu().numpy()
        avg_sim, max_sim = np.mean(sim_scores), np.max(sim_scores)
    else:
        avg_sim, max_sim = 0, 0

    # Reranker Scores (Cross-Encoder)
    rerank_scores = []
    for claim in claims:
        if input_text and claim and "No context found" not in claim:
            score = float(reranker.predict([(input_text, claim)]))
            rerank_scores.append(score)
        else:
            rerank_scores.append(-1.0)
    
    avg_rerank = np.mean(rerank_scores)

    return {
        'avg_sim_score': avg_sim,
        'max_sim_score': max_sim,
        'avg_rerank_score': avg_rerank,
        'claim_count': len(valid_claims),
        'has_no_context': sum(1 for c in claims if "No context found" in c),
        'interaction_score': avg_sim * avg_rerank
    }

# --- 3. Process Datasets ---
print("✨ Computing ML features...")
X_train_list = [extract_ml_features(row) for _, row in tqdm(train_df.iterrows(), total=len(train_df))]
X_test_list = [extract_ml_features(row) for _, row in tqdm(test_df.iterrows(), total=len(test_df))]

X = pd.DataFrame(X_train_list)
X_final_test = pd.DataFrame(X_test_list)
y = np.array([1 if l == 'consistent' else 0 for l in train_df['label']])

# Split for internal validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 4. Train XGBoost ---
print("🎯 Training XGBoost Classifier...")
# scale_pos_weight balances the classes if you have more consistent than inconsistent labels
ratio = len(y_train[y_train==0]) / max(1, len(y_train[y_train==1]))

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=ratio,
    random_state=42
)
model.fit(X_train, y_train)

# --- 5. Evaluation ---
y_pred = model.predict(X_val)
print("\n📊 Validation Performance:")
print(classification_report(y_val, y_pred, target_names=['inconsistent', 'consistent']))

# --- 6. Generate Final Results ---
print("\n🔮 Generating final results with numeric labels...")
test_preds = model.predict(X_final_test) # Already 1s and 0s from XGBoost
test_probs = model.predict_proba(X_final_test)

# Create the dossier list
dossier_list = []
for i, row in test_df.iterrows():
    # test_preds[i] is already 1 for consistent and 0 for inconsistent
    numeric_label = int(test_preds[i]) 
    
    dossier_list.append({
        "id": row.get('id'),
        "book_name": row.get('book_name'),
        "character": row.get('char'),
        "label": numeric_label,  # 1 = consistent, 0 = inconsistent
        "confidence": round(float(np.max(test_probs[i])), 4),
        "evidence_1": row.get('claim1'),
        "evidence_2": row.get('claim2'),
        "evidence_3": row.get('claim3')
    })

# Convert to DataFrame
report_df = pd.DataFrame(dossier_list)

# --- 7. Save to CSV ---
# Full Evidence Report
report_df.to_csv("consistency_evidence_report.csv", index=False)

# Simple Competition Format (id, label)
submission_df = report_df[['id', 'label']]
submission_df.to_csv("submission_numeric.csv", index=False)

print(f"✅ Success! Detailed report saved to: consistency_evidence_report.csv")
print(f"✅ Success! Numeric submission saved to: submission_numeric.csv")

# --- 8. Final Distribution Check ---
print("\n📊 Final Prediction Distribution:")
print(submission_df['label'].value_counts())