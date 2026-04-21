import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import matplotlib.gridspec as gridspec
from sklearn.metrics import (silhouette_score, silhouette_samples,davies_bouldin_score, calinski_harabasz_score)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix



warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)

# ── 1. LOAD
print("Loading data...")
df = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
print(f"  Raw shape: {df.shape}")

# ── 2. CLEAN
df = df.dropna(subset=["Customer ID"])
df["IsCancellation"] = df["Invoice"].astype(str).str.startswith("C")
df = df[df["Price"] > 0]
df = df[(df["Quantity"] > 0) | (df["IsCancellation"])]
df["LineRevenue"] = df["Quantity"] * df["Price"]
df["Customer ID"] = df["Customer ID"].astype(int)
print(f"  After cleaning: {df.shape} | Unique customers: {df['Customer ID'].nunique()}")

# ── 3. AGGREGATE TO CUSTOMER LEVEL
SNAPSHOT  = df["InvoiceDate"].max() + pd.Timedelta(days=1)
purchases = df[~df["IsCancellation"]].copy()
cancels   = df[df["IsCancellation"]].copy()

inv_counts    = purchases.groupby("Customer ID")["Invoice"].nunique().rename("Frequency")
cancel_counts = (cancels.groupby("Customer ID")["Invoice"]
                 .nunique().rename("CancelCount").reindex(inv_counts.index, fill_value=0))
monetary      = purchases.groupby("Customer ID")["LineRevenue"].sum().rename("Monetary")
recency       = ((SNAPSHOT - purchases.groupby("Customer ID")["InvoiceDate"].max()).dt.days).rename("Recency")
unique_prods  = purchases.groupby("Customer ID")["StockCode"].nunique().rename("UniqueProducts")
basket        = (purchases.groupby(["Customer ID","Invoice"]).size()
                 .groupby("Customer ID").mean().rename("AvgBasketSize"))

purchases["IsDiscountSeason"] = purchases["InvoiceDate"].dt.month.isin([11, 12]).astype(int)
seasonal_grp  = purchases.groupby(["Customer ID","Invoice"])["IsDiscountSeason"].max().groupby("Customer ID")
seasonal_conc = (seasonal_grp.sum() / seasonal_grp.count()).rename("SeasonalConcentration")

purchase_spread = (purchases.groupby("Customer ID")["InvoiceDate"]
                   .apply(lambda x: x.dt.to_period("M").nunique()).rename("PurchaseSpread"))
bulk            = purchases.groupby("Customer ID")["Quantity"].mean().rename("BulkBuyer")
price_std       = purchases.groupby("Customer ID")["Price"].std().fillna(0).rename("PriceVariance")

def avg_gap(dates):
    d = sorted(dates.dt.normalize().unique())
    if len(d) < 2: return np.nan
    return np.mean([(d[i+1]-d[i]).days for i in range(len(d)-1)])

avg_days = purchases.groupby("Customer ID")["InvoiceDate"].apply(avg_gap).rename("AvgDaysBetweenOrders")

cust = pd.concat([inv_counts, cancel_counts, monetary, recency,
                  unique_prods, basket, seasonal_conc, purchase_spread,
                  bulk, price_std, avg_days], axis=1).dropna()
cust["ReturnRate"]    = cust["CancelCount"] / (cust["Frequency"] + cust["CancelCount"])
cust["AvgOrderValue"] = cust["Monetary"] / cust["Frequency"]
print(f"  Customer-level shape: {cust.shape}")

# ── 4. TARGET VARIABLE
# Seasonal Deal Seeker if >= 2 of 3 behavioral conditions:
#   C1: SeasonalConcentration >= 60th percentile (heavy Nov-Dec buyer)
#   C2: PurchaseSpread <= 35th percentile (sporadic, not year-round)
#   C3: ReturnRate >= 65th percentile (high cancellation rate)
t1 = cust["SeasonalConcentration"].quantile(0.65)
t2 = cust["PurchaseSpread"].quantile(0.30)
t3 = cust["ReturnRate"].quantile(0.70)

c1 = (cust["SeasonalConcentration"] >= t1).astype(int)
c2 = (cust["PurchaseSpread"]        <= t2).astype(int)
c3 = (cust["ReturnRate"]            >= t3).astype(int)
cust["Customer_Type"] = ((c1 + c2 + c3) >= 2).astype(int)

n_dh = cust["Customer_Type"].sum()
print(f"\n  Deal Seekers: {n_dh} ({n_dh/len(cust):.1%}) | Loyal: {len(cust)-n_dh} ({1-n_dh/len(cust):.1%})")
print(f"  Thresholds → SeasonalConc>={t1:.2f}, Spread<={t2:.0f}mo, ReturnRate>={t3:.2f}")

# ── 5. FEATURE SELECTION ──────────────────────────────────────────────────────
selected = ["Recency","Frequency","Monetary","AvgOrderValue",
            "SeasonalConcentration","PurchaseSpread","UniqueProducts",
            "AvgBasketSize","BulkBuyer","PriceVariance","ReturnRate","AvgDaysBetweenOrders"]

X = cust[selected].copy()
y = cust["Customer_Type"]
for col in X.columns:
    X[col] = X[col].clip(upper=X[col].quantile(0.99))  # cap outliers at 99th pct

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  K-MEANS
inertias = [KMeans(n_clusters=k,random_state=42,n_init=10).fit(X_scaled).inertia_ for k in range(1,11)]

plt.figure(figsize=(8,5))
plt.plot(range(1,11), inertias, marker="o", color="#58a6ff", linewidth=2.5, markersize=8)
plt.axvline(x=3, color="#f78166", linestyle="--", linewidth=1.5, label="Selected k=3")
plt.xlabel("Number of Clusters (k)"); plt.ylabel("Within-Cluster Sum of Squares")
plt.title("Elbow Method: Justifying K=3"); plt.xticks(range(1,11)); plt.legend()
plt.tight_layout(); plt.savefig("figures/elbow_method.png", dpi=150); plt.close()

km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
cust["Cluster"] = km3.fit_predict(X_scaled)
dh_r = cust.groupby("Cluster")["Customer_Type"].mean().sort_values()
cmap = {dh_r.index[0]:"Loyal", dh_r.index[1]:"Mixed", dh_r.index[2]:"Deal Seeker"}
cust["Cluster_Label"] = cust["Cluster"].map(cmap)
cluster_order = ["Loyal","Mixed","Deal Seeker"]

k_range   = range(2, 8)
inertias  = []
sil_scores = []
db_scores  = []
ch_scores  = []

print("\n=== CLUSTER VALIDATION METRICS ===")
print(f"{'k':>4} {'Inertia':>12} {'Silhouette':>12} {'Davies-Bouldin':>16} {'Calinski-Harabasz':>20}")
print("-" * 68)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)           # X_scaled from main script
    inertias.append(km.inertia_)
    sil  = silhouette_score(X_scaled, labels)
    db   = davies_bouldin_score(X_scaled, labels)
    ch   = calinski_harabasz_score(X_scaled, labels)
    sil_scores.append(sil)
    db_scores.append(db)
    ch_scores.append(ch)
    marker = " ◄ SELECTED" if k == 3 else ""
    print(f"{k:>4} {km.inertia_:>12.1f} {sil:>12.4f} {db:>16.4f} {ch:>20.1f}{marker}")

print("-" * 68)
print(f"\nFor k=3 specifically:")
km3_labels = km3.labels_                        # km3 from main script
sil_k3 = silhouette_score(X_scaled, km3_labels)
db_k3  = davies_bouldin_score(X_scaled, km3_labels)
ch_k3  = calinski_harabasz_score(X_scaled, km3_labels)
print(f"  Silhouette Score     : {sil_k3:.4f}  (range -1 to 1; higher = better)")
print(f"  Davies-Bouldin Index : {db_k3:.4f}  (lower = better)")
print(f"  Calinski-Harabasz    : {ch_k3:.1f}  (higher = better)")

# ── SILHOUETTE SAMPLE SCORES PER CLUSTER
sil_samples = silhouette_samples(X_scaled, km3_labels)
cust["SilhouetteScore"] = sil_samples

print("\n=== PER-CLUSTER SILHOUETTE SCORES ===")
for lbl in ["Loyal", "Mixed", "Deal Seeker"]:
    mask = cust["Cluster_Label"] == lbl
    mean_s = cust.loc[mask, "SilhouetteScore"].mean()
    print(f"  {lbl:12s}: mean silhouette = {mean_s:.4f}")

# ── COMBINED FIGURE: ELBOW + SILHOUETTE
fig = plt.figure(figsize=(14, 5))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# LEFT: Elbow + dual-axis silhouette overlay
ax1 = fig.add_subplot(gs[0])
ax2 = ax1.twinx()

k_vals = list(k_range)
color_inertia = "#58a6ff"
color_sil     = "#f87171"

line1, = ax1.plot(k_vals, inertias, marker="o", color=color_inertia,
                  linewidth=2.5, markersize=8, label="Inertia (WCSS)")
ax1.set_xlabel("Number of Clusters (k)", fontsize=11)
ax1.set_ylabel("Within-Cluster Sum of Squares", color=color_inertia, fontsize=10)
ax1.tick_params(axis="y", labelcolor=color_inertia)
ax1.set_xticks(k_vals)

line2, = ax2.plot(k_vals, sil_scores, marker="s", color=color_sil,
                  linewidth=2.5, markersize=8, linestyle="--", label="Silhouette Score")
ax2.set_ylabel("Silhouette Score", color=color_sil, fontsize=10)
ax2.tick_params(axis="y", labelcolor=color_sil)

ax1.axvline(x=3, color="#888", linestyle=":", linewidth=1.5, label="k = 3 selected")
ax1.set_title("Elbow Method & Silhouette Scores\nJustifying k = 3", fontsize=12, fontweight="bold")
lines = [line1, line2]
labels_leg = [l.get_label() for l in lines]
ax1.legend(lines, labels_leg, loc="upper right", fontsize=9)

# Annotate k=3 point
idx3 = k_vals.index(3)
ax1.annotate(f"k=3\nWCSS={inertias[idx3]:,.0f}",
             xy=(3, inertias[idx3]), xytext=(4, inertias[idx3]*1.05),
             arrowprops=dict(arrowstyle="->", color="#58a6ff"), color="#58a6ff", fontsize=8)
ax2.annotate(f"Sil={sil_scores[idx3]:.3f}",
             xy=(3, sil_scores[idx3]), xytext=(4.2, sil_scores[idx3]-0.01),
             arrowprops=dict(arrowstyle="->", color=color_sil), color=color_sil, fontsize=8)

# RIGHT: Silhouette subplot plot per cluster
ax3 = fig.add_subplot(gs[1])
cluster_colors = {"Loyal": "#4ade80", "Mixed": "#facc15", "Deal Seeker": "#f87171"}
y_lower = 10
ytick_positions = []
ytick_labels    = []

for lbl in ["Loyal", "Mixed", "Deal Seeker"]:
    mask   = cust["Cluster_Label"] == lbl
    vals   = np.sort(cust.loc[mask, "SilhouetteScore"].values)
    n      = len(vals)
    y_upper = y_lower + n
    ax3.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                      facecolor=cluster_colors[lbl], alpha=0.8, edgecolor="none")
    ytick_positions.append(y_lower + n / 2)
    ytick_labels.append(f"{lbl}\n(n={n})")
    y_lower = y_upper + 10

ax3.axvline(x=sil_k3, color="red", linestyle="--", linewidth=1.5,
            label=f"Mean = {sil_k3:.3f}")
ax3.set_xlabel("Silhouette Coefficient", fontsize=11)
ax3.set_ylabel("Cluster", fontsize=11)
ax3.set_title(f"Silhouette Analysis for k = 3\nMean Score = {sil_k3:.3f}", fontsize=12, fontweight="bold")
ax3.set_yticks(ytick_positions)
ax3.set_yticklabels(ytick_labels, fontsize=9)
ax3.legend(fontsize=9)
ax3.set_xlim([-0.3, 1.0])

plt.suptitle("Clustering Validation — UCI Online Retail II", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("figures/elbow_silhouette_combined.png", dpi=200, bbox_inches="tight")
plt.close()

print("\n✓ Saved: figures/elbow_silhouette_combined.png")

print("\n=== CLUSTER COMPOSITION ===")
for lbl in cluster_order:
    s = cust[cust["Cluster_Label"]==lbl]
    print(f"  {lbl:12s}: n={len(s):4d} | DH={s['Customer_Type'].mean():.1%} "
          f"| Freq={s['Frequency'].mean():.1f} | £{s['Monetary'].mean():.0f} "
          f"| Spread={s['PurchaseSpread'].mean():.1f}mo "
          f"| SeasonConc={s['SeasonalConcentration'].mean():.2f}")



profile_cols = ["Frequency","SeasonalConcentration","PurchaseSpread","ReturnRate"]
profile_lbls = ["Order Frequency","Seasonal Concentration","Purchase Spread (months)","Return Rate"]
colors = ["#4ade80","#facc15","#f87171"]


plt.figure(figsize=(16,5))
for i,(col,lbl) in enumerate(zip(profile_cols, profile_lbls)):
    plt.subplot(1,4,i+1)
    means = [cust[cust["Cluster_Label"]==c][col].mean() for c in cluster_order]
    bars = plt.bar(cluster_order, means, color=colors, edgecolor="white")
    plt.title(lbl, fontsize=10, fontweight="bold"); plt.ylabel("Mean Value")
    for bar,val in zip(bars,means):
        plt.text(bar.get_x()+bar.get_width()/2, val*1.02, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
plt.suptitle("Cluster Behavioral Profiles — UCI Online Retail II", fontweight="bold")
plt.tight_layout(); plt.savefig("figures/cluster_profiles.png", dpi=150); plt.close()

# ── 7. RANDOM FOREST
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
cv_f1     = cross_val_score(rf, X, y, cv=cv, scoring="f1")
cv_prec   = cross_val_score(rf, X, y, cv=cv, scoring="precision")
cv_rec    = cross_val_score(rf, X, y, cv=cv, scoring="recall")
cv_auc    = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")

print("\n=== RANDOM FOREST — 5-FOLD CROSS-VALIDATION ===")
print(f"  Accuracy  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Precision : {cv_prec.mean():.4f} ± {cv_prec.std():.4f}")
print(f"  Recall    : {cv_rec.mean():.4f} ± {cv_rec.std():.4f}")
print(f"  F1 Score  : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
print(f"  ROC-AUC   : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
print("\n  (Note: metrics reflect the heuristic label quality, not a held-out ground truth)")

# In-sample report for reference
y_pred = rf.predict(X)
print("\n  In-sample classification report:")
print(classification_report(y, y_pred, target_names=["Loyal", "Deal Seeker"]))

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)

print("\n=== FULL FEATURE IMPORTANCES ===")
for feat, score in importances.sort_values(ascending=False).items():
    print(f"  {feat:30s}: {score:.4f}")

print("\n=== CLUSTER PROFILES (all features) ===")
print(cust.groupby("Cluster_Label")[selected].mean().round(3).loc[cluster_order].to_string())



plt.figure(figsize=(10,7))
bar_colors = ["#58a6ff" if f in importances.tail(4).index else "#555" for f in importances.index]
importances.plot(kind="barh", color=bar_colors, edgecolor="black")
plt.title("Feature Importance: Deal Seekers vs. Loyal Customers\n(UCI Online Retail II)")
plt.xlabel("Importance Score")
plt.tight_layout(); plt.savefig("figures/feature_importance.png", dpi=150); plt.close()

# ── 8. MONTHLY REVENUE BY SEGMENT ────────────────────────────────────────────
purchases_lbl = purchases.merge(cust[["Cluster_Label"]], left_on="Customer ID", right_index=True, how="inner")
purchases_lbl["Month"] = purchases_lbl["InvoiceDate"].dt.month
monthly = purchases_lbl.groupby(["Month","Cluster_Label"])["LineRevenue"].sum().unstack(fill_value=0)

plt.figure(figsize=(12,5))
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
for lbl, color in zip(cluster_order, colors):
    if lbl in monthly.columns:
        plt.plot(range(1,13), [monthly[lbl].get(m,0) for m in range(1,13)],
                 marker="o", label=lbl, color=color, linewidth=2)
plt.xticks(range(1,13), months); plt.xlabel("Month"); plt.ylabel("Total Revenue (£)")
plt.title("Monthly Revenue by Customer Segment"); plt.legend()
plt.tight_layout(); plt.savefig("figures/monthly_revenue.png", dpi=150); plt.close()

# ── 9. SAVE ───────────────────────────────────────────────────────────────────
cust.to_csv("Enhanced_Retail_Analysis.csv")
print("\n4 figures saved to figures/")
print("Customer data saved to Enhanced_Retail_Analysis.csv")

print("\n── RESULTS SUMMARY ──────────────────────────────────────────────")
print(f"  Customers analysed : {len(cust):,}")
print(f"  Deal Seekers       : {n_dh} ({n_dh/len(cust):.1%})")
print(f"  Loyal Customers    : {len(cust)-n_dh} ({1-n_dh/len(cust):.1%})")
for lbl in cluster_order:
    s = cust[cust["Cluster_Label"]==lbl]
    print(f"  {lbl:12s}: n={len(s)} | DH={s['Customer_Type'].mean():.1%}")
print("\nTop 5 discriminating features:")
for feat, score in importances.sort_values(ascending=False).head(5).items():
    print(f"  {feat:30s}: {score:.4f}")

if __name__ == "__main__":
    print("\nPipeline complete.")