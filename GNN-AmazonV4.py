# =====================================================
# HETERO-GNN: USER + PRODUCT CLUSTERING + CLUSTER NAMING
# =====================================================

import re
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sqlalchemy import create_engine
from itertools import combinations
from collections import Counter
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =====================================================
# 1. DATABASE CONNECTION
# =====================================================

engine = create_engine(
    "postgresql://postgres:sa@localhost:5432/postgres"
)

# =====================================================
# 2. LOAD & CLEAN DATA
# =====================================================

query = """
SELECT
  product_id,
  user_id,
  category,
  product_name,
  NULLIF(REGEXP_REPLACE(discounted_price, '[^0-9\.]', '', 'g'), '')::DOUBLE PRECISION AS discounted_price,
  NULLIF(REGEXP_REPLACE(actual_price, '[^0-9\.]', '', 'g'), '')::DOUBLE PRECISION AS actual_price,
  NULLIF(REGEXP_REPLACE(discount_percentage, '[^0-9\.]', '', 'g'), '')::DOUBLE PRECISION AS discount_percentage,
  NULLIF(REGEXP_REPLACE(rating, '[^0-9\.]', '', 'g'), '')::DOUBLE PRECISION AS rating,
  NULLIF(REGEXP_REPLACE(rating_count, '[^0-9]', '', 'g'), '')::INT AS rating_count
FROM my_table
"""

df = pd.read_sql(query, engine)
df.fillna(0, inplace=True)

# =====================================================
# 3. MAP IDS → INDICES
# =====================================================

product_ids = df["product_id"].unique()
user_ids = df["user_id"].unique()

product_map = {p: i for i, p in enumerate(product_ids)}
user_map = {u: i for i, u in enumerate(user_ids)}

df["product_idx"] = df["product_id"].map(product_map)
df["user_idx"] = df["user_id"].map(user_map)

num_products = len(product_ids)
num_users = len(user_ids)

print(df.head())
# =====================================================
# 4. NODE FEATURES
# =====================================================

product_features = (
    df.groupby("product_idx")
      .agg(
          discounted_price=("discounted_price", "mean"),
          discount_percentage=("discount_percentage", "mean"),
          rating=("rating", "mean"),
          rating_count=("rating_count", "mean")
      )
      .reindex(range(num_products), fill_value=0)
)
print(product_features.head()  )
user_features = (
    df.groupby("user_idx")
      .agg(
          avg_rating_given=("rating", "mean"),
          review_count=("rating", "count")
      )
      .reindex(range(num_users), fill_value=0)
)
print(user_features.head()  )
X_product = torch.tensor(product_features.values, dtype=torch.float)
X_user = torch.tensor(user_features.values, dtype=torch.float)

# =====================================================
# 5. EDGES
# =====================================================

edge_user_product = torch.tensor(
    [df["user_idx"].values, df["product_idx"].values],
    dtype=torch.long
)

edge_product_user = torch.stack([
    edge_user_product[1],
    edge_user_product[0]
])

edges_pp = set()
for _, g in df.groupby("user_idx"):
    prods = g["product_idx"].unique()
    for p1, p2 in combinations(prods, 2):
        edges_pp.add((p1, p2))
        edges_pp.add((p2, p1))

edge_product_product = (
    torch.tensor(list(edges_pp), dtype=torch.long).T
    if edges_pp else torch.empty((2, 0), dtype=torch.long)
)

# =====================================================
# 6. BUILD HETERODATA
# =====================================================

data = HeteroData()
data["product"].x = X_product
data["user"].x = X_user

data["user", "reviews", "product"].edge_index = edge_user_product
data["product", "rev_reviews", "user"].edge_index = edge_product_user
data["product", "similar", "product"].edge_index = edge_product_product

# =====================================================
# 7. HETERO-GNN MODEL
# =====================================================

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_dim=32, out_dim=16):
        super().__init__()

        self.conv1 = HeteroConv({
            ("user", "reviews", "product"): SAGEConv((X_user.shape[1], X_product.shape[1]), hidden_dim),
            ("product", "rev_reviews", "user"): SAGEConv((X_product.shape[1], X_user.shape[1]), hidden_dim),
            ("product", "similar", "product"): SAGEConv((X_product.shape[1], X_product.shape[1]), hidden_dim),
        }, aggr="sum")

        self.conv2 = HeteroConv({
            ("user", "reviews", "product"): SAGEConv((hidden_dim, hidden_dim), out_dim),
            ("product", "rev_reviews", "user"): SAGEConv((hidden_dim, hidden_dim), out_dim),
            ("product", "similar", "product"): SAGEConv((hidden_dim, hidden_dim), out_dim),
        }, aggr="sum")

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        return self.conv2(x_dict, edge_index_dict)

# =====================================================
# 8. TRAIN MODEL
# =====================================================

model = HeteroGNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for _ in range(200):
    optimizer.zero_grad()
    z = model(data.x_dict, data.edge_index_dict)
    prod_z = z["product"]

    if edge_product_product.numel() > 0:
        src, dst = edge_product_product
        loss = ((prod_z[src] - prod_z[dst]) ** 2).mean()
    else:
        loss = prod_z.norm(dim=1).mean()

    loss.backward()
    optimizer.step()

print("✅ GNN training complete")

# =====================================================
# 9. EXTRACT & SCALE EMBEDDINGS
# =====================================================

with torch.no_grad():
    z = model(data.x_dict, data.edge_index_dict)
    product_emb = StandardScaler().fit_transform(z["product"].cpu().numpy())
    user_emb = StandardScaler().fit_transform(z["user"].cpu().numpy())

# =====================================================
# 10. CLUSTER USERS & PRODUCTS
# =====================================================

product_clusters = KMeans(n_clusters=30, n_init=10, random_state=42).fit_predict(product_emb)
user_clusters = KMeans(n_clusters=10, n_init=10, random_state=42).fit_predict(user_emb)

# =====================================================
# 11. SAVE EMBEDDINGS + CLUSTERS
# =====================================================

product_results = pd.DataFrame({
    "product_id": product_ids,
    "product_cluster_id": product_clusters,
    "embedding": product_emb.tolist()
})

user_results = pd.DataFrame({
    "user_id": user_ids,
    "user_cluster_id": user_clusters,
    "embedding": user_emb.tolist()
})

product_results.to_sql("gnn_product_embeddings", engine, if_exists="replace", index=False)
user_results.to_sql("gnn_user_embeddings", engine, if_exists="replace", index=False)

# =====================================================
# 12. GENERATE PRODUCT CLUSTER NAMES
# =====================================================

def extract_keywords(texts, top_k=3):
    words = []
    for t in texts:
        if isinstance(t, str):
            words += re.findall(r"[A-Za-z]{4,}", t.lower())
    return [w.capitalize() for w, _ in Counter(words).most_common(top_k)]

product_meta = df[["product_id", "category", "product_name"]].drop_duplicates()

clustered_products = product_results.merge(product_meta, on="product_id", how="left")

cluster_name_rows = []

for pc, grp in clustered_products.groupby("product_cluster_id"):
    top_category = grp["category"].value_counts().idxmax() if grp["category"].notnull().any() else "Mixed Products"
    keywords = extract_keywords(grp["product_name"].tolist())

    name = (
        f"{top_category} – {' & '.join(keywords[:2])}"
        if keywords else f"{top_category} – Cluster {pc}"
    )

    cluster_name_rows.append({
        "product_cluster_id": pc,
        "product_cluster_name": name
    })

product_cluster_names = pd.DataFrame(cluster_name_rows)

product_cluster_names.to_sql(
    "gnn_product_cluster_labels",
    engine,
    if_exists="replace",
    index=False
)

print("✅ Product cluster names generated & saved")
