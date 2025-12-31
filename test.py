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
