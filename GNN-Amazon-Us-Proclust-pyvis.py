# =====================================================
# USER ↔ PRODUCT CLUSTER INTERACTION GRAPH (PYVIS)
# Uses clusters already learned by GNN
# =====================================================

import pandas as pd
from sqlalchemy import create_engine
from collections import Counter
from pyvis.network import Network

# =====================================================
# DB CONNECTION
# =====================================================

engine = create_engine(
    "postgresql://postgres:sa@localhost:5432/postgres"
)

# =====================================================
# 1. LOAD USER CLUSTERS (FROM GNN OUTPUT)
# =====================================================

user_df = pd.read_sql(
    """
    SELECT
        user_id,
        user_cluster_id AS user_cluster
    FROM gnn_user_embeddings
    """,
    engine
)

# =====================================================
# 2. LOAD PRODUCT CLUSTERS + PRODUCT NAMES
# =====================================================

product_df = pd.read_sql(
    """
    SELECT
        p.product_id,
        p.product_cluster_id AS product_cluster,
        m.product_name,pc.product_cluster_name
    FROM gnn_product_embeddings p
    join gnn_product_cluster_labels pc
      ON p.product_cluster_id = pc.product_cluster_id
    JOIN my_table m
      ON m.product_id = p.product_id
    """,
    engine
)

# =====================================================
# 3. LOAD USER–PRODUCT INTERACTIONS
# =====================================================

interaction_df = pd.read_sql(
    """
    SELECT user_id, product_id
    FROM my_table
    """,
    engine
)

# =====================================================
# 4. BUILD USER–PRODUCT CLUSTER INTERACTIONS
# =====================================================

merged = (
    interaction_df
    .merge(user_df, on="user_id", how="inner")
    .merge(product_df[["product_id", "product_cluster"]], on="product_id", how="inner")
)

# Count interactions at (user_cluster, product_cluster) level
interaction_counts = Counter(
    zip(merged["user_cluster"], merged["product_cluster"])
)

# =====================================================
# 5. PRODUCT CLUSTER TOOLTIP CONTENT
# =====================================================

product_cluster_examples = {}

for pc, grp in product_df.groupby("product_cluster"):
    product_cluster_examples[pc] = (
        grp["product_name"]
        .dropna()
        .head(3)
        .tolist()
    )

# =====================================================
# 6. USER CLUSTER LABELS (BEHAVIOR SUMMARY)
# =====================================================

user_cluster_labels = {}

for uc, grp in merged.groupby("user_cluster"):
    top_pc = grp["product_cluster"].value_counts().idxmax()
    #user_cluster_labels[uc] = f"Users{uc} → Prod Clust {top_pc}"
    user_cluster_labels[uc] = f"Consumer Group {uc}"

# =====================================================
# 7. BUILD PYVIS NETWORK
# =====================================================

net = Network(
    height="900px",
    width="100%",
    bgcolor="#ffffff",
    font_color="black",
    directed=False
)

net.barnes_hut()

# --- USER CLUSTER NODES ---
for uc in sorted(user_df["user_cluster"].unique()):
    label = user_cluster_labels.get(uc, f"User Clust {uc}")

    net.add_node(
        f"U{uc}",
        label=label,
        title=f"<b>User Cluster {uc}</b><br>{label}",
        color="#ff9999",
        shape="ellipse",
        size=45,
        font={"size": 20}
    )

# --- PRODUCT CLUSTER NODES ---

pairs = (
    product_df[["product_cluster", "product_cluster_name"]]
    .drop_duplicates()
    .sort_values("product_cluster")
)
for pc, name in pairs.itertuples(index=False):
    label = f"{pc}: {name}"
    examples = product_cluster_examples.get(pc, [])

    net.add_node(
        f"P{pc}",
        label=label,
        title=(
            f"<b>{label}</b><br><br>"
            f"Example products:<br>" +
            "<br>".join(examples)
        ),
        color="#99ccff",
        shape="box",
        size=55,
        font={"size": 20}
    )

# --- EDGES (INTERACTIONS) ---
for (uc, pc), weight in interaction_counts.items():
    net.add_edge(
        f"U{uc}",
        f"P{pc}",
        value=weight,
        title=f"Interactions: {weight}",
        color="#000000"
    )

# =====================================================
# 8. COMPACT + HIGH-ZOOM VISUAL TUNING
# =====================================================

net.set_options("""
{
  "interaction": {
    "zoomView": true,
    "dragView": true,
    "zoomSpeed": 0.6,
    "minZoom": 0.1,
    "maxZoom": 6
  },
  "nodes": {
    "font": {
      "strokeWidth": 1,
      "strokeColor": "#ffffff"
    }
  },
  "physics": {
    "barnesHut": {
      "gravitationalConstant": -65000,
      "centralGravity": 40,
      "springLength": 50,
      "springConstant": 0.02,
      "damping": 0.35,
      "avoidOverlap": 1
    },
    "minVelocity": 0.02,
    "timestep": 0.5
  }
}
""")


# =====================================================
# 9. SAVE INTERACTIVE HTML (SAFE METHOD)
# =====================================================

output_file = "user_product_cluster_interaction.html"

html = net.generate_html()
with open(output_file, "w", encoding="utf-8") as f:
    f.write(html)

print(f"✅ Interactive graph saved to: {output_file}")
print("➡️ Open this file manually in your browser")
