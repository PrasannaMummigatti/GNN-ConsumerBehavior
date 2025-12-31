# GNN-ConsumerBehavior

🚀 Consumer behavior isn’t just about products — it’s about relationships.

Most analytics still look at users/consumers and products separately.
But real buying behavior happens in the connections between them.

That’s where Heterogeneous Graph Neural Networks (HGNNs) change the game.

📊 Data used
This work is based on the Amazon Sales Dataset (public Kaggle data):
👉 https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset

🔗 What I’m building
Instead of flat tables, I modeled the ecosystem as a heterogeneous graph:

👤 Users as one node type

📦 Products as another node type

🧩 Interactions (views, purchases, co-occurrence) as typed edges

This allows the model to learn who buys what, how products relate, and how consumer groups emerge — all at once.

🧠 What the GNN learns

Consumer (user) clustering
→ Identifies behavioral segments based on interaction patterns, not demographics

Product clustering
→ Groups products by how they are actually consumed together, not by catalog labels

Cross-influence effects
→ How certain users influence product popularity and how products bridge consumer segments

📈 What the visualizations reveal

Hidden consumer communities with shared buying logic

Products that act as connectors across segments

Cross-category demand patterns invisible to traditional clustering

Behavioral overlap between seemingly unrelated products

💡 Why this matters for business

Segmentation becomes behavior-driven, not rule-based

Recommendations improve because users and products are learned jointly

Merchandising, bundling, and pricing strategies become network-aware

You stop asking “What sells?” and start asking “What influences what?”

📌 Key insight
Consumer behavior is not linear.
It’s heterogeneous, interconnected, and relational.
And Heterogeneous GNNs are built exactly for that reality.

Note: The model captures “who interacts with what” and “how popular a product is”, but not “what users actually say about the product”. this I may take in next step/post.

#ConsumerBehavior
#GraphNeuralNetworks
#HeterogeneousGNN
#UserClustering
#ProductClustering
#RetailAnalytics
#EcommerceAI
#DataScience
#AIInRetail
