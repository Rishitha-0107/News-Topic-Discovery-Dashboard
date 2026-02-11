import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

st.set_page_config(layout="wide")

# -----------------------------------------------------
# 🟣 TITLE
# -----------------------------------------------------

st.title("🟣 News Topic Discovery Dashboard")

st.markdown("""
This system uses **Hierarchical Clustering** to automatically group similar news articles based on textual similarity.

👉 Discover hidden themes without defining categories upfront.
""")

# -----------------------------------------------------
# 📂 SIDEBAR - INPUT CONTROLS
# -----------------------------------------------------

st.sidebar.header("📂 Dataset Upload")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# If no file uploaded
if uploaded_file is None:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

# Load data
df = pd.read_csv(uploaded_file, encoding="latin1")

# Auto-detect text column
text_columns = df.select_dtypes(include=["object"]).columns.tolist()

if len(text_columns) == 0:
    st.error("No text column detected.")
    st.stop()

text_column = st.sidebar.selectbox("Select Text Column", text_columns)

# -----------------------------------------------------
# 📝 TEXT VECTORIZATION CONTROLS
# -----------------------------------------------------

st.sidebar.header("📝 Text Vectorization")

max_features = st.sidebar.slider("Maximum TF-IDF Features", 100, 2000, 1000)

use_stopwords = st.sidebar.checkbox("Remove English Stopwords", value=True)

ngram_option = st.sidebar.selectbox(
    "N-gram Range",
    ["Unigrams", "Bigrams", "Unigrams + Bigrams"]
)

if ngram_option == "Unigrams":
    ngram_range = (1, 1)
elif ngram_option == "Bigrams":
    ngram_range = (2, 2)
else:
    ngram_range = (1, 2)

# -----------------------------------------------------
# 🌳 HIERARCHICAL CONTROLS
# -----------------------------------------------------

st.sidebar.header("🌳 Hierarchical Clustering")

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ["ward", "complete", "average", "single"]
)

distance_metric = st.sidebar.selectbox(
    "Distance Metric",
    ["euclidean"]
)

dendro_samples = st.sidebar.slider(
    "Number of Articles for Dendrogram",
    20, 200, 100
)

# -----------------------------------------------------
# TF-IDF PROCESSING
# -----------------------------------------------------

vectorizer = TfidfVectorizer(
    max_features=max_features,
    stop_words="english" if use_stopwords else None,
    ngram_range=ngram_range
)

X = vectorizer.fit_transform(df[text_column].astype(str))

# -----------------------------------------------------
# 🟦 GENERATE DENDROGRAM
# -----------------------------------------------------

if st.button("🟦 Generate Dendrogram"):

    st.subheader("🌳 Dendrogram")

    X_subset = X[:dendro_samples].toarray()

    linked = linkage(X_subset, method=linkage_method)

    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linked)
    plt.xlabel("Article Index")
    plt.ylabel("Distance")

    st.pyplot(fig)

    st.info("Look for large vertical gaps to decide number of clusters.")

# -----------------------------------------------------
# 🟩 APPLY CLUSTERING
# -----------------------------------------------------

st.subheader("🟩 Apply Clustering")

num_clusters = st.slider("Number of Clusters", 2, 10, 4)

if st.button("Apply Clustering"):

    model = AgglomerativeClustering(
        n_clusters=num_clusters,
        linkage=linkage_method
    )

    clusters = model.fit_predict(X.toarray())

    df["Cluster"] = clusters

    # -----------------------------------------------------
    # 📊 SILHOUETTE SCORE
    # -----------------------------------------------------

    sil_score = silhouette_score(X.toarray(), clusters)

    st.metric("📊 Silhouette Score", round(sil_score, 4))

    st.markdown("""
    **Interpretation:**
    - Close to 1 → Well-separated clusters  
    - Close to 0 → Overlapping clusters  
    - Negative → Poor clustering  
    """)

    # -----------------------------------------------------
    # 📉 PCA VISUALIZATION
    # -----------------------------------------------------

    st.subheader("📉 Cluster Visualization (PCA 2D)")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())

    plot_df = pd.DataFrame({
        "PCA1": X_pca[:, 0],
        "PCA2": X_pca[:, 1],
        "Cluster": clusters,
        "Snippet": df[text_column].astype(str).str[:120]
    })

    fig = px.scatter(
        plot_df,
        x="PCA1",
        y="PCA2",
        color="Cluster",
        hover_data=["Snippet"]
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------
    # 📋 CLUSTER SUMMARY
    # -----------------------------------------------------

    st.subheader("📋 Cluster Summary")

    feature_names = vectorizer.get_feature_names_out()

    summary_data = []

    for c in range(num_clusters):
        cluster_indices = np.where(clusters == c)[0]

        cluster_tfidf = X[cluster_indices].mean(axis=0)
        top_indices = np.argsort(cluster_tfidf.A1)[-10:]

        top_keywords = [feature_names[i] for i in top_indices]

        representative_article = df.iloc[cluster_indices[0]][text_column][:150]

        summary_data.append({
            "Cluster ID": c,
            "Number of Articles": len(cluster_indices),
            "Top Keywords": ", ".join(top_keywords),
            "Sample Article": representative_article
        })

    summary_df = pd.DataFrame(summary_data)

    st.dataframe(summary_df)

    # -----------------------------------------------------
    # 🧠 BUSINESS INTERPRETATION
    # -----------------------------------------------------

    st.subheader("🧠 Business Insights")

    st.markdown("""
Articles grouped in the same cluster share similar vocabulary and themes.

These clusters can be used for:
- Automatic tagging
- Content recommendation
- Editorial organization
- Trend discovery
""")

