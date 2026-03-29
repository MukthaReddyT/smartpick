import streamlit as st
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Electronics Recommender", page_icon="📱", layout="wide")

# ─── LOAD & MERGE DATA ───────────────────────────────────────────────
@st.cache_data
def load_and_merge():
    # Load both datasets — update filenames if yours differ
    df1 = pd.read_csv("dataset1.csv")   # Amazon/Flipkart reviews
    df2 = pd.read_csv("dataset2.csv")   # Product category + features

    # ── Standardize df1: create a clean product name from brand + model
    df1["product"] = (df1["brand"].astype(str) + " " + df1["model"].astype(str)).str.strip().str.title()

    # ── Standardize df2: clean product name
    df2["product"] = df2["product"].astype(str).str.strip().str.title()

    # ── Aggregate df2 to one row per product (most common category/feature)
    df2_agg = df2.groupby("product").agg(
        category=("category", lambda x: x.mode()[0]),
        top_feature=("feature_mentioned", lambda x: x.mode()[0]),
        top_attribute=("attribute_mentioned", lambda x: x.mode()[0]),
        ds2_sentiment=("sentiment", lambda x: (x == "positive").mean())
    ).reset_index()

    # ── Aggregate df1 to one row per product
    df1_agg = df1.groupby("product").agg(
        brand=("brand", "first"),
        model=("model", "first"),
        avg_rating=("rating", "mean"),
        review_count=("review_id", "count"),
        avg_price_usd=("price_usd", "mean"),
        avg_battery=("battery_life_rating", "mean"),
        avg_camera=("camera_rating", "mean"),
        avg_performance=("performance_rating", "mean"),
        avg_design=("design_rating", "mean"),
        avg_display=("display_rating", "mean"),
        helpful_votes=("helpful_votes", "sum"),
        verified_ratio=("verified_purchase", lambda x: (x == True).mean()),
        sources=("source", lambda x: ", ".join(x.unique())),
        all_reviews=("review_text", lambda x: " ".join(x.dropna().astype(str).head(20)))
    ).reset_index()

    # ── Merge: left join so all df1 products are kept
    merged = pd.merge(df1_agg, df2_agg, on="product", how="left")

    # Fill missing category/feature for unmatched products
    merged["category"] = merged["category"].fillna("Electronics")
    merged["top_feature"] = merged["top_feature"].fillna("general")
    merged["top_attribute"] = merged["top_attribute"].fillna("overall")

    return merged, df1


@st.cache_data
def run_sentiment(merged):
    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment_score(text):
        if pd.isna(text) or text == "":
            return 0.0
        score = analyzer.polarity_scores(str(text))["compound"]
        return round((score + 1) / 2, 4)   # normalize -1..1 → 0..1

    merged["vader_sentiment"] = merged["all_reviews"].apply(get_sentiment_score)

    # ── Final weighted score (0–100)
    # Rating: 30%, VADER Sentiment: 35%, Sub-ratings avg: 20%, Verified+Helpful: 15%
    merged["sub_rating_avg"] = merged[
        ["avg_battery", "avg_camera", "avg_performance", "avg_design", "avg_display"]
    ].mean(axis=1).fillna(merged["avg_rating"])

    rating_norm     = merged["avg_rating"] / 5
    sentiment_norm  = merged["vader_sentiment"]
    subrating_norm  = merged["sub_rating_avg"] / 5
    helpful_norm    = (merged["helpful_votes"] / (merged["helpful_votes"].max() + 1))
    verified_norm   = merged["verified_ratio"].fillna(0)
    trust_norm      = (helpful_norm * 0.5 + verified_norm * 0.5)

    merged["final_score"] = (
        rating_norm    * 0.30 +
        sentiment_norm * 0.35 +
        subrating_norm * 0.20 +
        trust_norm     * 0.15
    ) * 100

    merged["final_score"] = merged["final_score"].round(1)
    merged["avg_rating"]  = merged["avg_rating"].round(2)
    merged["avg_price_usd"] = merged["avg_price_usd"].round(2)
    merged["vader_sentiment"] = (merged["vader_sentiment"] * 100).round(1)

    return merged.sort_values("final_score", ascending=False).reset_index(drop=True)


# ─── LOAD ─────────────────────────────────────────────────────────────
with st.spinner("Loading and processing data..."):
    merged_raw, df1_raw = load_and_merge()
    df = run_sentiment(merged_raw)

# ─── SIDEBAR ──────────────────────────────────────────────────────────
st.sidebar.title("Filters")

categories = ["All"] + sorted(df["category"].dropna().unique().tolist())
selected_cat = st.sidebar.selectbox("Category", categories)

sources_available = []
for s in df["sources"].dropna():
    for src in s.split(", "):
        if src not in sources_available:
            sources_available.append(src)
selected_source = st.sidebar.selectbox("Platform", ["All"] + sorted(sources_available))

price_min = float(df["avg_price_usd"].min())
price_max = float(df["avg_price_usd"].max())
price_range = st.sidebar.slider("Price range (USD)", price_min, price_max, (price_min, price_max))

top_n = st.sidebar.slider("Show top N products", 5, 50, 10)

sort_by = st.sidebar.selectbox("Sort by", ["final_score", "avg_rating", "vader_sentiment", "avg_price_usd"])

# ─── FILTER ───────────────────────────────────────────────────────────
filtered = df.copy()
if selected_cat != "All":
    filtered = filtered[filtered["category"] == selected_cat]
if selected_source != "All":
    filtered = filtered[filtered["sources"].str.contains(selected_source, na=False)]
filtered = filtered[
    (filtered["avg_price_usd"] >= price_range[0]) &
    (filtered["avg_price_usd"] <= price_range[1])
]
filtered = filtered.sort_values(sort_by, ascending=(sort_by == "avg_price_usd")).head(top_n)

# ─── HEADER ───────────────────────────────────────────────────────────
st.title("Consumer Electronics Recommendation System")
st.caption("Sentiment analysis + multi-source rating fusion | Amazon × Flipkart")

# ─── KPI CARDS ────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Products analysed", f"{len(df):,}")
k2.metric("Avg final score",   f"{df['final_score'].mean():.1f}/100")
k3.metric("Avg sentiment",     f"{df['vader_sentiment'].mean():.1f}%")
k4.metric("Datasets merged",   "2 (Amazon/Flipkart + Feature)")

st.divider()

# ─── TOP PRODUCTS TABLE ───────────────────────────────────────────────
st.subheader(f"Top {len(filtered)} recommended products")

def score_color(val):
    if val >= 75:   return "background-color:#d4edda; color:#155724"
    elif val >= 55: return "background-color:#fff3cd; color:#856404"
    else:           return "background-color:#f8d7da; color:#721c24"

display_cols = {
    "product":        "Product",
    "category":       "Category",
    "brand":          "Brand",
    "avg_price_usd":  "Price (USD)",
    "avg_rating":     "Avg Rating",
    "vader_sentiment":"Sentiment %",
    "review_count":   "Reviews",
    "top_feature":    "Top Feature",
    "sources":        "Platforms",
    "final_score":    "Final Score"
}

table = filtered[list(display_cols.keys())].rename(columns=display_cols)
st.dataframe(
    table.style.applymap(score_color, subset=["Final Score"]),
    use_container_width=True,
    hide_index=True
)

st.divider()

# ─── CHARTS ───────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.subheader("Final score — top products")
    fig1 = px.bar(
        filtered.sort_values("final_score"),
        x="final_score", y="product",
        orientation="h",
        color="final_score",
        color_continuous_scale="teal",
        labels={"final_score": "Score", "product": ""},
        text="final_score"
    )
    fig1.update_traces(textposition="outside")
    fig1.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=0, r=20, t=10, b=10),
        height=420
    )
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.subheader("Rating vs sentiment score")
    fig2 = px.scatter(
        filtered,
        x="avg_rating", y="vader_sentiment",
        size="review_count",
        color="final_score",
        hover_name="product",
        color_continuous_scale="viridis",
        labels={
            "avg_rating": "Avg Rating (1–5)",
            "vader_sentiment": "Sentiment (%)",
            "review_count": "# Reviews"
        }
    )
    fig2.update_layout(margin=dict(l=0, r=0, t=10, b=10), height=420)
    st.plotly_chart(fig2, use_container_width=True)

# ─── SUB-RATING RADAR ─────────────────────────────────────────────────
st.subheader("Feature breakdown — top 5 products")
top5 = filtered.head(5)
categories_radar = ["Battery", "Camera", "Performance", "Design", "Display"]
cols_radar = ["avg_battery", "avg_camera", "avg_performance", "avg_design", "avg_display"]

fig3 = go.Figure()
for _, row in top5.iterrows():
    vals = [row[c] if not pd.isna(row[c]) else 0 for c in cols_radar]
    vals += [vals[0]]
    fig3.add_trace(go.Scatterpolar(
        r=vals,
        theta=categories_radar + [categories_radar[0]],
        fill="toself",
        name=str(row["product"])[:25]
    ))
fig3.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
    height=420,
    margin=dict(l=20, r=20, t=20, b=20)
)
st.plotly_chart(fig3, use_container_width=True)

# ─── SENTIMENT DISTRIBUTION ───────────────────────────────────────────
st.subheader("Sentiment distribution across products")
fig4 = px.histogram(
    df, x="vader_sentiment", nbins=30,
    color_discrete_sequence=["#1D9E75"],
    labels={"vader_sentiment": "Sentiment score (%)"}
)
fig4.update_layout(margin=dict(l=0, r=0, t=10, b=10), height=300)
st.plotly_chart(fig4, use_container_width=True)

# ─── DETAILED PRODUCT CARD ────────────────────────────────────────────
st.divider()
st.subheader("Inspect a product")
product_list = filtered["product"].tolist()
selected_product = st.selectbox("Select product", product_list)

if selected_product:
    row = df[df["product"] == selected_product].iloc[0]
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Final score",  f"{row['final_score']}/100")
    p2.metric("Avg rating",   f"{row['avg_rating']}/5")
    p3.metric("Sentiment",    f"{row['vader_sentiment']}%")
    p4.metric("Reviews",      f"{int(row['review_count']):,}")
    p5.metric("Price",        f"${row['avg_price_usd']:.0f}")

    st.caption(f"**Category:** {row['category']} | **Top feature:** {row['top_feature']} | **Platforms:** {row['sources']}")

# ─── FOOTER ───────────────────────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit · VADER Sentiment · Plotly · Pandas | Mini Project — Consumer Electronics Recommendation System")