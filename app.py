import streamlit as st
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
import re, warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="SmartPick", page_icon="⚡", layout="wide")

# ══════════════════════════════════════════════════════════════════════
# ASPECT-BASED SENTIMENT — keyword approach
# ══════════════════════════════════════════════════════════════════════
ASPECT_KEYWORDS = {
    "Battery":     {
        "pos": ["long battery","great battery","battery life","lasts all day","good battery","battery backup","fast charge","quick charge"],
        "neg": ["bad battery","poor battery","drains fast","battery drain","short battery","dies quickly","low battery"]
    },
    "Display":     {
        "pos": ["great display","beautiful screen","sharp display","bright screen","vivid","amoled","crisp","good screen"],
        "neg": ["poor display","dim screen","bad screen","low brightness","blurry","pixelated","bad display"]
    },
    "Camera":      {
        "pos": ["great camera","good camera","excellent camera","sharp photos","clear photos","amazing camera","best camera"],
        "neg": ["bad camera","poor camera","blurry photos","grainy","low quality camera","camera issues","disappointing camera"]
    },
    "Performance": {
        "pos": ["fast","smooth","no lag","great performance","snappy","responsive","powerful","runs well"],
        "neg": ["slow","lag","stutters","freezes","poor performance","hangs","crashes","sluggish"]
    },
    "Design":      {
        "pos": ["sleek","beautiful design","premium","great build","elegant","lightweight","slim","good design"],
        "neg": ["cheap","plasticky","heavy","bulky","bad design","poor build","flimsy"]
    },
    "Value":       {
        "pos": ["worth the price","good value","affordable","budget friendly","value for money","great deal"],
        "neg": ["overpriced","expensive","not worth","poor value","costly","too expensive"]
    },
}

def aspect_sentiment(text):
    """Return dict of aspect -> (pos_count, neg_count, pct_positive)"""
    if not text or pd.isna(text):
        return {}
    t = str(text).lower()
    result = {}
    for aspect, kws in ASPECT_KEYWORDS.items():
        pos = sum(1 for k in kws["pos"] if k in t)
        neg = sum(1 for k in kws["neg"] if k in t)
        total = pos + neg
        if total > 0:
            result[aspect] = {
                "pos": pos, "neg": neg,
                "pct": round((pos / total) * 100)
            }
    return result

def merge_aspect_dicts(dict_list):
    """Aggregate aspect dicts across multiple reviews into one."""
    merged = {}
    for d in dict_list:
        for asp, vals in d.items():
            if asp not in merged:
                merged[asp] = {"pos": 0, "neg": 0}
            merged[asp]["pos"] += vals["pos"]
            merged[asp]["neg"] += vals["neg"]
    final = {}
    for asp, vals in merged.items():
        total = vals["pos"] + vals["neg"]
        if total > 0:
            final[asp] = {
                "pos": vals["pos"], "neg": vals["neg"],
                "pct": round((vals["pos"] / total) * 100)
            }
    return final

# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════
def compute_weighted_rating(df):
    star_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        for i in range(1, 6):
            if cl == f"{i} stars":
                star_map[i] = col
    if len(star_map) < 5:
        return pd.Series([np.nan] * len(df))
    total    = sum(df[star_map[i]].fillna(0) for i in range(1, 6))
    weighted = sum(i * df[star_map[i]].fillna(0) for i in range(1, 6))
    avg = weighted / total.replace(0, np.nan)
    return avg.round(2)

def compute_review_count(df):
    star_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        for i in range(1, 6):
            if cl == f"{i} stars":
                star_map[i] = col
    if len(star_map) < 5:
        return pd.Series([0] * len(df))
    return sum(df[star_map[i]].fillna(0) for i in range(1, 6)).astype(int)

def clean_price(series):
    extracted = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"([\d.]+)")[0]
    )
    return pd.to_numeric(extracted, errors="coerce")

def normalize_series(s):
    mn, mx = s.min(), s.max()
    if mx == mn:
        return s * 0 + 0.5
    return (s - mn) / (mx - mn)

# ══════════════════════════════════════════════════════════════════════
# LOAD & MERGE
# ══════════════════════════════════════════════════════════════════════
@st.cache_data
def load_all():
    analyzer = SentimentIntensityAnalyzer()

    # ── Dataset 1: phones with review text ──────────────────────────
    d1 = pd.read_csv("dataset1.csv")
    d1["product"] = (
        d1["brand"].astype(str) + " " + d1["model"].astype(str)
    ).str.title().str.strip()

    # per-product aggregation
    def agg_reviews(x):
        return " ".join(x.dropna().astype(str).head(40))

    d1_agg = d1.groupby("product").agg(
        brand           = ("brand",            "first"),
        avg_rating      = ("rating",           "mean"),
        review_count    = ("review_id",        "count"),
        avg_price       = ("price_usd",        "mean"),
        avg_battery     = ("battery_life_rating",   "mean"),
        avg_camera      = ("camera_rating",         "mean"),
        avg_performance = ("performance_rating",     "mean"),
        avg_design      = ("design_rating",          "mean"),
        avg_display     = ("display_rating",         "mean"),
        helpful_votes   = ("helpful_votes",    "sum"),
        verified_ratio  = ("verified_purchase",lambda x: (x == True).mean()),
        platforms       = ("source",           lambda x: ", ".join(sorted(x.unique()))),
        all_reviews     = ("review_text",      agg_reviews),
    ).reset_index()

    # VADER sentiment
    def vader_score(text):
        if not text or pd.isna(text):
            return 0.5
        s = analyzer.polarity_scores(str(text))["compound"]
        return round((s + 1) / 2, 4)

    d1_agg["sentiment_score"] = d1_agg["all_reviews"].apply(vader_score)

    # Aspect-based sentiment
    d1_agg["aspects"] = d1_agg["all_reviews"].apply(
        lambda t: str(merge_aspect_dicts([aspect_sentiment(t)]))
    )
    d1_agg["category"] = "Smartphones"
    d1_agg.drop(columns=["all_reviews"], inplace=True)

    # ── Spec datasets (star-based) ───────────────────────────────────
    spec_files = {
        "Cameras":               "dataset3_cams.csv",
        "Headphones & Speakers": "dataset4_hpsp.csv",
        "Tablets":               "dataset5_tab.csv",
        "Wearables":             "dataset6_wear.csv",
        "Laptops":               "dataset7_lap.csv",
        "Gaming Consoles":       "dataset8_gam.csv",
        "TVs":                   "dataset9_tv.csv",
    }

    spec_frames = []
    for cat, fname in spec_files.items():
        try:
            df = pd.read_csv(fname)
        except FileNotFoundError:
            continue

        name_col = next(
            (c for c in df.columns if c.strip().lower() == "product name"), None
        )
        if name_col is None:
            continue

        df["product"]       = df[name_col].astype(str).str.strip().str.title()
        df["brand"]         = (
            df["Brand"].astype(str).str.strip().str.title()
            if "Brand" in df.columns else "Unknown"
        )
        df["avg_rating"]    = compute_weighted_rating(df)
        df["review_count"]  = compute_review_count(df)
        df["avg_price"]     = clean_price(df["Price in India"]) / 83  # INR→USD
        df["category"]      = cat
        df["platforms"]     = "Amazon/Flipkart India"
        df["sentiment_score"] = df["avg_rating"].fillna(3) / 5
        df["aspects"]       = "{}"
        for col in ["avg_battery","avg_camera","avg_performance","avg_design","avg_display"]:
            df[col] = df["avg_rating"]
        df["helpful_votes"]  = 0
        df["verified_ratio"] = 0.5

        spec_frames.append(df[[
            "product","brand","avg_rating","review_count","avg_price","category",
            "platforms","sentiment_score","aspects",
            "avg_battery","avg_camera","avg_performance",
            "avg_design","avg_display","helpful_votes","verified_ratio"
        ]])

    all_data = pd.concat([d1_agg] + spec_frames, ignore_index=True)

    # ── Clean NaNs ──────────────────────────────────────────────────
    all_data = all_data[all_data["product"].str.strip().str.len() > 2]
    all_data = all_data.dropna(subset=["avg_rating"])
    all_data["avg_rating"]    = all_data["avg_rating"].clip(1, 5)
    all_data["review_count"]  = all_data["review_count"].fillna(0).astype(int)
    all_data["avg_price"]     = all_data["avg_price"].fillna(
                                    all_data["avg_price"].median()
                                ).clip(lower=0)
    all_data["sentiment_score"] = all_data["sentiment_score"].fillna(0.5).clip(0, 1)
    all_data["helpful_votes"] = all_data["helpful_votes"].fillna(0)
    all_data["verified_ratio"]= all_data["verified_ratio"].fillna(0.5)
    for col in ["avg_battery","avg_camera","avg_performance","avg_design","avg_display"]:
        all_data[col] = all_data[col].fillna(all_data["avg_rating"])

    # ── Normalize & score ───────────────────────────────────────────
    r  = normalize_series(all_data["avg_rating"])
    s  = normalize_series(all_data["sentiment_score"])
    sr = normalize_series(
         all_data[["avg_battery","avg_camera","avg_performance",
                   "avg_design","avg_display"]].mean(axis=1))
    h  = normalize_series(all_data["helpful_votes"])
    v  = all_data["verified_ratio"].clip(0, 1)
    t  = h * 0.5 + v * 0.5

    all_data["final_score"]   = (r*0.30 + s*0.35 + sr*0.20 + t*0.15) * 100
    all_data["final_score"]   = all_data["final_score"].round(1).clip(0, 100)
    all_data["avg_rating"]    = all_data["avg_rating"].round(2)
    all_data["avg_price"]     = all_data["avg_price"].round(2)
    all_data["sentiment_pct"] = (all_data["sentiment_score"] * 100).round(1)

    return all_data.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
# SEARCH + FILTER
# ══════════════════════════════════════════════════════════════════════
CAT_KEYWORDS = {
    "phone": "Smartphones", "mobile": "Smartphones", "smartphone": "Smartphones",
    "samsung": "Smartphones", "apple": "Smartphones", "iphone": "Smartphones",
    "xiaomi": "Smartphones", "oneplus": "Smartphones", "realme": "Smartphones",
    "motorola": "Smartphones", "pixel": "Smartphones",
    "laptop": "Laptops", "notebook": "Laptops", "macbook": "Laptops",
    "dell": "Laptops", "lenovo": "Laptops", "asus": "Laptops", "acer": "Laptops", "hp laptop": "Laptops",
    "headphone": "Headphones & Speakers", "earphone": "Headphones & Speakers",
    "speaker": "Headphones & Speakers", "earbuds": "Headphones & Speakers",
    "bose": "Headphones & Speakers", "jbl": "Headphones & Speakers", "boat": "Headphones & Speakers",
    "tv": "TVs", "television": "TVs", "smart tv": "TVs", "qled": "TVs", "oled": "TVs",
    "tablet": "Tablets", "ipad": "Tablets",
    "watch": "Wearables", "smartwatch": "Wearables", "fitbit": "Wearables", "band": "Wearables",
    "camera": "Cameras", "dslr": "Cameras", "mirrorless": "Cameras", "canon": "Cameras", "nikon": "Cameras",
    "gaming": "Gaming Consoles", "console": "Gaming Consoles",
    "playstation": "Gaming Consoles", "xbox": "Gaming Consoles", "nintendo": "Gaming Consoles",
}

USE_CASE_BOOST = {
    "Gaming":         {"avg_performance": 0.4, "avg_display": 0.3, "avg_battery": 0.3},
    "Office / Work":  {"avg_performance": 0.4, "avg_battery": 0.3, "avg_design": 0.3},
    "Photography":    {"avg_camera": 0.6, "avg_display": 0.2, "avg_performance": 0.2},
    "Entertainment":  {"avg_display": 0.5, "avg_battery": 0.3, "avg_performance": 0.2},
    "Budget Pick":    {},   # only price filter matters
    "All-rounder":    {},   # default scoring
}

def search_and_filter(df, query, budget_usd, use_case, top_n):
    q = query.lower().strip()

    # category detect
    matched_cat = None
    for kw, cat in CAT_KEYWORDS.items():
        if kw in q:
            matched_cat = cat
            break

    if matched_cat:
        results = df[df["category"] == matched_cat].copy()
    else:
        mask = (
            df["product"].str.lower().str.contains(q, na=False) |
            df["brand"].str.lower().str.contains(q, na=False) |
            df["category"].str.lower().str.contains(q, na=False)
        )
        results = df[mask].copy()

    if results.empty:
        return results, None

    detected_cat = results["category"].mode()[0]

    # budget filter
    results = results[results["avg_price"] <= budget_usd]
    if results.empty:
        return results, detected_cat

    # use-case re-scoring
    boost = USE_CASE_BOOST.get(use_case, {})
    if boost:
        r  = normalize_series(results["avg_rating"])
        s  = normalize_series(results["sentiment_score"])
        h  = normalize_series(results["helpful_votes"])
        v  = results["verified_ratio"].clip(0, 1)
        t  = h * 0.5 + v * 0.5

        feature_score = sum(
            normalize_series(results[col]) * weight
            for col, weight in boost.items()
        )
        results["final_score"] = (
            r * 0.25 + s * 0.30 + feature_score * 0.30 + t * 0.15
        ) * 100
        results["final_score"] = results["final_score"].round(1).clip(0, 100)

    results = results.sort_values("final_score", ascending=False).reset_index(drop=True)
    return results.head(top_n), detected_cat


# ══════════════════════════════════════════════════════════════════════
# ASPECT CHART
# ══════════════════════════════════════════════════════════════════════
def render_aspect_chart(aspects_str):
    try:
        aspects = eval(aspects_str)
    except Exception:
        aspects = {}

    if not aspects:
        st.caption("No aspect data available for this product (spec-only dataset).")
        return

    asp_names, pos_pcts, colors = [], [], []
    for asp, vals in aspects.items():
        asp_names.append(asp)
        pct = vals["pct"]
        pos_pcts.append(pct)
        colors.append("#28a745" if pct >= 60 else "#ffc107" if pct >= 40 else "#dc3545")

    fig = go.Figure(go.Bar(
        x=pos_pcts, y=asp_names, orientation="h",
        marker_color=colors,
        text=[f"{p}% positive" for p in pos_pcts],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 115], title="% positive mentions"),
        margin=dict(l=0, r=60, t=10, b=10),
        height=220 + len(asp_names) * 20,
        showlegend=False,
    )
    fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.4)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════
with st.spinner("Loading SmartPick — merging 9 datasets..."):
    df = load_all()

st.markdown("""
<h1 style='text-align:center;margin-bottom:0'>⚡ SmartPick</h1>
<p style='text-align:center;color:gray;margin-top:4px'>
  Consumer Electronics Recommendation System &nbsp;·&nbsp;
  Aspect-Based Sentiment · Multi-Dataset Fusion · User-Centered Scoring
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# Search bar
_, mid, _ = st.columns([1, 3, 1])
with mid:
    query = st.text_input(
        "", placeholder="Try: laptop, headphones, Samsung phone, 4K TV, smartwatch...",
        label_visibility="collapsed"
    )

# User inputs: budget + use case
col_b, col_u, col_n = st.columns([2, 2, 1])
with col_b:
    budget = st.slider("Your budget (USD)", 10, 3000, 500, step=10)
with col_u:
    use_case = st.selectbox(
        "Use case",
        ["All-rounder", "Gaming", "Office / Work", "Photography", "Entertainment", "Budget Pick"]
    )
with col_n:
    top_n = st.slider("Top N", 3, 20, 6)

# KPIs
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Products",        f"{len(df):,}")
k2.metric("Categories",      df["category"].nunique())
k3.metric("Brands",          df["brand"].nunique())
k4.metric("Platforms",       "5")
k5.metric("Datasets merged", "9")

st.markdown("---")

# ── RESULTS ──────────────────────────────────────────────────────────
if query:
    results, detected_cat = search_and_filter(df, query, budget, use_case, top_n)

    if detected_cat is None or results is None:
        st.warning("No products found. Try: laptop, headphones, Samsung, TV, camera...")
    elif results.empty:
        st.warning(f"No products found under ${budget}. Try increasing your budget.")
    else:
        st.subheader(f"Top recommendations — {detected_cat}")
        st.caption(
            f"Query: '{query}'  ·  Budget: ≤${budget}  ·  "
            f"Use case: {use_case}  ·  {len(results)} results"
        )

        # ── Product cards ────────────────────────────────────────────
        cols_per_row = 3
        for i in range(0, len(results), cols_per_row):
            row_items = results.iloc[i:i + cols_per_row]
            cols = st.columns(cols_per_row)
            for j, (_, row) in enumerate(row_items.iterrows()):
                with cols[j]:
                    score = row["final_score"]
                    color = "#28a745" if score >= 75 else "#ffc107" if score >= 55 else "#dc3545"
                    st.markdown(f"""
                    <div style='border:1px solid #ddd;border-radius:12px;
                                padding:16px;margin-bottom:10px;'>
                        <div style='font-size:12px;color:gray;'>#{i+j+1} · {row['category']}</div>
                        <div style='font-size:15px;font-weight:600;margin:4px 0;
                                    line-height:1.3;'>{str(row['product'])[:45]}</div>
                        <div style='font-size:13px;color:gray;margin-bottom:8px;'>{row['brand']}</div>
                        <div style='font-size:32px;font-weight:700;color:{color};'>
                            {score}<span style='font-size:14px;font-weight:400;'>/100</span>
                        </div>
                        <div style='font-size:12px;margin-top:6px;'>
                            ⭐ {row['avg_rating']}/5 &nbsp; 💬 {int(row['review_count']):,} reviews
                        </div>
                        <div style='font-size:12px;'>😊 Sentiment: {row['sentiment_pct']}%</div>
                        <div style='font-size:12px;color:gray;margin-top:4px;'>
                            💰 ${row['avg_price']:.0f} &nbsp; 🛒 {row['platforms']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Charts ───────────────────────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Score comparison")
            fig1 = px.bar(
                results.sort_values("final_score"),
                x="final_score", y="product", orientation="h",
                color="final_score", color_continuous_scale="teal",
                text="final_score",
                labels={"final_score": "SmartPick Score", "product": ""}
            )
            fig1.update_traces(textposition="outside")
            fig1.update_layout(
                coloraxis_showscale=False,
                margin=dict(l=0, r=30, t=10, b=10), height=420,
                yaxis=dict(tickfont=dict(size=10))
            )
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            st.subheader("Rating vs sentiment")
            fig2 = px.scatter(
                results, x="avg_rating", y="sentiment_pct",
                size="review_count", color="final_score",
                hover_name="product",
                color_continuous_scale="viridis",
                labels={"avg_rating": "Avg Rating (1–5)", "sentiment_pct": "Sentiment (%)"}
            )
            fig2.update_layout(margin=dict(l=0, r=0, t=10, b=10), height=420)
            st.plotly_chart(fig2, use_container_width=True)

        # ── Aspect-based sentiment ───────────────────────────────────
        st.markdown("---")
        st.subheader("Aspect-based sentiment analysis")
        st.caption(
            "Feature-level sentiment extracted from review text using keyword-based NLP. "
            "Green = mostly positive mentions, Red = mostly negative."
        )

        asp_cols = st.columns(min(3, len(results)))
        for idx, (col, (_, row)) in enumerate(
            zip(asp_cols, results.head(3).iterrows())
        ):
            with col:
                st.markdown(f"**{str(row['product'])[:30]}**")
                render_aspect_chart(row["aspects"])

        # ── Full table ───────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Full comparison table")
        display = results[[
            "product","brand","category","avg_rating","sentiment_pct",
            "review_count","avg_price","platforms","final_score"
        ]].rename(columns={
            "product":"Product","brand":"Brand","category":"Category",
            "avg_rating":"Rating","sentiment_pct":"Sentiment %",
            "review_count":"Reviews","avg_price":"Price (USD)",
            "platforms":"Platforms","final_score":"SmartPick Score"
        })

        def color_score(val):
            if val >= 75: return "background-color:#d4edda;color:#155724"
            if val >= 55: return "background-color:#fff3cd;color:#856404"
            return "background-color:#f8d7da;color:#721c24"

        st.dataframe(
            display.style.applymap(color_score, subset=["SmartPick Score"]),
            use_container_width=True, hide_index=True
        )

# ── HOMEPAGE (no query) ───────────────────────────────────────────────
else:
    st.subheader("Browse categories")
    icons = {
        "Smartphones":"📱","Laptops":"💻","TVs":"📺",
        "Headphones & Speakers":"🎧","Tablets":"📟",
        "Cameras":"📷","Wearables":"⌚","Gaming Consoles":"🎮"
    }
    cat_stats = (
        df.groupby("category")
        .agg(products=("product","count"),
             avg_score=("final_score","mean"),
             avg_rating=("avg_rating","mean"))
        .reset_index()
        .sort_values("avg_score", ascending=False)
    )
    cat_stats["avg_score"]  = cat_stats["avg_score"].round(1)
    cat_stats["avg_rating"] = cat_stats["avg_rating"].round(2)

    cols = st.columns(4)
    for i, (_, row) in enumerate(cat_stats.iterrows()):
        icon = icons.get(row["category"], "🔌")
        with cols[i % 4]:
            st.markdown(f"""
            <div style='border:1px solid #ddd;border-radius:10px;padding:16px;
                        margin-bottom:10px;text-align:center;'>
                <div style='font-size:28px;'>{icon}</div>
                <div style='font-size:14px;font-weight:600;margin:4px 0;'>{row['category']}</div>
                <div style='font-size:26px;font-weight:700;color:#1D9E75;'>{row['avg_score']}</div>
                <div style='font-size:11px;color:gray;'>avg SmartPick score</div>
                <div style='font-size:12px;margin-top:6px;'>
                    {int(row['products'])} products · ⭐ {row['avg_rating']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Top 10 products overall")
    top10 = df.nlargest(10, "final_score")[[
        "product","brand","category","avg_rating",
        "sentiment_pct","review_count","avg_price","final_score"
    ]]
    top10.columns = ["Product","Brand","Category","Rating",
                     "Sentiment %","Reviews","Price (USD)","Score"]
    st.dataframe(top10, use_container_width=True, hide_index=True)

# ── FOOTER ────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "SmartPick · 9 datasets merged · VADER Sentiment · Aspect-Based NLP · "
    "User-Centered Scoring (Budget + Use Case) · Streamlit + Plotly"
)

