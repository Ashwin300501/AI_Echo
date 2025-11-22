# ─────────────────────────────────────────────────────────────
# AI Echo — Streamlit App (MODEL ONLY: negative / neutral / positive)
# ─────────────────────────────────────────────────────────────

import os
from pathlib import Path
from collections import Counter
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)

# --------------------------------------------------------------
st.set_page_config(page_title="AI Echo Sentiment Dashboard", page_icon="🧠", layout="wide")

# --------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "clean" / "reviews_clean.parquet"

MODEL_ID="Ashwin20015/ai-echo-distilbert-sentiment"

# --------------------------------------------------------------
# Loaders
# --------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: Path):
    if not path.exists():
        st.error(f"Dataset not found at:\n{path}")
        st.stop()
    df = pd.read_parquet(path)
    if "review_clean" in df.columns:
        df["text"] = df["review_clean"].fillna("").astype(str)
    elif "review" in df.columns:
        df["text"] = df["review"].fillna("").astype(str)
    else:
        st.error("Neither 'review_clean' nor 'review' found.")
        st.stop()
    return df

@st.cache_resource(show_spinner=False)
def load_model(model_id: str):
    # This will download from Hugging Face the first time, then cache
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)

    device = 0 if torch.cuda.is_available() else -1
    pipe = TextClassificationPipeline(
        model=mdl,
        tokenizer=tok,
        device=device,
        return_all_scores=True,
    )

    id2label = mdl.config.id2label
    label_order = [id2label[i] for i in sorted(id2label.keys())]
    return pipe, label_order

@st.cache_data(show_spinner=True)
def add_model_predictions(df, _pipe):
    texts = df["text"].astype(str).tolist()
    preds = []
    for i in range(0, len(texts), 64):
        out = _pipe(texts[i:i+64], truncation=True, max_length=384)
        for row in out:
            best = max(row, key=lambda d: d["score"])
            preds.append(best["label"])
    df = df.copy()
    df["sentiment_model"] = preds
    return df

def make_wordcloud(texts, title):
    text = " ".join([t for t in texts if isinstance(t, str)])
    if not text.strip():
        st.info(f"No text for {title}")
        return
    wc = WordCloud(width=900, height=350, background_color="white").generate(text)
    st.image(wc.to_array(), caption=title, use_container_width=True)

# --------------------------------------------------------------
# Load
# --------------------------------------------------------------
df_raw = load_data(DATA_PATH)
pipe, label_order = load_model(MODEL_ID)
df = add_model_predictions(df_raw, _pipe=pipe)

# --------------------------------------------------------------
# Layout
# --------------------------------------------------------------
st.title("🧠 AI Echo —Review Sentiment Dashboard")
st.caption("All insights below come from the fine-tuned DistilBERT model.")

tab1, tab2 = st.tabs(["💬 Sentiment Prediction", "📊 Key Questions & Insights"])

# ============================================================
# TAB 1 — Single prediction + distribution
# ============================================================
with tab1:
    st.subheader("🔍 Predict sentiment for a single review")

    user_input = st.text_area(
        "Enter your review:",
        placeholder="Example: I love using ChatGPT, it’s very helpful for coding!",
        height=140
    )

    if st.button("Analyze Sentiment", type="primary"):
        if user_input.strip():
            out = pipe(user_input[:2000])
            scores = {d["label"]: d["score"] for d in out[0]}
            best_label = max(scores, key=scores.get)
            st.success(f"Predicted Sentiment: **{best_label.capitalize()}**")

            fig_pred = px.bar(
                x=label_order,
                y=[scores.get(lbl, 0.0) for lbl in label_order],
                labels={"x": "Class", "y": "Score"},
                title="Class probabilities"
            )
            st.plotly_chart(fig_pred, use_container_width=True, key="single_pred_chart")
        else:
            st.warning("Please enter a review to analyze.")

    st.markdown("---")
    st.subheader("📈 Overall Sentiment (Model Predictions)")
    sentiment_counts = df["sentiment_model"].value_counts(normalize=True).rename_axis("Sentiment").reset_index(name="Proportion")
    sentiment_counts["Proportion (%)"] = (sentiment_counts["Proportion"] * 100).round(1)
    fig_overall = px.bar(
        sentiment_counts,
        x="Sentiment", y="Proportion (%)", text="Proportion (%)",
        title="Overall Sentiment Distribution (%) — Model"
    )
    fig_overall.update_traces(textposition="outside")
    st.plotly_chart(fig_overall, use_container_width=True, key="overall_dist")

# ============================================================
# TAB 2 — Key Questions (model predictions)
# ============================================================
with tab2:
    st.subheader("❓ Key Questions for Sentiment Analysis (Model-based)")

    # 1) Overall sentiment
    st.markdown("### 1) What is the overall sentiment of user reviews?")
    st.plotly_chart(fig_overall, use_container_width=True, key="overall_sentiment_bar")

    st.markdown("---")
    # 2) Sentiment vs rating
    st.markdown("### 2) How does sentiment vary by rating? Any mismatch?")
    if "rating" in df.columns:
        ct = pd.crosstab(df["rating"], df["sentiment_model"]).reset_index().melt(id_vars="rating", var_name="pred", value_name="count")
        fig2 = px.bar(ct, x="rating", y="count", color="pred", barmode="group", title="Model Sentiment across Ratings")
        st.plotly_chart(fig2, use_container_width=True, key="sentiment_vs_rating")
    else:
        st.info("No 'rating' column to compare against.")

    st.markdown("---")
    # 3) Word clouds
    st.markdown("### 3) Which keywords or phrases are most associated with each sentiment?")
    c1, c2, c3 = st.columns(3)
    with c1:
        make_wordcloud(df.loc[df["sentiment_model"] == "positive", "text"], "Positive Keywords")
    with c2:
        make_wordcloud(df.loc[df["sentiment_model"] == "neutral", "text"], "Neutral Keywords")
    with c3:
        make_wordcloud(df.loc[df["sentiment_model"] == "negative", "text"], "Negative Keywords")

    st.markdown("---")
    # 4) Sentiment trend over time
    st.markdown("### 4) How has sentiment changed over time?")
    if "date" in df.columns:
        dfx = df.copy()
        dfx["date"] = pd.to_datetime(dfx["date"], errors="coerce")
        dfx = dfx.dropna(subset=["date"])
        if not dfx.empty:
            dfx["month"] = dfx["date"].dt.to_period("M").astype(str)
            trend = dfx.groupby(["month", "sentiment_model"]).size().reset_index(name="count")
            fig3 = px.line(trend, x="month", y="count", color="sentiment_model", markers=True, title="Sentiment Trend by Month (Model)")
            st.plotly_chart(fig3, use_container_width=True, key="trend_over_time")
        else:
            st.info("No valid dates to plot trend.")
    else:
        st.info("No 'date' column found.")

    st.markdown("---")
    # 5) Verified users
    st.markdown("### 5) Do verified users tend to leave more positive or negative reviews?")
    if "verified_purchase" in df.columns:
        vf = df.groupby(["verified_purchase", "sentiment_model"]).size().reset_index(name="count")
        fig4 = px.bar(vf, x="verified_purchase", y="count", color="sentiment_model", barmode="group",
                      title="Sentiment by Verified Purchase (Model)")
        st.plotly_chart(fig4, use_container_width=True, key="verified_users")
    else:
        st.info("No 'verified_purchase' column.")

    st.markdown("---")
    # 6) Review length vs sentiment
    st.markdown("### 6) Are longer reviews more likely to be negative or positive?")
    if "review_length" not in df.columns:
        df["review_length"] = df["text"].astype(str).str.len()
    fig5 = px.box(df, x="sentiment_model", y="review_length", points="outliers", title="Review Length by Sentiment (Model)")
    st.plotly_chart(fig5, use_container_width=True, key="length_by_sentiment")

    st.markdown("---")
    # 7) Locations
    st.markdown("### 7) Which locations show the most positive or negative sentiment?")
    if "location" in df.columns:
        loc = df.groupby(["location", "sentiment_model"]).size().reset_index(name="count")
        top = loc.groupby("location")["count"].sum().nlargest(12).index
        loc = loc[loc["location"].isin(top)]
        fig6 = px.bar(loc, x="location", y="count", color="sentiment_model", barmode="group",
                      title="Top Locations — Sentiment Breakdown (Model)")
        st.plotly_chart(fig6, use_container_width=True, key="location_chart")
    else:
        st.info("No 'location' column available.")

    st.markdown("---")
    # 8) Platform differences
    st.markdown("### 8) Is there a difference across platforms (Web vs Mobile)?")
    if "platform" in df.columns:
        plat = df.groupby(["platform", "sentiment_model"]).size().reset_index(name="count")
        fig7 = px.bar(plat, x="platform", y="count", color="sentiment_model", barmode="group",
                      title="Sentiment by Platform (Model)")
        st.plotly_chart(fig7, use_container_width=True, key="platform_chart")
    else:
        st.info("No 'platform' column found.")

    st.markdown("---")
    # 9) Version impact
    st.markdown("### 9) Which ChatGPT versions are associated with higher/lower sentiment?")
    if "version" in df.columns:
        ver = df.groupby(["version", "sentiment_model"]).size().reset_index(name="count")
        fig8 = px.bar(ver, x="version", y="count", color="sentiment_model", barmode="group",
                      title="Sentiment by Version (Model)")
        st.plotly_chart(fig8, use_container_width=True, key="version_chart")
    else:
        st.info("No 'version' column found.")

    st.markdown("---")
    # 10) Negative themes
    st.markdown("### 10) What are the most common negative feedback themes?")
    neg_texts = df.loc[df["sentiment_model"] == "negative", "text"].astype(str).tolist()
    if len(neg_texts) >= 5:
        vocab_counts = Counter(" ".join(neg_texts).split())
        top_terms = pd.DataFrame(vocab_counts.most_common(20), columns=["term", "freq"])
        fig9 = px.bar(top_terms, x="freq", y="term", orientation="h", title="Top Terms in Negative Reviews (Model)")
        st.plotly_chart(fig9, use_container_width=True, key="negative_themes")
    else:
        st.info("Not enough negative reviews to extract themes.")

st.markdown("---")
