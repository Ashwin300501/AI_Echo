import os
import pandas as pd
import plotly.express as px
from transformers import pipeline
from wordcloud import WordCloud
import streamlit as st

st.set_page_config(page_title="AI Echo Sentiment Dashboard", layout="wide")


ROOT = r"D:\Data Science\AI_Echo"
DATA_PATH = os.path.join(ROOT, "data", "clean", "reviews_clean.parquet")
MODEL_PATH = os.path.join(ROOT, "models", "distilbert_sentiment")

@st.cache_data
def load_data(path: str):
    df = pd.read_parquet(path)
    return df

@st.cache_resource
def load_model(model_dir: str):
    # DistilBERT fine-tuned folder from your Colab training
    return pipeline("text-classification", model=model_dir, tokenizer=model_dir, return_all_scores=True)

def rating_to_sentiment(r: int) -> str:
    if r is None:
        return "neutral"
    try:
        r = int(r)
    except Exception:
        return "neutral"
    if r <= 2: return "negative"
    if r == 3: return "neutral"
    return "positive"

def make_wordcloud(texts, title: str):
    text = " ".join([t for t in texts if isinstance(t, str)])
    if not text.strip():
        st.info(f"No text available for {title}")
        return
    wc = WordCloud(width=900, height=350, background_color="white").generate(text)
    st.image(wc.to_array(), caption=title, use_container_width=True)


st.title("🎯 AI Echo — ChatGPT Review Sentiment Dashboard")

# Load dataset & model
if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found at:\n{DATA_PATH}")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error(f"Model folder not found at:\n{MODEL_PATH}")
    st.stop()

df = load_data(DATA_PATH)
pipe = load_model(MODEL_PATH)

# Prepare helpful derived columns
text_col = "review_clean" if "review_clean" in df.columns else ("review" if "review" in df.columns else None)
if text_col is None:
    st.error("Neither 'review_clean' nor 'review' column found in dataset.")
    st.stop()

df = df.copy()
df["text"] = df[text_col].fillna("").astype(str)
df["sentiment"] = df["rating"].apply(rating_to_sentiment)

# Tabs
tab1, tab2 = st.tabs(["💬 Sentiment Prediction", "📊 Key Questions & Insights"])

with tab1:
    st.subheader("🔍 Predict Sentiment for a Review")

    user_input = st.text_area(
        "Enter your review:",
        placeholder="Example: I love using ChatGPT, it’s very helpful for coding!",
        height=140
    )

    if st.button("Analyze Sentiment"):
        if user_input.strip():
            out = pipe(user_input[:2000])
            scores = {d["label"]: d["score"] for d in out[0]}
            best_label = max(scores, key=scores.get)
            st.success(f"Predicted Sentiment: **{best_label.capitalize()}**")
            order = ["negative", "neutral", "positive"]
            y_vals = [scores.get(lbl, 0.0) for lbl in order]
            fig = px.bar(x=order, y=y_vals, labels={"x": "Class", "y": "Score"}, title="Class probabilities")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please enter a review to analyze.")

    st.markdown("---")
    st.subheader("📈 Overall Sentiment (Dataset — rating-mapped)")
    sentiment_counts = df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    fig = px.pie(
        sentiment_counts, names="Sentiment", values="Count",
        title="Overall Sentiment Distribution (by rating mapping)"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("📊 Key Questions for Sentiment Analysis")

    # 1) Overall sentiment of user reviews
    st.markdown("### 1) What is the overall sentiment of user reviews?")
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(sentiment_counts.set_index("Sentiment"))
    with col2:
        st.dataframe(sentiment_counts, use_container_width=True)

    st.markdown("---")
    # 2) Sentiment vs rating (mismatch? — here we just show rating vs rating-mapped sentiment)
    st.markdown("### 2) How does sentiment vary by rating? Any mismatch?")
    fig2 = px.histogram(df, x="rating", color="sentiment", barmode="group", title="Sentiment by Rating")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    # 3) Keywords per sentiment (word clouds)
    st.markdown("### 3) Which keywords or phrases are most associated with each sentiment class?")
    c1, c2, c3 = st.columns(3)
    with c1:
        make_wordcloud(df.loc[df["sentiment"] == "positive", "text"].head(3000), "Positive Keywords")
    with c2:
        make_wordcloud(df.loc[df["sentiment"] == "neutral", "text"].head(3000), "Neutral Keywords")
    with c3:
        make_wordcloud(df.loc[df["sentiment"] == "negative", "text"].head(3000), "Negative Keywords")

    st.markdown("---")
    # 4) Sentiment over time (rating avg by month)
    st.markdown("### 4) How has sentiment changed over time?")
    if "date" in df.columns:
        dfx = df.copy()
        dfx["date"] = pd.to_datetime(dfx["date"], errors="coerce")
        dfx = dfx.dropna(subset=["date"])
        if not dfx.empty:
            dfx["month"] = dfx["date"].dt.to_period("M").astype(str)
            trend = dfx.groupby("month")["rating"].mean().reset_index()
            fig3 = px.line(trend, x="month", y="rating", markers=True, title="Average Rating Over Time (Monthly)")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No valid dates found to plot trend.")
    else:
        st.info("No 'date' column found.")

    st.markdown("---")
    # 5) Verified users positivity/negativity
    st.markdown("### 5) Do verified users tend to leave more positive or negative reviews?")
    if "verified_purchase" in df.columns:
        vp = df.groupby("verified_purchase")["rating"].mean().reset_index()
        fig4 = px.bar(vp, x="verified_purchase", y="rating", title="Verified vs Non-Verified — Average Rating")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No 'verified_purchase' column in dataset.")

    st.markdown("---")
    # 6) Review length vs sentiment
    st.markdown("### 6) Are longer reviews more likely to be negative or positive?")
    if "review_length" not in df.columns:
        # fallback if you didn't save review_length during Phase 1
        df["review_length"] = df["text"].astype(str).str.len()
    fig5 = px.box(df, x="sentiment", y="review_length", points="outliers", title="Review Length by Sentiment")
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")
    # 7) Location sentiment
    st.markdown("### 7) Which locations show the most positive or negative sentiment?")
    if "location" in df.columns:
        loc_stats = (
            df.groupby("location")["rating"]
            .mean()
            .reset_index()
            .sort_values("rating", ascending=False)
            .head(12)
        )
        fig6 = px.bar(loc_stats, x="rating", y="location", orientation="h", title="Top Locations by Average Rating")
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info("No 'location' column available.")

    st.markdown("---")
    # 8) Platform differences
    st.markdown("### 8) Is there a difference across platforms (Web vs Mobile)?")
    if "platform" in df.columns:
        plat_stats = df.groupby("platform")["rating"].mean().reset_index()
        fig7 = px.bar(plat_stats, x="platform", y="rating", title="Average Rating by Platform")
        st.plotly_chart(fig7, use_container_width=True)
    else:
        st.info("No 'platform' column found.")

    st.markdown("---")
    # 9) Version impact
    st.markdown("### 9) Which ChatGPT versions are associated with higher/lower sentiment?")
    if "version" in df.columns:
        ver_stats = (
            df.groupby("version")["rating"]
            .mean()
            .reset_index()
            .sort_values("rating", ascending=False)
            .head(15)
        )
        fig8 = px.bar(ver_stats, x="version", y="rating", title="Average Rating by ChatGPT Version")
        st.plotly_chart(fig8, use_container_width=True)
    else:
        st.info("No 'version' column found.")

    st.markdown("---")
    # 10) Negative themes (quick terms)
    st.markdown("### 10) What are the most common negative feedback themes?")
    from collections import Counter
    neg_texts = df.loc[df["sentiment"] == "negative", "text"]
    vocab_counts = Counter(" ".join(neg_texts).split())
    top_terms = pd.DataFrame(vocab_counts.most_common(20), columns=["term", "freq"])
    if not top_terms.empty:
        fig9 = px.bar(top_terms, x="freq", y="term", orientation="h", title="Top Terms in Negative Reviews")
        st.plotly_chart(fig9, use_container_width=True)
    else:
        st.info("Not enough negative reviews to extract themes.")

st.markdown("---")
