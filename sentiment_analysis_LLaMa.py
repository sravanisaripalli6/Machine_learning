import praw
import pandas as pd
from transformers import pipeline
import spacy
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import nltk
from nltk.corpus import stopwords

# Load spaCy English model for text processing
nlp = spacy.load("en_core_web_sm")

# Load English stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Set Output Directory
output_dir = "/Users/sravanisaripalli/Downloads/sentiment_analysis"
os.makedirs(output_dir, exist_ok=True)

# Initialize Sentiment Analysis Pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Initialize Reddit API (PRAW)
reddit = praw.Reddit(
    client_id="F8_2CeJuWfZEzAs7IHRZrg",
    client_secret="9cHbpnEITPz6OMfIBgxu2bK8-35Sxg",
    user_agent="LLM_Sentiment_Analysis"
)

# Fetch Reddit Posts (Top 1000)
def fetch_reddit_posts():
    subreddit = reddit.subreddit("technology+MachineLearning+artificial+ai")
    posts = subreddit.search("LLaMA", limit=1000)

    results = []
    for post in posts:
        if post.selftext:
            text = post.title + " " + post.selftext
        else:
            text = post.title

        # Perform Sentiment Analysis
        sentiment = sentiment_analyzer(text[:500])[0]
        results.append({
            "Title": post.title,
            "Text": text,
            "Sentiment": sentiment['label'],
            "Score": sentiment['score'],
            "Upvotes": post.score,
            "Comments": post.num_comments,
            "URL": post.url
        })

    return pd.DataFrame(results)

# Run Reddit Scraping
df = fetch_reddit_posts()
csv_path = os.path.join(output_dir, "llama_reddit_sentiment_analysis.csv")
df.to_csv(csv_path, index=False)
print(f"\n✅ Reddit data scraped and saved to '{csv_path}'.")

# Cleaning Text for Word Cloud
def clean_text(text):
    doc = nlp(text.lower())
    clean_words = [token.text for token in doc if token.is_alpha and token.text not in stop_words and token.pos_ != "PRON"]
    return " ".join(clean_words)

# Create a Clean Text Corpus
all_text = " ".join(df['Text'].dropna().tolist())
cleaned_text = clean_text(all_text)

# Generate Word Cloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(cleaned_text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud for LLaMA (Reddit Analysis - Top 1000 Posts)")
plt.show()

# Sentiment Distribution Analysis
sentiment_counts = df['Sentiment'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(
    sentiment_counts, 
    labels=sentiment_counts.index, 
    autopct='%1.1f%%', 
    startangle=90, 
    colors=['#4caf50', '#f44336', '#2196f3']
)
plt.title("Audience Sentiment Distribution (Top 1000 Reddit Posts)")
plt.show()

# Identify Top 5 Positive and Negative Posts
top_positive = df[df['Sentiment'] == 'POSITIVE'].sort_values(by="Score", ascending=False).head(5)
top_negative = df[df['Sentiment'] == 'NEGATIVE'].sort_values(by="Score", ascending=True).head(5)

# Display Top Positive and Negative Posts
print("\n🔍 Top 5 Positive Reddit Posts:")
for index, row in top_positive.iterrows():
    print(f"Title: {row['Title']} | Sentiment: {row['Sentiment']} | Score: {row['Score']} | Upvotes: {row['Upvotes']} | URL: {row['URL']}")

print("\n🔍 Top 5 Negative Reddit Posts:")
for index, row in top_negative.iterrows():
    print(f"Title: {row['Title']} | Sentiment: {row['Sentiment']} | Score: {row['Score']} | Upvotes: {row['Upvotes']} | URL: {row['URL']}")

# Save Report Summary
report_path = os.path.join(output_dir, "llama_reddit_report_summary.txt")
with open(report_path, "w") as file:
    file.write("🔍 Top 5 Positive Reddit Posts:\n")
    for index, row in top_positive.iterrows():
        file.write(f"Title: {row['Title']} | Sentiment: {row['Sentiment']} | Score: {row['Score']} | Upvotes: {row['Upvotes']} | URL: {row['URL']}\n")

    file.write("\n🔍 Top 5 Negative Reddit Posts:\n")
    for index, row in top_negative.iterrows():
        file.write(f"Title: {row['Title']} | Sentiment: {row['Sentiment']} | Score: {row['Score']} | Upvotes: {row['Upvotes']} | URL: {row['URL']}\n")

print(f"\n✅ Report Generated: '{report_path}'.")
