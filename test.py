import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from yake import KeywordExtractor
import nltk

# Download NLTK data
nltk.download("stopwords")
nltk.download("wordnet")

# Preprocessing configuration
stop_words = set(stopwords.words("french"))
lemmatizer = WordNetLemmatizer()


def preprocess_tweet(tweet):
    """Preprocess a tweet by removing noise and standardizing text."""
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)  # Remove URLs
    tweet = re.sub(r"\@\w+|\#", "", tweet)  # Remove mentions and hashtags
    tweet = re.sub(r"[^\w\s]", "", tweet)  # Remove punctuations
    tweet = tweet.lower()  # Lowercasing
    tweet_tokens = tweet.split()
    return " ".join(
        [lemmatizer.lemmatize(word) for word in tweet_tokens if word not in stop_words]
    )


# Keyword extraction
def extract_keywords(tweet, max_keywords=3):
    """Extract keywords from a tweet using YAKE."""
    kw_extractor = KeywordExtractor(lan="fr", top=max_keywords)
    keywords = kw_extractor.extract_keywords(tweet)
    return [kw[0] for kw in sorted(keywords, key=lambda x: x[1])[:max_keywords]]


# Clustering and topic determination
def cluster_keywords(all_keywords, n_clusters):
    """Cluster all keywords into topics."""
    vectorizer = TfidfVectorizer()
    keyword_vectors = vectorizer.fit_transform(all_keywords)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(keyword_vectors)

    return kmeans, vectorizer


def assign_topic_to_tweet(keywords, kmeans, vectorizer, topic_clusters):
    """Assign a topic to a tweet based on its keywords."""
    if not keywords:
        return "Uncategorized"
    keyword_vectors = vectorizer.transform(keywords)
    cluster_indices = kmeans.predict(keyword_vectors)
    # Get the most frequent cluster for this tweet
    assigned_cluster = max(set(cluster_indices), key=list(cluster_indices).count)
    return topic_clusters[assigned_cluster]


# Main application logic
def extract_tweet_topics(tweets, topic_clusters):
    """Extract topics from a list of tweets."""
    preprocessed_tweets = [preprocess_tweet(tweet) for tweet in tweets]
    all_keywords = [
        keyword for tweet in preprocessed_tweets for keyword in extract_keywords(tweet)
    ]

    # Cluster all keywords
    kmeans, vectorizer = cluster_keywords(all_keywords, len(topic_clusters))

    # Determine topics for each tweet
    results = []
    for tweet, preprocessed_tweet in zip(tweets, preprocessed_tweets):
        keywords = extract_keywords(preprocessed_tweet)
        topic = assign_topic_to_tweet(keywords, kmeans, vectorizer, topic_clusters)
        results.append((tweet, topic))
    return results


# Example use case
if __name__ == "__main__":
    # Define some example tweets
    tweets = [
        "Les Merdias nous controlent avec les gouvernements",
        "@complotiste tu dis de la merde, c'est les illuminatis",
        "Au moins on est d'accord, le réchauffement climatique c'est du flan",
        "J'aime les enfants.",
        "J'aime mes enfants et ma femme, je suis heureux, aimant et aimé",
    ]

    # Define potential topics
    topic_clusters = ["Complot", "Amour", "Medias", "Environnement"]

    # Extract topics for each tweet
    topics = extract_tweet_topics(tweets, topic_clusters)
    for tweet, topic in topics:
        print(f"Tweet: {tweet}\nPredicted Topic: {topic}\n")
