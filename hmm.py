from nltk.corpus import brown
from sklearn.model_selection import train_test_split

# Ensure you have downloaded the Brown corpus
import nltk
nltk.download('brown')

# Extract words from the "news" category
news_sentences = brown.sents(categories='news')

# Create train and test splits
split_index = int(0.9 * len(news_sentences))
train_sentences = news_sentences[:split_index]
test_sentences = news_sentences[split_index:]

