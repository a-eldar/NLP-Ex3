from nltk.corpus import brown
from nltk.lm import MLE
from collections import defaultdict, Counter
from typing import Callable

# Ensure you have downloaded the Brown corpus
import nltk
nltk.download('brown')

# Extract words from the "news" category
news_sentences = brown.tagged_sents(categories='news')

# Create train and test splits
split_index = int(0.9 * len(news_sentences))
train_sentences = news_sentences[:split_index]
test_sentences = news_sentences[split_index:]

def compute_probabilities(sentences: list[list[str]]):
    """Compute for each word the tag that yeilds max P(tag|word)
    This is for Q3b.i

    Args
    ----
        sentences (list[list[str]])

    Returns
    -------
        dict[str, str]
    """

    # Step 1: Count tag occurrences for each word
    word_tag_counts = defaultdict(Counter)

    for sentence in sentences:
        for word, tag in sentence:
            word_tag_counts[word.lower()][tag] += 1  # Use lowercase to avoid case sensitivity

    # Step 2: Find the tag that maximizes P(tag|word) for each word
    most_likely_tags = {word: tag_counts.most_common(1)[0][0] for word, tag_counts in word_tag_counts.items()}
    return most_likely_tags

def compute_model_error_rates(model: Callable[[str], tuple[str, bool]], test_sentences: list[list[str]]):
    """Given tagging model,
    compute error for known words, unknown words, and in total.

    Args:
        model (Callable[[str], tuple[str, bool]]): Given word,\
              returns tag and whether it is a known word
        test_sentences (list[list[str]])
    
    Return:
        tuple[float, float, float]: known words error, unknown words error, total error
    """
    known_words_error = 0
    total_known_words = 0
    unknown_words_error = 0
    total_unknown_words = 0

    for sentence in test_sentences:
        for word, tag in sentence:
            guess, is_known = model(word)
            if is_known:
                total_known_words += 1
            else:
                total_unknown_words += 1
            
            if guess == tag:
                continue
            
            if is_known:
                known_words_error += 1
            else:
                unknown_words_error += 1
    
    total_error = known_words_error + unknown_words_error
    known_words_error /= total_known_words
    unknown_words_error /= total_unknown_words
    total_error /= total_known_words + total_unknown_words
    return known_words_error, unknown_words_error, total_error
            



if __name__ == "__main__":
    MOST_COMMON_TAG = 'NN'
    model_dict = compute_probabilities(train_sentences)
    def model(word: str) -> tuple[str, bool]:
        if word in model_dict:
            return model_dict[word], True
        return MOST_COMMON_TAG, False

    print(compute_model_error_rates(model, test_sentences))