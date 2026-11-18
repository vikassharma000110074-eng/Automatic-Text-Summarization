
from nltk.tokenize import sent_tokenize

from summarizer import calculate_sentence_scores


text = """
Artificial Intelligence is transforming the world.
Machine learning is an important branch of Artificial Intelligence.
Machine learning helps computers learn from data.
Natural Language Processing helps computers understand human language.
Text summarization is an important application of Natural Language Processing.
Text summarization generates shorter versions of long documents.
Artificial Intelligence is used in healthcare and education.
Machine learning improves data analysis and prediction.
"""

# Tokenize sentences
sentences = sent_tokenize(text)

# Calculate TF-IDF sentence scores
score_df, word_frequencies, normalized_frequencies = (
    calculate_sentence_scores(sentences)
)

# Display results
print("\nTF-IDF Sentence Importance Scores")

print(
    score_df[
        [
            "Sentence Number",
            "Sentence",
            "TF-IDF Score",
            "Importance Score"
        ]
    ].to_string(index=False)
)
