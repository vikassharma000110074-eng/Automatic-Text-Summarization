
import math
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import preprocess_text


# -----------------------------------------
# 1. WORD FREQUENCY ANALYSIS
# -----------------------------------------

def calculate_word_frequencies(sentences):

    words = []

    for sentence in sentences:
        words.extend(preprocess_text(sentence))

    frequencies = Counter(words)

    if not frequencies:
        return {}, {}

    max_frequency = max(frequencies.values())

    normalized_frequencies = {
        word: count / max_frequency
        for word, count in frequencies.items()
    }

    return dict(frequencies), normalized_frequencies


# -----------------------------------------
# 2. TF-IDF SENTENCE SCORING
# -----------------------------------------

def calculate_sentence_scores(sentences):

    word_frequencies, normalized_frequencies = (
        calculate_word_frequencies(sentences)
    )

    cleaned_sentences = [
        " ".join(preprocess_text(sentence))
        for sentence in sentences
    ]

    columns = [
        "Sentence Number",
        "Sentence",
        "Word Count",
        "TF-IDF Score",
        "Position Bonus",
        "Selection Score"
    ]

    if not sentences:
        return (
            pd.DataFrame(columns=columns),
            word_frequencies,
            normalized_frequencies
        )

    # Handle documents without meaningful words
    if not any(cleaned_sentences):

        rows = []

        for i, sentence in enumerate(sentences):

            rows.append({
                "Sentence Number": i + 1,
                "Sentence": sentence,
                "Word Count": 0,
                "TF-IDF Score": 0.0,
                "Position Bonus": 0.0,
                "Selection Score": 0.0
            })

        return (
            pd.DataFrame(rows),
            word_frequencies,
            normalized_frequencies
        )

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False
    )

    tfidf_matrix = vectorizer.fit_transform(
        cleaned_sentences
    )

    total_sentences = len(sentences)

    rows = []

    for i, sentence in enumerate(sentences):

        # Get non-zero TF-IDF values
        tfidf_values = tfidf_matrix.getrow(i).data

        if len(tfidf_values) > 0:
            tfidf_score = float(tfidf_values.mean())
        else:
            tfidf_score = 0.0

        # Calculate sentence position bonus
        if total_sentences > 1:
            position_bonus = (
                0.10 * (1 - i / (total_sentences - 1))
            )
        else:
            position_bonus = 0.0

        # Calculate final selection score
        selection_score = (
            tfidf_score * (1 + position_bonus)
        )

        rows.append({
            "Sentence Number": i + 1,
            "Sentence": sentence,
            "Word Count": len(preprocess_text(sentence)),
            "TF-IDF Score": round(tfidf_score, 4),
            "Position Bonus": round(position_bonus, 4),
            "Selection Score": round(selection_score, 4)
        })

    score_df = pd.DataFrame(rows)

    return (
        score_df,
        word_frequencies,
        normalized_frequencies
    )


# -----------------------------------------
# 3. SELECT IMPORTANT SENTENCES
# -----------------------------------------

def select_important_sentences(
    score_df,
    summary_ratio=0.30,
    similarity_threshold=0.85
):

    if score_df.empty:
        return score_df.copy()

    if not 0 < summary_ratio <= 1:
        raise ValueError(
            "Summary ratio must be greater than 0 and at most 1."
        )

    # Determine number of sentences to select
    number_to_select = max(
        1,
        math.ceil(len(score_df) * summary_ratio)
    )

    # Rank sentences by selection score
    ranked_sentences = score_df.sort_values(
        by="Selection Score",
        ascending=False,
        kind="stable"
    )

    original_sentences = score_df["Sentence"].tolist()

    cleaned_sentences = [
        " ".join(preprocess_text(sentence))
        for sentence in original_sentences
    ]

    # Build vectors for redundancy checking
    tfidf_matrix = None

    if any(cleaned_sentences):

        vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            lowercase=False
        )

        tfidf_matrix = vectorizer.fit_transform(
            cleaned_sentences
        )

    sentence_indices = {
        int(number): index
        for index, number in enumerate(
            score_df["Sentence Number"]
        )
    }

    selected_numbers = []

    # Select sentences while avoiding duplicates
    for _, candidate in ranked_sentences.iterrows():

        sentence_number = int(
            candidate["Sentence Number"]
        )

        candidate_index = sentence_indices[sentence_number]

        is_redundant = False

        if tfidf_matrix is not None:

            for selected_number in selected_numbers:

                selected_index = sentence_indices[
                    selected_number
                ]

                similarity = cosine_similarity(
                    tfidf_matrix[candidate_index],
                    tfidf_matrix[selected_index]
                )[0, 0]

                if similarity >= similarity_threshold:
                    is_redundant = True
                    break

        if not is_redundant:
            selected_numbers.append(sentence_number)

        if len(selected_numbers) >= number_to_select:
            break

    # Fill remaining positions if redundancy filtering
    # removed too many sentences
    if len(selected_numbers) < number_to_select:

        for _, candidate in ranked_sentences.iterrows():

            sentence_number = int(
                candidate["Sentence Number"]
            )

            if sentence_number not in selected_numbers:
                selected_numbers.append(sentence_number)

            if len(selected_numbers) >= number_to_select:
                break

    # Restore original document order
    selected_sentences = score_df[
        score_df["Sentence Number"].isin(selected_numbers)
    ].sort_values(
        by="Sentence Number"
    )

    return selected_sentences.reset_index(drop=True)


# -----------------------------------------
# 4. GENERATE FINAL SUMMARY
# -----------------------------------------

def generate_extractive_summary(selected_sentences):

    if selected_sentences.empty:
        return ""

    summary = " ".join(
        selected_sentences["Sentence"].astype(str).tolist()
    )

    return summary


# -----------------------------------------
# 5. SUMMARY STATISTICS
# -----------------------------------------

def calculate_summary_statistics(
    original_text,
    summary_text,
    original_sentences=None,
    summary_sentences=None
):

    original_words = len(original_text.split())

    summary_words = len(summary_text.split())

    if original_words > 0:

        text_reduction = (
            (original_words - summary_words)
            / original_words
        ) * 100

    else:
        text_reduction = 0.0

    return {
        "Original Words": original_words,
        "Summary Words": summary_words,
        "Original Sentences": original_sentences or 0,
        "Summary Sentences": summary_sentences or 0,
        "Compression Ratio": round(text_reduction, 2)
    }


# -----------------------------------------
# 6. ROUGE EVALUATION
# -----------------------------------------

def calculate_rouge_scores(
    reference_summary,
    generated_summary
):

    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    scores = scorer.score(
        reference_summary,
        generated_summary
    )

    results = {}

    for metric, values in scores.items():

        results[metric.upper()] = {
            "Precision": round(values.precision, 4),
            "Recall": round(values.recall, 4),
            "F1": round(values.fmeasure, 4)
        }

    return results
