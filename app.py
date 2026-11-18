
import io

import nltk
import pandas as pd
import streamlit as st

from nltk.tokenize import sent_tokenize
from pypdf import PdfReader

from preprocessing import preprocess_text

from summarizer import (
    calculate_sentence_scores,
    select_important_sentences,
    generate_extractive_summary,
    calculate_summary_statistics,
    calculate_rouge_scores
)


# =========================================
# 1. STREAMLIT CONFIGURATION
# =========================================

st.set_page_config(
    page_title="Automatic Text Summarization",
    page_icon="📄",
    layout="wide"
)


# =========================================
# 2. NLTK RESOURCE SETUP
# =========================================

@st.cache_resource
def setup_nltk():

    for resource in ["punkt", "punkt_tab"]:

        try:
            nltk.data.find(f"tokenizers/{resource}")

        except LookupError:
            nltk.download(resource, quiet=True)


setup_nltk()


# =========================================
# 3. APPLICATION TITLE
# =========================================

st.title("Automatic Text Summarization System")

st.write(
    "Generate concise summaries from long documents "
    "using TF-IDF-based extractive text summarization."
)

st.info(
    "This application uses TF-IDF to calculate sentence "
    "importance, position adjustment and cosine similarity "
    "to reduce redundant sentence selection."
)


# =========================================
# 4. PDF TEXT EXTRACTION
# =========================================

def extract_pdf_text(uploaded_file):

    pdf_reader = PdfReader(
        io.BytesIO(uploaded_file.getvalue())
    )

    extracted_pages = []

    for page in pdf_reader.pages:

        page_text = page.extract_text() or ""

        extracted_pages.append(page_text)

    return "\n".join(extracted_pages).strip()


# =========================================
# 5. DOCUMENT UPLOAD
# =========================================

st.header("Upload Your Document")

uploaded_file = st.file_uploader(
    "Upload a TXT or PDF file",
    type=["txt", "pdf"]
)

uploaded_text = ""

if uploaded_file is not None:

    try:

        if uploaded_file.name.lower().endswith(".txt"):

            uploaded_text = uploaded_file.getvalue().decode(
                "utf-8-sig"
            )

        elif uploaded_file.name.lower().endswith(".pdf"):

            uploaded_text = extract_pdf_text(
                uploaded_file
            )

        if uploaded_text.strip():

            st.success(
                f"Successfully loaded: {uploaded_file.name}"
            )

        else:

            st.warning(
                "No readable text was found. "
                "Scanned PDFs require OCR."
            )

    except Exception as error:

        st.error(
            f"Unable to read the uploaded document: {error}"
        )


# =========================================
# 6. DOCUMENT INPUT
# =========================================

st.header("Enter Your Document")

document_text = st.text_area(
    "Paste your document below:",
    value=uploaded_text,
    height=250,
    placeholder="Enter your article or document here..."
)


# =========================================
# 7. SUMMARY LENGTH
# =========================================

st.header("Summary Configuration")

summary_percentage = st.slider(
    "Choose percentage of sentences to include (%)",
    min_value=10,
    max_value=80,
    value=30,
    step=5
)

st.caption(
    "The percentage determines approximately how many "
    "sentences will be included in the final summary."
)


# =========================================
# 8. OPTIONAL ROUGE REFERENCE SUMMARY
# =========================================

with st.expander(
    "Optional: ROUGE Evaluation"
):

    st.write(
        "Enter a human-written reference summary "
        "to evaluate the generated summary."
    )

    reference_summary = st.text_area(
        "Reference Summary",
        height=120,
        placeholder="Enter your reference summary here..."
    )


# =========================================
# 9. GENERATE SUMMARY BUTTON
# =========================================

generate_button = st.button(
    "Generate Summary",
    type="primary",
    use_container_width=False
)


if generate_button:

    if not document_text.strip():

        st.warning(
            "Please enter a document or upload a file."
        )

        st.stop()


    # =====================================
    # 10. SENTENCE TOKENIZATION
    # =====================================

    try:

        sentences = sent_tokenize(
            document_text.strip()
        )

        sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    except LookupError:

        st.error(
            "NLTK tokenizer resources are missing. "
            "Run: python -m nltk.downloader punkt punkt_tab"
        )

        st.stop()


    if not sentences:

        st.warning(
            "No sentences were detected in the document."
        )

        st.stop()


    # =====================================
    # 11. TEXT PREPROCESSING
    # =====================================

    preprocessed_words = preprocess_text(
        document_text
    )

    preprocessed_text = " ".join(
        preprocessed_words
    )


    # =====================================
    # 12. WORD FREQUENCY AND TF-IDF
    # =====================================

    (
        score_df,
        word_frequencies,
        normalized_frequencies
    ) = calculate_sentence_scores(
        sentences
    )


    if not word_frequencies:

        st.warning(
            "No meaningful English words were found. "
            "Please try another document."
        )

        st.stop()


    # =====================================
    # 13. SELECT IMPORTANT SENTENCES
    # =====================================

    selected_sentences = select_important_sentences(
        score_df,
        summary_ratio=summary_percentage / 100
    )


    # =====================================
    # 14. GENERATE FINAL SUMMARY
    # =====================================

    final_summary = generate_extractive_summary(
        selected_sentences
    )


    # =====================================
    # 15. SUMMARY STATISTICS
    # =====================================

    statistics = calculate_summary_statistics(
        document_text,
        final_summary,
        original_sentences=len(sentences),
        summary_sentences=len(selected_sentences)
    )


    actual_percentage = (
        len(selected_sentences) / len(sentences)
    ) * 100


    # =====================================
    # DISPLAY RESULTS
    # =====================================

    st.divider()

    # -------------------------------------
    # RESULT 1: ORIGINAL DOCUMENT
    # -------------------------------------

    st.header("1. Original Document")

    st.write(document_text)


    # -------------------------------------
    # RESULT 2: PREPROCESSED TEXT
    # -------------------------------------

    st.header("2. Preprocessed Text")

    st.write(preprocessed_text)


    # -------------------------------------
    # RESULT 3: WORD FREQUENCY ANALYSIS
    # -------------------------------------

    st.header("3. Word Frequency Analysis")

    frequency_df = pd.DataFrame(
        [
            {
                "Word": word,
                "Frequency": frequency,
                "Normalized Frequency": round(
                    normalized_frequencies[word], 4
                )
            }

            for word, frequency in word_frequencies.items()
        ]
    )

    frequency_df = frequency_df.sort_values(
        by="Frequency",
        ascending=False
    )

    st.dataframe(
        frequency_df,
        use_container_width=True,
        hide_index=True
    )


    # -------------------------------------
    # RESULT 4: TF-IDF SENTENCE SCORES
    # -------------------------------------

    st.header("4. TF-IDF Sentence Importance Scores")

    st.write(
        "TF-IDF Score measures the importance of words "
        "within each sentence. Position Bonus gives a "
        "small preference to earlier sentences."
    )

    display_score_df = score_df.sort_values(
        by="Selection Score",
        ascending=False
    )

    st.dataframe(
        display_score_df,
        use_container_width=True,
        hide_index=True
    )


    # -------------------------------------
    # RESULT 5: SELECTED SENTENCES
    # -------------------------------------

    st.header("5. Selected Important Sentences")

    for _, row in selected_sentences.iterrows():

        sentence_number = int(
            row["Sentence Number"]
        )

        st.markdown(
            f"**{sentence_number}.** {row['Sentence']}"
        )


    st.success(
        f"Selected {len(selected_sentences)} "
        f"out of {len(sentences)} sentences. "
        f"Actual selection: {actual_percentage:.2f}%."
    )


    # -------------------------------------
    # RESULT 6: FINAL SUMMARY
    # -------------------------------------

    st.header("6. Final Generated Extractive Summary")

    st.success(final_summary)


    # -------------------------------------
    # RESULT 7: DOCUMENT COMPARISON
    # -------------------------------------

    st.header(
        "7. Original Document vs Generated Summary"
    )

    original_column, summary_column = st.columns(2)

    with original_column:

        st.subheader("Original Document")

        st.write(document_text)

    with summary_column:

        st.subheader("Generated Summary")

        st.write(final_summary)


    # -------------------------------------
    # RESULT 8: SUMMARY STATISTICS
    # -------------------------------------

    st.header("8. Summary Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Original Words",
        statistics["Original Words"]
    )

    col2.metric(
        "Summary Words",
        statistics["Summary Words"]
    )

    col3.metric(
        "Text Reduction",
        f"{statistics['Compression Ratio']}%"
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Original Sentences",
        statistics["Original Sentences"]
    )

    col5.metric(
        "Summary Sentences",
        statistics["Summary Sentences"]
    )

    col6.metric(
        "Actual Sentence Selection",
        f"{actual_percentage:.2f}%"
    )


    # -------------------------------------
    # RESULT 9: VISUAL ANALYSIS
    # -------------------------------------

    st.header("9. Visual Analysis")

    st.subheader("Sentence Score Comparison")

    sentence_chart = score_df.set_index(
        "Sentence Number"
    )[
        ["TF-IDF Score", "Selection Score"]
    ]

    st.bar_chart(sentence_chart)


    st.subheader("Top 15 Most Frequent Words")

    top_words = frequency_df.head(15).set_index(
        "Word"
    )[["Frequency"]]

    st.bar_chart(top_words)


    # -------------------------------------
    # RESULT 10: DOWNLOAD SUMMARY
    # -------------------------------------

    st.header("10. Download Results")

    st.download_button(
        label="Download Summary (.txt)",

        data=final_summary,

        file_name="generated_summary.txt",

        mime="text/plain"
    )


    # Download sentence scores

    st.download_button(
        label="Download Sentence Scores (.csv)",

        data=score_df.to_csv(
            index=False
        ).encode("utf-8"),

        file_name="sentence_scores.csv",

        mime="text/csv"
    )


    # Download complete report

    report = f"""
AUTOMATIC TEXT SUMMARIZATION REPORT
===================================

Original Words:
{statistics['Original Words']}

Summary Words:
{statistics['Summary Words']}

Text Reduction:
{statistics['Compression Ratio']}%

Original Sentences:
{statistics['Original Sentences']}

Summary Sentences:
{statistics['Summary Sentences']}


GENERATED SUMMARY
=================

{final_summary}
"""

    st.download_button(
        label="Download Complete Report (.txt)",

        data=report,

        file_name="summarization_report.txt",

        mime="text/plain"
    )


    # -------------------------------------
    # RESULT 11: ROUGE EVALUATION
    # -------------------------------------

    if reference_summary.strip():

        st.header("11. ROUGE Evaluation")

        try:

            rouge_results = calculate_rouge_scores(
                reference_summary,
                final_summary
            )

            rouge_df = pd.DataFrame(
                rouge_results
            ).T

            st.write(
                "ROUGE measures the overlap between the "
                "generated summary and the reference summary."
            )

            st.dataframe(
                rouge_df,
                use_container_width=True
            )

        except ImportError:

            st.error(
                "ROUGE package is missing. "
                "Install it using: pip install rouge-score"
            )

        except Exception as error:

            st.error(
                f"ROUGE evaluation failed: {error}"
            )


    # -------------------------------------
    # SUCCESS MESSAGE
    # -------------------------------------

    st.divider()

    st.success(
        "Your TF-IDF-based summary has been generated successfully!"
    )
