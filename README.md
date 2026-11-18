# Automatic Text Summarization System

An NLP-based extractive text summarization application developed using Python, Scikit-learn, NLTK, and Streamlit. The system identifies important sentences from a document using TF-IDF scoring and generates a shorter summary while preserving key information.

## Project Overview

Reading lengthy documents can be time-consuming. This project aims to simplify that process by automatically identifying important sentences and generating concise summaries.

The application uses an extractive summarization approach. Instead of generating entirely new sentences, it selects important sentences from the original document based on their calculated importance scores.

Users can enter text directly or upload TXT and PDF documents through the Streamlit interface.

## Objectives

- Develop an automatic extractive text summarization system.
- Apply NLP preprocessing techniques to clean and prepare text.
- Calculate sentence importance using TF-IDF.
- Improve sentence ranking through position-based adjustment.
- Reduce redundant sentence selection using cosine similarity.
- Generate concise summaries based on a user-selected summary percentage.
- Provide summary statistics, visualizations, and downloadable results.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| NLTK | Text processing and sentence tokenization |
| Scikit-learn | TF-IDF vectorization and cosine similarity |
| Pandas | Data manipulation and tabular results |
| Matplotlib / Streamlit charts | Data visualization |
| PyPDF | Extracting text from PDF documents |
| ROUGE Score | Optional summary evaluation |

## System Methodology

The application follows these main processing stages:

1. **Document Input:** Users enter text or upload a TXT/PDF document.
2. **Text Extraction:** Text is extracted from the uploaded document.
3. **Text Preprocessing:** Text is converted to lowercase, tokenized, and cleaned by removing stopwords. Stemming is available in the preprocessing module.
4. **Word Frequency Analysis:** Word frequencies are calculated to understand the occurrence of terms.
5. **TF-IDF Calculation:** TF-IDF scores are calculated for individual sentences to estimate their importance.
6. **Position-Based Adjustment:** A small preference is given to earlier sentences.
7. **Redundancy Reduction:** Cosine similarity is used to reduce the selection of highly similar sentences.
8. **Sentence Selection:** Important sentences are selected according to the user-defined summary percentage.
9. **Summary Generation:** Selected sentences are arranged in their original order to generate the final summary.
10. **Evaluation and Visualization:** The application displays summary statistics, charts, and optional ROUGE evaluation.

## Key Features

- Text input through the application interface.
- TXT and PDF document upload.
- Text preprocessing and word frequency analysis.
- TF-IDF-based sentence importance scoring.
- Position-based sentence score adjustment.
- Cosine similarity-based redundancy reduction.
- Adjustable summary percentage.
- Original document and generated summary comparison.
- Summary statistics and visualizations.
- Optional ROUGE evaluation using a reference summary.
- Downloadable summary, sentence scores, and complete report.

## Project Structure

```text
Automatic-Text-Summarization/
│
├── app.py                 # Streamlit application
├── summarizer.py          # Summarization algorithms
├── preprocessing.py       # Text preprocessing functions
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
├── .gitignore             # Files excluded from Git
│
├── data/                  # Sample input documents
│
└── outputs/               # Generated output directory
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Automatic-Text-Summarization.git
```

### 2. Navigate to the project directory

```bash
cd Automatic-Text-Summarization
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install the dependencies

```bash
pip install -r requirements.txt
```

### 5. Download the required NLTK resources

```bash
python -m nltk.downloader punkt punkt_tab
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your web browser, usually at:

```text
http://localhost:8501
```

## Evaluation

The application provides optional ROUGE-based evaluation when a reference summary is supplied.

The supported metrics include:

- ROUGE-1
- ROUGE-2
- ROUGE-L

These metrics help compare the generated summary with a reference summary.

## Limitations

- The system uses extractive summarization and selects sentences from the original document.
- It does not generate new sentences or fully understand document meaning.
- Pronoun references and contextual relationships may not always be preserved.
- PDF extraction works best with text-based PDFs; scanned documents may require OCR.
- Summary quality depends on the input document and sentence selection settings.

## Future Scope

- Explore transformer-based summarization models.
- Improve context awareness and summary coherence.
- Support additional document formats.
- Add OCR support for scanned PDF documents.
- Evaluate performance on larger datasets using standard summarization metrics.
- Compare extractive and abstractive summarization approaches.

## Author

**Vikas Sharma**

MSc Data Science Student

Reva University, Bengaluru, India

## Acknowledgements

This project was developed as part of academic learning in Natural Language Processing and Data Science.

---

If you find this project useful, feel free to explore the repository and share your feedback.