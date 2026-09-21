# Text Preprocessing and Normalization — Virtual Laboratory

## Knowledge Graph and Information Retrieval Systems

A Streamlit-based Virtual Laboratory experiment for performing and understanding **text preprocessing and normalization** techniques used in **Information Retrieval (IR), Natural Language Processing (NLP), and text-based knowledge systems**.

The application provides an interactive laboratory environment where students can apply preprocessing operations to a text corpus, observe every transformation step, record experimental trials, assess their understanding through a quiz, and generate a PDF laboratory report.

---

## Experiment

### Title

**Text Preprocessing and Normalization**

### Aim

To perform **tokenization, stop-word removal, stemming, lemmatization, and text normalization** on a text corpus and obtain clean standardized text suitable for indexing and analysis.

---

## Objectives

The experiment enables students to:

- Understand the purpose of text preprocessing in NLP and Information Retrieval.
- Perform text normalization such as lowercasing, punctuation removal, and whitespace normalization.
- Divide a text corpus into meaningful tokens using tokenization.
- Remove common stop words.
- Apply stemming and observe reduced word forms.
- Apply lemmatization and observe linguistically meaningful base forms.
- Compare intermediate preprocessing outputs.
- Analyze preprocessing statistics.
- Prepare a clean and standardized corpus for indexing and downstream analysis.

---

## Features

### 1. Theory

The **Theory** section provides the conceptual background required to understand the experiment.

It covers:

- Introduction to text preprocessing
- Text normalization
- Tokenization
- Stop-word removal
- Stemming
- Lemmatization
- Stemming vs. lemmatization
- Preprocessing pipeline
- Importance of preprocessing in Information Retrieval
- Key terminology
- Experimental procedure
- References

---

### 2. Interactive Simulation

The **Simulation** section provides an interactive text preprocessing sandbox.

Students can select from predefined corpora or enter their own text.

Available examples include:

- **Basic NLP**
- **Information Retrieval**
- **E-Commerce**
- **Custom Input**

The preprocessing pipeline is:

```text
Raw Text
   ↓
Normalization
   ↓
Tokenization
   ↓
Stop-Word Removal
   ↓
Stemming
   ↓
Lemmatization
   ↓
Final Standardized Corpus
````

Each intermediate stage is displayed separately so that students can observe how the original text changes throughout the preprocessing pipeline.

---

## Preprocessing Operations

### Text Normalization

The application performs basic text normalization operations including:

* Converting text to lowercase
* Removing punctuation and special symbols
* Normalizing whitespace
* Producing a consistent textual representation

Example:

```text
"The QUICK, brown foxes!"
```

becomes:

```text
"the quick brown foxes"
```

---

### Tokenization

The normalized text is divided into individual word or number tokens.

Example:

```text
"Natural language processing is useful"
```

becomes:

```text
Natural | language | processing | is | useful
```

Tokenization provides the basic textual units used by the subsequent preprocessing operations.

---

### Stop-Word Removal

Common English stop words can be removed before stemming and lemmatization.

Examples include:

```text
the
is
and
of
to
```

Students can enable or disable stop-word removal using the simulation checkbox.

---

### Stemming

The application uses the **Porter Stemmer** to reduce words to stem forms.

Stemming generally applies rule-based suffix removal and may produce forms that are not complete dictionary words.

---

### Lemmatization

The application uses the **WordNet Lemmatizer** to obtain dictionary-oriented base forms.

The experiment uses the default noun-based lemmatization because POS tagging is outside the scope of the experiment.

---

## Preprocessing Statistics

After processing the corpus, the application calculates:

* Original characters
* Normalized characters
* Total tokens
* Stop words removed
* Tokens after stop-word removal
* Unique tokens
* Unique stems
* Unique lemmas
* Final corpus words

A chart is also generated to compare the number of terms across preprocessing stages.

---

## Token-Level Comparison

The application provides a token-level comparison table showing the relationship between filtered tokens, their stemmed forms, and their lemmatized forms.

Example:

| Original / Filtered | Stem    | Lemma      |
| ------------------- | ------- | ---------- |
| connected           | connect | connected  |
| cars                | car     | car        |
| processing          | process | processing |

The actual values in the table depend on the selected input corpus and preprocessing results.

---

## Experimental Data Log Book

Students can record multiple experimental trials.

Each recorded trial stores:

* Trial number
* Number of tokens
* Tokens remaining after stop-word removal
* Number of unique terms
* Input preview
* Timestamp

The recorded trials are displayed in a table and can also be downloaded as a CSV file.

This allows students to compare the effect of preprocessing on different text inputs.

---

## Quiz

The application includes a **10-question conceptual assessment** covering:

* Purpose of text preprocessing
* Tokenization
* Stop words
* Stemming
* Lemmatization
* Normalization
* Information Retrieval
* Difference between stemming and lemmatization
* Index size
* Final preprocessing output

The quiz provides:

* Instant grading
* Correct/incorrect feedback
* Explanations for answers
* Final score

---

## PDF Laboratory Report

Students can generate an official PDF laboratory report containing:

* Student name
* Roll number / ID
* Experiment date
* Quiz score
* Aim
* Learning objectives
* Experimental corpus
* Preprocessing outputs
* Preprocessing statistics
* Recorded experimental trial
* Student observations
* Conclusion
* Signature section

The generated report is designed as a compact two-page laboratory report.

---

## Technology Stack

| Technology        | Purpose                                  |
| ----------------- | ---------------------------------------- |
| Python            | Application development                  |
| Streamlit         | Interactive Virtual Laboratory interface |
| NLTK              | NLP preprocessing                        |
| PorterStemmer     | Stemming                                 |
| WordNetLemmatizer | Lemmatization                            |
| Pandas            | Data processing and tables               |
| Plotly            | Statistics visualization                 |
| FPDF              | PDF laboratory report generation         |

---

## Project Structure

The project is intentionally implemented as a **single Python file** suitable for a Virtual Laboratory submission.

```text
text-preprocessing-virtual-lab/
│
├── text_preprocessing_lab.py
├── requirements.txt
└── README.md
```

The Python application is organized into the following components:

```text
Experiment Configuration
        ↓
Theory & Educational Content
        ↓
References
        ↓
NLP Dependency Setup
        ↓
Preprocessing Engine
        ↓
Quiz Content
        ↓
PDF Report Generator
        ↓
Section Renderers
        ↓
Session State
        ↓
Main Streamlit Application
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sabhansali/text_preprocessing_vlab_kgirs
cd text_preprocessing_vlab_kgirs
```

---

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

---

### 3. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

---

### 4. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Run the Streamlit application using:

```bash
streamlit run text_preprocessing_lab.py
```

The application will open in a web browser.

If Streamlit does not automatically open the browser, use the local URL displayed in the terminal.

---

## Using the Virtual Laboratory

### Step 1 — Theory

Open the **Theory** section and study:

* Text preprocessing
* Normalization
* Tokenization
* Stop-word removal
* Stemming
* Lemmatization
* Information Retrieval applications
* Key terminology
* Experimental procedure

---

### Step 2 — Simulation

1. Select an example corpus or choose **Custom Input**.
2. Enter or modify the text.
3. Select whether stop words should be removed.
4. Click **Run Preprocessing Pipeline**.
5. Observe each preprocessing stage.
6. Hover over the preprocessing stage headings to view a short explanation of each operation.
7. Analyze the preprocessing statistics.
8. Examine the token-level comparison.
9. Record experimental trials.

---

### Step 3 — Quiz

Open the **Quiz** section and complete the 10-question conceptual assessment.

After submission, the application displays:

* Correct and incorrect answers
* Explanations
* Final score

---

### Step 4 — Report Generation

Open **Report Generation** and enter:

* Student name
* Roll number / ID
* Experiment date
* Discussion and observations

The application generates a PDF containing the current experiment results, statistics, trial information, quiz score, observations, and conclusion.

---

### Step 5 — References

The **References** section provides the academic and technical references used for the theoretical and methodological content of the experiment.

---

## NLTK Resource Handling

The application attempts to automatically obtain the required NLTK resources:

* `stopwords`
* `wordnet`
* `omw-1.4`

If these resources cannot be downloaded, the application contains a built-in fallback English stop-word list so that the experiment can still operate.

---

## Example Input

The following example corpus can be used to demonstrate the complete preprocessing pipeline:

```text
The QUICK brown foxes are running quickly!
They jumped over the lazy dogs.
```

The application progressively transforms the input through:

```text
Original Text
      ↓
Normalization
      ↓
Tokenization
      ↓
Stop-Word Removal
      ↓
Stemming
      ↓
Lemmatization
      ↓
Final Standardized Corpus
```

The intermediate results and statistics can be inspected directly in the **Simulation** section.

---

## Educational Relevance

Text preprocessing is an important component of **Information Retrieval systems** because raw textual data contains variations that can affect indexing and retrieval.

A consistent representation of text can support downstream operations such as:

* Document indexing
* Search
* Information retrieval
* Text classification
* Text clustering
* Corpus analysis
* Knowledge extraction

The experiment demonstrates the preprocessing stage that can occur before text is supplied to larger **Information Retrieval and knowledge-processing pipelines**.

---

## Relation to Knowledge Graph and Information Retrieval Systems

This experiment forms a preprocessing foundation for systems that work with textual information.

Before documents, entities, relationships, or other textual information are indexed or incorporated into information retrieval and knowledge-processing workflows, the raw text may need to be cleaned and standardized.

The output of this experiment can therefore serve as a conceptual preprocessing stage before tasks such as:

```text
Raw Documents
      ↓
Text Preprocessing
      ↓
Normalized / Tokenized Text
      ↓
Indexing
      ↓
Information Retrieval
      ↓
Knowledge Extraction / Knowledge Graph Processing
```

The experiment focuses specifically on the **text preprocessing stage** of this broader workflow.

---

## References

1. A. Tyagi, V. K. Jain and V. Kumar, "Benchmark Text Preprocessing Techniques in Natural Language Processing," *2024 4th International Conference on Innovative Sustainable Computational Technologies (CISCT)*, Dehradun, India, 2024, pp. 1–6, doi: 10.1109/CISCT62494.2024.11134188.

2. J. L. Gastaldi, J. Terilla, L. Malagutti, B. DuSell, T. Vieira and R. Cotterell, "The Foundations of Tokenization: Statistical and Computational Concerns," in *Proc. International Conference on Learning Representations (ICLR)*, 2025.

3. J. Camacho-Collados and M. T. Pilehvar, "On the Role of Text Preprocessing in Neural Network Architectures: An Evaluation Study on Text Categorization and Sentiment Analysis," in *Proc. 2018 EMNLP Workshop BlackboxNLP*, Brussels, Belgium, 2018, pp. 40–46, doi: 10.18653/v1/W18-5406.

4. N. Babanejad, A. Agrawal, A. An and M. Papagelis, "A Comprehensive Analysis of Preprocessing for Word Representation Learning in Affective Tasks," in *Proc. 58th Annual Meeting of the Association for Computational Linguistics (ACL)*, 2020, pp. 5799–5810, doi: 10.18653/v1/2020.acl-main.514.

5. M. F. Porter, "Snowball: A Language for Stemming Algorithms," Oct. 2001.

6. Virtual Labs, Indian Institute of Technology Kharagpur, "Virtual Laboratory."
   [https://vlabs.iitkgp.ac.in/](https://vlabs.iitkgp.ac.in/)

---

## Author

Developed as an academic **Virtual Laboratory experiment** for the course:

**Knowledge Graph and Information Retrieval Systems**

The application is designed to provide an interactive, student-oriented environment for learning and experimenting with fundamental text preprocessing techniques used in **Information Retrieval and NLP**.
