"""
Virtual Laboratory Experiment
Text Preprocessing and Normalization
Streamlit single-file application

Sections:
1. Theory
2. Simulation
3. Quiz
4. Report Generation
"""

import os
import re
import time
import random
import html
import textwrap
from datetime import datetime
from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF

# =============================================================================
# 1. EXPERIMENT CONFIGURATION & EDUCATIONAL CONTENT
# =============================================================================

EXPERIMENT_CONFIG = {
    "title": "Text Preprocessing and Normalization",
    "aim": (
        "To perform tokenization, stop-word removal, stemming, lemmatization, "
        "and text normalization on a text corpus and obtain clean standardized "
        "text suitable for indexing and analysis."
    ),
    "objectives": [
        "Understand the purpose of text preprocessing in Natural Language Processing and Information Retrieval.",
        "Perform text normalization such as lowercasing, punctuation removal, whitespace normalization, and number handling.",
        "Divide a text corpus into meaningful tokens using tokenization.",
        "Remove common stop words to reduce unnecessary terms.",
        "Apply stemming and observe how words are reduced to their stems.",
        "Apply lemmatization and observe linguistically meaningful base forms.",
        "Compare the intermediate outputs and statistics produced by each preprocessing stage.",
        "Prepare a clean and standardized corpus suitable for indexing and downstream analysis.",
    ],
}

THEORY_CONTENT = {
    "background": """
### 1. Introduction

Text preprocessing is the process of converting raw, unstructured text into a
clean and standardized representation that can be efficiently used by Natural
Language Processing (NLP), Information Retrieval (IR), text mining, and
machine-learning systems.

Raw text may contain uppercase and lowercase variations, punctuation, numbers,
extra spaces, stop words, inflected word forms, special characters, and other
noise. Preprocessing reduces these inconsistencies before indexing or analysis.

### 2. Text Normalization

Normalization converts different surface forms into a more consistent format.
Typical operations include:

- Converting text to lowercase.
- Removing or handling punctuation and special characters.
- Normalizing repeated whitespace.
- Handling numbers and unwanted symbols.
- Removing unnecessary formatting noise.

For example:

`"The QUICK, brown foxes!"`

can be normalized to:

`"the quick brown foxes"`

### 3. Tokenization

Tokenization divides a text into smaller units called **tokens**. In a
word-tokenization task, the tokens are generally words or word-like units.

Example:

`"Natural language processing is useful."`

becomes:

`["Natural", "language", "processing", "is", "useful"]`

Tokenization provides the basic units on which subsequent preprocessing
operations are performed.

### 4. Stop-Word Removal

Stop words are frequent function words that may contribute relatively little
to many retrieval or text-analysis tasks. Examples include words such as
`the`, `is`, `and`, `of`, and `to`.

Removing stop words can reduce the number of terms that must be stored and
processed. However, whether a word should be removed depends on the application.

### 5. Stemming

Stemming reduces related words to a common stem, usually by applying
rule-based suffix removal. A stem is not necessarily a valid dictionary word.

For example, a stemmer may reduce:

`connect`, `connected`, `connecting`

to a related form such as:

`connect`

Stemming is computationally simple but can sometimes be aggressive.

### 6. Lemmatization

Lemmatization converts a word to its **lemma**, or dictionary base form, using
linguistic information.

Examples include:

- `running` → `run`
- `cars` → `car`
- `better` → `good` when the appropriate linguistic analysis is available

Compared with stemming, lemmatization generally aims to produce valid words and
uses more linguistic information.

### 7. Stemming vs Lemmatization

| Stemming | Lemmatization |
|---|---|
| Usually rule-based | Uses linguistic/dictionary information |
| Often faster | Usually more computationally involved |
| May produce non-words | Aims to produce valid base words |
| Can be aggressive | Usually more linguistically meaningful |

### 8. Preprocessing Pipeline

A typical pipeline used in this experiment is:

**Raw Text → Normalization → Tokenization → Stop-Word Removal → Stemming / Lemmatization → Clean Corpus**

The exact order can vary according to the application and preprocessing
requirements.

### 9. Importance in Information Retrieval

Search engines and indexing systems need a consistent representation of text.
Preprocessing can reduce vocabulary variation and make related forms easier to
match during indexing and retrieval.

The output of this experiment is therefore a clean, standardized corpus that
can be used as input to indexing, search, classification, clustering, or other
text-analysis tasks.
""",
    "procedure": [
        "Read the theoretical background and understand each preprocessing operation.",
        "Enter or select a text corpus in the Simulation section.",
        "Apply text normalization to standardize the raw input.",
        "Tokenize the normalized text into word tokens.",
        "Remove stop words from the token list.",
        "Apply stemming to observe stemmed word forms.",
        "Apply lemmatization to obtain base word forms.",
        "Compare the intermediate outputs and preprocessing statistics.",
        "Record at least three trials using different input text or preprocessing configurations.",
        "Complete the conceptual quiz.",
        "Enter student details and generate the final PDF laboratory report.",
    ],
    "key_terms": {
        "Corpus": "A collection of text documents or text data used for processing and analysis.",
        "Normalization": "Conversion of text into a consistent standardized representation.",
        "Token": "A basic textual unit produced by a tokenization process.",
        "Stop Word": "A frequent word that may be removed when it provides limited value for a particular task.",
        "Stem": "A reduced word form produced by a stemming algorithm; it may not be a valid dictionary word.",
        "Lemma": "A linguistically meaningful dictionary base form of a word.",
        "Vocabulary": "The set of unique terms occurring in a processed corpus.",
        "Preprocessing": "The sequence of operations used to clean and standardize raw text before analysis.",
    },
}

# =============================================================================
# REFERENCES
# =============================================================================

REFERENCES = [
    {
        "citation": (
            'A. Tyagi, V. K. Jain and V. Kumar, '
            '"Benchmark Text Preprocessing Techniques in Natural Language Processing," '
            '2024 4th International Conference on Innovative Sustainable Computational '
            'Technologies (CISCT), Dehradun, India, 2024, pp. 1-6, '
            'doi: 10.1109/CISCT62494.2024.11134188.'
        ),
        "link": "https://doi.org/10.1109/CISCT62494.2024.11134188",
        "label": "IEEE Xplore / DOI"
    },
    {
        "citation": (
            'J. L. Gastaldi, J. Terilla, L. Malagutti, B. DuSell, '
            'T. Vieira and R. Cotterell, '
            '"The Foundations of Tokenization: Statistical and Computational Concerns," '
            'in Proc. International Conference on Learning Representations (ICLR), 2025.'
        ),
        "link": "https://arxiv.org/abs/2407.11606",
        "label": "arXiv"
    },
    {
        "citation": (
            'J. Camacho-Collados and M. T. Pilehvar, '
            '"On the Role of Text Preprocessing in Neural Network Architectures: '
            'An Evaluation Study on Text Categorization and Sentiment Analysis," '
            'in Proc. 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting '
            'Neural Networks for NLP, Brussels, Belgium, 2018, pp. 40-46, '
            'doi: 10.18653/v1/W18-5406.'
        ),
        "link": "https://aclanthology.org/W18-5406/",
        "label": "ACL Anthology"
    },
    {
        "citation": (
            'N. Babanejad, A. Agrawal, A. An and M. Papagelis, '
            '"A Comprehensive Analysis of Preprocessing for Word Representation '
            'Learning in Affective Tasks," in Proc. 58th Annual Meeting of the '
            'Association for Computational Linguistics (ACL), 2020, pp. 5799-5810, '
            'doi: 10.18653/v1/2020.acl-main.514.'
        ),
        "link": "https://aclanthology.org/2020.acl-main.514/",
        "label": "ACL Anthology"
    },
    {
        "citation": (
            'M. F. Porter, "Snowball: A Language for Stemming Algorithms," '
            'Oct. 2001.'
        ),
        "link": "https://snowballstem.org/texts/introduction.html",
        "label": "Snowball"
    },
    {
        "citation": (
            'Virtual Labs, Indian Institute of Technology Kharagpur, '
            '"Virtual Laboratory."'
        ),
        "link": "https://vlabs.iitkgp.ac.in/",
        "label": "IIT Kharagpur Virtual Labs"
    },
]


def render_references():
    """Display the experiment references in IEEE style."""
    st.header("References")

    st.write(
        "The following references were consulted for the concepts and "
        "methodologies used in this experiment."
    )

    for i, reference in enumerate(REFERENCES, start=1):
        st.markdown(
            f"**[{i}]** {reference['citation']}  \n"
            f"[{reference['label']}]({reference['link']})"
        )

# =============================================================================
# 2. NLP DEPENDENCY SETUP
# =============================================================================

@st.cache_resource
def setup_nlp():
    """
    Prepare NLTK resources. If downloads are unavailable, the application still
    provides a fallback stop-word list and regex tokenization.
    """
    try:
        import nltk
        from nltk.stem import PorterStemmer, WordNetLemmatizer
        from nltk.corpus import stopwords

        resources = [
            ("corpora/stopwords", "stopwords"),
            ("corpora/wordnet", "wordnet"),
            ("corpora/omw-1.4", "omw-1.4"),
        ]

        for resource_path, resource_name in resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                try:
                    nltk.download(resource_name, quiet=True)
                except Exception:
                    pass

        try:
            stop_words = set(stopwords.words("english"))
        except Exception:
            stop_words = fallback_stop_words()

        try:
            lemmatizer = WordNetLemmatizer()
        except Exception:
            lemmatizer = None

        return PorterStemmer(), lemmatizer, stop_words

    except Exception:
        return None, None, fallback_stop_words()


def fallback_stop_words():
    """Small built-in fallback so the app remains usable without downloads."""
    return {
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
        "by", "for", "from", "had", "has", "have", "he", "her", "here",
        "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
        "into", "is", "it", "its", "itself", "me", "more", "most", "my",
        "myself", "no", "nor", "not", "of", "on", "or", "our", "ours",
        "ourselves", "she", "so", "some", "such", "than", "that", "the",
        "their", "theirs", "them", "themselves", "then", "there", "these",
        "they", "this", "those", "to", "too", "under", "until", "up", "very",
        "was", "we", "were", "what", "when", "where", "which", "while",
        "who", "whom", "why", "will", "with", "you", "your", "yours",
        "yourself", "yourselves"
    }


STEMMER, LEMMATIZER, STOP_WORDS = setup_nlp()

# =============================================================================
# 3. PREPROCESSING ENGINE
# =============================================================================

def normalize_text(text: str) -> str:
    """Lowercase, normalize whitespace, and remove punctuation/special symbols."""
    text = str(text or "")
    text = text.lower()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text: str):
    """Tokenize normalized English text into word/number tokens."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def remove_stop_words(tokens):
    """Remove English stop words."""
    return [token for token in tokens if token not in STOP_WORDS]


def stem_tokens(tokens):
    """Apply Porter stemming."""
    if STEMMER is None:
        return tokens
    return [STEMMER.stem(token) for token in tokens]


def lemmatize_tokens(tokens):
    """
    Apply WordNet lemmatization. Nouns are used as the default POS because
    POS tagging is outside the scope of this experiment.
    """
    if LEMMATIZER is None:
        return tokens
    output = []
    for token in tokens:
        try:
            output.append(LEMMATIZER.lemmatize(token))
        except Exception:
            output.append(token)
    return output


def preprocess_text(text: str, remove_stops=True):
    """Run the complete preprocessing pipeline and return all intermediate data."""
    normalized = normalize_text(text)
    tokens = tokenize_text(normalized)
    removed = [t for t in tokens if t in STOP_WORDS]
    filtered = remove_stop_words(tokens) if remove_stops else tokens.copy()
    stems = stem_tokens(filtered)
    lemmas = lemmatize_tokens(filtered)

    return {
        "original": str(text or ""),
        "normalized": normalized,
        "tokens": tokens,
        "removed_stop_words": removed,
        "filtered_tokens": filtered,
        "stems": stems,
        "lemmas": lemmas,
        "final_corpus": " ".join(lemmas),
    }


def preprocessing_statistics(result):
    """Calculate simple corpus statistics."""
    return {
        "Original characters": len(result["original"]),
        "Normalized characters": len(result["normalized"]),
        "Tokens": len(result["tokens"]),
        "Stop words removed": len(result["removed_stop_words"]),
        "Tokens after stop-word removal": len(result["filtered_tokens"]),
        "Unique tokens": len(set(result["filtered_tokens"])),
        "Unique stems": len(set(result["stems"])),
        "Unique lemmas": len(set(result["lemmas"])),
        "Final corpus words": len(result["lemmas"]),
    }


# =============================================================================
# 4. QUIZ CONTENT
# =============================================================================

QUESTION_BANK = [
    {
        "id": 1,
        "question": "What is the main purpose of text preprocessing?",
        "options": [
            "A) To increase random noise",
            "B) To clean and standardize text before analysis",
            "C) To convert text into images",
            "D) To remove all meaningful words"
        ],
        "answer_index": 1,
        "explanation": "Text preprocessing cleans and standardizes raw text before analysis."
    },
    {
        "id": 2,
        "question": "What does tokenization do?",
        "options": [
            "A) Divides text into smaller textual units called tokens",
            "B) Encrypts the document",
            "C) Translates the document",
            "D) Deletes the document"
        ],
        "answer_index": 0,
        "explanation": "Tokenization divides text into smaller units called tokens."
    },
    {
        "id": 3,
        "question": "Which of the following is a stop word?",
        "options": [
            "A) algorithm",
            "B) database",
            "C) the",
            "D) information"
        ],
        "answer_index": 2,
        "explanation": "'The' is a common English stop word."
    },
    {
        "id": 4,
        "question": "Which operation converts text into lowercase?",
        "options": [
            "A) Tokenization",
            "B) Normalization",
            "C) Stemming",
            "D) Lemmatization"
        ],
        "answer_index": 1,
        "explanation": "Lowercasing is a common normalization operation."
    },
    {
        "id": 5,
        "question": "What is the main purpose of stemming?",
        "options": [
            "A) Translate words",
            "B) Reduce words to their stems",
            "C) Add punctuation",
            "D) Remove documents"
        ],
        "answer_index": 1,
        "explanation": "Stemming reduces related words to a common stem."
    },
    {
        "id": 6,
        "question": "What is the goal of lemmatization?",
        "options": [
            "A) Produce a meaningful base form",
            "B) Add punctuation",
            "C) Remove nouns",
            "D) Increase vocabulary"
        ],
        "answer_index": 0,
        "explanation": "Lemmatization aims to produce linguistically meaningful base forms."
    },
    {
        "id": 7,
        "question": "Which operation can remove words such as 'the', 'is', and 'and'?",
        "options": [
            "A) Stemming",
            "B) Tokenization",
            "C) Stop-word removal",
            "D) Lemmatization"
        ],
        "answer_index": 2,
        "explanation": "Stop-word removal removes selected common words."
    },
    {
        "id": 8,
        "question": "Which is generally faster?",
        "options": [
            "A) Stemming",
            "B) Lemmatization",
            "C) Both always take the same time",
            "D) Neither"
        ],
        "answer_index": 0,
        "explanation": "Stemming is generally simpler and faster than lemmatization."
    },
    {
        "id": 9,
        "question": "Which method uses linguistic information?",
        "options": [
            "A) Stemming",
            "B) Lemmatization",
            "C) Tokenization",
            "D) Normalization"
        ],
        "answer_index": 1,
        "explanation": "Lemmatization uses linguistic and dictionary information."
    },
    {
        "id": 10,
        "question": "What is a corpus?",
        "options": [
            "A) A collection of text",
            "B) A database server",
            "C) A programming language",
            "D) An image file"
        ],
        "answer_index": 0,
        "explanation": "A corpus is a collection of text documents or text data."
    },

    {
        "id": 11,
        "question": "What is normalization useful for?",
        "options": [
            "A) Increasing text variation",
            "B) Creating a consistent text representation",
            "C) Adding random symbols",
            "D) Encrypting text"
        ],
        "answer_index": 1,
        "explanation": "Normalization reduces unnecessary variation in text."
    },
    {
        "id": 12,
        "question": "Which is an example of punctuation?",
        "options": [
            "A) hello",
            "B) 123",
            "C) !",
            "D) token"
        ],
        "answer_index": 2,
        "explanation": "An exclamation mark is punctuation."
    },
    {
        "id": 13,
        "question": "Which stage usually comes first?",
        "options": [
            "A) Lemmatization",
            "B) Normalization",
            "C) Stemming",
            "D) Stop-word removal"
        ],
        "answer_index": 1,
        "explanation": "Normalization is performed early in the preprocessing pipeline."
    },
    {
        "id": 14,
        "question": "What does whitespace normalization do?",
        "options": [
            "A) Removes repeated unnecessary spaces",
            "B) Adds random spaces",
            "C) Converts text into images",
            "D) Deletes all words"
        ],
        "answer_index": 0,
        "explanation": "Whitespace normalization makes spacing consistent."
    },
    {
        "id": 15,
        "question": "Which of these is a token?",
        "options": [
            "A) A word or word-like unit",
            "B) Only a paragraph",
            "C) Only a document",
            "D) Only punctuation"
        ],
        "answer_index": 0,
        "explanation": "A token is a basic textual unit produced by tokenization."
    },
    {
        "id": 16,
        "question": "Which operation may produce a non-dictionary word?",
        "options": [
            "A) Lemmatization",
            "B) Stemming",
            "C) Tokenization",
            "D) Normalization"
        ],
        "answer_index": 1,
        "explanation": "Stemming can produce stems that are not valid dictionary words."
    },
    {
        "id": 17,
        "question": "Which operation aims to produce dictionary base forms?",
        "options": [
            "A) Stemming",
            "B) Lemmatization",
            "C) Tokenization",
            "D) Stop-word removal"
        ],
        "answer_index": 1,
        "explanation": "Lemmatization aims to produce meaningful dictionary base forms."
    },
    {
        "id": 18,
        "question": "Why is preprocessing useful in search engines?",
        "options": [
            "A) It increases random noise",
            "B) It creates consistent representations",
            "C) It removes search results",
            "D) It disables indexing"
        ],
        "answer_index": 1,
        "explanation": "Consistent text representations can improve indexing and retrieval."
    },
    {
        "id": 19,
        "question": "What is vocabulary in NLP?",
        "options": [
            "A) Set of unique terms",
            "B) Number of documents only",
            "C) Number of paragraphs",
            "D) Number of characters"
        ],
        "answer_index": 0,
        "explanation": "Vocabulary is the set of unique terms in the processed corpus."
    },
    {
        "id": 20,
        "question": "Which stage divides text into individual words?",
        "options": [
            "A) Tokenization",
            "B) Stemming",
            "C) Lemmatization",
            "D) Stop-word removal"
        ],
        "answer_index": 0,
        "explanation": "Tokenization divides text into tokens."
    },

    {
        "id": 21,
        "question": "Which preprocessing stage removes unwanted punctuation?",
        "options": [
            "A) Normalization",
            "B) Stemming",
            "C) Lemmatization",
            "D) Tokenization"
        ],
        "answer_index": 0,
        "explanation": "Punctuation removal is commonly part of normalization."
    },
    {
        "id": 22,
        "question": "What does lowercasing achieve?",
        "options": [
            "A) Reduces case-based variation",
            "B) Increases vocabulary",
            "C) Adds punctuation",
            "D) Removes all nouns"
        ],
        "answer_index": 0,
        "explanation": "Lowercasing treats uppercase and lowercase forms consistently."
    },
    {
        "id": 23,
        "question": "Which word is most likely to be removed as a stop word?",
        "options": [
            "A) computer",
            "B) information",
            "C) the",
            "D) retrieval"
        ],
        "answer_index": 2,
        "explanation": "'The' is a common stop word."
    },
    {
        "id": 24,
        "question": "What does stemming commonly use?",
        "options": [
            "A) Rule-based suffix removal",
            "B) Image processing",
            "C) Audio processing",
            "D) Encryption"
        ],
        "answer_index": 0,
        "explanation": "Stemming commonly applies rules to remove suffixes."
    },
    {
        "id": 25,
        "question": "Which produces linguistically meaningful base forms?",
        "options": [
            "A) Lemmatization",
            "B) Tokenization",
            "C) Stop-word removal",
            "D) Normalization"
        ],
        "answer_index": 0,
        "explanation": "Lemmatization aims for linguistically meaningful base forms."
    },
    {
        "id": 26,
        "question": "Which is NOT normally a text preprocessing operation?",
        "options": [
            "A) Lowercasing",
            "B) Tokenization",
            "C) Stop-word removal",
            "D) CPU overclocking"
        ],
        "answer_index": 3,
        "explanation": "CPU overclocking is unrelated to text preprocessing."
    },
    {
        "id": 27,
        "question": "What is the output of tokenization?",
        "options": [
            "A) Tokens",
            "B) Images",
            "C) Audio signals",
            "D) Hardware settings"
        ],
        "answer_index": 0,
        "explanation": "Tokenization produces tokens."
    },
    {
        "id": 28,
        "question": "What can happen to vocabulary after stop-word removal?",
        "options": [
            "A) It can decrease",
            "B) It always doubles",
            "C) It becomes infinite",
            "D) It becomes an image"
        ],
        "answer_index": 0,
        "explanation": "Removing common words can reduce vocabulary size."
    },
    {
        "id": 29,
        "question": "Which operation is generally more linguistically informed?",
        "options": [
            "A) Stemming",
            "B) Lemmatization",
            "C) Tokenization",
            "D) Lowercasing"
        ],
        "answer_index": 1,
        "explanation": "Lemmatization uses linguistic information."
    },
    {
        "id": 30,
        "question": "What is a major purpose of an information retrieval system?",
        "options": [
            "A) Find relevant information",
            "B) Compress images",
            "C) Edit videos",
            "D) Control hardware"
        ],
        "answer_index": 0,
        "explanation": "Information retrieval systems find relevant information from collections."
    },

    {
        "id": 31,
        "question": "Which sequence represents the preprocessing pipeline?",
        "options": [
            "A) Raw Text ΓåÆ Normalization ΓåÆ Tokenization",
            "B) Tokenization ΓåÆ Raw Text ΓåÆ Normalization",
            "C) Lemmatization ΓåÆ Raw Text ΓåÆ Tokenization",
            "D) Stemming ΓåÆ Raw Text ΓåÆ Stop Words"
        ],
        "answer_index": 0,
        "explanation": "Normalization and tokenization are early stages of the pipeline."
    },
    {
        "id": 32,
        "question": "Why are stop words sometimes retained?",
        "options": [
            "A) Their importance depends on the application",
            "B) They are always meaningful",
            "C) They are never common",
            "D) They are images"
        ],
        "answer_index": 0,
        "explanation": "Stop-word removal depends on the requirements of the application."
    },
    {
        "id": 33,
        "question": "What does the Porter Stemmer perform?",
        "options": [
            "A) Stemming",
            "B) Translation",
            "C) Token counting only",
            "D) Image classification"
        ],
        "answer_index": 0,
        "explanation": "Porter Stemmer is a stemming algorithm."
    },
    {
        "id": 34,
        "question": "Which Python library is commonly used for NLP resources in this experiment?",
        "options": [
            "A) NLTK",
            "B) NumPy only",
            "C) Matplotlib only",
            "D) Flask"
        ],
        "answer_index": 0,
        "explanation": "The application uses NLTK for stemming, lemmatization, and stop words."
    },
    {
        "id": 35,
        "question": "What does WordNet provide in this experiment?",
        "options": [
            "A) Linguistic information for lemmatization",
            "B) Image files",
            "C) Audio files",
            "D) Network connections"
        ],
        "answer_index": 0,
        "explanation": "WordNet is used by the lemmatizer for linguistic information."
    },
    {
        "id": 36,
        "question": "Which metric counts all tokens before stop-word removal?",
        "options": [
            "A) Tokens",
            "B) Unique stems",
            "C) Unique lemmas",
            "D) Final corpus words"
        ],
        "answer_index": 0,
        "explanation": "The Tokens statistic counts tokens generated during tokenization."
    },
    {
        "id": 37,
        "question": "What does 'Unique tokens' measure?",
        "options": [
            "A) Number of distinct filtered tokens",
            "B) Total characters only",
            "C) Number of documents",
            "D) Number of sentences only"
        ],
        "answer_index": 0,
        "explanation": "Unique tokens represent distinct terms after filtering."
    },
    {
        "id": 38,
        "question": "What is the final standardized corpus in this application?",
        "options": [
            "A) Space-separated lemmatized terms",
            "B) Raw input only",
            "C) Images",
            "D) Audio"
        ],
        "answer_index": 0,
        "explanation": "The final corpus is created by joining the lemmatized terms."
    },
    {
        "id": 39,
        "question": "What can preprocessing help reduce?",
        "options": [
            "A) Unnecessary textual variation",
            "B) Computer memory to zero",
            "C) Internet speed",
            "D) Screen size"
        ],
        "answer_index": 0,
        "explanation": "Preprocessing reduces unnecessary variation in textual data."
    },
    {
        "id": 40,
        "question": "Which task can use a clean standardized corpus?",
        "options": [
            "A) Classification",
            "B) Keyboard repair",
            "C) Hardware testing",
            "D) Screen calibration"
        ],
        "answer_index": 0,
        "explanation": "Clean text can be used for classification and other NLP tasks."
    },

    {
        "id": 41,
        "question": "Which task can use text preprocessing?",
        "options": [
            "A) Search",
            "B) Retrieval",
            "C) Text classification",
            "D) All of the above"
        ],
        "answer_index": 3,
        "explanation": "Preprocessing is useful in search, retrieval, classification, and many other tasks."
    },
    {
        "id": 42,
        "question": "What happens when punctuation is removed?",
        "options": [
            "A) Text becomes more standardized",
            "B) Text becomes an image",
            "C) All words disappear",
            "D) Documents are deleted"
        ],
        "answer_index": 0,
        "explanation": "Removing punctuation can create a cleaner standardized representation."
    },
    {
        "id": 43,
        "question": "Which stage produces stems?",
        "options": [
            "A) Stemming",
            "B) Tokenization",
            "C) Normalization",
            "D) Stop-word removal"
        ],
        "answer_index": 0,
        "explanation": "Stemming produces reduced word stems."
    },
    {
        "id": 44,
        "question": "Which stage produces lemmas?",
        "options": [
            "A) Lemmatization",
            "B) Stemming",
            "C) Tokenization",
            "D) Normalization"
        ],
        "answer_index": 0,
        "explanation": "Lemmatization produces lemma or dictionary base forms."
    },
    {
        "id": 45,
        "question": "What is text mining?",
        "options": [
            "A) Extracting useful information from text",
            "B) Mining physical materials",
            "C) Compressing images",
            "D) Creating hardware"
        ],
        "answer_index": 0,
        "explanation": "Text mining extracts useful information and patterns from textual data."
    },
    {
        "id": 46,
        "question": "Why can preprocessing improve indexing?",
        "options": [
            "A) It creates more consistent terms",
            "B) It removes the index",
            "C) It deletes all documents",
            "D) It stops searching"
        ],
        "answer_index": 0,
        "explanation": "Consistent terms can make indexing more efficient and useful."
    },
    {
        "id": 47,
        "question": "What is an inflected word form?",
        "options": [
            "A) A grammatical variation of a word",
            "B) An image",
            "C) A database",
            "D) A programming language"
        ],
        "answer_index": 0,
        "explanation": "Inflected forms are grammatical variations such as plural or tense forms."
    },
    {
        "id": 48,
        "question": "Which operation can help handle different cases such as 'Apple' and 'apple'?",
        "options": [
            "A) Lowercasing",
            "B) Stemming",
            "C) Lemmatization",
            "D) Token counting"
        ],
        "answer_index": 0,
        "explanation": "Lowercasing makes different letter cases consistent."
    },
    {
        "id": 49,
        "question": "Which application can benefit from text preprocessing?",
        "options": [
            "A) E-commerce search",
            "B) Information retrieval",
            "C) NLP systems",
            "D) All of the above"
        ],
        "answer_index": 3,
        "explanation": "All these applications can use standardized text."
    },
    {
        "id": 50,
        "question": "What is the overall goal of this experiment?",
        "options": [
            "A) Convert raw text into a clean standardized representation",
            "B) Convert text into images",
            "C) Increase unnecessary words",
            "D) Remove all text"
        ],
        "answer_index": 0,
        "explanation": "The experiment demonstrates how raw text can be cleaned and standardized."
    },
]

# =============================================================================
# 5. LAB REPORT PDF EXPORTER
# =============================================================================

class LabReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(
            0, 10,
            f"Page {self.page_no()}/{{nb}} | Virtual Laboratory Report",
            align="C"
        )


def pdf_safe(value):
    """Convert Unicode text into FPDF-compatible text."""
    return str(value).encode("latin-1", "replace").decode("latin-1")


def add_wrapped_pdf_text(pdf, text, size=9, bold=False, width=0, line_height=5):
    pdf.set_font("Helvetica", "B" if bold else "", size)
    pdf.multi_cell(width, line_height, pdf_safe(text))


def generate_pdf_report(
    student_name: str,
    student_id: str,
    date_str: str,
    trials_df: pd.DataFrame,
    quiz_score: int,
    quiz_total: int,
    student_notes: str,
    current_result: dict,
) -> bytes:
    """Generate a compact 2-page laboratory PDF report.

    The report intentionally uses short previews of long token lists so that
    one experiment entry does not become a very long multi-page document.
    """

    pdf = LabReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(12, 12, 12)

    # ---------- PAGE 1: Experiment + preprocessing result ----------
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 9, pdf_safe(EXPERIMENT_CONFIG["title"]))
    pdf.ln(5)

    # Student information in a compact 2-row table
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.set_font("Helvetica", "B", 8)

    info_rows = [
        (("Student Name", student_name or "N/A"), ("Roll / ID", student_id or "N/A")),
        (("Date", date_str), ("Quiz Score", f"{quiz_score} / {quiz_total}")),
    ]

    for left, right in info_rows:
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(35, 6, pdf_safe(left[0]), 1, 0, "L", True)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(60, 6, pdf_safe(left[1]), 1, 0, "L")
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(35, 6, pdf_safe(right[0]), 1, 0, "L", True)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(60, 6, pdf_safe(right[1]), 1, 1, "L")

    pdf.ln(3)

    def section_heading(title):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 58, 138)
        pdf.cell(0, 6, title)
        pdf.ln(6)
        pdf.set_text_color(51, 65, 85)

    section_heading("1. Aim")
    add_wrapped_pdf_text(pdf, EXPERIMENT_CONFIG["aim"], 8.5, False, 0, 4.5)
    pdf.ln(2)

    section_heading("2. Learning Objectives")
    for obj in EXPERIMENT_CONFIG["objectives"]:
        pdf.set_font("Helvetica", "", 8.5)
        pdf.cell(4, 4.5, "-", 0)
        pdf.multi_cell(0, 4.5, pdf_safe(obj))
    pdf.ln(2)

    section_heading("3. Experimental Corpus")

    # Keep the report readable: long token/stem/lemma lists are previews.
    stages = [
        ("Original Text", current_result.get("original", "")),
        ("Normalized Text", current_result.get("normalized", "")),
        ("Tokens", ", ".join(current_result.get("tokens", []))),
        ("Stop Words Removed", ", ".join(current_result.get("removed_stop_words", []))),
        ("Filtered Tokens", ", ".join(current_result.get("filtered_tokens", []))),
        ("Stems", ", ".join(current_result.get("stems", []))),
        ("Lemmas", ", ".join(current_result.get("lemmas", []))),
        ("Final Clean Corpus", current_result.get("final_corpus", "")),
    ]

    for label, value in stages:
        value = pdf_safe(value)
        # 500 chars is enough to demonstrate the transformation without
        # creating pages of repeated token lists.
        if len(value) > 500:
            value = value[:500].rstrip() + " ..."
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.cell(38, 4.5, pdf_safe(label), 0)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.multi_cell(0, 4.5, value)

    # ---------- PAGE 2: Statistics + trial + conclusion ----------
    pdf.add_page()

    section_heading("4. Preprocessing Statistics")
    stats = preprocessing_statistics(current_result)

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(241, 245, 249)
    for key, value in stats.items():
        pdf.cell(70, 5.5, pdf_safe(key), 1, 0, "L", True)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(45, 5.5, pdf_safe(value), 1, 1, "L")
        pdf.set_font("Helvetica", "B", 8)

    pdf.ln(4)
    section_heading("5. Recorded Experimental Trial")

    if trials_df.empty:
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.cell(0, 5, "No trial was recorded during this session.")
        pdf.ln(6)
    else:
        # For a lab report, one compact row is more useful than a huge table.
        row = trials_df.iloc[-1]
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(241, 245, 249)
        pdf.cell(35, 5.5, "Trial", 1, 0, "L", True)
        pdf.cell(35, 5.5, "Tokens", 1, 0, "L", True)
        pdf.cell(45, 5.5, "After Stop Words", 1, 0, "L", True)
        pdf.cell(40, 5.5, "Unique Terms", 1, 0, "L", True)
        pdf.cell(25, 5.5, "Date/Time", 1, 1, "L", True)

        pdf.set_font("Helvetica", "", 8)
        pdf.cell(35, 5.5, pdf_safe(row.get("Trial #", "-")), 1)
        pdf.cell(35, 5.5, pdf_safe(row.get("Tokens", "-")), 1)
        pdf.cell(45, 5.5, pdf_safe(row.get("After Stop Words", "-")), 1)
        pdf.cell(40, 5.5, pdf_safe(row.get("Unique Terms", "-")), 1)
        timestamp = str(row.get("Timestamp", "-"))[:16]
        pdf.cell(25, 5.5, pdf_safe(timestamp), 1, 1)

        pdf.ln(2)
        preview = str(row.get("Input Preview", ""))
        if len(preview) > 300:
            preview = preview[:300].rstrip() + " ..."
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(30, 5, "Input Preview:", 0)
        pdf.set_font("Helvetica", "", 8)
        pdf.multi_cell(0, 5, pdf_safe(preview))

    pdf.ln(4)
    section_heading("6. Observations and Analysis")
    notes = student_notes.strip() if student_notes.strip() else (
        "The preprocessing pipeline standardized the input text, generated "
        "tokens, removed stop words, and produced stemmed and lemmatized forms. "
        "The resulting corpus is suitable for further indexing and text analysis."
    )
    add_wrapped_pdf_text(pdf, notes, 8.5, False, 0, 4.5)

    pdf.ln(4)
    section_heading("7. Conclusion")
    conclusion = (
        "Text preprocessing and normalization were successfully performed using "
        "normalization, tokenization, stop-word removal, stemming, and "
        "lemmatization. The experiment demonstrates how raw text can be converted "
        "into a cleaner and more consistent representation for information "
        "retrieval and text analysis."
    )
    add_wrapped_pdf_text(pdf, conclusion, 8.5, False, 0, 4.5)

    pdf.ln(7)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(135, pdf.get_y() + 8, 195, pdf.get_y() + 8)
    pdf.set_xy(135, pdf.get_y() + 10)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(60, 4, "Instructor / Student Signature", align="C")

    # Export as actual PDF bytes for both FPDF 1.x and FPDF2.
    pdf_output = pdf.output(dest="S")
    if isinstance(pdf_output, str):
        return pdf_output.encode("latin-1")
    if isinstance(pdf_output, (bytes, bytearray)):
        return bytes(pdf_output)
    return str(pdf_output).encode("latin-1")



# =============================================================================
# 5A. INTERACTIVE PREPROCESSING SIMULATOR
# =============================================================================

def _sim_format(value):
    """Format simulator values for readable display."""
    if isinstance(value, list):
        return "[" + ", ".join(f'"{item}"' for item in value) + "]"
    return str(value or "(empty)")


def build_simulation_stages(text: str, remove_stops=True):
    """
    Build educational preprocessing stages separately so lowercasing and
    punctuation removal can be demonstrated as distinct operations.
    """
    original = str(text or "")

    lowercase = original.lower()
    lowercase = re.sub(r"[\r\n\t]+", " ", lowercase)
    lowercase = re.sub(r"\s+", " ", lowercase).strip()

    # Keep punctuation visible during tokenization so students can see that
    # punctuation removal is a separate step.
    tokenized = re.findall(r"\b[\w']+\b|[^\w\s]", lowercase)

    # Stop-word removal applies only to word tokens; punctuation remains here.
    if remove_stops:
        after_stopwords = [
            token for token in tokenized
            if not (re.fullmatch(r"[a-z0-9']+", token) and token in STOP_WORDS)
        ]
    else:
        after_stopwords = tokenized.copy()

    punctuation_removed = [
        token for token in after_stopwords
        if re.fullmatch(r"[a-z0-9]+", token)
    ]

    stems = stem_tokens(punctuation_removed)
    lemmas = lemmatize_tokens(punctuation_removed)
    final_text = " ".join(lemmas)

    return [
        {
            "short": "Original",
            "title": "Original Sentence",
            "before": original,
            "after": original,
            "action": "Reading the raw sentence...",
            "explanation": (
                "This is the unprocessed input exactly as entered by the user. "
                "It may contain uppercase letters, punctuation, stop words, and "
                "different grammatical word forms."
            ),
            "tooltip": "Raw input before any preprocessing is applied.",
        },
        {
            "short": "Lowercase",
            "title": "Lowercasing",
            "before": original,
            "after": lowercase,
            "action": "Converting all letters to lowercase...",
            "explanation": (
                "Lowercasing converts uppercase and lowercase variants into a "
                "consistent form. For example, 'Students' and 'students' become "
                "the same textual form."
            ),
            "tooltip": "Converts uppercase characters to lowercase.",
        },
        {
            "short": "Tokenize",
            "title": "Tokenization",
            "before": lowercase,
            "after": tokenized,
            "action": "Splitting the sentence into tokens...",
            "explanation": (
                "Tokenization divides the sentence into smaller textual units. "
                "Here, words and punctuation are temporarily represented as "
                "separate tokens so later operations can be demonstrated."
            ),
            "tooltip": "Splits the sentence into individual textual units called tokens.",
        },
        {
            "short": "Stop Words",
            "title": "Stop-word Removal",
            "before": tokenized,
            "after": after_stopwords,
            "action": (
                "Removing common English stop words..."
                if remove_stops
                else "Stop-word removal is disabled..."
            ),
            "explanation": (
                "Common function words such as 'the', 'are', and 'in' can be "
                "removed when they contribute little to the retrieval task."
                if remove_stops
                else
                "Stop-word removal is disabled, so the word tokens are preserved."
            ),
            "tooltip": "Removes frequent function words such as the, is, are, in, and of.",
        },
        {
            "short": "Punctuation",
            "title": "Punctuation Removal",
            "before": after_stopwords,
            "after": punctuation_removed,
            "action": "Removing punctuation and special-symbol tokens...",
            "explanation": (
                "Punctuation marks and special symbols are removed so that the "
                "remaining data contains clean word and number tokens."
            ),
            "tooltip": "Removes punctuation and special-symbol tokens.",
        },
        {
            "short": "Stem/Lemma",
            "title": "Stemming & Lemmatization",
            "before": punctuation_removed,
            "after": {
                "Stems": stems,
                "Lemmas": lemmas,
            },
            "action": "Reducing words to simpler base forms...",
            "explanation": (
                "Stemming applies rule-based reduction and may produce non-words. "
                "Lemmatization aims to produce linguistically meaningful base "
                "forms. Both results are shown for comparison."
            ),
            "tooltip": "Compares Porter stems with WordNet lemma forms.",
        },
        {
            "short": "Final",
            "title": "Final Preprocessed Text",
            "before": lemmas,
            "after": final_text,
            "action": "Building the standardized corpus...",
            "explanation": (
                "The processed lemma tokens are joined to create the final clean "
                "corpus that can be used for indexing and further text analysis."
            ),
            "tooltip": "Joins the processed tokens into the final standardized corpus.",
        },
    ]


def render_preprocessing_simulator(text: str, remove_stops=True):
    """Render the interactive preprocessing animation box."""
    stages = build_simulation_stages(text, remove_stops)
    total = len(stages)

    if "sim_step" not in st.session_state:
        st.session_state["sim_step"] = 0
    if "sim_playing" not in st.session_state:
        st.session_state["sim_playing"] = False
    if "sim_speed" not in st.session_state:
        st.session_state["sim_speed"] = 1.5
    if "sim_signature" not in st.session_state:
        st.session_state["sim_signature"] = None

    signature = (text, bool(remove_stops))
    if st.session_state["sim_signature"] != signature:
        st.session_state["sim_signature"] = signature
        st.session_state["sim_step"] = 0
        st.session_state["sim_playing"] = False

    st.session_state["sim_step"] = min(
        max(int(st.session_state["sim_step"]), 0),
        total - 1
    )

    st.markdown("### Interactive Preprocessing Simulation")

    # Pipeline CSS. Dedented and stripped so Markdown never mistakes the
    # leading whitespace of this triple-quoted string for a code block.
    pipeline_css = textwrap.dedent("""
        <style>
        .sim-pipeline {
            display: flex;
            align-items: stretch;
            gap: 7px;
            width: 100%;
            margin: 0.4rem 0 1rem 0;
            overflow-x: auto;
            padding-bottom: 4px;
        }
        .sim-node {
            flex: 1 0 105px;
            min-width: 105px;
            border: 1px solid rgba(128,128,128,.35);
            border-radius: 10px;
            padding: 10px 7px;
            text-align: center;
            font-size: 0.82rem;
            font-weight: 650;
            background: rgba(128,128,128,.08);
            transition: all .2s ease;
        }
        .sim-node.done {
            border-color: rgba(34,197,94,.75);
            background: rgba(34,197,94,.12);
        }
        .sim-node.active {
            border: 2px solid #2563eb;
            background: rgba(37,99,235,.15);
            box-shadow: 0 0 0 3px rgba(37,99,235,.08);
            transform: translateY(-2px);
        }
        .sim-node.future {
            opacity: .70;
        }
        .sim-arrow {
            align-self: center;
            font-size: 1.1rem;
            opacity: .55;
        }
        .sim-card {
            border: 1px solid rgba(128,128,128,.35);
            border-radius: 16px;
            padding: 1.2rem 1.3rem;
            background: rgba(128,128,128,.035);
            margin-bottom: 0.8rem;
        }
        .sim-step-label {
            font-size: .85rem;
            letter-spacing: .08em;
            font-weight: 750;
            opacity: .72;
        }
        .sim-title {
            font-size: 1.45rem;
            font-weight: 750;
            margin: .15rem 0 .8rem 0;
        }
        .sim-box-label {
            font-size: .78rem;
            font-weight: 750;
            opacity: .68;
            margin-bottom: .25rem;
        }
        .sim-value {
            border-radius: 10px;
            padding: .75rem .9rem;
            background: rgba(128,128,128,.09);
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
        }
        .sim-action {
            text-align: center;
            padding: .7rem;
            font-weight: 700;
            color: #2563eb;
        }
        .sim-explain {
            border-left: 4px solid #2563eb;
            padding: .65rem .9rem;
            margin-top: .8rem;
            background: rgba(37,99,235,.07);
            border-radius: 0 8px 8px 0;
        }
        .sim-next {
            margin-top: .8rem;
            font-weight: 700;
        }
        </style>
    """).strip()
    st.markdown(pipeline_css, unsafe_allow_html=True)

    current_index = st.session_state["sim_step"]
    current = stages[current_index]

    # title= creates the browser tooltip on hover.
    pipeline_html = '<div class="sim-pipeline">'
    for i, stage in enumerate(stages):
        if i < current_index:
            state = "done"
        elif i == current_index:
            state = "active"
        else:
            state = "future"

        pipeline_html += (
            f'<div class="sim-node {state}" title="{stage["tooltip"]}">'
            f'{i + 1}. {stage["short"]}</div>'
        )
        if i < total - 1:
            pipeline_html += '<div class="sim-arrow">→</div>'
    pipeline_html += "</div>"
    st.markdown(pipeline_html, unsafe_allow_html=True)

    before_value = _sim_format(current["before"])
    after_value = current["after"]
    if isinstance(after_value, dict):
        after_value = (
            "STEMS:\n" + _sim_format(after_value["Stems"]) +
            "\n\nLEMMAS:\n" + _sim_format(after_value["Lemmas"])
        )
    else:
        after_value = _sim_format(after_value)

    next_text = (
        f'Next Step → {stages[current_index + 1]["title"]}'
        if current_index < total - 1
        else "Pipeline Complete ✓"
    )

    # Built as a single concatenated string (no blank lines, no leading
    # indentation on any line) so Markdown never mistakes this HTML block
    # for a preformatted code block.
    card_html = (
        '<div class="sim-card">'
        f'<div class="sim-step-label">STEP {current_index + 1} OF {total}</div>'
        f'<div class="sim-title">{html.escape(current["title"])}</div>'
        '<div class="sim-box-label">BEFORE</div>'
        f'<div class="sim-value">{html.escape(before_value)}</div>'
        f'<div class="sim-action">↓ &nbsp; {html.escape(current["action"])} &nbsp; ↓</div>'
        '<div class="sim-box-label">AFTER</div>'
        f'<div class="sim-value">{html.escape(after_value)}</div>'
        '<div class="sim-explain">'
        '<strong>What is happening?</strong><br>'
        f'{html.escape(current["explanation"])}'
        '</div>'
        f'<div class="sim-next">{html.escape(next_text)}</div>'
        '</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    # Controls.
    previous_col, play_col, next_col, restart_col = st.columns(4)

    with previous_col:
        previous = st.button(
            "◀ Previous",
            key="sim_previous",
            use_container_width=True,
            disabled=current_index == 0,
        )

    with play_col:
        play_label = "⏸ Pause" if st.session_state["sim_playing"] else "▶ Play"
        play_pause = st.button(
            play_label,
            key="sim_play_pause",
            type="primary",
            use_container_width=True,
        )

    with next_col:
        next_clicked = st.button(
            "Next ▶",
            key="sim_next",
            use_container_width=True,
            disabled=current_index >= total - 1,
        )

    with restart_col:
        restart = st.button(
            "↻ Restart",
            key="sim_restart",
            use_container_width=True,
        )

    if previous:
        st.session_state["sim_playing"] = False
        st.session_state["sim_step"] = max(0, current_index - 1)
        st.rerun()

    if play_pause:
        if st.session_state["sim_playing"]:
            st.session_state["sim_playing"] = False
        else:
            if current_index >= total - 1:
                st.session_state["sim_step"] = 0
            st.session_state["sim_playing"] = True
        st.rerun()

    if next_clicked:
        st.session_state["sim_playing"] = False
        st.session_state["sim_step"] = min(total - 1, current_index + 1)
        st.rerun()

    if restart:
        st.session_state["sim_playing"] = False
        st.session_state["sim_step"] = 0
        st.rerun()

    st.markdown("**Visualization Speed**")
    speed = st.slider(
        "Animation delay (seconds per step)",
        min_value=0.5,
        max_value=4.0,
        value=float(st.session_state["sim_speed"]),
        step=0.5,
        key="sim_speed_slider",
        help="Smaller values make the automatic animation faster.",
    )
    st.session_state["sim_speed"] = speed
    st.caption(
        f"Fast  ←  {speed:.1f} second{'s' if speed != 1 else ''} per step  →  Slow"
    )

    progress = (current_index + 1) / total
    st.progress(progress)

    # Automatic progression. The short sleep intentionally happens only while
    # Play is active; each rerun advances exactly one stage.
    if st.session_state["sim_playing"]:
        if current_index < total - 1:
            time.sleep(float(st.session_state["sim_speed"]))
            st.session_state["sim_step"] = current_index + 1
            st.rerun()
        else:
            st.session_state["sim_playing"] = False


# =============================================================================
# 6. SECTION RENDERERS
# =============================================================================


def render_theory_section():
    st.header("Theoretical Framework & Background")

    st.subheader("Aim")
    st.info(EXPERIMENT_CONFIG["aim"])

    st.markdown(THEORY_CONTENT["background"])

    st.divider()
    st.subheader("Learning Objectives")
    for i, obj in enumerate(EXPERIMENT_CONFIG["objectives"], start=1):
        st.write(f"**{i}.** {obj}")

    st.divider()
    st.subheader("Experimental Procedure")
    for i, step in enumerate(THEORY_CONTENT["procedure"], start=1):
        st.write(f"**Step {i}:** {step}")

    st.divider()
    with st.expander("Key Terminology"):
        terms_df = pd.DataFrame(
            list(THEORY_CONTENT["key_terms"].items()),
            columns=["Term", "Definition"]
        )
        st.table(terms_df)

    st.divider()
    render_references()
def render_case_study_intro():
    """Attention-grabbing real-world case study shown at the top of the Simulation tab."""

    st.markdown(
        """
        <style>
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.6); }
            70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(14px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .case-study-hero {
            background: linear-gradient(-45deg, #0f172a, #1e3a8a, #2563eb, #0ea5e9);
            background-size: 300% 300%;
            animation: gradientShift 8s ease infinite;
            padding: 32px 34px;
            border-radius: 16px;
            color: white;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(30, 58, 138, 0.35);
        }
        .live-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #ef4444;
            color: white;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
            padding: 4px 10px;
            border-radius: 20px;
            animation: pulse 2s infinite;
            margin-bottom: 12px;
        }
        .live-dot {
            width: 7px; height: 7px;
            background: white;
            border-radius: 50%;
        }
        .case-study-hero h2 {
            margin: 0 0 8px 0;
            font-size: 28px;
            font-weight: 800;
        }
        .case-study-hero p {
            margin: 0;
            font-size: 15.5px;
            opacity: 0.94;
            max-width: 900px;
            line-height: 1.5;
        }
        .case-study-hero .highlight {
            background: rgba(250, 204, 21, 0.25);
            padding: 1px 6px;
            border-radius: 4px;
            font-weight: 700;
            color: #fde68a;
        }

        .stat-strip {
            display: flex;
            gap: 14px;
            margin: 18px 0 22px 0;
            flex-wrap: wrap;
        }
        .stat-box {
            flex: 1;
            min-width: 150px;
            background: linear-gradient(135deg, #eff6ff, #dbeafe);
            border: 1px solid #bfdbfe;
            border-radius: 12px;
            padding: 14px 16px;
            text-align: center;
            animation: fadeInUp 0.6s ease;
        }
        .stat-number {
            font-size: 24px;
            font-weight: 800;
            color: #1e3a8a;
            display: block;
        }
        .stat-label {
            font-size: 11.5px;
            color: #475569;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }

        .app-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 18px 18px 16px 18px;
            height: 100%;
            margin-bottom: 10px;
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
            box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        }
        .app-card:hover {
            transform: translateY(-6px) scale(1.02);
            box-shadow: 0 12px 24px rgba(37, 99, 235, 0.18);
            border-color: #93c5fd;
        }
        .app-card h4 {
            margin: 0 0 6px 0;
            font-size: 16px;
            color: #1e3a8a;
            font-weight: 700;
        }
        .app-card p {
            margin: 0;
            font-size: 13.5px;
            color: #334155;
            line-height: 1.45;
        }
        .app-icon {
            font-size: 30px;
            margin-bottom: 8px;
            display: block;
        }
        </style>

        <div class="case-study-hero">
            <div class="live-badge"><span class="live-dot"></span> RUNNING RIGHT NOW, BILLIONS OF TIMES A DAY</div>
            <h2>🔍 The Invisible Engine Behind Every Search, Text & Tap</h2>
            <p>
                Right now — as you read this — <span class="highlight">over 100,000 Google searches</span>
                are being processed every second. Before any of them return a single result, your query
                gets torn apart, cleaned, and rebuilt by the exact same steps you're about to run below:
                <b>normalization → tokenization → stop-word removal → stemming → lemmatization.</b>
                This isn't a toy exercise — it's the hidden layer powering search engines, voice
                assistants, spam filters, and every AI chatbot you've ever used.
            </p>
        </div>

        <div class="stat-strip">
            <div class="stat-box">
                <span class="stat-number">8.5B+</span>
                <span class="stat-label">Google searches / day</span>
            </div>
            <div class="stat-box">
                <span class="stat-number">347K</span>
                <span class="stat-label">Tweets posted / minute</span>
            </div>
            <div class="stat-box">
                <span class="stat-number">45%</span>
                <span class="stat-label">Of all emails are spam</span>
            </div>
            <div class="stat-box">
                <span class="stat-number">4.3B</span>
                <span class="stat-label">Voice assistant users worldwide</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 🌍 Where This Exact Pipeline Runs in the Real World")

    apps = [
        {
            "icon": "🔎",
            "title": "Search Engines",
            "desc": (
                "\"best affordable running shoes\" gets normalized, tokenized, and stemmed "
                "so Google can match it against billions of pages — regardless of tense, "
                "case, or spelling."
            ),
        },
        {
            "icon": "🤖",
            "title": "Voice Assistants",
            "desc": (
                "Siri and Alexa strip filler words before intent detection, so "
                "\"Can you PLEASE turn the lights on?!\" is understood the same as "
                "\"turn lights on.\""
            ),
        },
        {
            "icon": "🚫",
            "title": "Spam & Fraud Filters",
            "desc": (
                "Email providers preprocess millions of messages per second to spot "
                "\"WIN\", \"winning\", \"winner!!!\" as one underlying signal."
            ),
        },
        {
            "icon": "📊",
            "title": "Sentiment Analysis",
            "desc": (
                "Brands preprocess tweets and reviews at scale — removing noise and "
                "normalizing slang — before classifying sentiment automatically."
            ),
        },
    ]

    cols = st.columns(4)
    for col, app in zip(cols, apps):
        with col:
            st.markdown(
                f"""
                <div class="app-card">
                    <span class="app-icon">{app['icon']}</span>
                    <h4>{app['title']}</h4>
                    <p>{app['desc']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.success(
        "⚡ **Your turn:** The sandbox below runs the identical pipeline — at a scale you "
        "can actually *see*, step by step. Type something in and watch raw text become "
        "machine-ready in real time."
    )

    st.divider()
def render_simulation_section():
    st.header("Interactive Text Preprocessing Sandbox")
    st.write(
        "Enter a text corpus below and execute the preprocessing pipeline. "
        "The output of every stage is displayed so that the transformation "
        "from raw text to standardized text can be observed."
    )
    render_case_study_intro()
    sample_texts = {
        "Sample 1 - Basic NLP": (
            "The QUICK brown foxes are running quickly! They jumped over the lazy dogs."
        ),
        "Sample 2 - Information Retrieval": (
            "Information retrieval systems retrieve relevant documents from large "
            "collections of text. The systems should process and index documents efficiently."
        ),
        "Sample 3 - E-Commerce": (
            "Customers are searching for affordable products, comparing prices, "
            "and buying useful products online."
        ),
        "Custom Input": "",
    }

    selected_sample = st.selectbox(
        "Choose an example corpus or select Custom Input",
        list(sample_texts.keys())
    )

    default_text = sample_texts[selected_sample]

    text_input = st.text_area(
        "Input Text Corpus",
        value=default_text,
        height=150,
        placeholder="Enter or paste your text corpus here..."
    )

    col1, col2 = st.columns(2)
    with col1:
        remove_stops = st.checkbox(
            "Remove stop words",
            value=True,
            help="Remove common English stop words before stemming and lemmatization."
        )
    with col2:
        run_button = st.button(
            "Run Preprocessing Pipeline",
            type="primary",
            use_container_width=True
        )

    # Run automatically if no result exists, or when requested.
    if run_button or not st.session_state.get("current_result"):
        if not text_input.strip():
            st.warning("Please enter some text before running the experiment.")
            return
        st.session_state["current_result"] = preprocess_text(
            text_input,
            remove_stops=remove_stops
        )

    result = st.session_state["current_result"]

    st.divider()
    render_preprocessing_simulator(
        result["original"],
        remove_stops=remove_stops
    )

    st.divider()
    st.subheader("Step-by-Step Transformation")

    stage_data = [
        ("1. Original Text", result["original"]),
        ("2. Normalization", result["normalized"]),
        ("3. Tokenization", " | ".join(result["tokens"])),
        ("4. Stop-Word Removal", " | ".join(result["filtered_tokens"])),
        ("5. Stemming", " | ".join(result["stems"])),
        ("6. Lemmatization", " | ".join(result["lemmas"])),
    ]

    for title, content in stage_data:
        with st.expander(title, expanded=True):
            st.code(content if content else "(empty)", language="text")

    st.subheader("Final Standardized Corpus")
    st.success(result["final_corpus"] if result["final_corpus"] else "(empty)")

    # Stop words specifically removed
    st.subheader("Removed Stop Words")
    if result["removed_stop_words"]:
        st.write(", ".join(result["removed_stop_words"]))
    else:
        st.write("No stop words were removed.")

    # Statistics
    st.divider()
    st.subheader("Preprocessing Statistics")

    stats = preprocessing_statistics(result)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original Tokens", stats["Tokens"])
    c2.metric("Stop Words Removed", stats["Stop words removed"])
    c3.metric("Remaining Tokens", stats["Tokens after stop-word removal"])
    c4.metric("Unique Terms", stats["Unique tokens"])

    # Comparison chart
    chart_df = pd.DataFrame({
        "Stage": [
            "Original Tokens",
            "After Stop-Word Removal",
            "Unique Terms",
            "Unique Stems",
            "Unique Lemmas",
        ],
        "Count": [
            stats["Tokens"],
            stats["Tokens after stop-word removal"],
            stats["Unique tokens"],
            stats["Unique stems"],
            stats["Unique lemmas"],
        ],
    })

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=chart_df["Stage"],
            y=chart_df["Count"],
            text=chart_df["Count"],
            textposition="auto",
            name="Token Count"
        )
    )
    fig.update_layout(
        title="Corpus Statistics Across Preprocessing Stages",
        xaxis_title="Stage",
        yaxis_title="Count",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detailed token comparison
    st.subheader("Token-Level Comparison")
    comparison_len = max(
        len(result["filtered_tokens"]),
        len(result["stems"]),
        len(result["lemmas"])
    )

    comparison = pd.DataFrame({
        "Original / Filtered": result["filtered_tokens"] + [""] * (
            comparison_len - len(result["filtered_tokens"])
        ),
        "Stem": result["stems"] + [""] * (
            comparison_len - len(result["stems"])
        ),
        "Lemma": result["lemmas"] + [""] * (
            comparison_len - len(result["lemmas"])
        ),
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    # Record trial
    st.divider()
    st.subheader("Experimental Data Log Book")
    st.caption(
        "Record different input corpora or configurations to create an "
        "experimental observation table."
    )

    if st.button("Record Current Trial", type="primary", use_container_width=True):
        trial = {
            "Trial #": len(st.session_state["trials"]) + 1,
            "Tokens": stats["Tokens"],
            "After Stop Words": stats["Tokens after stop-word removal"],
            "Unique Terms": stats["Unique tokens"],
            "Input Preview": re.sub(r"\s+", " ", result["original"]).strip()[:80],
            "Timestamp": datetime.now().strftime("%H:%M:%S"),
        }
        st.session_state["trials"].append(trial)
        st.toast(f"Trial #{trial['Trial #']} successfully recorded.")

    if st.session_state["trials"]:
        trials_df = pd.DataFrame(st.session_state["trials"])
        st.dataframe(trials_df, use_container_width=True, hide_index=True)

        col_a, col_b = st.columns(2)
        with col_a:
            csv_data = trials_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Trials as CSV",
                data=csv_data,
                file_name="text_preprocessing_trials.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_b:
            if st.button("Clear Logged Trials", use_container_width=True):
                st.session_state["trials"] = []
                st.rerun()
    else:
        st.info("No trials recorded yet.")


def render_quiz_section():
    st.header("Concept Assessment Quiz")

    st.write(
        "This quiz contains 50 questions in the question bank. "
        "Each attempt randomly selects 10 questions."
    )

    # Create a new random quiz
    if (
        "current_quiz" not in st.session_state
        or st.button("Start New Quiz / Shuffle Questions")
    ):
        st.session_state["current_quiz"] = random.sample(QUESTION_BANK, 10)
        st.session_state["quiz_answers"] = {}
        st.session_state["quiz_submitted"] = False
        st.session_state["quiz_score"] = 0
        st.rerun()

    current_quiz = st.session_state["current_quiz"]

    with st.form("lab_quiz_form"):

        user_responses = {}

        for i, q in enumerate(current_quiz, start=1):

            st.subheader(f"Question {i}")

            selected = st.radio(
                q["question"],
                q["options"],
                index=st.session_state["quiz_answers"].get(q["id"], 0),
                key=f"quiz_radio_{q['id']}",
            )

            user_responses[q["id"]] = q["options"].index(selected)

        submitted = st.form_submit_button(
            "Submit Quiz for Grading",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        score = 0

        st.session_state["quiz_answers"] = user_responses
        st.session_state["quiz_submitted"] = True

        st.divider()
        st.subheader("Evaluation Results and Feedback")

        for i, q in enumerate(current_quiz, start=1):

            user_ans = user_responses[q["id"]]
            correct_ans = q["answer_index"]

            if user_ans == correct_ans:

                score += 1

                st.success(
                    f"Question {i}: Correct\n\n"
                    f"{q['explanation']}"
                )

            else:

                st.error(
                    f"Question {i}: Incorrect\n\n"
                    f"Your answer: {q['options'][user_ans]}\n\n"
                    f"Correct answer: {q['options'][correct_ans]}\n\n"
                    f"Reason: {q['explanation']}"
                )

        st.session_state["quiz_score"] = score

        percentage = (score / len(current_quiz)) * 100

        st.info(
            f"Final Score: **{score} / {len(current_quiz)} "
            f"({percentage:.0f}%)**"
        )

    elif st.session_state.get("quiz_submitted", False):

        score = st.session_state.get("quiz_score", 0)

        st.success(
            f"Quiz already submitted. Current score: "
            f"**{score} / {len(current_quiz)}**"
        )

def render_report_section():
    st.header("Report Generation")
    st.write(
        "Enter your student details and compile the current experiment "
        "results, recorded trials, observations, and quiz evaluation into a PDF."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        student_name = st.text_input(
            "Student Name",
            value=st.session_state["student_info"].get("name", "")
        )

    with col2:
        student_id = st.text_input(
            "Student Roll / ID",
            value=st.session_state["student_info"].get("id", "")
        )

    with col3:
        lab_date = st.date_input(
            "Experiment Date",
            value=datetime.now().date()
        )

    st.session_state["student_info"]["name"] = student_name
    st.session_state["student_info"]["id"] = student_id
    st.session_state["student_info"]["date"] = str(lab_date)

    st.subheader("Discussion & Observations")

    student_notes = st.text_area(
        "Enter your interpretation of the experimental results:",
        value=st.session_state.get("student_notes", ""),
        height=150,
        placeholder=(
            "Example: The number of tokens decreased after stop-word removal. "
            "Stemming reduced several related word forms, while lemmatization "
            "produced linguistically meaningful base forms."
        ),
    )

    st.session_state["student_notes"] = student_notes

    current_result = st.session_state.get("current_result")

    if current_result is None:
        st.warning(
            "Run the preprocessing experiment at least once before generating "
            "the final report."
        )
        return

    trials_df = (
        pd.DataFrame(st.session_state["trials"])
        if st.session_state["trials"]
        else pd.DataFrame()
    )

    st.divider()
    st.subheader("Report Summary Preview")

    st.write(f"**Experiment:** {EXPERIMENT_CONFIG['title']}")
    st.write(
        f"**Student:** {student_name or 'N/A'} | "
        f"**ID:** {student_id or 'N/A'} | "
        f"**Date:** {lab_date}"
    )
    st.write(
        f"**Quiz Score:** {st.session_state.get('quiz_score', 0)} / "
        f"{len(QUESTION_BANK)}"
    )

    st.write("**Final Standardized Corpus:**")
    st.code(current_result["final_corpus"] or "(empty)", language="text")

    st.write("**Preprocessing Statistics:**")
    stats_df = pd.DataFrame(
        list(preprocessing_statistics(current_result).items()),
        columns=["Metric", "Value"]
    )
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    if not trials_df.empty:
        st.write("**Recorded Trials:**")
        st.dataframe(trials_df, hide_index=True, use_container_width=True)
    else:
        st.info("No experimental trials have been recorded yet.")

    pdf_bytes = generate_pdf_report(
        student_name=student_name,
        student_id=student_id,
        date_str=str(lab_date),
        trials_df=trials_df,
        quiz_score=st.session_state.get("quiz_score", 0),
        quiz_total=len(QUESTION_BANK),
        student_notes=student_notes,
        current_result=current_result,
    )

    st.divider()
    st.subheader("Download Official Lab Report (.pdf)")

    st.download_button(
        label="Download Lab Report PDF",
        data=pdf_bytes,
        file_name="text_preprocessing_lab_report.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
    )


# =============================================================================
# 7. SESSION STATE
# =============================================================================

def init_session_state():
    if "trials" not in st.session_state:
        st.session_state["trials"] = []

    if "quiz_answers" not in st.session_state:
        st.session_state["quiz_answers"] = {}

    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False

    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0

    if "student_info" not in st.session_state:
        st.session_state["student_info"] = {
            "name": "",
            "id": "",
            "date": str(datetime.now().date()),
        }

    if "student_notes" not in st.session_state:
        st.session_state["student_notes"] = ""

    if "current_result" not in st.session_state:
        st.session_state["current_result"] = None

    if "sim_step" not in st.session_state:
        st.session_state["sim_step"] = 0

    if "sim_playing" not in st.session_state:
        st.session_state["sim_playing"] = False

    if "sim_speed" not in st.session_state:
        st.session_state["sim_speed"] = 1.5

    if "sim_signature" not in st.session_state:
        st.session_state["sim_signature"] = None


# =============================================================================
# 8. MAIN ENTRYPOINT
# =============================================================================

def main():
    st.set_page_config(
        page_title="Text Preprocessing and Normalization",
        page_icon="",
        layout="wide",
    )

    init_session_state()

    st.title(EXPERIMENT_CONFIG["title"])
    st.caption(
        "Virtual Laboratory | Information Retrieval / Natural Language Processing"
    )

    # Sidebar navigation
    st.sidebar.title("Virtual Lab")
    section = st.sidebar.radio(
        "Lab Navigator",
        options=["Theory", "Simulation", "Quiz", "Report Generation", "References"],
    )

    st.sidebar.divider()
    st.sidebar.subheader("Experiment Progress")

    simulation_status = (
        "Completed"
        if st.session_state.get("current_result")
        else "Pending"
    )
    quiz_status = (
        "Completed"
        if st.session_state.get("quiz_submitted", False)
        else "Pending"
    )
    trial_count = len(st.session_state.get("trials", []))

    st.sidebar.write(f"Preprocessing: **{simulation_status}**")
    st.sidebar.write(f"Recorded Trials: **{trial_count}**")
    st.sidebar.write(f"Quiz: **{quiz_status}**")

    if st.session_state.get("quiz_submitted", False):
        st.sidebar.write(
            f"Quiz Score: **{st.session_state.get('quiz_score', 0)} / "
            f"{len(QUESTION_BANK)}**"
        )

    st.sidebar.divider()
    st.sidebar.caption(
        "Perform multiple trials with different text inputs, record the "
        "observations, complete the quiz, and generate the final report."
    )

    # Dispatcher
    if section == "Theory":
        render_theory_section()
    elif section == "Simulation":
        render_simulation_section()
    elif section == "Quiz":
        render_quiz_section()
    elif section == "Report Generation":
        render_report_section()
    elif section == "References":
        render_references()


if __name__ == "__main__":
    main()