import readline
import nltk
nltk.download('stopwords')

from dotenv import load_dotenv
load_dotenv()

def llm(messages, temperature=1):
    '''
    This function is my interface for calling the LLM.
    The messages argument should be a list of dictionaries.

    >>> llm([
    ... {'role': 'system', 'content': 'You are a helpful assistant.'},
    ... {'role': 'user', 'content': 'What is the capital of France?'}
    ... ], temperature=0)
    'The capital of France is Paris!'
    '''
    import groq
    client = groq.Groq()

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
        temperature=temperature
        )
    return chat_completion.choices[0].message.content


def chunk_document(text, max_chunk_size=500, overlap=50):
    """
    Splits a text document into overlapping chunks for RAG.

    Examples:
        >>> text = "abcdefghijklmnopqrstuvwxyz"
        >>> chunks = chunk_document(text, max_chunk_size=10, overlap=2)
        >>> for chunk in chunks:
        ...     print(chunk)
        abcdefghij
        ijklmnopqr
        qrstuvwxyz

        >>> text = "Hello world!"
        >>> chunk_document(text, max_chunk_size=5, overlap=0)
        ['Hello', ' worl', 'd!']

        >>> text = "1234567890"
        >>> chunk_document(text, max_chunk_size=4, overlap=1)
        ['1234', '4567', '7890']

        >>> chunk_document("", max_chunk_size=10, overlap=2)
        []
    """
    if max_chunk_size <= overlap:
        raise ValueError("max_chunk_size must be greater than overlap")

    chunks = []
    start = 0
    text_length = len(text)

    while start + overlap < text_length:
        end = min(start + max_chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += max_chunk_size - overlap  # slide window

    return chunks


import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

def score_chunk_relevance(chunk: str, query: str) -> float:
    """
    Scores the relevance of a text chunk to a user query using Jaccard similarity,
    with stopword removal using NLTK and regex-based tokenization.

    Examples:
        >>> round(score_chunk_relevance("Python is a programming language.", "What is Python?"), 2)
        0.5

        >>> round(score_chunk_relevance("Bananas are yellow fruits.", "What is quantum mechanics?"), 2)
        0.0

        >>> round(score_chunk_relevance("The mitochondria is the powerhouse of the cell.", "What is the mitochondria?"), 2)
        0.5
    """
    stop_words = set(stopwords.words('english'))

    def tokenize(text):
        tokens = re.findall(r'\b\w+\b', text.lower())
        return set(token for token in tokens if token not in stop_words)

    chunk_tokens = tokenize(chunk)
    query_tokens = tokenize(query)

    if not chunk_tokens or not query_tokens:
        return 0.0

    intersection = chunk_tokens.intersection(query_tokens)
    union = chunk_tokens.union(query_tokens)

    return len(intersection) / len(union)


def score_chunks(chunks, query):
    """
    Scores the relevance of a list of chunks to a query using overlap of lemmatized, non-stopword tokens.

    >>> round(score_chunks(["Python is a programming language.", "The mitochondria is the powerhouse of the cell."], "What is Python?"), 2)
    0.33
    >>> round(score_chunks(["Python is a programming language.", "The mitochondria is the powerhouse of the cell."], "What is the mitochondria?"), 2)
    0.33
    >>> round(score_chunks(["The mitochondria is the powerhouse of the cell."], "What is the powerhouse of the cell?"), 2)
    0.5
    """

    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    import nltk
    import re

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess(text):
        """
        Preprocesses a text by lowercasing, tokenizing, removing stopwords and non-alphabetic tokens,
        and lemmatizing the remaining tokens.
        """
        tokens = re.findall(r'\b\w+\b', text.lower())
        return {
            lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalpha() and token not in stop_words
    }

    # Preprocess the query text
    query_lemmas = preprocess(query)

    # If no meaningful lemmatized tokens in the query, return a score of 0 for all chunks
    if not query_lemmas:
        return [0.0] * len(chunks)

    scores = []
    for chunk in chunks:
        chunk_lemmas = preprocess(chunk)
        overlap = chunk_lemmas.intersection(query_lemmas)
        score = len(overlap) / len(query_lemmas)
        scores.append(score)

    return scores

import pprint
if __name__ == '__main__':
    messages = []
    messages.append({
        'role': 'system',
        'content': "You are a helpful assistant. You always speak like a pirate. You always answer in one sentence.",
    })
    while True:
        text = input('docchat> ')
        messages.append({
            'role': 'user',
            'content': text,
        })
        result = llm(messages)
        messages.append({
            'role': 'assistant',
            'content': result
        })
        print('result=', result)
        pprint.pprint(messages)