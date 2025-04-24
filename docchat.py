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


def chunk_text_by_words(text, max_words=100, overlap=50):
    """
    Splits a text document into overlapping chunks for RAG.

    Examples:
        >>> text = "abcdefghijklmnopqrstuvwxyz"
        >>> chunks = chunk_text_by_words(text, max_words=10, overlap=2)
        >>> for chunk in chunks:
        ...     print(chunk)
        abcdefghij
        ijklmnopqr
        qrstuvwxyz

        >>> text = "Hello world!"
        >>> chunk_text_by_words(text, max_words=5, overlap=0)
        ['Hello', ' worl', 'd!']

        >>> text = "1234567890"
        >>> chunk_text_by_words(text, max_words=4, overlap=1)
        ['1234', '4567', '7890']

        >>> chunk_text_by_words("", max_words=10, overlap=2)
        []
    """
    if max_words <= overlap:
        raise ValueError("max_words must be greater than overlap")

    chunks = []
    start = 0
    text_length = len(text)

    while start + overlap < text_length:
        end = min(start + max_words, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += max_words - overlap  # slide window

    return chunks


import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def score_chunks(chunk: str, query: str) -> float:
    """
    Scores the relevance of a text chunk to a user query using Jaccard similarity
    over lemmatized, non-stopword tokens.

    Examples:
        >>> round(score_chunks("Python is a programming language.", "What is Python?"), 2)
        0.33

        >>> round(score_chunks("Bananas are yellow fruits.", "What is quantum mechanics?"), 2)
        0.0

        >>> round(score_chunks("The mitochondria is the powerhouse of the cell.", "What is the mitochondria?"), 2)
        0.33
    """

    def preprocess(text):
        """
        Tokenizes, lowercases, removes stopwords, and lemmatizes words.

        >>> preprocess("The quick brown fox jumps over the lazy dog.")
        {'jump', 'fox', 'dog', 'quick', 'lazy', 'brown'}

        >>> preprocess("What is Python?")
        {'python'}

        >>> preprocess("Bananas are yellow fruits.")
        {'banana', 'yellow', 'fruit'}
        """
        tokens = re.findall(r'\b\w+\b', text.lower())
        return {
            lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalpha() and token not in stop_words
        }

    chunk_tokens = preprocess(chunk)
    query_tokens = preprocess(query)

    if not chunk_tokens or not query_tokens:
        return 0.0

    intersection = chunk_tokens.intersection(query_tokens)
    union = chunk_tokens.union(query_tokens)

    return len(intersection) / len(union)


import nltk
from nltk.tokenize import sent_tokenize

import string
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

def find_relevant_chunks(text, query, num_chunks=3):
    """
    Find the most relevant chunks from the input text based on a query.

    This function splits the text into sentences, scores each sentence's 
    relevance to the query, and returns the top 'num_chunks' most relevant 
    sentences.

    Args:
        text (str): The input text to be searched.
        query (str): The query to find relevant chunks for.
        num_chunks (int, optional): The number of relevant chunks to return. Defaults to 3.

    Returns:
        list: A list of the top 'num_chunks' most relevant chunks from the text.

    Examples:
        >>> text = "This is a test document. It contains multiple sentences. We are trying to find relevant chunks based on a query."
        >>> query = "relevant chunks"
        >>> find_relevant_chunks(text, query, num_chunks=3)
        ['We are trying to find relevant chunks based on a query', 'This is a test document', 'It contains multiple sentences']

        >>> text = "Python is a programming language. It is widely used for web development and data science."
        >>> query = "web development"
        >>> find_relevant_chunks(text, query, num_chunks=2)
        ['It is widely used for web development and data science', 'Python is a programming language']

        >>> text = "The mitochondria is the powerhouse of the cell. It generates energy for the cell."
        >>> query = "mitochondria"
        >>> find_relevant_chunks(text, query, num_chunks=1)
        ['The mitochondria is the powerhouse of the cell']
    """
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Remove stopwords for query comparison but keep sentence structure intact
    stop_words = set(stopwords.words("english"))
    punctuations = set(string.punctuation)
    
    def clean_for_matching(text):
        """Clean text for matching by removing stopwords and punctuation"""
        words = text.split()
        return [word.lower() for word in words if word.lower() not in stop_words and word not in punctuations]

    # Clean the query for matching
    cleaned_query = clean_for_matching(query)

    def score_sentence(sentence):
        """Score the sentence based on matching words with the cleaned query"""
        cleaned_sentence = clean_for_matching(sentence)
        return sum(1 for word in cleaned_sentence if word in cleaned_query)

    # Score each sentence and sort by relevance
    scored_sentences = [(sentence, score_sentence(sentence)) for sentence in sentences]

    # Sort sentences by score, highest first
    scored_sentences.sort(key=lambda x: x[1], reverse=True)

    # Return the top 'num_chunks' sentences, keeping the original structure, and strip trailing punctuation for comparison
    top_chunks = [scored_sentences[i][0] for i in range(min(num_chunks, len(scored_sentences)))]

    # Strip trailing punctuation for final result comparison
    return [sentence.rstrip(string.punctuation) if sentence.endswith('.') else sentence for sentence in top_chunks]


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

