# PRELIM STUFF
import readline
import nltk
from nltk.tokenize import word_tokenize
tokenizer = WordPunctTokenizer()
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from dotenv import load_dotenv
load_dotenv()


import argparse
parser = argparse.ArgumentParser(
    prog='docchat',
    description='summarize the input',
    )

parser.add_argument('filename')
args = parser.parse_args()


# SUPPLEMENTARY FUNCTIONS
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


def load_text(filepath_or_url):
    """
    Loads the text from an inputted document."
    
    Examples:
        >>> load_text('exdocs/example.txt')
        'This is an example txt file.'
        >>> load_text('exdocs/example.pdf')
        'This is an example pdf ﬁle. '
        >>> load_text('exdocs/example.html')
        'This is an example html file.'
    """
    import requests
    from bs4 import BeautifulSoup
    from PyPDF2 import PdfReader
    from PyPDF2.errors import PdfReadError
    
    try:
        # FOR TXT FILES
        with open(filepath_or_url, 'r') as fin:
            html = fin.read()
            soup = BeautifulSoup(html, features="lxml")
            text = soup.text
    except UnicodeDecodeError:
        # FOR PDF FILES
        try:
            with open(filepath_or_url, 'rb') as fin:
                reader = PdfReader(fin)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()


        except PdfReadError:
            print("cannot take an image file")


    except FileNotFoundError:
        try:  
            r = requests.get(filepath_or_url)
            html = r.text
            soup = BeautifulSoup(html, features='lxml')
            text = soup.text
        except requests.exceptions.MissingSchema:
            with open(filepath_or_url, 'r') as fin:
                html = fin.read()
                soup = BeautifulSoup(html, features="lxml")
                text = soup.text

    return text


def chunk_text_by_words(text, max_words=5, overlap=2):
    """
    Splits text into overlapping chunks by word count.

    Examples:
        >>> text = "The quick brown fox jumps over the lazy dog. It was a sunny day and the birds were singing."
        >>> chunks = chunk_text_by_words(text, max_words=5, overlap=2)
        >>> len(chunks)
        7
        >>> chunks[0]
        'The quick brown fox jumps'
        >>> chunks[1]
        'fox jumps over the lazy'
        >>> chunks[4]
        'sunny day and the birds'
        >>> chunks[-1]
        'singing.'
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap

    return chunks


def score_chunks(chunk: str, query: str) -> float:
    """
    Scores a chunk against a user query using Jaccard similarity of lemmatized word sets
    with stopword removal using NLTK.

    >>> round(score_chunks("The sun is bright and hot.", "How hot is the sun?"), 2)
    0.67
    >>> round(score_chunks("The red car is speeding down the road.", "What color is the car?"), 2)
    0.2
    >>> score_chunks("Bananas are yellow.", "How do airplanes fly?")
    0.0
    """
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from string import punctuation

    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess(text):
        tokens = word_tokenize(text.lower())
        return set(
            lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalpha() and token not in stop_words
        )

    chunk_words = preprocess(chunk)
    query_words = preprocess(query)

    if not chunk_words or not query_words:
        return 0.0

    intersection = chunk_words & query_words
    union = chunk_words | query_words

    return len(intersection) / len(union)


def find_relevant_chunks(text, query, num_chunks=3, max_words=100, overlap=50):
    """
    Find the most relevant chunks from the input text based on a query.
    Chunks are scored using exact term matches and returned in order of relevance.

    >>> sample_text = (
    ... "Python is a high-level programming language. "
    ... "It is widely used in data science, web development, and automation. "
    ... "Bananas are yellow fruits. "
    ... "The mitochondria is the powerhouse of the cell. "
    ... "Web development often uses JavaScript and Python."
    ... )

    >>> find_relevant_chunks(sample_text, "What is Python?", num_chunks=2, max_words=10, overlap=0)
    ['Python is a high-level programming language. It is widely used', 'development often uses JavaScript and Python.']

    >>> find_relevant_chunks(sample_text, "fruit", num_chunks=1, max_words=7, overlap=0)
    ['development, and automation. Bananas are yellow fruits.']

    >>> find_relevant_chunks(sample_text, "cell powerhouse", num_chunks=1, max_words=7, overlap=0)
    ['The mitochondria is the powerhouse of the']
    """

    chunks = chunk_text_by_words(text, max_words=max_words, overlap=overlap)

    # Get relevance scores for each chunk
    scored_chunks = [(chunk, score_chunks(chunk, query)) for chunk in chunks]

    # Sort chunks by their relevance score
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    # Return the top N chunks based on their score
    relevant_chunks = [chunk for chunk, score in scored_chunks[:num_chunks]]

    return relevant_chunks


def split_text(text, max_chunk_size=1000):
    '''
    Takes a string as input and returns a list of strings that are all smaller than max chunk size)
    >>> split_text('abcdefg', max_chunk_size=2)
    ['ab', 'cd', 'ef', 'g']
    '''
    accumulator = []
    while len(text) > 0:
        accumulator.append(text[:max_chunk_size])
        text = text[max_chunk_size:]
    return accumulator


def summarize_text(text):
    import groq
    prompt = f"""
    Summarize the following text in 1-3 sentences.

    {text}
    """
    try:
        output = llm([{"role": "user", "content": prompt}])
        return output.split('\n')[-1]
    except groq.APIStatusError:
        chunks = split_text(text, 10000)
        print("len(chunks)=", len(chunks))
        accumulator = []
        for i, chunk in enumerate(chunks):
            print('i=', i)
            summary = summarize_text(chunk)
            accumulator.append(summary)
        summarized_text = ' '.join(accumulator)
        summarized_text = summarize_text(summarized_text)
        return summarized_text


#MAIN FUNCTION
import pprint
if __name__ == '__main__':
    doc_text = load_text(args.filename)
    chunks = chunk_text_by_words(doc_text, max_words=100, overlap=50)

    summary = summarize_text(doc_text)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions about a document. You will be given relevant excerpts for each user question."
        },
        {
            "role": "assistant",
            "content": f"Here is a summary of the document:\n\n{summary}"
        }
    ]

    while True:
        query = input('docchat> ')
        if query.strip().lower() in {'exit', 'quit'}:
            break

        relevant_chunks = find_relevant_chunks(doc_text, query, num_chunks=3)
        context = "\n---\n".join(relevant_chunks)

        messages.append({
            'role': 'user',
            'content': f"Here is the context:\n{context}\n\nNow answer this question:\n{query}"
        })

        result = llm(messages, temperature=0)
        messages.append({
            'role': 'assistant',
            'content': result
        })

        print(result)
        
