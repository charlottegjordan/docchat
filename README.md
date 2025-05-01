# docchat: chat with your documents!



The docchat program allows you to input a document and ask an LLM-based chatbot questions about the inputted document. The inputted document can be a PDF, an html link or file, or a txt file.

Here is an example of docchat answering questions well:

```
$ docchat exdocs/cnn.pdf
docchat> what is this document?
Based on the provided excerpts, this document appears to be a news article from CNN Business, titled "California overtakes Japan to become the world's fourth-largest economy". The article reports on California's economic growth and surpassing Japan's economy, as well as the potential threats posed by President Trump's tariffs to the state's economy.
docchat> what language is the document written in?
The document appears to be written in English.
docchat>
```

However, docchat cannot answer every question, especially if the answer relies on information not included in the inputted document:
```
docchat> what is the fifth largest economy in the world?
I apologize, but the provided document does not mention the fifth-largest economy in the world. The document only discusses California's economy surpassing Japan to become the world's fourth-largest economy and the lawsuit filed by Governor Gavin Newsom against President Donald Trump's tariffs. If you need information on the fifth-largest economy, I can try to find it for you from a different source.
docchat> 
```

## Requirements
To install the required dependencies for this program, run this command:
```$ pip3 install -r requirements.txt```
