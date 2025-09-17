# YouTube Video Question-Answering Tool  

## Overview  
This project builds a **question-answering (QA) tool** that extracts and summarizes information from YouTube videos. Using **LangChain** and a **Large Language Model (LLM)**, the tool allows users to ask natural language questions and receive accurate answers based on the video's transcript.  

The system leverages **video transcript loaders, text processing, embeddings, FAISS vector databases, and retrievers**, while providing a simple **Gradio interface** for interaction.  

---

##  Features  
- **Transcript Extraction** – Fetches transcripts directly from YouTube videos.  
- **Contextual Search** – Embeds text and uses **FAISS** for fast similarity search.  
- **Question Answering** – Uses an LLM to generate answers from transcript context.  
- **Summarization** – Produces concise summaries of lengthy transcripts.  
- **Interactive UI** – Built with **Gradio**   support.  

---

##  Tech Stack  
- [LangChain](https://www.langchain.com/) – LLM orchestration  
- [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/) – Transcript fetching  
- [FAISS](https://github.com/facebookresearch/faiss) – Vector similarity search  
- [IBM WatsonX AI](https://www.ibm.com/watsonx/ai) – Optional LLM backend   
- [Gradio](https://www.gradio.app/) – Alternative lightweight interface  

---

## Usage

1.  ### Create a virtual environment (recommended)
    ```bash
    python -m venv my_env
    source my_env/bin/activate   # On Mac/Linux
    my_env\Scripts\activate      # On Windows
    ```

2. ### Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

3. ### Run the app
    ```bash
    python app.py
    ```

4. ### After starting, Gradio will display something like:
    ```bash
    Running on local URL:  http://127.0.0.1:7860
    Running on public URL: https://xxxx.gradio.live
    ```

5. ### Open the provided link in your browser to access the app.