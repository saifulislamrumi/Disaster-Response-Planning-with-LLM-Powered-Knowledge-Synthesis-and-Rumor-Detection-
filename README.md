# Disaster Response Planning with LLM-Powered Knowledge Synthesis and Rumor Detection

![Disaster Response Dashboard] This repository contains the source code and resources for the B.Sc. thesis project, "Disaster Response Planning with LLM-Powered Knowledge Synthesis and Rumor Detection", developed at the International Islamic University Chittagong (IIUC).

The project introduces a retrieval-first pipeline that transforms unstructured crisis tweets into auditable, structured intelligence. It leverages Large Language Models (LLMs) to enhance situational awareness for emergency responders by filtering misinformation and extracting actionable data.

## ✨ Key Features

-   **LLM-Powered Headline Generation**: Compresses raw tweets into concise, search-ready headlines for efficient information retrieval.
-   **Retrieval-Augmented Generation (RAG)**: Grounds the model's analysis in verifiable external sources, retrieving context from a dynamic knowledge base to reduce hallucinations and improve accuracy.
-   **Automated Rumor Detection**: Classifies tweets as "Rumor" or "Non-Rumor" based on contradictory evidence found in retrieved articles, providing explanations for its decisions.
-   **Structured Information Extraction**: For verified posts, it extracts key fields like **severity**, **primary need** (e.g., food, medical), **location**, and **time** into a structured JSON format.
-   **Dynamic Knowledge Base**: Uses a FAISS vector store that "warms up" by caching context, reducing redundant web searches and lowering latency over time.
-   **Interactive Dashboard**: A Streamlit-based user interface for uploading data, running the pipeline, and visualizing results through interactive maps, charts, and tables.

## ⚙️ System Architecture

The pipeline follows a multi-stage workflow to process each tweet:

1.  **Ingestion**: User uploads a CSV file containing disaster-related tweets.
2.  **Headline Generation**: A Llama 3.1 model generates a concise, searchable headline from the tweet text.
3.  **Context Retrieval**: The system first queries a local FAISS vector database for relevant context. If no sufficient context is found, it performs a web search using the Serper.dev API.
4.  **Knowledge Enrichment**: New information retrieved from the web is parsed, embedded, and used to update the FAISS vector database.
5.  **Analysis and Classification**: The LLM analyzes the original tweet along with the retrieved context to classify it and extract structured data.
6.  **Output & Visualization**: The structured data is stored in JSON files and presented on the Streamlit dashboard for analysis.

## 🛠️ Technology Stack

This project is built with the following technologies and libraries:

-   **LLM**: Llama 3.1 (served locally via Ollama)
-   **Embeddings**: `sentence-transformers`
-   **Vector Store**: FAISS (Facebook AI Similarity Search)
-   **Web Search**: Serper.dev API
-   **Dashboard**: Streamlit
-   **Backend**: Python 3.11
-   **Core Libraries**: pandas, scikit-learn, plotly, geopy, newspaper3k, python-dotenv

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

-   Python 3.10+
-   [Ollama](https://ollama.com/) installed and running.
-   An API key from [Serper.dev](https://serper.dev/) for web search.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/disaster-response-llm-rumor-detection.git](https://github.com/your-username/disaster-response-llm-rumor-detection.git)
    cd disaster-response-llm-rumor-detection
    ```

2.  **Set up a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Pull the Llama 3.1 model with Ollama:**
    ```bash
    ollama pull llama3.1
    ```

5.  **Set up your environment variables:**
    -   Create a file named `.env` in the root directory.
    -   Add your Serper API key to this file:
    ```
    SERPER_API_KEY="your_serper_api_key_here"
    ```

### Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run dashboard.py
    ```

2.  **Open your web browser** and navigate to the local URL provided (usually `http://localhost:8501`).

3.  **Upload a CSV file** containing tweet data. The CSV should have columns like `tweet`, `location`, `timestamp`, etc.

4.  **Click the "Run Pipeline" button** to start the analysis.

5.  **Explore the results** on the interactive dashboard, including maps, need/severity distributions, and a table of detected rumors.

## 📊 Evaluation Results

The system was evaluated on a manually curated dataset of 491 disaster-related tweets. The rumor detection model achieved the following performance:

| Metric               | Score  |
| -------------------- | :----: |
| **Accuracy** | 0.98   |
| **Rumor F1-Score** | 0.78   |
| **Non-Rumor F1-Score**| 0.99   |
| **ROC AUC** | 0.87   |

## 🔭 Future Work

Future enhancements for this system include:
-   **Real-time Data Ingestion**: Integrate with social media APIs for live, streaming data processing.
-   **Multilingual and Multimodal Capabilities**: Extend the pipeline to process tweets in other languages and analyze images/videos for a more comprehensive understanding.
-   **Advanced Media Forensics**: Add modules to detect deepfakes and manipulated images to counter sophisticated visual misinformation.
-   **Automated Alerting**: Implement a real-time alerting system to notify emergency responders of critical events or spikes in needs.
