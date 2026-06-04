## GEOpulse — copyWriter

An opinionated, production-oriented assistant for marketing content and research built on Cohere (LLM + embeddings), Pinecone (vector DB) and Llama-Index. The repository contains utilities to crawl and index web pages, store RAG context in a vector store, and run a marketing copy agent that produces focused, SEO- and conversion-oriented content.

Note: some source files contain strings and prompts in Hebrew; the code is ready to work with multilingual embeddings and Cohere's models.

## Project layout

- `copyWriter/`
  - `indexer.py` — web page scraping + indexing script. Pulls HTML, extracts text and stores Document objects in a Pinecone-backed vector store using Cohere embeddings.
  - `copy_agent.py` — marketing copy agent. Loads the vector index, constructs a strict prompt template (zero-hallucination policy), queries the index, and uses Cohere to generate marketing copy and placement recommendations.
  - `.env` — local environment file (contains API keys). Do NOT commit real secrets to remote repositories. Remove from git history if already pushed.
- `ai_env/` — an optional local Python virtual environment used when the project was developed here (not required to commit).

## Key features

- RAG-enabled marketing agent: retrieves the most relevant content from a vector index and provides data-anchored marketing copy.
- Multilingual embeddings: configured to use Cohere's multilingual embedding model.
- Pinecone vector store backend with serverless index spec.
- A strict prompt template enforcing zero-hallucination and structured outputs (placement strategy + content).

## Quick start (Windows PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install required Python packages (examples of packages used in the project):

```powershell
pip install python-dotenv pinecone-client llama-index cohere requests beautifulsoup4
```

3. Create a `.env` at the project root with the following variables (do not paste secrets into public places):

```
COHERE_API_KEY=<your_cohere_api_key>
PINECONE_API_KEY=<your_pinecone_api_key>
PINECONE_INDEX_NAME=<desired_index_name>
```

4. Index sample sites (run the indexer):

```powershell
python .\copyWriter\indexer.py
```

5. Run the marketing agent (demo):

```powershell
python .\copyWriter\copy_agent.py
```

The agent includes a small demo JSON payload in `copy_agent.py` that runs when the file is executed as `__main__`. Replace that sample JSON with your own `category`, `vulnerability` and `action_plan` fields when you want bespoke output.

## Environment variables

- `COHERE_API_KEY` — required by Cohere SDK for both embeddings and LLM calls.
- `PINECONE_API_KEY` — required to create/connect to Pinecone indexes.
- `PINECONE_INDEX_NAME` — the index name used by the scripts (e.g., `bituach-yashir-knowledge`).

Security: Never commit an `.env` with real API keys. If you accidentally committed keys, rotate them immediately and remove the file from the repository and its history.

## How it works (high level)

1. Indexing: `indexer.py` scrapes configured URLs with `requests` and `BeautifulSoup`, converts page text to Llama-Index `Document` objects, embeds with Cohere, and writes vectors to Pinecone.
2. Retrieval: `copy_agent.py` initializes Llama-Index and a Cohere LLM client, wraps a hand-crafted prompt template (with zero-hallucination rules), retrieves the top-k contextual nodes from Pinecone, and asks Cohere to produce marketing copy + placement recommendations.
3. Output: The agent returns a structured piece of marketing copy that includes a placement suggestion and a content block, intended to be copy/pasted or adapted into a web CMS.

## Example input (JSON)

Use the same shape as the in-file demo in `copy_agent.py`:

```json
{
  "category": "Comprehensive Car Insurance - Claims & Service",
  "vulnerability": "Users can't find clear information about claims speed and replacement car availability, causing trust friction.",
  "action_plan": {
    "technical": "Add structured FAQ schema for Google AI Overviews",
    "marketing": "Highlight speed and digital-first processing; surface specific time metrics from policy documents."
  }
}
```

## Troubleshooting

- If you get authentication errors, verify your `.env` values and that your API keys are active.
- If requests to target websites fail, check network connectivity and whether the site blocks automated clients. Try setting `headers` in `indexer.py` (it already sets a browser UA string).
- If embeddings/queries return no relevant context, index more documents or increase `similarity_top_k` in the retriever call.


