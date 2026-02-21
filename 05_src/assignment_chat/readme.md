# Byte - AI & Tech Assistant

A conversational AI chatbot built with Gradio that can answer questions about AI/ML topics, check the weather, and have general tech conversations.

## What is Byte?

Byte is a friendly, slightly nerdy AI assistant that specializes in tech and AI topics. It has a warm and approachable personality - gets genuinely excited when you ask about machine learning or deployment concepts, and likes to explain things with real-world analogies.

## Services

### 1. Weather API (Open-Meteo)

When you ask about weather, Byte calls the [Open-Meteo API](https://open-meteo.com/) to get real-time weather data. It first geocodes the city name to coordinates, then fetches current temperature, humidity, wind speed, and conditions. The raw JSON response gets transformed into a natural conversational message instead of just dumping numbers.

I chose Open-Meteo because it doesn't need an API key and has reliable uptime.

### 2. AI Knowledge Base (ChromaDB)

Byte has a knowledge base of ~30 entries about AI and tech topics stored in ChromaDB. When you ask about concepts like "what is RAG" or "explain transformers", it does a semantic search to find the most relevant entries and uses them to give an informed answer.

The data is in `data/ai_knowledge.csv` and covers topics like machine learning basics, LLMs, deployment, MLOps, and more. Embeddings are generated using ChromaDB's default embedding function (all-MiniLM-L6-v2) and persisted in the `chroma_db/` directory.

### 3. Function Calling (OpenAI)

The two services above are wired up through OpenAI's function calling. Instead of hardcoding when to use each service, the model decides based on the conversation context whether it needs to check the weather, search the knowledge base, or just respond directly. This makes the routing feel natural.

## Guardrails

- **System prompt protection**: Byte won't reveal or discuss its system prompt. Prompt injection attempts get a polite redirect.
- **Topic restrictions**: Byte refuses to discuss cats/dogs, horoscopes/zodiac signs, and Taylor Swift. It redirects to tech topics instead.
- Input is checked for blocked keywords before being sent to the model as an extra safety layer.

## How to Run

1. Make sure you have the course virtual environment activated
2. Make sure `05_src/.secrets` has your `API_GATEWAY_KEY` and `OPENAI_API_KEY`
3. From this directory:

```bash
python app.py
```

4. Open the URL shown in terminal (usually `http://127.0.0.1:7860`)

## Regenerating Embeddings

If you want to regenerate the ChromaDB embeddings (not required, the pre-built ones are included):

```bash
python embed_data.py
```

This reads from `data/ai_knowledge.csv` and creates the `chroma_db/` directory.

## Decisions and Notes

- **Why Open-Meteo?** Free, no API key, good documentation. It returns structured JSON that I can transform into natural language.
- **Why ChromaDB default embeddings?** Using the built-in all-MiniLM-L6-v2 model avoids extra API calls for embeddings and works well for this dataset size. It's also included in the course dependencies via sentence-transformers.
- **Why function calling for Service 3?** It ties everything together naturally. The model decides when to use tools based on context, which feels more conversational than hardcoded routing.
- **Memory**: Gradio's ChatInterface handles conversation history automatically. The full history is passed to the API on each turn so the model has context of the whole conversation.
- **Topic blocking**: I check for keywords in the user input before it goes to the model. This catches most cases. The system prompt also instructs the model to refuse these topics as a second layer.
