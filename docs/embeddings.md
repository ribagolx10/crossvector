# Embeddings

This document covers embedding adapters for OpenAI and Google Gemini.

## Configuration

- **OpenAI**: set `OPENAI_API_KEY`
- **Gemini**: set `GOOGLE_API_KEY` (or `GEMINI_API_KEY`)

Missing API keys raise `MissingConfigError` in adapters.

## Error Behavior

- API request failures are re-raised as their original exception types to preserve details.
- Configuration issues (missing keys or packages) raise `MissingConfigError` with guidance.

## Dimensions

- **OpenAI**: Uses known dimensions, unknown models raise `InvalidFieldError`.
- **Gemini**:
  - Defaults to 768 for standard models.
  - `gemini-embedding-001` supports `768`, `1536`, `3072`; invalid dimensionality raises `InvalidFieldError`.
