# 🗄️ Text to SQL Assistant (LangChain + Groq + Streamlit)

Ask in plain English -> get SQL, explanation and live results.

**LangChain pieces:** `ChatGroq` (model) · `ChatPromptTemplate` (prompt) · `PydanticOutputParser` (parser) · chain = `prompt | llm | parser`

## Features
- Structured output: `sql_query` + `explanation`
- Safe: SELECT-only, single statement, read-only DB connection
- Auto-repair of failed SQL (one retry with the error message)
- Sample employees DB **or upload your own CSV**
- Query history, CSV download, quick bar chart, schema/data preview

## Run locally
```bash
python -m venv .venv
.venv\Scripts\activate        # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
set GROQ_API_KEY=your_key     # Mac/Linux: export GROQ_API_KEY=your_key
streamlit run app.py
```
Free key: https://console.groq.com  (or paste it in the sidebar)

## Deploy (Streamlit Community Cloud)
1. Push to GitHub. **Never commit your key.**
2. share.streamlit.io -> New app -> main file `app.py`.
3. App settings -> Secrets:
   ```toml
   GROQ_API_KEY = "your_key"
   ```
