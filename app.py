import streamlit as st

from src.database import get_database
from src.chain import create_sql_chain

st.set_page_config(
    page_title="Text-to-SQL Assistant",
    page_icon="🗄️",
    layout="wide"
)

st.title("🗄️ Text-to-SQL Assistant")
st.caption("Ask questions about your database using natural language.")

try:
    db = get_database()
    chain = create_sql_chain(db)

    schema = db.get_table_info()

    with st.expander("View Database Schema"):
        st.code(schema, language="sql")

    question = st.text_input(
        "Ask a question",
        placeholder="Example: Show the top 5 employees by salary"
    )

    if question:
        with st.spinner("Generating SQL and executing query..."):
            result = chain.invoke({"question": question})

        st.subheader("Answer")
        st.write(result)

except Exception as e:
    st.error(f"Application error: {e}")
