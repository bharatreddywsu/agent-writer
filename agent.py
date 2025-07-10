import os

# ── Disable Chromadb-based embedding configurator so CrewAI works without chromadb ──
os.environ["CREWAI_DISABLE_EMBEDDING_CONFIGURATOR"] = "1"

import streamlit as st
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

# ── Load OpenAI key from environment ────────────────────────────────────────────────
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    st.error(
        "❌ OPENAI_API_KEY environment variable not found.\n"
        "Set it in ~/.zshrc for local runs or in Streamlit Cloud Secrets when deployed."
    )
    st.stop()

llm = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-3.5-turbo", temperature=0.5)

# ── Streamlit UI ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Multi-Agent Article Writer", layout="centered")
st.title("🧠 Multi-Agent Article Writer")

topic = st.text_input("Enter a topic to generate an article:")
generate = st.button("🚀 Generate Article")

if generate and topic.strip():
    with st.spinner("🤖 Agents collaborating..."):
        # ── Define Agents ──
        researcher = Agent(
            role="Research Analyst",
            goal=f"Collect the most important insights about '{topic}'",
            backstory="An expert researcher scouring the latest data, tools and trends.",
            verbose=True,
            llm=llm,
        )
        writer = Agent(
            role="Technical Writer",
            goal="Write a clear, engaging article from the research",
            backstory="Transforms complex findings into readable prose.",
            verbose=True,
            llm=llm,
        )
        editor = Agent(
            role="Language Editor",
            goal="Polish the article for flawless grammar and flow",
            backstory="Ensures publication-ready quality.",
            verbose=True,
            llm=llm,
        )

        # ── Define Tasks ──
        research_task = Task(
            description=f"Produce at least five key bullet-point insights on: {topic}",
            expected_output="A bullet list summarizing each key point.",
            agent=researcher,
        )
        write_task = Task(
            description="Draft a 600-800 word Markdown article using the research.",
            expected_output="Article with intro, body (covering each insight) and conclusion.",
            agent=writer,
            depends_on=[research_task],
        )
        edit_task = Task(
            description="Refine the article for tone, clarity and grammar.",
            expected_output="Polished final article ready to publish.",
            agent=editor,
            depends_on=[write_task],
        )

        # ── Run crew ──
        crew = Crew(
            agents=[researcher, writer, editor],
            tasks=[research_task, write_task, edit_task],
            verbose=False,
        )

        article = str(crew.kickoff()).strip()

        if article:
            st.success("✅ Article generated!")
            st.markdown("---")
            st.markdown(article)
        else:
            st.warning("⚠️ Agents returned an empty result. Try another topic.")
elif generate:
    st.error("Please enter a topic before clicking Generate.")
