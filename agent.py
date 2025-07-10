import os
import streamlit as st
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

# ── Load OpenAI key from environment ────────────────────────────────────────
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    st.error("❌ OPENAI_API_KEY environment variable not found. "
             "Please set it in your shell or in Streamlit Cloud secrets.")
    st.stop()

llm = ChatOpenAI(
    openai_api_key=openai_key,
    model_name="gpt-3.5-turbo",
    temperature=0.5
)

# ── Streamlit UI ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Multi-Agent Article Writer", layout="centered")
st.title("🧠 Multi-Agent Article Writer")

topic = st.text_input("Enter a topic to generate an article:")
generate = st.button("🚀 Generate Article")

if generate and topic.strip():
    with st.spinner("Agents are collaborating..."):
        # Agents
        researcher = Agent(
            role="Research Analyst",
            goal=f"Gather key insights about '{topic}'",
            backstory="Expert at digging up the latest tools, facts, and trends.",
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )
        writer = Agent(
            role="Technical Writer",
            goal="Create a well-structured article from the research",
            backstory="Turns complex research into engaging prose.",
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )
        editor = Agent(
            role="Editor",
            goal="Polish the article for grammar, tone, and clarity",
            backstory="Ensures every sentence is crisp and error-free.",
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

        # Tasks
        research_task = Task(
            description=f"Produce at least 5 bullet-point insights on {topic}.",
            expected_output="Bullet list with explanations.",
            agent=researcher,
        )
        write_task = Task(
            description="Draft an article with intro, body, and conclusion using the research.",
            expected_output="600–800-word Markdown article.",
            agent=writer,
            depends_on=[research_task],
        )
        edit_task = Task(
            description="Refine the article for publication quality.",
            expected_output="Polished final article.",
            agent=editor,
            depends_on=[write_task],
        )

        # Crew execution
        crew = Crew(
            agents=[researcher, writer, editor],
            tasks=[research_task, write_task, edit_task],
            verbose=False,
        )

        result = crew.kickoff()
        article = str(result).strip()

        if article:
            st.success("✅ Article ready!")
            st.markdown("---")
            st.markdown(article)
        else:
            st.warning("⚠️ Agents returned an empty result. Try another topic.")
elif generate:
    st.error("Please enter a topic before clicking Generate.")
