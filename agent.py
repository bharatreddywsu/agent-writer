import os
import streamlit as st
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import traceback
import time

# ── Hardcoded OpenAI Key ──
os.environ["OPENAI_API_KEY"] = (
    "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"
)

# ── Streamlit Page Setup ──
st.set_page_config(page_title="🧠 CrewAI Agent Writer", layout="centered")
st.title("🧠 Multi-Agent Article Writer")

topic = st.text_input("Enter a topic to generate an article:")
generate = st.button("🚀 Generate Article")

if generate and topic.strip():
    with st.spinner("Agents at work... Please wait."):
        try:
            print("🔥 Agents starting...")
            start_time = time.time()

            # ── LLM Setup ──
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

            # ── Agents ──
            researcher = Agent(
                role="Research Analyst",
                goal="Find key tools and insights related to the topic",
                backstory="You're an AI tools expert always up to date with the latest innovations.",
                verbose=True,
                allow_delegation=False,
                llm=llm,
            )

            writer = Agent(
                role="Content Writer",
                goal="Write a well-structured article using the research",
                backstory="You're skilled at writing tech blogs that are clear and engaging.",
                verbose=True,
                allow_delegation=False,
                llm=llm,
            )

            editor = Agent(
                role="Editor",
                goal="Polish the article for grammar, tone, and flow",
                backstory="You fine-tune articles for smooth reading and professional quality.",
                verbose=True,
                allow_delegation=False,
                llm=llm,
            )

            # ── Tasks ──
            research_task = Task(
                description=f"Research at least 5 facts, tools, or trends related to: {topic}",
                expected_output="Bullet points with 5+ items: name, short description, and importance.",
                agent=researcher,
            )

            write_task = Task(
                description="Write an article with intro, sections per item, and conclusion based on the research.",
                expected_output="A well-formatted Markdown article around 600–800 words.",
                agent=writer,
                depends_on=[research_task],
            )

            edit_task = Task(
                description="Edit the article for tone, grammar, and flow.",
                expected_output="A polished final article ready to publish.",
                agent=editor,
                depends_on=[write_task],
            )

            # ── Crew Run ──
            crew = Crew(
                agents=[researcher, writer, editor],
                tasks=[research_task, write_task, edit_task],
                verbose=False,
            )

            result = crew.kickoff()
            final_output = str(result).strip()

            print("✅ Agents finished in", round(time.time() - start_time, 2), "seconds")
            print("📝 Raw Result:", repr(final_output))

            if final_output:
                st.success("✅ Done! Here's your article:")
                st.markdown("---")
                st.markdown(final_output)
            else:
                st.warning("⚠️ The agents completed but gave an empty result.")
                st.info("Try rephrasing the topic, or check your OpenAI key and internet connection.")

        except Exception as e:
            st.error("❌ An error occurred during execution:")
            st.code(traceback.format_exc())

elif generate and not topic.strip():
    st.error("❗ Please enter a topic.")
