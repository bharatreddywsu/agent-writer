import os
import streamlit as st
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

# Load OpenAI API key from environment
openai_key = os.getenv("OPENAI_API_KEY")

# LLM Configuration
llm = ChatOpenAI(
    openai_api_key=openai_key,
    model_name="gpt-3.5-turbo",
    temperature=0.5
)

# Streamlit UI
st.set_page_config(page_title="Multi-Agent Article Writer", layout="centered")
st.title("🧠 Multi-Agent Article Writer")
topic = st.text_input("Enter a topic to generate an article:")

if topic:
    with st.spinner("🤖 Agents working together..."):
        # Agents
        researcher = Agent(
            role="AI Research Analyst",
            goal=f"Research about '{topic}' in detail",
            backstory="An expert researcher with deep AI knowledge",
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

        writer = Agent(
            role="Technical Content Writer",
            goal="Write a full article on the topic",
            backstory="A professional writer who converts research into clear, engaging content",
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

        editor = Agent(
            role="Language Editor",
            goal="Edit the article to ensure perfect grammar, tone, and clarity",
            backstory="An English expert who finalizes articles before publication",
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

        # Tasks
        research_task = Task(
            description=f"Find at least five insights on: {topic}. Include key facts, tools, and relevance.",
            agent=researcher,
            expected_output="A structured list of key points about the topic."
        )

        write_task = Task(
            description="Write a complete blog article based on the research findings.",
            agent=writer,
            depends_on=[research_task],
            expected_output="A full article with intro, body (covering each insight), and conclusion."
        )

        edit_task = Task(
            description="Edit the article for tone, grammar, and clarity.",
            agent=editor,
            depends_on=[write_task],
            expected_output="A final polished article ready for publishing."
        )

        # Crew
        crew = Crew(
            agents=[researcher, writer, editor],
            tasks=[research_task, write_task, edit_task],
            verbose=True
        )

        result = crew.kickoff()
        st.success("✅ Final article generated!")
        st.markdown("---")
        st.markdown(result.output if hasattr(result, "output") else result)

