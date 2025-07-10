import os
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI   # wrapper for LangChain ≥ 0.3

# ── HARD-CODED KEY (for quick tests) ────────────────────────────────────────────
os.environ["OPENAI_API_KEY"] = (
    "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"
)
# ────────────────────────────────────────────────────────────────────────────────

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

# ── AGENTS ──────────────────────────────────────────────────────────────────────
researcher = Agent(
    role="AI Research Analyst",
    goal="Gather top AI tools trending in 2025",
    backstory="Expert in AI tools, trends, and evaluations.",
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

writer = Agent(
    role="Tech Writer",
    goal="Write an article using the research",
    backstory="Writes beginner-friendly tech content with clarity.",
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

editor = Agent(
    role="Language Editor",
    goal="Polish the article for grammar, tone, and flow",
    backstory="Professional editor who makes content shine.",
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

# ── TASKS ───────────────────────────────────────────────────────────────────────
topic = "Top AI tools in 2025"

research_task = Task(
    description=(
        f"Find at least five trending AI tools in 2025. "
        f"Explain briefly what each tool does and why it matters."
    ),
    expected_output=(
        "A bullet-point list with ≥5 tools. "
        "For each: Name • One-line description • Why it matters."
    ),
    agent=researcher,
)

write_task = Task(
    description=(
        "Write a structured article based on the research. "
        "Include an introduction, a section for each tool, and a conclusion."
    ),
    expected_output="A Markdown article of roughly 600–800 words.",
    agent=writer,
    depends_on=[research_task],
)

edit_task = Task(
    description="Edit the article for tone, grammar, and flow.",
    expected_output="A polished, publication-ready article.",
    agent=editor,
    depends_on=[write_task],
)

# ── RUN THE CREW ────────────────────────────────────────────────────────────────
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, write_task, edit_task],
    verbose=True,
)

result = crew.kickoff()

print("\n✅ FINAL ARTICLE:\n")
print(result)
