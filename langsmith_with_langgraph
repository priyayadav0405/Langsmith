import operator
import json
import os
from typing import TypedDict, Annotated, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from langchain_groq import ChatGroq
from langsmith import traceable
from langgraph.graph import StateGraph, START, END

# --------------------------------------------------
# ENV SETUP
# --------------------------------------------------

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "Langsmith_Langgraph"

api_key = os.getenv("CHAT_GROQ_KEY")

model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=api_key,
    temperature=0
)

# --------------------------------------------------
# SCHEMA (USED FOR VALIDATION, NOT TOOL CALLING)
# --------------------------------------------------

class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback")
    score: int = Field(description="Score out of 10", ge=0, le=10)

# --------------------------------------------------
# SAFE JSON INVOCATION (CRITICAL FIX)
# --------------------------------------------------

def safe_evaluate(prompt: str) -> EvaluationSchema:
    """
    Calls Groq normally (NO tool calling),
    parses JSON manually,
    validates with Pydantic.
    """
    raw = model.invoke(prompt).content.strip()

    try:
        data = json.loads(raw)
        return EvaluationSchema(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        raise RuntimeError(
            f"Model returned invalid structured output.\n\nRaw output:\n{raw}"
        ) from e

# --------------------------------------------------
# SAMPLE ESSAY
# --------------------------------------------------

essay = """India and AI Time

Now world change very fast because new tech call Artificial Intel… something (AI).
India also want become big in this AI thing. If work hard, India can go top.
But if no careful, India go back.

India have many good. We have smart student, many engine-ear, and good IT peoples.
Big company like TCS, Infosys, Wipro already use AI.
Government also do program “AI for All”.

But problem come also. Many villager no have phone or internet.
Many people lose job because AI.

India must careful and make rule.
AI must help all people, not only rich.
"""

# --------------------------------------------------
# LANGGRAPH STATE
# --------------------------------------------------

class UPSCState(TypedDict, total=False):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[List[int], operator.add]
    avg_score: float

# --------------------------------------------------
# EVALUATION NODES
# --------------------------------------------------

@traceable(name="evaluate_language", tags=["dimension:language"])
def evaluate_language(state: UPSCState):
    prompt = f"""
Return ONLY valid JSON in this exact format:
{{
  "feedback": "string",
  "score": integer (0-10)
}}

Rules:
- No extra text
- No explanation
- No markdown

Evaluate LANGUAGE quality of this essay:

{state["essay"]}
"""
    out = safe_evaluate(prompt)
    return {
        "language_feedback": out.feedback,
        "individual_scores": [out.score]
    }

@traceable(name="evaluate_analysis", tags=["dimension:analysis"])
def evaluate_analysis(state: UPSCState):
    prompt = f"""
Return ONLY valid JSON in this exact format:
{{
  "feedback": "string",
  "score": integer (0-10)
}}

Rules:
- No extra text
- No explanation
- No markdown

Evaluate DEPTH OF ANALYSIS of this essay:

{state["essay"]}
"""
    out = safe_evaluate(prompt)
    return {
        "analysis_feedback": out.feedback,
        "individual_scores": [out.score]
    }

@traceable(name="evaluate_clarity", tags=["dimension:clarity"])
def evaluate_clarity(state: UPSCState):
    prompt = f"""
Return ONLY valid JSON in this exact format:
{{
  "feedback": "string",
  "score": integer (0-10)
}}

Rules:
- No extra text
- No explanation
- No markdown

Evaluate CLARITY OF THOUGHT of this essay:

{state["essay"]}
"""
    out = safe_evaluate(prompt)
    return {
        "clarity_feedback": out.feedback,
        "individual_scores": [out.score]
    }

@traceable(name="final_evaluation", tags=["aggregate"])
def final_evaluation(state: UPSCState):
    summary_prompt = f"""
Create a concise overall feedback based on:

Language: {state.get("language_feedback", "")}
Analysis: {state.get("analysis_feedback", "")}
Clarity: {state.get("clarity_feedback", "")}
"""

    overall = model.invoke(summary_prompt).content
    scores = state.get("individual_scores", [])
    avg = round(sum(scores) / len(scores), 2) if scores else 0.0

    return {
        "overall_feedback": overall,
        "avg_score": avg
    }

# --------------------------------------------------
# BUILD LANGGRAPH
# --------------------------------------------------

graph = StateGraph(UPSCState)

graph.add_node("language", evaluate_language)
graph.add_node("analysis", evaluate_analysis)
graph.add_node("clarity", evaluate_clarity)
graph.add_node("final", final_evaluation)

graph.add_edge(START, "language")
graph.add_edge(START, "analysis")
graph.add_edge(START, "clarity")

graph.add_edge("language", "final")
graph.add_edge("analysis", "final")
graph.add_edge("clarity", "final")

graph.add_edge("final", END)

workflow = graph.compile()

# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    result = workflow.invoke(
        {"essay": essay},
        config={
            "run_name": "upsc_essay_evaluation",
            "tags": ["upsc", "essay", "langgraph"],
        }
    )

    print("\n=== RESULTS ===\n")
    print("Language Feedback:\n", result["language_feedback"], "\n")
    print("Analysis Feedback:\n", result["analysis_feedback"], "\n")
    print("Clarity Feedback:\n", result["clarity_feedback"], "\n")
    print("Overall Feedback:\n", result["overall_feedback"], "\n")
    print("Scores:", result["individual_scores"])
    print("Average Score:", result["avg_score"])