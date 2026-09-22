import src.config 

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from pydantic import BaseModel, Field

#HELPER FUNCTIONS & SCHEMAS
class BinaryScoreModel(BaseModel):
    binary_score: str = Field(description="Binary score: 'yes' or 'no'")

def create_structured_llm(model: str, schema: BaseModel):
    """Initializes a Gemini LLM and forces it to conform to a specific output structure."""
    llm = ChatGoogleGenerativeAI(model=model, temperature=0)
    return llm.with_structured_output(schema)

def create_grading_prompt(system_message: str, human_template: str):
    """Helper to generate standard chat-based grading prompts."""
    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_template),
    ])

# CORE EVALUATOR & GRADERS SETUP
retrieval_evaluator_llm = create_structured_llm("gemini-3.5-flash", BinaryScoreModel)
retrieval_evaluator_prompt = create_grading_prompt(
    "You are a document retrieval evaluator responsible for checking the relevancy of a retrieved document to the user's question. \n"
    "If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n"
    "Output a binary score 'yes' or 'no'.",
    "Retrieved document: \n\n {document} \n\n User question: {question}"
)
retrieval_grader = retrieval_evaluator_prompt | retrieval_evaluator_llm


hallucination_grader_prompt = create_grading_prompt(
    "You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n"
    "Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.",
    "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"
)
hallucination_grader = hallucination_grader_prompt | create_structured_llm("gemini-3.5-flash", BinaryScoreModel)


answer_grader_prompt = create_grading_prompt(
    "You are a grader assessing whether an answer addresses / resolves a question. \n"
    "Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question.",
    "User question: \n\n {question} \n\n LLM generation: {generation}"
)
answer_grader = answer_grader_prompt | create_structured_llm("gemini-3.5-flash", BinaryScoreModel)


question_rewriter_prompt = create_grading_prompt(
    "You are a question re-writer that converts an input question to a better version optimized for vectorstore retrieval. "
    "Look at the input and try to reason about the underlying semantic intent / meaning.",
    "Here is the initial question: \n\n {question} \n Formulate an improved question."
)
question_rewriter = question_rewriter_prompt | ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0) | StrOutputParser()


# BASE RAG GENERATION CHAIN
rag_prompt = hub.pull("rlm/rag-prompt")
rag_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

rag_chain = rag_prompt | rag_llm | StrOutputParser()