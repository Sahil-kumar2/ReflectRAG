from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START

from src.retriever import retriever
from src.components import (
    rag_chain,
    retrieval_grader,
    hallucination_grader,
    answer_grader,
    question_rewriter
)

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        question: the user's current question
        generation: the LLM's generated response
        documents: list of retrieved documents
    """
    question: str
    generation: str
    documents: List[str]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



def retrieve(state):
    """Retrieves documents relevant to the question."""
    print("---NODE: RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state):
    """Filters retrieved documents based on relevance."""
    print("---NODE: CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    for d in documents:
        grade = retrieval_grader.invoke({
            "question": question, 
            "document": d.page_content
        }).binary_score
        
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            
    return {"documents": filtered_docs, "question": question}

def generate(state):
    """Generates an answer using the formatted context."""
    print("---NODE: GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    generation = rag_chain.invoke({
        "context": format_docs(documents), 
        "question": question
    })
    
    return {"documents": documents, "question": question, "generation": generation}

def transform_query(state):
    """Rewrites the question to optimize for vectorstore retrieval."""
    print("---NODE: TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def decide_to_generate(state):
    """Decides whether to generate an answer or rewrite the query."""
    print("---EDGE: ASSESS GRADED DOCUMENTS---")
    
    if not state["documents"]:
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT. TRANSFORM QUERY---")
        return "transform_query"
        
    print("---DECISION: DOCUMENTS RELEVANT. PROCEED TO GENERATE---")
    return "generate"

def grade_generation_v_documents_and_question(state):
    """Checks for hallucinations and ensures the question is answered."""
    print("---EDGE: CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    hallucination_score = hallucination_grader.invoke({
        "documents": format_docs(documents), 
        "generation": generation
    }).binary_score
    
    if hallucination_score == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        
        answer_score = answer_grader.invoke({
            "question": question, 
            "generation": generation
        }).binary_score
        
        if answer_score == "yes":
            print("---DECISION: GENERATION FULLY ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION. TRANSFORM QUERY---")
            return "not useful"
    else:
        print("---DECISION: GENERATION CONTAINS HALLUCINATIONS. RE-TRY GENERATION---")
        return "not supported"


workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve) 
workflow.add_node("grade_documents", grade_documents) 
workflow.add_node("generate", generate) 
workflow.add_node("transform_query", transform_query) 

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents", 
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)

workflow.add_edge("transform_query", "retrieve")

workflow.add_conditional_edges(
    "generate", 
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

app = workflow.compile()