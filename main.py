import sys
from src.graph import app

def run_self_rag_demo():
    print("=" * 60)
    
    config_limit = {"recursion_limit": 10}

    # Scenario 1: A query where relevant documents exist in the knowledge base
    print("\n[SCENARIO 1] Querying topic present in the database...")
    question_1 = "How to improve relationships"
    print(f"User Prompt: '{question_1}'\n")
    
    try:
        res_1 = app.invoke({"question": question_1}, config=config_limit)
        print("\n" + "=" * 40)
        print("FINAL RESULT (Scenario 1):")
        print("=" * 40)
        print(res_1.get('generation', "No response generated."))
    except Exception as e:
        print(f"\nExecution stopped: {e}")

    print("\n" + "=" * 60)

    # Scenario 2: A query completely absent from the knowledge base (Triggers Rewrite Loop)
    print("\n[SCENARIO 2] Querying topic completely absent from the database...")
    question_2 = "How to cook a bagel?"
    print(f"User Prompt: '{question_2}'\n")
    
    try:
        res_2 = app.invoke({"question": question_2}, config=config_limit)
        print("\n" + "=" * 40)
        print("FINAL RESULT (Scenario 2):")
        print("=" * 40)
        print(res_2.get('generation', "No response generated."))
    except Exception as e:
        
        print(f"\nExecution stopped as expected: {e}")
        print("The system successfully identified that no relevant context was available after multiple refinement attempts.")
    
    print("=" * 60)

if __name__ == "__main__":
    run_self_rag_demo()