import sys
import pandas as pd
import json
import time
from openai import OpenAI
from tqdm import tqdm

# --- CONFIGURATION ---
API_KEY = "openai_api_key"
MODEL_NAME = "gpt-5.2" # 
INPUT_PATH = "v2_train.parquet"
OUTPUT_PATH = "v2_train_compliance_checked.parquet"

client = OpenAI(api_key=API_KEY)

# --- REFINED PROMPT ---
SYSTEM_PROMPT = """
### ROLE
You are a Legal Compliance Auditor specializing in Copyright and Intellectual Property. Your task is to audit "Training Data Pairs" for violations of Well-Known Commercial Entity (WKCE) rights.

### OBJECTIVE
Determine if the provided content attempts to replicate, extend, or generate creative material owned by a major commercial entity (e.g., Disney, Marvel, Nintendo, Warner Bros, J.K. Rowling).

### AUDIT CRITERIA

1. FLAG (Violation) - "Derivative or Substitutive":
- Verbatim Reproductions: Quoting substantial portions (>3 sentences) of a book, script, or song lyrics.
- Creative Extension: Fan-fiction, role-playing as protected characters, or generating "new" scenes/dialogue for existing IP.
- Style Mimicry: Explicitly instructing the model to write "in the style of" a protected author or brand to create new works.

2. SAFE (Compliance) - "Descriptive or Analytical":
- Factual Reference: Mentioning characters, titles, or entities in a biographical or encyclopedic context.
- High-Level Summary: Brief plot synopses that describe the work without reproducing the creative expression.
- Critical Analysis: Reviews, academic discussions, or commentary about the cultural impact of the IP.
- Public Domain: Works published before 1928 (e.g., original Grimm's Fairytales, Shakespeare).

### OUTPUT FORMAT
Return ONLY a JSON object:
{
 "status": "Safe" | "Flagged",
 "reason": "Briefly state why (e.g., 'Creative dialogue generation for Disney-owned character')",
 "entity_detected": "Name of the WKCE or IP (e.g., 'Star Wars / Lucasfilm')",
 "confidence_score": 0.0-1.0
}
"""

def check_compliance(row):
    """
    Constructs the prompt and calls the LLM for a single row.
    """
    question = row.get('query', '')
    document = row.get('pos', '') # 'pos' column from your screenshot

    user_content = f"""
    Please analyze the following data pair:
    
    [Question]:
    {question}
    
    [Document]:
    {document}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"}, # Ensures valid JSON back
            temperature=0.0 # Keep it deterministic
        )
        
        result_json = response.choices[0].message.content
        return json.loads(result_json)

    except Exception as e:
        # Return a neutral error object so the script doesn't crash
        return {
            "status": "Error",
            "reason": str(e),
            "entity_detected": None,
            "confidence_score": 0.0
        }

# --- MAIN EXECUTION ---

def main():
    # Load Data
    print(f"Loading data from {INPUT_PATH}...")
    try:
        df = pd.read_parquet(INPUT_PATH)
        # Test on a small sample first
        df = df.head(10) 
        print(f"Loaded {len(df)} rows.")
    except FileNotFoundError:
        print("File not found. Please check the path.")
        return

    
    results = []
    print("Starting compliance check...")
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        result = check_compliance(row)
        print(result)  
        
        # Flatten the result into the row for easy saving
        results.append({
            "compliance_status": result.get("status"),
            "compliance_reason": result.get("reason"),
            "compliance_entity": result.get("entity_detected"),
            "compliance_score": result.get("confidence_score")
        })
        

    # Merge Results back to DataFrame
    results_df = pd.DataFrame(results)
    final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

    # Save
    print(f"Saving results to {OUTPUT_PATH}...")
    final_df.to_parquet(OUTPUT_PATH)
    print("Done!")

    # Quick Summary
    flagged_count = final_df[final_df['compliance_status'] == 'Flagged'].shape[0]
    print(f"\nSummary: {flagged_count} rows were flagged.")

if __name__ == "__main__":
    main()