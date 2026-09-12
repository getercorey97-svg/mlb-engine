import os
import glob
import google.generativeai as genai

# 1. Authenticate with Gemini
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY is not set.")
    exit(1)

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# 2. Find all code files in the repository (Change .py to your language if needed)
files_to_scan = glob.glob("**/*.py", recursive=True)

for file_path in files_to_scan:
    # Skip checking the AI script itself
    if "gemini_fixer.py" in file_path:
        continue 
        
    print(f"Scanning and fixing {file_path}...")
    
    # Read the original code
    with open(file_path, "r", encoding="utf-8") as file:
        original_code = file.read()

    # 3. Ask Gemini to fix it
    prompt = f"""
    You are an expert software engineer and mathematician. Review the following code.
    Fix any bugs, logic errors, or math/algorithm inefficiencies.
    
    CRITICAL INSTRUCTIONS:
    - Return ONLY the raw, corrected code.
    - Do not include any explanations or conversational text.
    - Do not include markdown formatting blocks (like ```python). 
    Just give me the code so I can save it directly to the file.
    
    Here is the code:
    {original_code}
    """
    
    try:
        response = model.generate_content(prompt)
        new_code = response.text.strip()
        
        # Failsafe: Clean up markdown if the AI disobeys instructions
        if new_code.startswith("```"):
            new_code = "\n".join(new_code.split("\n")[1:])
        if new_code.endswith("```"):
            new_code = "\n".join(new_code.split("\n")[:-1])
            
        # 4. Overwrite the file with the perfected code
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(new_code)
            
        print(f"Successfully optimized {file_path}")
        
    except Exception as e:
        print(f"Failed to process {file_path}. Error: {e}")
