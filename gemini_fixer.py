import os
import glob
from google import genai
from google.genai import errors

# 1. Authenticate with Gemini
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY environment variable is not set.")
    exit(1)

# Initialize the client using the new SDK format
client = genai.Client(api_key=api_key)

# The updated model based on the GitHub Actions error log
MODEL_ID = 'gemini-3.6-flash'

# 2. Find all python code files in the repository
# (If your MLB engine uses a different language, change "*.py" to "*.js", etc.)
files_to_scan = glob.glob("**/*.py", recursive=True)

for file_path in files_to_scan:
    # Skip checking this AI script itself or hidden virtual environments
    if "gemini_fixer.py" in file_path or ".venv" in file_path or ".git" in file_path:
        continue 
        
    print(f"Scanning and fixing {file_path}...")
    
    try:
        # Read the original code
        with open(file_path, "r", encoding="utf-8") as file:
            original_code = file.read()

        # Skip empty files
        if not original_code.strip():
            print(f"Skipping {file_path} (Empty file)")
            continue

        # 3. Ask Gemini to fix it
        prompt = f"""
        You are an expert software engineer and mathematician. Review the following code.
        Fix any bugs, logic errors, or math/algorithm inefficiencies.
        
        CRITICAL INSTRUCTIONS:
        - Return ONLY the raw, corrected code.
        - Do not include any explanations or conversational text.
        - Do not include markdown formatting blocks (like ```python). 
        Just give me the exact code so I can save it directly to the file.
        
        Here is the code:
        {original_code}
        """
        
        # Make the API call to Gemini
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )
        
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
        
    except errors.ClientError as e:
        print(f"Gemini API Error on {file_path}: {e}")
        exit(1) # Stop the script if the API fails
    except Exception as e:
        print(f"Failed to process {file_path}. Error: {e}")
