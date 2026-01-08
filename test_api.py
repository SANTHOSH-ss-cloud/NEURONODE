import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("GEMINI_API_KEY not found in .env file")
else:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content("test")
        if response.text:
            print("API key is working properly.")
        else:
            print("API key is not working properly.")
    except Exception as e:
        print(f"An error occurred: {e}")
