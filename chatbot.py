from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from chat_prompt_template import chat_with_history_template

# Load environment variables
load_dotenv()

# Get API key
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    print("Error: GEMINI_API_KEY not found in .env file")
    exit(1)

# Initialize model (✅ FIXED MODEL NAME)
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key=gemini_key
)

# Store chat history
chat_history = []

print("Chatbot ready! Use 'exit' to quit.")

while True:
    user_input = input('You: ')
    
    if user_input.lower() == 'exit':
        break

    try:
        # Format prompt with history
        formatted_prompt = chat_with_history_template.invoke({
            "history": chat_history,
            "input": user_input
        })

        # Get response
        result = model.invoke(formatted_prompt)

        # Save conversation
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=result.content))

        # Print response
        print("AI:", result.content)

    except Exception as e:
        print("Error:", e)

# Print full history
print("\nFinal chat history:")
for msg in chat_history:
    role = "Human" if isinstance(msg, HumanMessage) else "AI"
    print(f"{role}: {msg.content}")