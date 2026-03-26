from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    print("Error: GEMINI_API_KEY not found in .env file")
    exit(1)

def get_history_messages(history):
    return history

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    google_api_key=gemini_key
)

chat_history = []

print("Fixed Chatbot ready! Use 'exit' to quit.")

while True:
    user_input = input('You: ')
    if user_input.lower() == 'exit':
        break
    
    current_history = get_history_messages(chat_history)
    formatted_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]).invoke({"history": current_history, "input": user_input})
    
    result = model.invoke(formatted_prompt)
    
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=result.content))
    
    print("AI:", result.content)

print("\nFinal chat history:")
for msg in chat_history:
    role = "Human" if isinstance(msg, HumanMessage) else "AI"
    print(f"{role}: {msg.content}")

