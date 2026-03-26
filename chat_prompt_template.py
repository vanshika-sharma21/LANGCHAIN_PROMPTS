from langchain_core.prompts import ChatPromptTemplate

chat_with_history_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Provide responses based on the chat history."),
    ("human", "{input}"),
    ("placeholder", "{history}")
])
