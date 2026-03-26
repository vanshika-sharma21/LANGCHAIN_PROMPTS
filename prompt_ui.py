import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load env variables
load_dotenv()

st.set_page_config(page_title="Research Tool", page_icon="🔬")

st.header("🔬 Research Tool")

# ------------------ UI Inputs ------------------ #

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

template = load_prompt('template.json')

model_choice = st.radio(
    "Choose model:",
    ("HuggingFace (DialoGPT)", "Google Gemini 2.5 Flash")
)

# ------------------ Button Action ------------------ #

if st.button("Summarize"):

    # Fill template
    prompt = template.format(
        paper_input=paper_input,
        style_input=style_input,
        length_input=length_input
    )

    if model_choice == "HuggingFace (DialoGPT)":
        model_id = "microsoft/DialoGPT-medium"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_id)

        llm = ChatHuggingFace(llm=model, tokenizer=tokenizer)

        result = llm.invoke([HumanMessage(content=prompt)])

    else:
        gemini_key = os.getenv("GEMINI_API_KEY")

        if not gemini_key:
            st.error("GEMINI_API_KEY not found in .env")
            st.stop()

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=gemini_key
        )

        result = llm.invoke([HumanMessage(content=prompt)])

    # Output
    st.success("Response:")
    st.write(result.content)
