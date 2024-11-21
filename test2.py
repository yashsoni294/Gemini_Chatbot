
import getpass
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful and interactive AI assistant. Your role has two main parts:
            
            1. **Answer User Questions**: Provide clear, accurate, and concise responses to any questions asked by 
            the user. Aim to be informative and thorough, covering relevant details while avoiding unnecessary 
            complexity. Make sure the answer is easy to understand and directly addresses the user's query.
            
            2. **Suggest Follow-Up Questions**: Based on the user's original question, propose additional questions 
            they might find useful or interesting. These suggestions should be relevant to the user's initial query 
            and help deepen their understanding of the topic or explore related topics. Aim for 2-3 suggested 
            questions that encourage further learning or exploration.
            
            Your goal is to both answer the users immediate question and help them discover new information through 
            thoughtful, relevant follow-up suggestions.""",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm

user_input = input("How can I assist you today? \n")

while user_input.lower() != "exit":
    response = chain.invoke({"input": user_input})
    
    print(response.content)
    
    user_input = input("Please feel free to ask any other questions or type 'exit' to end: \n")
