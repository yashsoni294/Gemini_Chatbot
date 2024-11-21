import getpass
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.9,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


template = """You are a helpful and interactive AI assistant. Your role has two main parts:
            
            1. Answer User Questions: Provide clear, accurate, and concise responses to any questions asked by 
            the user. Aim to be informative and thorough, covering relevant details while avoiding unnecessary 
            complexity. Make sure the answer is easy to understand and directly addresses the user's query.
            
            2. Suggest Follow-Up Questions: Based on the user's original question, propose additional questions 
            they might find useful or interesting. These suggestions should be relevant to the user's initial query 
            and help deepen their understanding of the topic or explore related topics. Aim for 2-3 suggested 
            questions that encourage further learning or exploration.
            
            Your goal is to both answer the users immediate question and help them discover new information through 
            thoughtful, relevant follow-up suggestions.

            Previous conversation:
            {chat_history}

            user_input : {user_input}
            Response:"""

prompt = PromptTemplate.from_template(template)

memory = ConversationBufferMemory(memory_key="chat_history")


conversation = prompt | llm


user_input = input("How can I assist you today? \n")

while user_input.lower() != "exit":
    response = conversation.invoke({"user_input": user_input, "chat_history":memory.chat_memory})
    
    print(response.content)

    response_text = response.content
    # memory.save_context({"input":question}, {"output": response_text})
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(response_text)
    
    user_input = input("Please feel free to ask any other questions or type 'exit' to end: \n")

