import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage
from langchain.memory import ChatMessageHistory


def configure_genai():
    """Load environment variables and configure Google Generative AI."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.9,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )


def create_prompt_template():
    """Create and return the chat prompt template."""
    template = """
    You are a helpful and interactive AI assistant. Your role has two main parts:

    1. Answer User Questions: Provide clear, accurate, and concise responses to any questions asked by 
       the user. Aim to be informative and thorough, covering relevant details while avoiding unnecessary 
       complexity. Ensure the answer is easy to understand and directly addresses the user's query.

    2. Suggest Follow-Up Questions: Based on the user's original question, propose additional questions 
       they might find useful or interesting. These suggestions should be relevant to the user's initial query 
       and help deepen their understanding of the topic or explore related topics. Aim for 2-3 suggested 
       questions that encourage further learning or exploration.

    Your goal is to both answer the user's immediate question and help them discover new information through 
    thoughtful, relevant follow-up suggestions.

    Previous conversation:
    {chat_history}

    user_input: {user_input}
    Response:
    """
    return ChatPromptTemplate.from_template(template)


def main():
    """Main function to handle the interactive AI assistant conversation."""
    # Configure the LLM and set up the prompt
    llm = configure_genai()
    prompt = create_prompt_template()

    # Initialize chat history
    chat_history = ChatMessageHistory()

    # Interactive conversation loop
    print("Welcome to the AI Assistant! Type 'exit' to end the conversation.")
    user_input = input("How can I assist you today?\n")

    while user_input.lower() != "exit":
        # Prepare the chat history
        chat_history_str = "\n".join(
            f"{msg.type.capitalize()}: {msg.content}"
            for msg in chat_history.messages
        )

        # Format input for the prompt
        formatted_input = prompt.format(
            chat_history=chat_history_str,
            user_input=user_input,
        )

        # Generate response
        response = llm.predict(formatted_input)

        # Display the response
        print("\nAI Assistant:", response)

        # Update chat history
        chat_history.add_message(HumanMessage(content=user_input))
        chat_history.add_message(AIMessage(content=response))

        # Ask for the next input
        user_input = input("\nPlease feel free to ask any other questions or type 'exit' to end:\n")

    print("Thank you for using the AI Assistant. Goodbye!")


if __name__ == "__main__":
    main()
