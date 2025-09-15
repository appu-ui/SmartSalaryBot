# graph.py - State graph module
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in environment")

# Initialize Gemini LLM via LangChain
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)

# Define state structure
class State(dict):
    name: str
    salary: float
    step: str
    messages: list
    conversation_history: list
    initial_advice: str


# Gemini helper prompt invocation
def ask_gemini(prompt: str) -> str:
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        # Fallback response if Gemini API fails
        return f"I apologize, but I'm having trouble generating personalized advice right now. Here's some general financial guidance: Consider creating a budget, building an emergency fund with 3-6 months of expenses, and investing in your retirement. Please try again later for personalized advice."


# Conversation nodes
def ask_name(state: State):
    state["step"] = "ask_name"
    state.setdefault("messages", [])
    state["messages"].append({"role": "assistant", "content": "Hi! What is your name?"})
    return state


def ask_salary(state: State):
    state["step"] = "ask_salary"
    state.setdefault("messages", [])
    state["messages"].append(
        {
            "role": "assistant",
            "content": f"Namaste {state['name']}! 👋 I’m your friendly Indian financial advisor, here to guide you with smart budgeting, savings, investments, and insurance tips. To get started, could you please share your monthly salary?"
        }
    )
    return state


def give_advice(state: State):
    state["step"] = "advice"
    state.setdefault("messages", [])
    state.setdefault("conversation_history", [])

    prompt = (
        f"You are a friendly and knowledgeable Indian financial advisor. "
        f"The user's name is {state['name']} and their monthly salary is ₹{state['salary']}. "
        "Analyze their salary and create a personalized, detailed money management plan. "
        "Break down their salary into percentages and rupee amounts for needs, wants, savings, investments, and insurance. "
        "Include actionable advice for: \n"
        "1. Budgeting (using a 50-30-20 or similar rule)\n"
        "2. Building an emergency fund\n"
        "3. Suitable insurance coverage (term life, health, etc.)\n"
        "4. Investment options (SIPs, PPF, ELSS, index funds)\n"
        "5. Short-term and long-term savings tips\n"
        "6. Practical money habits to follow in India.\n"
        "Make the advice structured, clear, and motivational also End by asking if they have any other questions."
    )

    advice = ask_gemini(prompt)

    # Store the initial advice for future reference
    state["initial_advice"] = advice

    # Add to conversation history
    state["conversation_history"].append({
        "role": "system",
        "content": f"Initial advice given for {state['name']} (salary: ${state['salary']}): {advice}"
    })

    state["messages"].append({"role": "assistant", "content": advice})

    # Move to followup step instead of ending
    state["step"] = "followup"
    return state


def handle_followup(state: State):
    """Handle follow-up questions about financial advice"""
    state["step"] = "followup"
    state.setdefault("messages", [])
    state.setdefault("conversation_history", [])

    # Get the user's follow-up question from the last message
    user_question = state.get("followup_question", "")

    # Check if user wants to end the conversation
    end_phrases = ["thanks", "thank you", "bye", "goodbye", "that's all", "no more questions",
                   "that's enough", "done", "finish", "end", "exit", "quit"]

    if any(phrase in user_question.lower() for phrase in end_phrases):
        # User wants to end the conversation
        state["step"] = "conversation_ended"
        state["messages"].append({
            "role": "assistant",
            "content": f"You're welcome, {state['name']}! I'm glad I could help with your financial planning. Best of luck with your financial goals!"
        })
        return state

    # Build context for the follow-up question
    previous_context = "\n".join([f"- {item['content']}" for item in state['conversation_history'][-3:]])

    context = (
        f"You are a friendly and knowledgeable Indian financial advisor. Here's the context:\n"
        f"- User's name: {state['name']}\n"
        f"- Monthly salary: {state['salary']} INR\n"
        f"- Initial advice given: {state.get('initial_advice', 'N/A')}\n\n"
        f"Previous conversation:\n{previous_context}\n\n"
        f"The user is asking a follow-up question: \"{user_question}\"\n\n"
        f"Provide a helpful, detailed answer that builds on the previous advice. Keep it friendly and practical. "
        f"If the question is unclear, ask for clarification. End by asking if they have any other questions."
    )

    # Get AI response for the follow-up question
    followup_response = ask_gemini(context)

    # Add to conversation history
    state["conversation_history"].append({
        "role": "user",
        "content": f"Follow-up question: {user_question}"
    })
    state["conversation_history"].append({
        "role": "assistant",
        "content": f"Follow-up response: {followup_response}"
    })

    state["messages"].append({"role": "assistant", "content": followup_response})

    # Stay in followup step for more questions
    return state


# Graph construction
def build_graph():
    graph = StateGraph(State)
    graph.add_node("ask_name", ask_name)
    graph.add_node("ask_salary", ask_salary)
    graph.add_node("advice", give_advice)
    graph.add_node("followup", handle_followup)

    graph.set_entry_point("ask_name")
    graph.add_edge("ask_name", "ask_salary")
    graph.add_edge("ask_salary", "advice")
    graph.add_edge("advice", "followup")
    graph.add_edge("followup", "followup")  # Loop back for more questions
    # The actual ending happens when handle_followup sets step to "conversation_ended"
    return graph.compile()


graph = build_graph()