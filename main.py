from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uuid

from graph import ask_name, ask_salary, give_advice, handle_followup, State
import re

app = FastAPI()


def extract_name_from_text(text: str) -> str:
    """Extract actual name from natural language input like 'My name is John' or 'I'm Sarah'"""
    text = text.strip()

    # Common patterns for name introduction
    patterns = [
        r"my name is\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)",  # "My name is John" or "My name is John Smith"
        r"i'm\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)",  # "I'm Sarah" or "I'm Sarah Jones"
        r"i am\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)",  # "I am Mike" or "I am Mike Brown"
        r"call me\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)",  # "Call me David" or "Call me David Lee"
        r"name:\s*([a-zA-Z]+(?:\s+[a-zA-Z]+)*)",  # "Name: Lisa" or "Name: Lisa Wilson"
        r"it's\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)",  # "It's Alex" or "It's Alex Taylor"
    ]

    # Try each pattern (case insensitive)
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # If no pattern matches, assume the entire input is a name (but clean it)
    # Remove common non-name words and clean up
    cleaned = re.sub(r'\b(the|a|an|is|am|are|my|me|call|name)\b', '', text, flags=re.IGNORECASE)
    cleaned = re.sub(r'[^\w\s]', '', cleaned)  # Remove punctuation
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # Clean up spaces

    # If result is empty or too long, return the original text trimmed
    if not cleaned or len(cleaned) > 50:
        return text[:50].strip()

    return cleaned


class UserInput(BaseModel):
    name: Optional[str] = None
    salary: Optional[float] = None
    conversation_id: Optional[str] = None
    followup_question: Optional[str] = None


# In-memory conversation state (keyed by conversation ID)
conversation_states = {}


@app.post("/chat")
async def chat(user_input: UserInput):
    # Get or create conversation ID
    conversation_id = user_input.conversation_id
    if not conversation_id:
        conversation_id = str(uuid.uuid4())

    state = conversation_states.get(conversation_id, {})

    if not state:
        # Start conversation by calling ask_name
        try:
            state = State()
            state = ask_name(state)
            conversation_states[conversation_id] = state
            return {
                "messages": state.get("messages", []),
                "step": state.get("step"),
                "conversation_id": conversation_id
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error starting conversation: {str(e)}")

    step = state.get("step")

    if step == "ask_name":
        if user_input.name is None:
            # Return current messages (asking for name)
            return {
                "messages": state.get("messages", []),
                "step": step,
                "conversation_id": conversation_id
            }
        else:
            # User provided name, extract actual name from natural language input
            try:
                extracted_name = extract_name_from_text(user_input.name)
                state["name"] = extracted_name
                state = ask_salary(state)
                conversation_states[conversation_id] = state
                return {
                    "messages": state.get("messages", []),
                    "step": state.get("step"),
                    "conversation_id": conversation_id
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error processing name: {str(e)}")

    elif step == "ask_salary":
        if user_input.salary is None:
            # Return current messages (asking for salary)
            return {
                "messages": state.get("messages", []),
                "step": step,
                "conversation_id": conversation_id
            }
        else:
            # Validate salary input
            if not isinstance(user_input.salary, (int, float)) or user_input.salary <= 0:
                raise HTTPException(status_code=400, detail="Please provide a valid positive salary amount")

            # User provided salary, generate advice
            try:
                state["salary"] = user_input.salary
                state = give_advice(state)
                conversation_states[conversation_id] = state
                return {
                    "messages": state.get("messages", []),
                    "step": state.get("step"),
                    "conversation_id": conversation_id
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error generating advice: {str(e)}")

    elif step == "followup":
        if user_input.followup_question is None:
            # Return current messages (asking for follow-up or showing advice)
            return {
                "messages": state.get("messages", []),
                "step": step,
                "conversation_id": conversation_id
            }
        else:
            # User provided a follow-up question
            try:
                state["followup_question"] = user_input.followup_question
                state = handle_followup(state)
                conversation_states[conversation_id] = state

                # Check if conversation ended
                if state.get("step") == "conversation_ended":
                    final_response = {
                        "messages": state.get("messages", []),
                        "step": "conversation_ended",
                        "conversation_id": conversation_id,
                        "conversation_ended": True
                    }
                    # Clean up this conversation
                    conversation_states.pop(conversation_id, None)
                    return final_response
                else:
                    return {
                        "messages": state.get("messages", []),
                        "step": state.get("step"),
                        "conversation_id": conversation_id
                    }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error handling follow-up: {str(e)}")

    elif step == "conversation_ended":
        # This shouldn't normally happen, but handle gracefully
        final_response = {
            "messages": state.get("messages", []),
            "step": "conversation_ended",
            "conversation_id": conversation_id,
            "conversation_ended": True
        }
        # Clean up this conversation
        conversation_states.pop(conversation_id, None)
        return final_response

    else:
        # Invalid state, clean up and start over
        conversation_states.pop(conversation_id, None)
        raise HTTPException(status_code=400, detail="Invalid conversation state. Please start a new conversation.")


# Serve the HTML file
@app.get("/")
async def read_index():
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)