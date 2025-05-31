import os
import chainlit as cl
from dotenv import load_dotenv, find_dotenv

from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from fastapi import FastAPI
from pydantic import BaseModel

# Load environment variables
load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Provider and Model Setup (shared)
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider,
)

run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

# Agent Definition (unchanged)
agent1 = Agent(
    instructions="You are a helpful AI assistant that helps users find the best mobile phone to buy based on their budget, country, city, and personal preferences. Ask relevant questions such as budget range, preferred brands, usage priorities (e.g., gaming, camera, battery life), and any specific features needed. Use this information to recommend the best available phones in their location with reasoning behind each suggestion.",
    name="Support Agent"
)

# Chainlit Handlers (unchanged)
@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="I'm PhoneGenie, your smart assistant to help you find the perfect mobile phone.").send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})

    result = await Runner.run(
        agent1,
        input=history,
        run_config=run_config,
    )

    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)

    await cl.Message(content=result.final_output).send()

# FastAPI Setup (added)
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to PhoneGenie API"}

class PromptRequest(BaseModel):
    prompt: str

@app.post("/llm")
async def llm_endpoint(req: PromptRequest):
    result = await Runner.run(
        starting_agent=agent1,
        input=req.prompt,
        run_config=run_config
    )
    return {"response": result.final_output}

# Run FastAPI (only when not using Chainlit CLI)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
