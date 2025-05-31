import gradio as gr
import asyncio
import os
from dotenv import load_dotenv, find_dotenv

from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner

# Load environment variables
load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Setup provider and model
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
    tracing_disabled=True,
)

agent1 = Agent(
    instructions="You are a helpful AI assistant that helps users find the best mobile phone to buy based on their budget, country, city, and personal preferences. Ask relevant questions such as budget range, preferred brands, usage priorities (e.g., gaming, camera, battery life), and any specific features needed. Use this information to recommend the best available phones in their location with reasoning behind each suggestion.",
    name="Support Agent",
)

def chat_with_agent(user_input):
    # Run async Runner.run synchronously
    result = asyncio.run(
        Runner.run(agent1, input=user_input, run_config=run_config)
    )
    return result.final_output

iface = gr.Interface(fn=chat_with_agent, inputs="text", outputs="text", title="PhoneGenie AI Assistant")

iface.launch(server_name="0.0.0.0", server_port=7860)
