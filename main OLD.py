import os
import chainlit as cl

from dotenv import load_dotenv, find_dotenv
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner



load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")


# step 1 :Provider
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# step2:  
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client = provider,
)

# config: define at run level
run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True

)


# step 3: agent
agent1=Agent(
    instructions= "You are a helpful AI assistant that helps users find the best mobile phone to buy based on their budget, country, city, and personal preferences. Ask relevant questions such as budget range, preferred brands, usage priorities (e.g., gaming, camera, battery life), and any specific features needed. Use this information to recommend the best available phones in their location with reasoning behind each suggestion.",
    name="Support Agent"
)


@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content=" I'm PhoneGenie, your smart assistant to help you find the perfect mobile phone.").send()




@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")

    # standard Interface
    history.append({"role": "user","content": message.content})
    result = await Runner.run(
        agent1,
        input=history,
        run_config=run_config,
    )
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)

    await cl.Message(content=result.final_output).send()



    