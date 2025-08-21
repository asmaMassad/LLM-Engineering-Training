import os
import asyncio
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.language_models import LLM
import google.generativeai as genai
from dotenv import load_dotenv
import tavily

load_dotenv()  
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Configure Tavily
tavily_client = tavily.TavilyClient(api_key=TAVILY_API_KEY)

# ----------------------------
# Custom Gemini LLM Wrapper
class GeminiLLM(LLM):
    model_name: str = "gemini-2.5-flash"

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, stop=None) -> str:
        model = genai.GenerativeModel(self.model_name)
        resp = model.generate_content(prompt)
        return resp.text

    async def _acall(self, prompt: str, stop=None) -> str:
        return self._call(prompt, stop=stop)



# ---------------------------------------
# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -------------------------------------------------
# Helper Functions
def summarize_web_results(inputs):
    """Include web results in a way that preserves dates and URLs."""
    question = inputs.get("question", "")
    chat_history = inputs.get("chat_history", "")
    web_results = inputs.get("web_results", [])

    if not web_results or "results" not in web_results:
        summary_text = "No recent web results found."
        print("Web Summary:\n", summary_text)
        return {
            "question": question,
            "chat_history": chat_history,
            "web_summary": summary_text
        }

    web_texts = []
    for r in web_results.get("results", []):
        title = r.get("title", "No Title")
        url = r.get("url", "No URL")
        content = r.get("content", "")
        web_texts.append(f"Title: {title}\nURL: {url}\nContent: {content}")

    summary_text = "\n\n".join(web_texts)
    #print("Web Summary:+++++++++++++++++++++++++++++++++++++\n", summary_text)

    return {
        "question": question,
        "chat_history": chat_history,
        "web_summary": summary_text
    }


summarizer_runnable = RunnableLambda(summarize_web_results)

def attach_sources(answer, web_results):
    """Attach only real article URLs as sources."""
    if web_results and "results" in web_results:
        sources_text = "\nSources:\n" + "\n".join(
            [r.get("url", "No URL") for r in web_results["results"]]
        )
        return f"{answer}\n\n{sources_text}"
    return answer

# ------------------------------------------------
# Prompt Template 
prompt = PromptTemplate(
    input_variables=["question", "chat_history", "web_summary"],
    template=(
        "You are an expert assistant with access to both memory and web results.\n\n"
        "Follow this reasoning process:\n"
        "1. Review the conversation history.\n"
        "2. Consider the current question.\n"
        "3. Decide if the web search summary is useful or if memory/internal knowledge is enough.\n"
        "4. Use the date information from web summaries to reason about the current year and temporal context.\n"
        "5. Think step by step before answering.\n\n"
        "Chat History:\n{chat_history}\n\n"
        "Question: {question}\n\n"
        "Web Search Summary (may be empty if not useful):\n{web_summary}\n\n"
        "Now provide the best possible answer with clear reasoning."
    )
)

def format_prompt(inputs):
    
    return prompt.format(
        question=inputs["question"],
        chat_history=inputs.get("chat_history", ""),
        web_summary=inputs.get("web_summary", "No recent web results found.")
        
    )

prompt_runnable = RunnableLambda(format_prompt)

llm = GeminiLLM()
llm_runnable = RunnableLambda(lambda inputs: llm._call(inputs["prompt_text"]))

def build_prompt(inputs):
    inputs["prompt_text"] = format_prompt(inputs)
    return inputs

qa_chain = RunnableSequence(
    summarizer_runnable,     
    RunnableLambda(build_prompt), 
    llm_runnable              
)

# ---------------------------------------
#  Chat Loop
async def chat_loop():
    while True:
        question = input("Enter your question: ")
        if question.lower() in ["quit", "exit"]:
            break

        # Tavily Web Search 
        web_results = tavily_client.search(query=question, limit=3)
        #print("web results-------------\n", web_results)
        # Get chat history from memory 
        chat_history = memory.load_memory_variables({}).get("chat_history", "")

        # Prepare inputs
        inputs = {
            "question": question,
            "chat_history": chat_history,
            "web_results": web_results
        }

        #run the chain 
        answer = qa_chain.invoke(inputs)
        answer_with_sources = attach_sources(answer, web_results)

        #  Update memory
        memory.save_context({"input": question}, {"output": answer_with_sources})

        #Display answer
        print("\nAnswer:\n", answer_with_sources)
        print("-" * 50)

# -----------------------------------------------------------
# Run
if __name__ == "__main__":
    asyncio.run(chat_loop())
