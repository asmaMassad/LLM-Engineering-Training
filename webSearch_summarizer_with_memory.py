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
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# Configure Tavily
tavily_client = tavily.TavilyClient(api_key=TAVILY_API_KEY)

# ---------------------------------------------
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

# --------------------------------------------------------------------
# Prompt Template
prompt = PromptTemplate(
    input_variables=["question", "chat_history", "web_summary"],
    template=(
        "You are an expert assistant. Use the following information to answer the question step by step.\n\n"
        "Chat History:\n{chat_history}\n\n"
        "Question: {question}\n\n"
        "Web Summary (for internal use only): {web_summary}"
    )
)


def format_prompt(inputs):
    return prompt.format(
        question=inputs["question"],
        chat_history=inputs.get("chat_history", ""),
        web_summary=inputs.get("web_summary", "")
    )

prompt_runnable = RunnableLambda(format_prompt)
# -----------------------------------
# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --------------------------------------------------------
# Helper Functions
def filter_web_results(question, web_results, threshold=0.2):
    question_tokens = set(question.lower().split())
    relevant = []
    for snippet in web_results:
        snippet_lower = snippet.lower()
        overlap = len(question_tokens.intersection(snippet_lower.split())) / max(len(question_tokens), 1)
        if overlap >= threshold:
            relevant.append(snippet)
    return relevant

def summarize_web_results(inputs):
    question = inputs.get("question", "")
    chat_history = inputs.get("chat_history", "")
    web_results = inputs.get("web_results", [])

    if not web_results:
        return {"question": question, "chat_history": chat_history, "web_summary": ""}

    summary_prompt = (
        "Summarize the following web search results in 2-3 concise sentences, keeping key facts only:\n\n"
        + "\n".join(web_results)
    )
    llm = GeminiLLM()
    summary_text = llm._call(summary_prompt)

    
    return {
        "question": question,
        "chat_history": chat_history,
        "web_summary": summary_text
    }


summarizer_runnable = RunnableLambda(summarize_web_results)

def attach_sources(answer, web_results):
    if web_results:
        sources_text = "\nSources:\n" + "\n".join(web_results)
        return f"{answer}\n\n{sources_text}"
    return answer

# --------------------------------------

llm = GeminiLLM()
llm_runnable = RunnableLambda(lambda prompt_text: llm._call(prompt_text))
qa_chain = RunnableSequence(summarizer_runnable, prompt_runnable, llm_runnable)

# ----------------------------------------------------------------
#  Chat Loop
async def chat_loop():
    while True:
        question = input("Enter your question: ")
        if question.lower() in ["quit", "exit"]:
            break

        # Tavily Web Search 
        web_results = tavily_client.search(query=question, limit=3) 
        web_results = filter_web_results(question, web_results)

        # Get chat history from memory 
        chat_history = memory.load_memory_variables({}).get("chat_history", "")

        #Prepare inputs 
        inputs = {
            "question": question,
            "chat_history": chat_history,
            "web_results": web_results
        }

        # Run the chain
        answer = qa_chain.invoke(inputs)
        answer_with_sources = attach_sources(answer, web_results)

        #Update memory 
        memory.save_context({"input": question}, {"output": answer_with_sources})

        #Display answer 
        print("\nAnswer:\n", answer_with_sources)
        print("-" * 50)

# -------------------------
# Run
if __name__ == "__main__":
    asyncio.run(chat_loop())
