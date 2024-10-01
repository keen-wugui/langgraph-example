# %%  special for the notebook - remove in the final version

# add to path the parent 
import sys
sys.path.append('../')

# load the environment variables
from dotenv import load_dotenv
load_dotenv()

# %% 
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader

# LCEL docs
url = "https://python.langchain.com/docs/concepts/#langchain-expression-language-lcel"
loader = RecursiveUrlLoader(
    url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
)
docs = loader.load()

# Sort the list based on the URLs and get the text
d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc.page_content for doc in d_reversed]
)
# %%
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
# NOTE: you must use langchain-core >= 0.3 with Pydantic v2
from pydantic import BaseModel, Field

### OpenAI

# Grader prompt
code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a coding assistant with expertise in LCEL, LangChain expression language. \n 
    Here is a full set of LCEL documentation:  \n ------- \n  {context} \n ------- \n Answer the user 
    question based on the above provided documentation. Ensure any code you provide can be executed \n 
    with all required imports and variables defined. Structure your answer with a description of the code solution. \n
    Then list the imports. And finally list the functioning code block. Here is the user question:""",
        ),
        ("placeholder", "{messages}"),
    ]
)


# Data model
class code(BaseModel):
    """Schema for code solutions to questions about LCEL."""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")


expt_llm = "gpt-4o-mini"
llm = ChatOpenAI(temperature=0, model=expt_llm)
code_gen_chain_oai = code_gen_prompt | llm.with_structured_output(code)
question = "How do I build a RAG chain in LCEL?"
solution = code_gen_chain_oai.invoke({"context":concatenated_content,"messages":[("user",question)]})
solution
# %%
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# OpenAI Model Configuration
openai_model = "gpt-4"  # Replace with desired OpenAI model

# Define the prompt template for OpenAI
code_gen_prompt_openai = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """<instructions> You are a coding assistant with expertise in LCEL, LangChain expression language. \n 
            Here is the LCEL documentation:  \n ------- \n  {context} \n ------- \n Answer the user question based on the \n 
            above provided documentation. Ensure any code you provide can be executed with all required imports and variables \n
            defined. Structure your answer: 1) a prefix describing the code solution, 2) the imports, 3) the functioning code block. \n
            Invoke the code tool to structure the output correctly. </instructions> \n Here is the user question:""",
        ),
        ("placeholder", "{messages}"),
    ]
)

# Initialize the LLM with OpenAI
llm = ChatOpenAI(model=openai_model)

# Optional: Check for errors in case tool use is flaky
def check_openai_output(tool_output):
    """Check for parse error or failure to call the tool"""

    # Error with parsing
    if "parsing_error" in tool_output:
        # Report back output and parsing errors
        print("Parsing error!")
        raw_output = tool_output["raw"]
        error = tool_output["parsing_error"]
        raise ValueError(
            f"Error parsing your output! Be sure to invoke the tool. Output: {raw_output}. \n Parse error: {error}"
        )

    # Tool was not invoked
    if "parsed" not in tool_output:
        print("Failed to invoke tool!")
        raise ValueError(
            "You did not use the provided tool! Be sure to invoke the tool to structure the output."
        )
    return tool_output

# Chain with output check
def build_code_chain_openai_raw():
    structured_llm_openai = LLMChain(
        prompt=code_gen_prompt_openai, llm=llm
    )
    return structured_llm_openai

# Error handling fallback
def insert_errors(inputs):
    """Insert errors for tool parsing in the messages"""

    # Get errors safely (if the error key doesn't exist, add a default message)
    error = inputs.get("error", "Unknown error occurred")
    messages = inputs["messages"]
    messages.append(
        (
            "assistant",
            f"Retry. You are required to fix the parsing errors: {error} \n\n You must invoke the provided tool.",
        )
    )
    return {
        "messages": messages,
        "context": inputs["context"],
        "error": error  # Ensure the error key is present
    }

# Fallback chain and retry logic
N = 3  # Max re-tries

def build_fallback_chain():
    """This will handle retries and error management"""
    code_chain = build_code_chain_openai_raw()

    def fallback_handler(inputs):
        # First, insert errors into the messages
        fallback_inputs = insert_errors(inputs)
        # Then re-run the chain
        return code_chain.run(fallback_inputs)
    
    # Use fallback handler for retries
    for _ in range(N):
        try:
            return fallback_handler
        except ValueError as e:
            print(f"Retry failed with error: {e}")
            continue

# Parsing the output
def parse_output(solution):
    """When we add 'include_raw=True' to structured output,
    it will return a dict w 'raw', 'parsed', 'parsing_error'."""
    # Check if the solution is in dictionary form
    if isinstance(solution, dict):
        return solution.get("parsed", solution)
    # If it's not structured, return as is
    return solution

# Main code generation chain with retries and fallback
def code_gen_chain(inputs):
    code_chain = build_code_chain_openai_raw()
    fallback_chain = build_fallback_chain()

    try:
        # Try generating code
        result = code_chain.run(inputs)
        return parse_output(result)
    except Exception as e:
        # If something goes wrong, use the fallback
        print(f"Error encountered: {e}. Running fallback chain.")
        result = fallback_chain(inputs)
        return parse_output(result)

# Example input
inputs = {
    "messages": [("user", "Please generate code to solve problem X")],
    "context": "LCEL documentation goes here..."
}

# Execute the code generation
generated_code = code_gen_chain(inputs)
print(generated_code)
# %%
# Test
question = "How do I build a RAG chain in LCEL?"
solution = code_gen_chain.invoke(
    {"context": concatenated_content, "messages": [("user", question)]}
)
solution
# %%
