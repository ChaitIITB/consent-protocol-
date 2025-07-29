import langchain
from langchain import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


prompt = PromptTemplate(
    input_variables=["question"],
    template="""
    We are planning to do this for the following description 

    {task.description}

    Plan the tasks accordingly and create the most optimal flow, for using multiple agents, so make threads, meaning the tasks which are dependent on each other should be done by one agent and others should be done by others. Then also specify the total number of agents required, in the first line, and only should contain the number of agents required. 

    """,
    )



llm_chain = LLMChain(
    llm=OpenAI(),
    prompt=prompt
)

