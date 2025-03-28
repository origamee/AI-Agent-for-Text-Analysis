import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, END # provides foundation for agent structure
from langchain.prompts import PromptTemplate # This and ChatOpenAI gives us tools to interact with AI models
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

#Load env variables (to fetch API key)
load_dotenv()

#Designing agent's memory
class State(TypedDict): 
    text: str 
    classification: str 
    entities: List[str] 
    summary: str

#Initialize our LLM
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)

#Create an actual skill or a capability that an agent will use to perform a particular task
'''Notice how we use a prompt template to give clear, consistent instructions to our AI model. This function takes in our current state (which includes the text we're analyzing) and returns its classification.'''
def classification_node(state: State):
    ''' Classify the text into one of the categories: News, Blog, Research, or Other '''
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Classify the following text into one of the categories: News, Blog, Research, or Other.\n\nText:{text}\n\nCategory:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    classification = llm.invoke([message]).content.strip()
    return {"classification": classification}

#Create entity extraction capability
''' This function is like a careful reader who identifies and remembers all the important names, organizations, and places mentioned in the text. It processes the text and returns a list of these key entities. '''
def entity_extraction_node(state: State):
    ''' Extract all the entities (Person, Organization, Location) from the text      '''
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Extract all the entities (Person, Organization, Location) from the following text. Provide the result as a comma-separated list.\n\nText:{text}\n\nEntities:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    entities = llm.invoke([message]).content.strip().split(", ")
    return {"entities": entities}

#Implement summarization capability
def summarization_node(state: State):
    ''' Summarize the text in one short sentence '''
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in one short sentence.\n\nText:{text}\n\nSummary:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    summary = llm.invoke([message]).content.strip()
    return {"summary": summary}

#Now connect these capabilities into a co-ordinated system i.e. start -> classification -> entity_extraction -> summarization -> exit
workflow = StateGraph(State)

# Add nodes to the graph
workflow.add_node("classification_node", classification_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("summarization", summarization_node)

# Add edges to the graph
workflow.set_entry_point("classification_node") # Set the entry point of the graph
workflow.add_edge("classification_node", "entity_extraction")
workflow.add_edge("entity_extraction", "summarization")
workflow.add_edge("summarization", END)

# Compile the graph
app = workflow.compile()


#ACTION
sample_text1 = """
A comprehensive guide to implementing secure authentication mechanisms in modern web applications. This document outlines best practices for password hashing, two-factor authentication, and protection against common security vulnerabilities such as SQL injection and cross-site scripting (XSS)."""

# News Article Sample
sample_text2 = """
Artificial intelligence is evolving rapidly, but are we prioritizing ethics enough? As AI systems become more integrated into our daily lives, the risks of bias, misinformation, and misuse grow. Companies must take responsibility for ensuring fairness and transparency in AI models. Governments, too, should enact stricter regulations to prevent potential harm.

If we fail to act now, we risk creating systems that reinforce discrimination rather than eliminate it. It’s time for a global conversation about responsible AI development before it’s too late
# Personal Narrative Sample """

sample_text3 = """
The warm aroma of freshly baked bread, the rhythmic chopping of vegetables, and the soft humming of old folk songs – these are the sensory memories that transport me back to my grandmother's kitchen. Her culinary wisdom was more than just recipes; it was a form of love passed down through generations."""

# Test each sample
for sample_text in [sample_text1, sample_text2, sample_text3]:
    state_input = {"text": sample_text}
    result = app.invoke(state_input)
    
    print("Text Sample:", sample_text[:100] + "...")
    print("Classification:", result["classification"])
    print("Entities:", result["entities"])
    print("Summary:", result["summary"])
    print("\n---\n")

'''
'''
sample_text = """
OpenAI has announced the GPT-4 model, which is a large multimodal model that exhibits human-level performance on various professional benchmarks. It is developed to improve the alignment and safety of AI systems.
additionally, the model is designed to be more efficient and scalable than its predecessor, GPT-3. The GPT-4 model is expected to be released in the coming months and will be available to the public for research and development purposes.
"""

sample_text= """
The warm aroma of freshly baked bread, the rhythmic chopping of vegetables, and the soft humming of old folk songs – these are the sensory memories that transport me back to my grandmother's kitchen. Her culinary wisdom was more than just recipes; it was a form of love passed down through generations."""
'''
sample_text = """
A comprehensive guide to implementing secure authentication mechanisms in modern web applications. This document outlines best practices for password hashing, two-factor authentication, and protection against common security vulnerabilities such as SQL injection and cross-site scripting (XSS)."""

state_input = {"text": sample_text}
result = app.invoke(state_input)

print("Classification:", result["classification"])
print("\nEntities:", result["entities"])
print("\nSummary:", result["summary"])
'''
