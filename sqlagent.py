import os
from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import add_messages
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langchain.prompts import ChatPromptTemplate
from tools import db, llm, list_tables_tool, get_schema_tool
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

#state
class AgentState(TypedDict):
    messages: Annotated[AnyMessage, add_messages]
    question: str
    question_type: str
    tables_list: list[str]
    selected_tables: list[str]
    generated_query: str
    error_messages: str
    attempts: int
    user_feedback: str

#node
def classify(state: AgentState):
    system_prompt = """
    You are an assistant who can classify the input question from user to following
    1.Fact_Check
    2.Simulation
    3.None_of_the_above

    Here is the question
    {question}
    """
    prompt = ChatPromptTemplate.from_template(system_prompt)
    question = state['question']
    prompt = prompt.invoke(question)

    result = llm.invoke(prompt)

    return {"question_type": result.content,
            "messages": result}

#node
def list_tables(state: AgentState):
    tables_list =  ", ".join(db.get_usable_table_names())
    print(f"List of Tables: {tables_list}")
    return {"tables_list" : tables_list}

#node
def select_tables(state:AgentState):
    system_prompt = """You are an expert sql assistant who can identify the tables needed from the list of tables based on the input question
    List of available tables: {tables}
    DO NOT come up with table names. Only think from list of tables.
    If no tables identified return only NONE_IDENTIFIED
    Here is the question
    {question}
    """

    prompt = ChatPromptTemplate.from_template(system_prompt)
    prompt = prompt.invoke({"question":state['question'], "tables":state['tables_list']})

    result = llm.invoke(prompt)
    return {"selected_tables": [result.content]}

#router
def table_selection_router(state:AgentState):
    selected_tables = state['selected_tables']
    if state['selected_tables'] == "NONE_SELECTED":
        return END
    else:
        return "get_schema"

#router    
def feedback_router(state:AgentState):
    selected_tables = state['user_feedback']
    if state['user_feedback'] == "REGENERATE":
        return "generate_query"
    else:
        return END
#router    
def question_type_router(state:AgentState):
    question_type = state['question_type']
    if "Fact_Check" in question_type:
        return END
    return "list_tables"


#node
def get_schema(state: AgentState):
    tables = state['selected_tables']
    if tables:
        for table in tables:
            schema_desc =  f"\n{get_schema_tool(table)}"
    return {"messages": schema_desc}

#node
def generate_query(state:AgentState):
    system_prompt = """
            You are an database expert agent designed to generate sql query and interact with a SQL database.
            Given an input question, create a syntactically correct sqlite query.
            Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 3 results.
            You can order the results by a relevant column to return the most interesting examples in the database.
            Never query for all the columns from a specific table, only ask for the relevant columns given the question.
            You have access to tools for interacting with the database.
            Only use the below tools. Only use the information returned by the below tools to construct your final answer.
            You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
            DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
            To start you should ALWAYS look at the tables in the database to see what you can query.
            Do NOT skip this step.
            You must query the tables from schema mentioned.

            question: {question}
            schema_desc : {schema_desc}
    """
    # question = state['question']
    # schema_desc = state['messages'][-1]
    prompt = ChatPromptTemplate.from_template(system_prompt)
    prompt = prompt.invoke({"question":state['question'], "schema_desc":state['messages'][-1]})

    result = llm.invoke(prompt)
    # print(result)

    return {"generated_query": result.content,
            "messages": result}

#node
def human_review(state:AgentState):
    print("##################################")
    print(f"question: {state['question']}")
    # print(f"Please provide feedback from any of the following options 1.REGENERATE 2.CONTINUE 3.ADDITIONAL_INPUT")
    feedback = interrupt(
        {"task":"Please provide feedback from any of the following options 1.REGENERATE 2.CONTINUE 3.ADDITIONAL_INPUT",
         "Generated_Query": state['generated_query']}
    )
    print(f"$$$$$$$$$$$$$$$$: {feedback}")
    return {"user_feedback": feedback}

    

workflow = StateGraph(AgentState)
#nodes
workflow.add_node("classify", classify)
workflow.add_node("list_tables", list_tables)
workflow.add_node("select_tables", select_tables)
workflow.add_node("get_schema", get_schema)
workflow.add_node("generate_query", generate_query)
workflow.add_node("human_review", human_review)
#edges
workflow.add_edge(START, "classify")
workflow.add_conditional_edges("classify", question_type_router, path_map=[END, "list_tables"])
workflow.add_edge("list_tables","select_tables")
workflow.add_conditional_edges("select_tables", table_selection_router, path_map=[END, "get_schema"])
workflow.add_edge("get_schema","generate_query")
workflow.add_edge("generate_query","human_review")
workflow.add_conditional_edges("human_review",feedback_router, path_map=[END, "generate_query"])

config = {"configurable": {"thread_id": "1"}}
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# graph_builder.invoke({"question": "Generate a sql for moving employees from one designation to another and calculate savings"})
# graph.invoke({"question": "Generate a sql for the scenario : increasing the "}, config)
for chunk in graph.stream({"question": "Generate a sql for the scenario: get all the employees from india"}, config):
    print(chunk)
human_feedback = input("Please provide feedback from any of the following options 1.REGENERATE 2.CONTINUE 3.ADDITIONAL_INPUT")
graph.invoke(Command(resume=human_feedback), config=config)


from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

display(
    Image(
        graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
)