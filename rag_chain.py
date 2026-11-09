import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph,START,END
from typing import Annotated,Sequence,TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,ToolMessage
from langgraph.prebuilt import ToolNode,tools_condition

class State(TypedDict):
    messages : Annotated[Sequence[BaseMessage],add_messages]


def build_rag_workflow(vector_store):
    llm = ChatGoogleGenerativeAI(
        model='gemini-3.5-flash',
        temperature=0
    )

    retriever = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={'k':4}
    )

    @tool
    def hr_retrieval_tool(query:str):
        """
            Retrieve relevant information from company HR policies and employee HRIS data.
            Use this to answer questions about leave, PTO balances, and conduct rules.
        """

        result = retriever.invoke(query)
        context = [doc.page_content for doc in result]
        metadata = [doc.metadata for doc in result]

        return{
            'query':query,
            'context':context,
            'metadata':metadata
        }


    tools=[hr_retrieval_tool]

    llm_with_tools = llm.bind_tools(tools)

    def chat_node(state : State):
        messages = state['messages']
        response = llm_with_tools.invoke(messages)
        return {'messages':[response]}


    tool_node = ToolNode(tools)

    graph=StateGraph(State)

    graph.add_node('chatNode',chat_node)
    graph.add_node('tools',tool_node)

    graph.add_edge(START,'chatNode')
    graph.add_conditional_edges('chatNode',tools_condition)
    graph.add_edge('tools','chatNode')

    return graph.compile()