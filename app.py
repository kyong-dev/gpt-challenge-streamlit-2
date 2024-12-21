import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_unstructured import UnstructuredLoader
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser
import os
import json

CACHE_DIR = "./.cache"

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

llm = None

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

# @st.cache_resource(show_spinner="Making quiz...")
def run_quiz_chain(_docs, level):
    questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
                You are a helpful assistant that is role playing as a teacher.
                    
                Based ONLY on the following context make 5 (TEN) questions minimum to test the user's knowledge about the text.
                
                Each question should have 2 answers, three of them must be incorrect and one should be correct.
                The level is {level}

                Use (o) to signal the correct answer.
    
                Question examples:
                
                Question: What is the color of the ocean? ({level})
                Answers: Red|Yellow|Green|Blue(o)
                    
                Question: What is the capital or Georgia? ({level})
                Answers: Baku|Tbilisi(o)|Manila|Beirut
                    
                Question: When was Avatar released? ({level})
                Answers: 2007|2001|2009(o)|1998
                    
                Question: Who was Julius Caesar? ({level})
                Answers: A Roman Emperor(o)|Painter|Actor|Model

                Your turn!
                    
                Context: {{context}}
            """,
            )
        ]
    )

    questions_chain = {"context": format_docs} | questions_prompt | llm

    formatting_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                    You are a powerful formatting algorithm.
                    
                    You format exam questions into JSON format.
                    Answers with (o) are the correct ones.
                    
                    Example Input:

                    Question: What is the color of the ocean?
                    Answers: Red|Yellow|Green|Blue(o)
                        
                    Question: What is the capital or Georgia?
                    Answers: Baku|Tbilisi(o)|Manila|Beirut
                        
                    Question: When was Avatar released?
                    Answers: 2007|2001|2009(o)|1998
                        
                    Question: Who was Julius Caesar?
                    Answers: A Roman Emperor(o)|Painter|Actor|Model
                    

                    Questions: {context}

                """,
            )
        ]
    )

    formatting_chain = formatting_prompt | llm 

    class JsonOutputParser(BaseOutputParser):
        def parse(self, text):
            print("Raw response:", text)  # Debugging: Print the raw response
            text = text.replace("```", "").replace("json", "")
            return json.loads(text)

    # output_parser = JsonOutputParser()
    chain = questions_chain | formatting_chain
    response = chain.invoke(_docs)
    if 'function_call' in response.additional_kwargs:
        return json.loads(response.additional_kwargs['function_call']['arguments'].replace("```", "").replace("json", ""))
    else:
        return json.loads(response.content.replace("```", "").replace("json", ""))


# @st.cache_resource(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

api_key = None
processing = False

with st.sidebar:
    docs = None
    api_key = st.text_input("Write down your OpenAI key", placeholder="sk-proj-NDE*********")

    choice = st.selectbox(
        "Level",
        (
            None,
            "Hard",
            "Normal",
            "Easy"
        )
    )
    topic = st.text_input("Search Wikipedia...")
    search_button = st.button("Search")

    st.write("<a href='https://github.com/kyong-dev/gpt-challenge-streamlit-2'>https://github.com/kyong-dev/gpt-challenge-streamlit-2</a>", unsafe_allow_html=True)

    if search_button and topic and api_key and choice and not processing:
        processing = True
        llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-4o",
            streaming=True,
            api_key=api_key,
            callbacks=[StreamingStdOutCallbackHandler()],
        ).bind(
            function_call="auto",
            functions=[
                function
            ],
        )
        docs = wiki_search(topic)
        topic = None
    else:
        st.error("Please write down a topic and your OpenAI key.")

if choice:
    st.write(f"Level: {choice}")

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles to test your knowledge and help you study.
                
    Get started by setting up the level you want and searching on Wikipedia in the sidebar.
    """
    )
else:
    response = run_quiz_chain(docs, choice)
    with st.form("questions_form"):
        correct_count = 0
        questions = response["questions"]
        for idx, question in enumerate(questions):
            st.write(question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
                key=f"question_{idx}"
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
                correct_count += 1
            elif value is not None:
                st.error("Wrong!")
        processing = False
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write("Form submitted!")
        
        if correct_count == len(response["questions"]):
            st.balloons()