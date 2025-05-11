import os
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from get_vector_db import get_vector_db

LLM_MODEL = os.getenv('LLM_MODEL')
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

def get_prompt():
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI assistant. Generate five reworded versions of the user question to improve document retrieval. Original question: {question}""",
    )
    template = "Answer the question based ONLY on this context:\n{context}\nQuestion: {question}"
    prompt = ChatPromptTemplate.from_template(template)
    return QUERY_PROMPT, prompt

def query(input):
    if input:
        llm = ChatOllama(model=LLM_MODEL)

        db = get_vector_db()
        
        QUERY_PROMPT, prompt = get_prompt()

        # Tuning strategies: Enhance Retrieval
        # - fetch more relevant document chunks
        #
        # Option 1: Increase the number of retrieved documents
        # Increase the number of retrieved documents by setting search_kwargs
        #retriever = db.as_retriever(search_kwargs={"k": 5})  # Fetch 5 chunks instead of default
        #
        # Option 2: Keep the MultiQueryRetriever but increase results
        # Or if you want to keep the MultiQueryRetriever but increase results:
        retriever = MultiQueryRetriever.from_llm(
            retriever=db.as_retriever(search_kwargs={"k": 5}),
            llm=llm,
            prompt=QUERY_PROMPT
        )

        retriever = MultiQueryRetriever.from_llm(db.as_retriever(), llm,        prompt=QUERY_PROMPT)
        
        chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
        
        return chain.invoke(input)
    
    return None
