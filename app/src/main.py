from .pdf import Pdf
from .llm import Llm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List


# Initialize the Profile and Llm classes
pdf = Pdf()
llm = Llm()

# Initialize FastAPI app
app = FastAPI()

# Initialize the LLM
llm = Llm()

# Define request and response models
class QueryRequest(BaseModel):
    query_text: str

class Metadata(BaseModel):
    doc_name: str
    page: int

class QueryResponse(BaseModel):
    answer: str
    metadata: List[Metadata]
    paragraph: List[List[str]]
    
    
@app.post("/api/query", response_model=QueryResponse)
def query_llm(request: QueryRequest):
    """
    Endpoint to handle user queries and return the LLM-generated response based on the provided context.

    Args:
        request (QueryRequest): The user's query in the form of a `query_text`.

    Returns:
        QueryResponse: The chatbot's answer, metadata about the source documents, and relevant paragraphs of context.

    Workflow:
        1. Retrieve relevant context from the vector database using the `query_chroma` method.
        2. Use the LLM to generate an answer based on the query and context.
        3. Flatten and process metadata into a structured format.
        4. Return the response, including the answer, metadata, and context.
    """
    try:
        # Retrieve relevant context
        context = pdf.query_chroma(request.query_text, pdf.vector_store)
        # Generate the answer using the LLM
        answer = llm.generate_answer(request.query_text, context)
        
        # Flatten the nested metadatas list
        flattened_metadatas = [item for sublist in context['metadatas'] for item in sublist]

        # Convert flattened metadata dictionaries to Metadata objects
        metadata_list = [
            Metadata(doc_name=item['doc_name'], page=item['page_number'])
            for item in flattened_metadatas
        ]

        # Return the response
        return QueryResponse(answer=answer, metadata=metadata_list, paragraph=context['documents'])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")



