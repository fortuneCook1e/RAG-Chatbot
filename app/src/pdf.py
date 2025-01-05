import os
import fitz
import chromadb
import ollama

class Pdf:
    """
    A class to handle PDF text extraction, embedding generation, and storage in a vector database.
    """
    
    def __init__(self, resource_path="./resource", vector_store_path="./data/vector_store"):
        """
        Initializes the Pdf class.

        Args:
            resource_path (str): Path to the directory containing PDF files.
            vector_store_path (str): Path to the directory where the vector store is stored or will be created.
        """
        self.vector_store_path = vector_store_path
        # Load or create vector store
        self.vector_store = self._initialize_vector_store(resource_path)
        
    def _initialize_vector_store(self, resource_path):
        """
        Initializes or loads the vector store and processes the PDFs if necessary. 
        (PDFs processing will only be done during the first run, where the vector store does not exist yet.)

        Args:
            resource_path (str): Path to the directory containing PDF files.

        Returns:
            chroma.Collection: The collection for storing/retrieving embeddings.
        """
        persist_directory = self.vector_store_path
        
        # Ensure persist directory exists
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)  # Create the directory if it doesn't exist
            print(f"Created directory: {persist_directory}")
        
        # Create or load collection
        collection = self.initialize_collection(persist_directory)
        
        # Check if the collection is already populated
        if not collection.count():  # Check if the collection has any data
            print(f"No existing ChromaDB found. Creating a new one at '{persist_directory}'")
            
            # Process and populate the collection with PDFs
            for filename in os.listdir(resource_path):
                if filename.endswith(".pdf"):  # Only process .pdf files
                    pdf_path = os.path.join(resource_path, filename)
                    print(f"Processing file: {pdf_path}")
                    self.process_pdf_page_by_page(pdf_path, collection)
        else:
            print(f"Loading existing ChromaDB from '{persist_directory}'")
        
        return collection    
    
    def initialize_collection(self, persist_directory):
        """
        Initialize a ChromaDB client, loading the database if it exists, or creating it if not.

        Args:
            persist_directory (str): Path to the directory where the database is stored.

        Returns:
            chroma.Collection: The collection for storing/retrieving embeddings.
        """
        
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=persist_directory)

        # Create or get collection
        collection_name = "pdf_chunks"
        collection = client.get_or_create_collection(name=collection_name)
        return collection
    
    def generate_embedding_with_ollama(self, text):
        """
        Generates embeddings for a given text using the Ollama model.

        Args:
            text (str): The text to embed.

        Returns:
            list: A list of floats representing the embedding vector.
        """
        response = ollama.embeddings(model='nomic-embed-text', prompt=text)
        embedding = response["embedding"]
        return embedding


    def process_pdf_page_by_page(self, pdf_path, collection, chunk_size=3000, overlap=500):
        """
        Processes a PDF file page by page, extracts text, chunks it, generates embeddings, 
        and stores them in the vector database.

        Args:
            pdf_path (str): Path to the PDF file.
            collection (chroma.Collection): The ChromaDB collection to store embeddings.
            chunk_size (int): Number of characters in each chunk.
            overlap (int): Number of characters to overlap between chunks.
        """
        pdf_name = pdf_path.split("/")[-1]  # Get the filename

        # Open the PDF with PyMuPDF
        try:
            pdf_document = fitz.open(pdf_path)
        except Exception as e:
            print(f"Error opening PDF {pdf_path}: {e}")
            return

        # Process each page individually
        for page_number in range(len(pdf_document)):
            try:
                page = pdf_document[page_number]
                text = page.get_text()

                if not text.strip():  # Skip empty pages
                    continue

                text = text.replace('\n', ' ').strip()  # Normalize text
                start = 0

                # Process chunks for this page
                while start < len(text):
                    end = start + chunk_size
                    chunk = text[start:end]
                    start = end - overlap  # Ensure overlap for the next chunk

                    # Prepare metadata for this chunk
                    chunk_metadata = {
                        "chunk_id": f"{pdf_name}_page{page_number + 1}_chunk{start}",
                        "text": chunk,
                        "doc_name": pdf_name,
                        "page_number": page_number + 1
                    }

                    # Generate embedding and store directly in ChromaDB
                    embedding = self.generate_embedding_with_ollama(chunk)
                    if embedding:
                        collection.add(
                            documents=[chunk],
                            metadatas=[{
                                "doc_name": chunk_metadata["doc_name"],
                                "page_number": chunk_metadata["page_number"]
                            }],
                            ids=[chunk_metadata["chunk_id"]],
                            embeddings=[embedding]
                        )
                        print(f"Stored chunk: {chunk_metadata['chunk_id']}")

            except Exception as e:
                print(f"Error processing page {page_number + 1} of {pdf_path}: {e}")

        pdf_document.close()
        print(f"----------Finished processing {pdf_path}----------")
        
    def query_chroma(self, query_text, collection):
        """
        Queries the vector database for the most relevant chunks based on the input query.

        Args:
            query_text (str): The query text to search for.
            collection (chroma.Collection): The ChromaDB collection to search.

        Returns:
            dict: A dictionary containing the top results, including metadata and documents.
        """
        # Generate query embedding
        query_embedding = self.generate_embedding_with_ollama(query_text)
        # Perform a similarity search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3  # Number of top results to retrieve
        )
        return results