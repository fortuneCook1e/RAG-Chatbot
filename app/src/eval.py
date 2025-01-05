import time
import json
from pdf import Pdf
from llm import Llm

# Initialize Pdf and Llm classes
pdf = Pdf()
llm = Llm()

# List of queries for testing
queries = [
    "Explain the mechanism of action in anti-aging medicine.",
    "What is Mesenchymal Cell Therapy?",
    "What is the application of precursor stem cells?",
    "Who is Bill Gates?",
    "Tell me about thymic regeneration."
]

def measure_query_speed(queries, log_file="query_logs.json"):
    """
    Measures the speed of query_chroma and generate_answer for a list of queries,
    logs the answers and metadata, and calculates the average time for each operation.

    Args:
        queries (list): List of query strings.
        log_file (str): Path to the file where logs will be saved.

    Returns:
        dict: A dictionary containing individual timings, answers, metadata, and averages.
    """
    query_chroma_times = []
    generate_answer_times = []
    logs = []  # To store logs for each query

    for query in queries:
        print(f"Processing query: {query}")

        # Measure query_chroma time
        start_time = time.time()
        context = pdf.query_chroma(query, pdf.vector_store)
        end_time = time.time()
        query_chroma_time = end_time - start_time
        query_chroma_times.append(query_chroma_time)
        print(f"query_chroma time: {query_chroma_time:.4f} seconds")

        # Measure generate_answer time
        start_time = time.time()
        answer = llm.generate_answer(query, context)
        end_time = time.time()
        generate_answer_time = end_time - start_time
        generate_answer_times.append(generate_answer_time)
        print(f"generate_answer time: {generate_answer_time:.4f} seconds")

        # Extract metadata and log
        metadata = context['metadatas'] if 'metadatas' in context else []
        chunks = context['documents'] if 'documents' in context else []
        
        logs.append({
            "query": query,
            "answer": answer,
            "metadata": metadata,
            "paragraphs": chunks,
            "query_chroma_time": query_chroma_time,
            "generate_answer_time": generate_answer_time
        })

    # Save logs to a file
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=4)

    # Calculate averages
    avg_query_chroma_time = sum(query_chroma_times) / len(query_chroma_times)
    avg_generate_answer_time = sum(generate_answer_times) / len(generate_answer_times)

    # Return results
    return {
        "query_chroma_times": query_chroma_times,
        "generate_answer_times": generate_answer_times,
        "avg_query_chroma_time": avg_query_chroma_time,
        "avg_generate_answer_time": avg_generate_answer_time,
        "logs": logs
    }

# Run the measurement
results = measure_query_speed(queries, log_file="query_logs.json")

# Print results
print("\n--- Results ---")
print(f"Query Chroma Times: {results['query_chroma_times']}")
print(f"Generate Answer Times: {results['generate_answer_times']}")
print(f"Average Query Chroma Time: {results['avg_query_chroma_time']:.4f} seconds")
print(f"Average Generate Answer Time: {results['avg_generate_answer_time']:.4f} seconds")
print(f"Logs saved to query_logs.json")
