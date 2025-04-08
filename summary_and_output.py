import os
import re
from dotenv import load_dotenv
import openai
from weighted_vectordb_retriever import VectorDbRetriever, GraphDbRetriever, Reranker
from classify_query import QueryClassifierAgent
from source_router import SourceRouterAgent

load_dotenv()

class SummaryAgent:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def summarize_text(self, text, query):
        prompt = f"""
        Generate a comprehensive and context-aware summary of the provided document chunk, ensuring that it accurately addresses the user's query. Your summary should preserve all critical details, maintain the meaning, intent, and legal nuances of the original regulatory text, and avoid omitting any essential information to prevent regulatory misinterpretation.

        Document Chunk:
        {text}

        User's Query:
        {query}
        
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "developer", "content": "You are an expert specializing in summarizing financial regulatory texts. Your summaries must preserve critical details and context to prevent regulatory misinterpretation."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

class OutputAgent:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def output_response(self, source, classification, query, summarized_chunks=None):
        formatted_source = ''
        example_citation = ''

        # Format the data source and example citation 
        if source == 'ba':
            formatted_source = 'Banking Act 1970'
            example_citation = '(Banking Act 1970, 27.1)'
        elif source == 'mas':
            formatted_source = 'MAS Financial Regulation documents, which include the Fair Dealing Guidelines (30 May 2024) and MAS 626 Notice on Money Laundering Prevention'
            example_citation = '(MAS Notice 626, 12.1)'

        # Prompt engineering for factual vs reasoning query 
        if classification.lower() == "factual":
            # Standard prompt 
            prompt = f"""
            Answer the following question based on the relevant context from {formatted_source}. If you can not answer the question based on the context, just say that you do not know the answer. Do not try to make up an answer. 
            
            ## Question: 
            {query}

            ## Context: 
            {summarized_chunks}

            ## Answer format: 
            Answer the question based on information from the context. Add in-text citations (e.g: {example_citation}) at the end of each phrase matched. 
            """
        elif classification.lower() == "reasoning":
            # Chain-of-draft
            prompt = f"""
            You are a financial reasoning agent. Answer the following question using relevant information from the provided context {formatted_source}. Think step by step: break down your reasoning into concise steps (no more than 30 words per step). These steps should help structure your approach logically, but avoid lengthy drafts.

            At the end of your response, include a final, comprehensive answer after a separator line ####. This answer should fully address the question, incorporating insights from the reasoning steps and backed by the provided context. 
            
            Use in-text citations (e.g., {example_citation}) at the end of any phrase where specific context is used, including in the final answer.
            
            ## Question:
            {query}

            ## Context:
            {summarized_chunks}

            ## Answer format:
            1. [reasoning step 1]
            2. [reasoning step 2]
            3. [reasoning step 3]
            ####
            [final answer]
            """
        else:
            return "Error: Unrecognized classification: " + classification
        
        print(prompt)
        response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "developer", "content": "You are an RAG-powered chatbot for financial regulation checker."},
                    {"role": "user", "content": prompt}
                ]
            )
        
        return response.choices[0].message.content.strip()

if __name__ == "__main__":
    sample_query = "What is the limit on equity investments for banks in Singapore?"

    classifier = QueryClassifierAgent()
    classification = classifier.classify_query(sample_query)
    if 'factual' in classification.lower():
        classification = 'factual'
    elif 'reasoning' in classification.lower():
        classification = 'reasoning'
    # print("Classification:", classification)

    router = SourceRouterAgent()
    data_source = router.get_source(sample_query)
    # print("Chosen Data Source:", data_source)

    vectorretriever = VectorDbRetriever(top_k=10)
    top_k_chunks = vectorretriever.get_top_k_chunks(sample_query, data_source)
    # print(top_k_chunks)

    graphretriever = GraphDbRetriever(top_k=10, hops=1)
    appended_chunks = graphretriever.get_appended_chunks(top_k_chunks, data_source)
    # print(appended_chunks)

    reranker = Reranker(top_k=5)
    ranked_chunks = reranker.rerank(sample_query, appended_chunks)
    # print(ranked_chunks)

    summary_agent = SummaryAgent()
    summarized_chunks = []
    for chunk in ranked_chunks:
        id_pattern = re.match(r"\(([^)]+)\)", chunk)
        chunk_id = id_pattern.group(1)
        if data_source == 'ba':
            chunk_id = chunk_id.replace('ba', 'Banking Act')
        elif data_source == 'mas':
            chunk_id = chunk_id.replace('mas', 'MAS')

        before, sep, after = chunk_id.rpartition('-')
        chunk_id = before + ', ' + after
        chunk_id = chunk_id.replace('-', ' ').title()    

        summary = summary_agent.summarize_text(chunk, sample_query)
        summary = summary.replace('\n', ' ')
        summarized_chunks.append(f'{chunk_id}: {summary}')
    final_chunks = '\n\n'.join(summarized_chunks)
    # print(final_chunks)

    output_agent = OutputAgent()
    final_answer = output_agent.output_response(data_source, classification, sample_query, summarized_chunks=final_chunks)
    # print(f"\nFinal Answer:\n{final_answer}")
