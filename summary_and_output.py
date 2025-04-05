import os
from dotenv import load_dotenv
import openai
from vectordb_retriever import VectorDbRetriever, GraphDbRetriever, Reranker
from classify_query import QueryClassifierAgent
from source_router import SourceRouterAgent

load_dotenv()

class SummaryAgent:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def summarize_text(self, text, query):
        prompt = f"""
        Please provide a comprehensive and concise summary of the following text, keeping the user's query in mind.

        {text}
        {query}
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "developer", "content": "You are a summarization agent."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

class OutputAgent:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def output_response(self, classification, query, summarized_chunks=None):
        if classification.lower() == "factual":
            prompt = f"""
            From the following summarized chunks, choose the one that is most relevant to the user's query and produce a final, comprehensive answer:
            {summarized_chunks}
            {query}
            """
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "developer", "content": "You are a final output agent."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        elif classification.lower() == "reasoning":
            return None
        else:
            return "Error: Unrecognized classification: " + classification

if __name__ == "__main__":
    sample_query = "What is the limit on equity investments for banks in Singapore?"


    classifier = QueryClassifierAgent()
    classification = classifier.classify_query(sample_query)
    print("Classification:", classification)

    if classification.lower() == "factual":
        router = SourceRouterAgent()
        data_source = router.get_source(sample_query)
        print("Chosen Data Source:", data_source)
        vectorretriever = VectorDbRetriever(top_k=10)
        top_k_chunks = vectorretriever.get_top_k_chunks(sample_query, data_source)

        graphretriever = GraphDbRetriever(top_k=10, hops=1)
        appended_chunks = graphretriever.get_appended_chunks(top_k_chunks, data_source)

        reranker = Reranker(top_k=10)
        ranked_chunks = reranker.rerank(sample_query, appended_chunks)

        summary_agent = SummaryAgent()
        summarized_chunks = []
        for chunk, _ in ranked_chunks:
            summary = summary_agent.summarize_text(chunk, sample_query)
            summarized_chunks.append(summary)

        output_agent = OutputAgent()
        final_answer = output_agent.output_response(classification, sample_query, summarized_chunks=summarized_chunks)
        print("\nFinal Answer:\n", final_answer)

    else:
        print("The query requires reasoning or more context.")
