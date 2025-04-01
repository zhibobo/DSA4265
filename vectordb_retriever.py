import os
from dotenv import load_dotenv
from pinecone import Pinecone
import openai
from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

load_dotenv()

class VectorDbRetriever:
    def __init__(self, top_k):
        self.top_k = top_k

        self.api_key = os.getenv("PINECONE_API_KEY")
        self.pc = Pinecone(api_key=self.api_key)
        self.index = {'ba': self.pc.Index(os.getenv("PINECONE_INDEX_BA_NAME")),
                      'mas' : self.pc.Index(os.getenv("PINECONE_INDEX_MAS_NAME"))}
                      
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=self.openai_key)

    def get_top_k_chunks(self, query, index_name):
        #embed the query with same embedding as vectordb
        embedded_query = self.client.embeddings.create(
            input = query,
            model = "text-embedding-3-small"
        )

        embedded_vector = embedded_query.data[0].embedding

        results = self.index[index_name].query(
                    vector = embedded_vector,
                    top_k = self.top_k,
                    include_values = False)
        
        top_k_chunks = [chunk['id'] for chunk in results['matches']]
        return top_k_chunks


class GraphDbRetriever:
    def __init__(self, top_k, hops):
        self.top_k = top_k
        self.hops = hops

        self.uri = {'ba': os.getenv("NEO4J_URI_BA"),
                    'mas' : os.getenv("NEO4J_URI_MAS")}
        
        self.auth = {'ba': (os.getenv("NEO4J_USERNAME_BA"), os.getenv("NEO4J_PASSWORD_BA")),
                     'mas': (os.getenv("NEO4J_USERNAME_MAS"), os.getenv("NEO4J_PASSWORD_MAS"))}
        
        self.driver = {'ba': GraphDatabase.driver(self.uri['ba'], auth=self.auth['ba']),
                       'mas': GraphDatabase.driver(self.uri['mas'], auth=self.auth['mas'])}

    def get_neighbours(self, chunk_id, index_name):
        cypher_query = f"""
        MATCH (n {{id: $chunk_id}})-[:REFERS_TO*1..{self.hops}]-(m)
        RETURN DISTINCT n,m
        """
        with self.driver[index_name].session() as session:
            result = session.run(cypher_query, chunk_id=chunk_id)
            graph = []
            retrieved_context = ""
            for record in result:
                a = record["n"]["id"]
                b = record["m"]["id"]
                if a not in graph:
                    graph.append(a) 
                    retrieved_context += f"{record['n']['text']}///"
                if b not in graph:
                    graph.append(b)
                    retrieved_context += f"{record['m']['text']}///"
            return retrieved_context
        
    def get_appended_chunks(self, top_k_chunks, index_name):
        appended_top_k = []
        for chunk in top_k_chunks:
            appended_chunk = self.get_neighbours(chunk, index_name)
            appended_top_k.append(appended_chunk)
        return appended_top_k


class Reranker:
    def __init__(self, top_k):
        self.top_k = top_k

    #Rerank APPENDED top k chunks
    def rerank(self, query, appended_chunks):
        """
        Takes in the initial query and chunks that have been combined with their neighbouring nodes (1 hop), reranks the chunks and
        outputs the top chunk
        encoder model used: "BAAI/bge-reranker-base"

        :query: text
        :appendec_chunks: list of text of each "appended" chunk 

        :return: now set as top chunk. which will be summarized by summary agent
        """
        # Load tokenizer and model
        model_name = "BAAI/bge-reranker-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        scores = []

        for chunk in tqdm(appended_chunks, desc="Reranking chunks"):
            # Tokenize query-context pair
            inputs = tokenizer(query, chunk, return_tensors="pt", truncation=True) #can tweak the settings

            # Predict relevance score
            with torch.no_grad():
                logits = model(**inputs).logits

            score = logits[0].item()  # Extract relevance score
            scores.append((chunk, score))

        # Sort contexts by score in descending order
        ranked_contexts = sorted(scores, key=lambda x: x[1], reverse=True)

        return [context[0] for context in ranked_contexts[:self.top_k]]


if __name__ == "__main__":
    vectorretriever = VectorDbRetriever(top_k=10)

    sample_query = "What is the limit on equity investments for banks in Singapore?"
    top_k_chunks = vectorretriever.get_top_k_chunks(sample_query, 'ba')

    graphretriever = GraphDbRetriever(top_k=10, hops=1)
    appended_chunks = graphretriever.get_appended_chunks(top_k_chunks, 'ba')

    reranker = Reranker(top_k=10)
    #Run the line below to see the output for the whole flow
    print(reranker.rerank(sample_query,appended_chunks))

    """
    Some findings:
    - The hops function should work. There is an increase in the length of chunk after appending, but validation not done to see if appended the correct neighbours
    - Reranking does change the sequence of the top_k, based on the sample query.
    """
    