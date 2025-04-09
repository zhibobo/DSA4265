import os
from dotenv import load_dotenv
from pinecone import Pinecone
import openai
from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from langchain.schema import Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever
load_dotenv()
from sentence_transformers import CrossEncoder
from typing import List 

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
        """
        Takes user input query and get the top_k_chunks(context) with the highest similarity score from vector db

        :query: String from front end
        :index_name:  fixed string field('ba' or 'mas')
        :return: list of chunk_id corresponding to the top_k_chunks
        """
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
    def __init__(self, hops):
        self.hops = hops
        self.uri = {'ba': os.getenv("NEO4J_URI_BA"),
                    'mas' : os.getenv("NEO4J_URI_MAS")}
        self.auth = {'ba': (os.getenv("NEO4J_USERNAME_BA"), os.getenv("NEO4J_PASSWORD_BA")),
                     'mas': (os.getenv("NEO4J_USERNAME_MAS"), os.getenv("NEO4J_PASSWORD_MAS"))}
        self.driver = {'ba': GraphDatabase.driver(self.uri['ba'], auth=self.auth['ba']),
                       'mas': GraphDatabase.driver(self.uri['mas'], auth=self.auth['mas'])}

    def get_neighbours(self, chunk_id, index_name):
        """
        Takes one chunk id to query graph db and get all neighbours of n hops, append all their text into one
        :chunk_id: string. (eg. 'ba-1970-15b.1')
        :index_name:  fixed string field('ba' or 'mas')
        :return: string that contains chunk's text + all queried neighbour's text
        """

        cypher_query = {'ba': f"""
                        MATCH (n {{id: $chunk_id}}), (m)
                        WHERE elementId(n) <> elementId(m)
                        MATCH path = shortestPath((n)-[:REFERS_TO*1..{self.hops}]-(m))
                        RETURN DISTINCT n, m, length(path) AS hop
                    """,
                        'mas': f"""
                        MATCH (n {{id: $chunk_id}}), (m)
                        WHERE elementId(n) <> elementId(m)
                        MATCH path = shortestPath((n)-[:REFERRED_BY*1..{self.hops}]-(m))
                        RETURN DISTINCT n, m, length(path) AS hop
                    """
                       }
        
        with self.driver[index_name].session() as session:
            result = session.run(cypher_query[index_name], chunk_id=chunk_id)
            graph = []  
            retrieved_context = []
            context_str = ""
            
            for record in result:
                a = record["n"]["id"]
                b = record["m"]["id"]
                if a not in graph:
                    graph.append(a) 
                    hop = record["hop"]
                    weight = 1 - 0.1 * (hop - 1)
                    retrieved_context.append({
                         'id': record['n']['id'],
                         'text': record['n']['text'],
                         'hop': hop,
                         'weight': weight
                     })
                if b not in graph:
                    graph.append(b)
                    hop = record["hop"]
                    weight = 1 - 0.1 * (hop - 1)
                    retrieved_context.append({
                         'id': record['m']['id'],
                         'text': record['m']['text'],
                         'hop': hop,
                         'weight': weight
                     })
        
        return retrieved_context 
        
    def get_appended_chunks(self, top_k_chunks, index_name):
        """
        Iterates list of top_k_chunk ids to query the graph db. outputs the list of top_k_chunks with their text appended with 
        neighbouring node's text
        
        :top_k_chunks: list of chunk ids
        :index_name:  fixed string field('ba' or 'mas')
        :return: list of appended top_k_chunk's text (list of strings)
        """
        appended_top_k = []
        for chunk in top_k_chunks:
            appended_chunk = self.get_neighbours(chunk, index_name)
            appended_top_k.append(appended_chunk)
        return appended_top_k


class Reranker:
    def __init__(self, top_k):
        self.top_k = top_k
    
    @staticmethod
    def get_raw_scores(query, appended_chunks): #, cross_encoder
        #print(chunk for chunk in appended_chunks)
        pairs = [(query, chunk[0]['text']) for chunk in appended_chunks]  
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
        raw_scores = cross_encoder.predict(pairs)
        probabilities = torch.sigmoid(torch.tensor(raw_scores))
        return probabilities

    #Rerank APPENDED top k chunks
    def rerank(self, query, appended_chunks):
        """
        Retrieve top-k similar documents using vector similarity + neighbour weights + rerank them independently.
        Returns a list of reranked documents.
        """
        
        reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
      
        raw_scores = self.get_raw_scores(query, appended_chunks)
        docs = [
            Document(
                page_content=f"ID: {chunk[0]['id']}\n{chunk[0]['text']}",
                metadata={
                    'id': chunk[0]['id'], 
                    'hop': chunk[0]['hop'],
                    'weight': chunk[0]['weight'],
                    'raw_score': raw_scores[i],  # Add the raw score to metadata
                    'combined_score': chunk[0]['weight'] * raw_scores[i]
                }
            ) 
            for i, chunk in enumerate(appended_chunks)
        ]
        
        combined_scored_docs = sorted(docs, key=lambda doc: doc.metadata['combined_score'], reverse=True) 
        
        # for doc in combined_scored_docs:
        #     print(f"Document: {doc.page_content[:100]}...")  # Print the first 100 characters of the document content (for brevity)
        #     print(f"Combined Score: {doc.metadata['combined_score']}")
        #     print("-" * 50)  # Separator for readability
        
            
        return [doc.page_content for doc in combined_scored_docs[:self.top_k]]


if __name__ == "__main__":
    vectorretriever = VectorDbRetriever(top_k=10)

    sample_query = "How can we manage service quality and customer feedback?"
    top_k_chunks = vectorretriever.get_top_k_chunks(sample_query, 'mas')

    graphretriever = GraphDbRetriever(top_k=10, hops=10)
    appended_chunks = graphretriever.get_appended_chunks(top_k_chunks, 'mas')

    reranker = Reranker(top_k=5)
    #Run the line below to see the output for the whole flow
    print(reranker.rerank(sample_query,appended_chunks))

    
    """
    Some findings:
    - The scores are repeated for  a few document chunks
    Document: ID: ba-1970-31.1a
    31(1A) A bank incorporated outside Singapore must not, through a branch or office ...
    Combined Score: 0.8976913094520569
    --------------------------------------------------
    Document: ID: ba-1970-31.1
    31(1) A bank incorporated in Singapore must not acquire or hold any equity investme...
    Combined Score: 0.8972207903862
    --------------------------------------------------
    Document: ID: ba-1970-33.1
    33(1) A bank incorporated in Singapore must not acquire or hold interests in or rig...
    Combined Score: 0.7631070613861084
    --------------------------------------------------
    Document: ID: ba-1970-55t.3
    55T(3) Any amount of paid-up capital or capital funds of a merchant bank incorpora...
    Combined Score: 0.4621790051460266
    --------------------------------------------------
    """
    