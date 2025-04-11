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
    
    def get_single_node(self, chunk_id, index_name):
        """
        Takes one chunk id to query graph db and get that chunk

        :chunk_id: string. (eg. 'ba-1970-15b.1')
        :index_name:  fixed string field('ba' or 'mas')
        :return: string that contains chunk's text + queried text
        """
        #edit the queries below for ba and mas index respectively
        cypher_query = f"""
        MATCH (n {{id: $chunk_id}})
        RETURN n
        """
        with self.driver[index_name].session() as session:
            result = session.run(cypher_query, chunk_id=chunk_id)
            for record in result:
                node = record["n"]
                return node['id'], node['text']

    def get_neighbours(self, chunk_id, index_name):
        """
        Takes one chunk id to query graph db and get all neighbours of n hops, append all their text into one

        :chunk_id: string. (eg. 'ba-1970-15b.1')
        :index_name:  fixed string field('ba' or 'mas')
        :return: string that contains chunk's text + all queried neighbour's text
        """
        #edit the queries below for ba and mas index respectively
        cypher_query = {'ba': f"""
        MATCH (n {{id: $chunk_id}})-[:REFERS_TO*1..{self.hops}]-(m)
        RETURN DISTINCT n,m
        """,
        'mas': f"""
        MATCH (n {{id: $chunk_id}})-[:REFERRED_BY*1..{self.hops}]-(m)
        RETURN DISTINCT n,m
        """}
        with self.driver[index_name].session() as session:
            result = session.run(cypher_query[index_name], chunk_id=chunk_id)
            graph = []
            retrieved_context = ""
            for record in result:
                a = record["n"]["id"]
                b = record["m"]["id"]
                if a not in graph:
                    graph.append(a) 
                    retrieved_context += f"{record['n']['id']} {record['n']['text']}   "
                if b not in graph:
                    graph.append(b)
                    retrieved_context += f"{record['m']['id']} {record['m']['text']}   "
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

    #Rerank APPENDED top k chunks
    def rerank(self, query, appended_chunks):
        """
        Takes in the initial query and chunks that have been combined with their neighbouring nodes (1 hop), reranks the chunks and
        outputs the top chunk
        encoder model used: "BAAI/bge-reranker-base"

        :query: text
        :appended_chunks: list of text of each "appended" chunk 

        :return: now set as top chunk. which will be summarized by summary agent (list of strings)
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
    
class ChunkPath:
    def __init__(self, root_id, depth):
        self.root_id = root_id
        self.depth = depth
        self.path_ids = []
        self.path_chunks = []
        
    def add_root_chunk(self, root_chunk_id, root_text):
        self.path_ids.append(root_chunk_id)
        self.path_chunks.append(root_text)

    def add_next_seq(self, chunk_id, text):
        self.path_ids.append(chunk_id)
        self.path_chunks.append(text)

    def show_path(self):
        print(f"Root Node is {self.root_id}")
        print(f"Path IDs are {self.path_ids}")
        print(f"Path Text are {self.path_chunks}")

class DFSRetriever:
    def __init__(self, path_length):
        self.path_length = path_length
        self.retriever = GraphDbRetriever(hops = 1)
        self.reranker = Reranker(top_k = 10)
        self.path_storage = []
    
    def initiate_path_storage(self, top_k_chunks):
        for chunk_id in top_k_chunks:
            self.path_storage.append(ChunkPath(root_id = chunk_id,
                                            depth = self.path_length))
            
    def initiate_root_chunk(self, top_k_chunks, index_name):
        for i in range(len(top_k_chunks)):
            id, text = self.retriever.get_single_node(top_k_chunks[i], index_name)
            self.path_storage[i].add_root_chunk(id,text)
        
    #remove root chunk + rerank
    def get_next_seq(self, query, nearest_neighbours):
        next_k_chunks = []
        for i in range(len(nearest_neighbours)):
            neighbours = nearest_neighbours[i].split("   ")[1:-1]
            #curr_id, curr_text = curr.split(" ",1)
            #neighbours = neighbours.split("   ")
            #(f"This is the neighbour list: {neighbours}")
            #get top scored neighbour
            next_seq_chunk = self.reranker.rerank(query, neighbours)
            #print(f"This is the list of ranked neighbours for root node {i}: {next_seq_chunk}")
            for chunk in next_seq_chunk:
                #print(f"This is the next chunk {chunk}")
                next_id, next_text = chunk.split(" ",1)

                if next_id not in self.path_storage[i].path_ids:
                    next_k_chunks.append(next_id)

                    (self.path_storage[i]).add_next_seq(next_id, next_text)
                    break

        return next_k_chunks
    
    def get_appended_chunks_dfs(self):
        appended_chunks_df = []
        for path in self.path_storage:
            id = path.path_ids
            text = path.path_chunks
            chunk = ""
            for index, value in enumerate(id):
                 chunk += f"({value}) {text[index]}   "
            appended_chunks_df.append(chunk)
        return appended_chunks_df

    def run_DFS(self, query, top_k_chunks, index_name):
        self.initiate_path_storage(top_k_chunks)

        self.initiate_root_chunk(top_k_chunks, index_name)

        curr_k_chunks = top_k_chunks
        for i in tqdm(range(self.path_length), desc='Getting next chunk in path'):
            nearest_neighbours = self.retriever.get_appended_chunks(curr_k_chunks, index_name)
            #print(f"These are the next nearest_neighbours: {nearest_neighbours}")
            curr_k_chunks = self.get_next_seq(query, nearest_neighbours)
            #print(f"These are the next chunks to DFS: {curr_k_chunks}")
        
        for path in self.path_storage:
            print(f"Root: {path.root_id}//Path_id: {path.path_ids}//Text: {path.path_chunks}")
        return self.get_appended_chunks_dfs()

if __name__ == "__main__":
    vectorretriever = VectorDbRetriever(top_k=10)

    sample_query = "What is the limit on equity investments for banks in Singapore?"
    top_k_chunks = vectorretriever.get_top_k_chunks(sample_query, 'ba')

    dfsretriever = DFSRetriever(path_length=1)
    appended_chunks_dfs = dfsretriever.run_DFS(sample_query,top_k_chunks,'ba')
    
    #graphretriever = GraphDbRetriever(hops=1)
    #appended_chunks = graphretriever.get_appended_chunks(top_k_chunks, 'ba')

    reranker = Reranker(top_k=5)
    #Run the line below to see the output for the whole flow
    print(reranker.rerank(sample_query,appended_chunks_dfs))

    """
    Some findings:
    - Root Chunk would always be the first in the list of chunks
    - The hops function should work. There is an increase in the length of chunk after appending, but validation not done to see if appended the correct neighbours
    - Reranking does change the sequence of the top_k, based on the sample query.
    """