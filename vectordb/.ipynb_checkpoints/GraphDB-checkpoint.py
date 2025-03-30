import dotenv
import os
from neo4j import GraphDatabase
from py2neo import Graph, Node, Relationship
from py2neo import Graph, Node, Relationship
import json
import networkx as nx
import matplotlib.pyplot as plt

load_status = dotenv.load_dotenv("../Neo4j-66cb9e32-Created-2025-03-26.txt")
if load_status is False:
    raise RuntimeError('Environment variables not loaded.')

URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))


class GraphDB: 
    def __init__(self, json_file, visualisation):
        self.json_file = json_file
        self.visualisation = visualisation
        
    @staticmethod    
    def connect_to_database(): 
        driver = GraphDatabase.driver(URI, auth=AUTH)
        driver.verify_connectivity()
        print("Connection established.")
        return driver
    
    def load_json_to_graph(self):
        driver = self.connect_to_database()
    
        with open(self.json_file, "r") as file:
            data = json.load(file)
    
        with driver.session() as session:
            # Create nodes
            for data_section in data:
                text_id = data_section.get("id")
                text = data_section.get("text", "")
                metadata = data_section.get("metadata")
                references = metadata.get('references', []) if metadata else []
                
                session.run("""
                    MERGE (s:Section {id: $text_id})
                    SET s.text_id = $text_id, s.text = $text
                """, text_id=text_id, text=text) 
    
                # Create undirected relationships (current section → references)
                # MERGE (s1)-[:REFERS_TO]->(s2)
                # MERGE (s2)-[:REFERRED_BY]->(s1) 
                
                for ref_id in references:
                    session.run("""
                        MERGE (s1:Section {id: $text_id})
                        MERGE (s2:Section {id: $ref_id})
                        MERGE (s1)-[:REFERS_TO]-(s2)
                        
                    """, text_id=text_id, ref_id=ref_id)
    
        driver.close()

    @staticmethod
    def fetch_graph_data():
        driver = GraphDatabase.driver(URI, auth=AUTH)
        G = nx.DiGraph()
    
        with driver.session() as session:
            # Fetch nodes
            nodes = session.run("MATCH (n:Section) RETURN n.id")
            for record in nodes:
                G.add_node(record["n.id"])
    
            # Fetch edges
            edges = session.run("""
                MATCH (a:Section)-[:REFERS_TO]-(b:Section)
                RETURN a.id AS source, b.id AS target
                LIMIT 10
            """)
            # edges = session.run("""
            #     MATCH (a:Section)-[r:REFERS_TO]->(b:Section)
            #     RETURN a.id , b.id
            #     LIMIT 20
            # """)

            for record in edges:
                G.add_edge(record["source"], record["target"])

        driver.close()
        return G 

   
    def visualise_graph(self): 
        G = self.fetch_graph_data()
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.5)
        nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue", edge_color="gray", font_size=10, font_weight="bold")
        plt.show()
        

    def run(self): 
        ## This step does not need to be ran after the data is uploaded into the database.
        # self.load_json_to_graph()

        # Run this step to visualise relationships within the graph. 
        if self.visualisation == True: 
            self.visualise_graph()

if __name__ == "__main__":
    graphdb = GraphDB(json_file = "../ba_data_extraction/banking_act.json",
                      visualisation = True)
    graphdb.run()