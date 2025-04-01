from mas_data_extraction.mas_preprocessor import MASPreprocessor
from ba_data_extraction.banking_act_preprocessor.py import BankingActPreprocessor 
from graphdb.GraphDB import GraphDB 
from PineconeUploader import PineconeUploader
from scripts.fact_reasoning_agent import classify_query




def main(): 
    ## 1. pdf extraction 
    mas_preprocessor = MASPreprocessor()
    mas_chunks = mas_preprocessor.add_references_to_guidelines()
    ba_preprocessor = BankingActPreprocessor()
    ba_chunks = ba_preprocessor.get_chunks() 

    ## 2a. Add data extracted to GraphDB
    graphdb_mas = GraphDB(json_file = "../mas_data_extraction/mas_chunks.json",
                      visualisation = False)
    graphdb_mas.run()
    graphdb_ba = GraphDB(json_file = "../mas_data_extraction/banking_act.json",
                      visualisation = False)
    graphdb_ba.run()

    ## 2b. Add data extracted to  VectorDB
    uploader_mas = PineconeUploader(index_name=os.getenv("PINECONE_INDEX_MAS_NAME"))
    uploader_mas.upload_json_to_index("../mas_data_extraction/mas_chunks.json")
    uploader_ba = PineconeUploader(index_name=os.getenv("PINECONE_INDEX_BA_NAME"))
    uploader_ba.upload_json_to_index("../ba_data_extraction/banking_act.json")

    ## 3a. Determine Banking Act/MAS query Agent 

    ## 3b. Determine fact or reasoning query Agent
    fact_reasoning_output = classify_query()

    ## 4a. Retrieve top chunk with references Agent 

    ## 4b. Rerankers 

    ## 4c. Summary Agent 

    ## 5. Identify & Ask for additional user input Agent 

    ## 6. Answer output Agent
    


if __name__ == "__main__":
    main()