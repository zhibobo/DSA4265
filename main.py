from graphdb.GraphDB import GraphDB 
from mas_data_extraction.mas_preprocessor import MASPreprocessor
from ba_data_extraction.banking_act_preprocessor.py import BankingActPreprocessor 


def main(): 
    ## 1. pdf extraction 
    mas_preprocessor = MASPreprocessor()
    mas_chunks = mas_preprocessor.add_references_to_guidelines()
    ba_preprocessor = BankingActPreprocessor()
    ba_chunks = ba_preprocessor.get_chunks() 

    ## 2a. Add data extracted to GraphDB
    graphdb = GraphDB(json_file = "../mas_data_extraction/mas_chunks.json",
                      visualisation = True)
    graphdb.run()

    ## 2b. Add data extracted to  VectorDB 

    ## 3a. Determine Banking Act/MAS query Agent 

    ## 3b. Determine fact or reasoning query Agent

    ## 4a. Retrieve top chunk with references Agent 

    ## 4b. Reranking Agent 

    ## 4c. Summary Agent 

    ## 5. Identify & Ask for additional user input Agent 

    ## 6. Answer output Agent
    


if __name__ == "__main__":
    main()