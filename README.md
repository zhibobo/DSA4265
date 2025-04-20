# DSA4265 Team Project

This is a Retrieval-Augmented Generation (RAG) assistant designed to support natural language queries related to financial regulations in Singapore.

## Installing all necessary dependencies 

1. Make sure you `cd env`.
2. Enter `chmod +x bootstrap.sh`.
3. Run the script using `./bootstrap.sh` to download all necessary python libraries

## Usage

- To use the RAG model, run `app.py`
- For evaluation of the RAG model, prepare your sample query in `sample_query.csv` and run `evaluation.ipynb`. 
- Data Preprocessing:
    - Banking Act: `ba_data_extraction/banking_act_preprocessor.py`
    - MAS Documents: `mas_data_extraction/mas_preprocessor.py`
- LLM Agents: 
    - Classification Agent: `classify_query.py`
    - Source Router Agent: `source_router.py`
    - Reranking Agent and BFS retriever: `vectordb_retriever.py`
    - DFS retriever: `dfs_vectordb_retriever.py`
    - Weighted BFS retriever: `weighted_vectordb_retriever.py`
    - Summary Agent and Output Agent: `summary_and_output.py`


## Contributing Members

- Chien Shi Yun (A0238439U)
- Hans
- Hastuti Hera Hardiyanti (A0244837W)
- Terence Lai Yixuan (A0234538B)
- Victoria Magdalena (A0244492B)
- Yeo Zhi Hao (A0234031Y)


