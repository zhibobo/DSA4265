import json 
import pandas as pd
import re
from vectordb_retriever import VectorDbRetriever, GraphDbRetriever, Reranker
from weighted_vectordb_retriever import WeightedGraphDbRetriever, WeightedReranker
from dfs_vectordb_retriever import DFSRetriever
from classify_query import QueryClassifierAgent
from summary_and_output import SummaryAgent, OutputAgent
from source_router import SourceRouterAgent
from llm_as_judge import LLMAsJudge
from sentence_transformers import SentenceTransformer, util

class Evaluation:
    def __init__(self, file_path):
        with open(r"ba_data_extraction\banking_act.json", "r", encoding="utf-8") as file:
            ba_chunks = json.load(file)

        ba_dict = {}
        for chunk in ba_chunks:
            ba_dict[chunk['id']] = chunk['text']

        with open(r"mas_data_extraction\mas_chunks.json", "r", encoding="utf-8") as file:
            mas_chunks = json.load(file)

        mas_dict = {}
        for chunk in mas_chunks:
            mas_dict[chunk['id']] = chunk['text']

            combined_dict = {**ba_dict, **mas_dict}
        self.combined_dict = combined_dict
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.path = file_path
        self.df = pd.read_csv(self.path)

    def get_text(self, id):
        return self.combined_dict[id]
    
    def get_similarity_score(self, chunk1, chunk2):
        embeddings = self.model.encode([chunk1, chunk2])
        score = util.cos_sim(embeddings[0], embeddings[1]).item()
        return score
    
    def get_ranking_score(self, retrieved):
        # Initialize total penalty score
        correct = sorted(retrieved, reverse=True)

        retrieved = [correct.index(i) for i in retrieved]
        correct = [correct.index(i) for i in correct]
        
        total_penalty = 0
        max_penalty = 0
        n = len(correct)
        
        # Compare each pair of chunks in the retrieved list
        for i in range(n):
            for j in range(i + 1, n):
                # Check if there is an inversion (higher value chunk before lower value chunk)
                if correct.index(retrieved[i]) > correct.index(retrieved[j]):
                    # Calculate the penalty based on the difference in chunk values
                    penalty = abs(retrieved[i] - retrieved[j])
                    total_penalty += penalty  # Add penalty to total

        for i in range(n):
            for j in range(i + 1, n):
                # Maximum possible penalty when elements are completely reversed
                max_penalty += abs(correct[i] - correct[j])
            
        # Normalize the penalty score to be between 0 and 1
        normalized_penalty = total_penalty / max_penalty
        
        return 1 - normalized_penalty
    
    def get_retriever_score(self, reference_ids, top_k_chunks):
        # Calculate position score
        position_scores = [] 
        # Using semantic similarity score 
        max_scores = []
        rank_scores = []
        
        # 'reference_retrieval':['ba-1970-2.1', 'ba-1970-4.2']
        for ref_id in reference_ids:
            # Get position_scores 
            if ref_id in top_k_chunks:
                position = top_k_chunks.index(ref_id) 
            else:
                position = len(top_k_chunks)
            position_score = (len(top_k_chunks) - position) / len(top_k_chunks)
            position_scores.append(position_score)

            # Get semantic similarity score between the reference chunk and the retrieved chunk
            retrieved_chunks_score = [self.get_similarity_score(self.get_text(ref_id), self.get_text(chunk_id)) for chunk_id in top_k_chunks]
            print('Retrieved chunks similarity score:', retrieved_chunks_score)

            # Store max similarity score (range -1 to 1)
            max_score = max(retrieved_chunks_score)
            max_scores.append(max_score)
            
            # Evaluate the ranking/order  
            rank_score = self.get_ranking_score(retrieved_chunks_score)
            rank_scores.append(rank_score)

        print('Position scores:', position_scores)
        print('Max scores:', max_scores)
        print('Rank scores:', rank_scores)

        agg_position_score = max(position_scores)
        agg_max_score = max(max_scores)
        agg_rank_score = max(rank_scores)

        return agg_position_score, agg_max_score, agg_rank_score, position_scores, max_scores, rank_scores
    
    def get_evaluation(self, query, correct_source, correct_type, reference_retrieval, reference_generation, n_hop, retriever_method):  
        user_query = query
        
        output = {}
        output['query'] = query
        print('Query:', query)

        # Determine data source 
        output['correct_source'] = correct_source
        router = SourceRouterAgent()
        data_source = router.get_source(user_query)
        print("Raw Data Source:", data_source) 

        # Cleaning data source
        if 'ba' in data_source.lower():
            data_source = 'ba'
        elif 'mas' in data_source.lower():
            data_source = 'mas'
        else:
            output['source'] = data_source
            output['is_source_correct'] = 0
            return output 

        output['source'] = data_source
        if data_source == correct_source:
            output['is_source_correct'] = 1
        else: 
            output['is_source_correct'] = 0

        # Classify the query
        output['correct_type'] = correct_type
        classifier_agent = QueryClassifierAgent()
        classification = classifier_agent.classify_query(user_query)
        print("Raw Classification:", classification)

        if 'factual' in classification.lower():
            classification = 'factual'
        elif 'reasoning' in classification.lower():
            classification = 'reasoning'
        else: 
            # assign misclassification as factual for simplicity
            classification = 'factual'
        
        output['type'] = classification
        if classification == correct_type: 
            output['is_type_correct'] = 1
        else: 
            output['is_type_correct'] = 0 
            
        # Retrieve top-k chunks from VectorDB 
        vectorretriever = VectorDbRetriever(top_k=10)
        top_k_chunks = vectorretriever.get_top_k_chunks(user_query, data_source)
        print("VectorDB Chunks:", top_k_chunks)
        agg_position_score, agg_max_score, agg_rank_score, position_scores, max_scores, rank_scores = self.get_retriever_score(reference_retrieval, top_k_chunks)
        
        output['reference_retrieval'] = reference_retrieval
        output['generated_retrieval'] = top_k_chunks
        output['retrieval_position_scores'] = position_scores
        output['retrieval_max_position_score'] = agg_position_score
        output['retrieval_best_similarity_scores'] = max_scores
        output['retrieval_max_similarity_score'] = agg_max_score
        output['retrieval_rank_similarity_scores'] = rank_scores
        output['retrieval_max_rank_similarity_score'] = agg_rank_score
        
        # GraphDB Retriever 
        output['graphdb_retrieval_method'] = retriever_method
        output['n_hops'] = n_hop
        if retriever_method == 'bfs':
            graphretriever = GraphDbRetriever(hops=n_hop)
            appended_chunks = graphretriever.get_appended_chunks(top_k_chunks, data_source)
            reranker = Reranker(top_k=5)
        elif retriever_method == 'weighted':
            graphretriever = WeightedGraphDbRetriever(hops=n_hop)
            appended_chunks = graphretriever.get_appended_chunks(top_k_chunks, data_source)
            reranker = WeightedReranker(top_k=5)
        elif retriever_method == 'dfs':
            graphretriever = DFSRetriever(path_length=n_hop)
            appended_chunks = graphretriever.run_DFS(user_query, top_k_chunks, data_source)
            reranker = Reranker(top_k=5)

        # Rerank top_k appended chunks 
        reranked_chunks = reranker.rerank(user_query, appended_chunks)
        print(reranked_chunks)
        print('Length Reranked chunks:', len(reranked_chunks))

        # Extrack reranked chunk ids
        reranked_ids = []
        for chunk in reranked_chunks:
            if retriever_method == 'weighted':
                id_pattern = re.search(r'^ID:\s(.*?)(?=\n)', chunk)
                chunk_id = id_pattern.group(1) if id_pattern else ''
                if chunk_id != '':
                    reranked_ids.append(chunk_id)
            else:
                id_pattern = re.match(r"\(([^)]+)\)", chunk)
                chunk_id = id_pattern.group(1) if id_pattern else ''
                if chunk_id != '':
                    reranked_ids.append(chunk_id)

        print('Reranker ids:', reranked_ids)
        
        agg_position_score, agg_max_score, agg_rank_score, position_scores, max_scores, rank_scores = self.get_retriever_score(reference_retrieval, reranked_ids)
        output['reranker_generated_retrieval'] = reranked_ids
        output['reranker_position_scores'] = position_scores
        output['reranker_max_position_score'] = agg_position_score
        output['reranker_best_similarity_scores'] = max_scores
        output['reranker_max_similarity_score'] = agg_max_score
        output['reranker_rank_similarity_scores'] = rank_scores
        output['reranker_max_rank_similarity_score'] = agg_rank_score
        
        # Summary reranked appended chunks 
        summary_agent = SummaryAgent()
        judge = LLMAsJudge()
        summarized_chunks = []
        summary_scores = []
        for chunk in reranked_chunks:
            id_pattern = re.match(r"\(([^)]+)\)", chunk)

            if id_pattern:
                chunk_id = id_pattern.group(1) 
                if data_source == 'ba':
                    chunk_id = chunk_id.replace('ba', 'Banking Act')
                elif data_source == 'mas':
                    chunk_id = chunk_id.replace('mas', 'MAS')

                before, sep, after = chunk_id.rpartition('-')
                chunk_id = before + ', ' + after
                chunk_id = chunk_id.replace('-', ' ').title()
            else: 
                chunk_id = ''    

            summary = summary_agent.summarize_text(chunk, user_query)
            summary = summary.replace('\n', ' ')
            summarized_chunks.append(f'{chunk_id}: {summary}')

            # Evaluate summary using LLM As Judge
            summary_score = judge.judge_summary(user_query, chunk, summary)
            summary_score = re.findall(r'\d+', summary_score)
            summary_score = int(summary_score[0]) if summary_score else 0
            summary_scores.append(summary_score)
        final_chunks = '\n\n'.join(summarized_chunks)
        print('Summary scores:', summary_scores)
        output['summary_score'] = sum(summary_scores)/len(summary_scores)

        # Get final answer
        output_agent = OutputAgent()
        final_answer = output_agent.output_response(data_source, classification, user_query, summarized_chunks=final_chunks)
        output['reference_answer'] = reference_generation
        output['generated_answer'] = final_answer

        # Evaluate output agent with LLM As Judge 
        answer_score = judge.judge_answer(user_query, reference_generation, final_answer)
        answer_score = re.findall(r'\d+', answer_score)
        answer_score = int(answer_score[0]) if answer_score else 0
        output['answer_score'] = answer_score
        print('Answer score:', answer_score)

        print('---------------------------------------------------\n')
        return output
    
    def get_bfs_eval(self):
        bfs_queries_evaluation = []
        bfs_retriever_method = 'bfs'
        for id, row in self.df.iterrows():
            for n_hop in [1, 3, 5]:
                query = row['query']
                correct_source = row['correct_source']
                correct_type = row['correct_type']
                reference_retrieval = [row['reference_retrieval']] if ',' not in row['reference_retrieval'] else row['reference_retrieval'].split(', ')
                reference_generation = row['reference_generation']
                evaluation = self.get_evaluation(query, correct_source, correct_type, reference_retrieval, reference_generation, n_hop, bfs_retriever_method)
                bfs_queries_evaluation.append(evaluation)
        df_bfs = pd.DataFrame(bfs_queries_evaluation)
        return df_bfs

    def get_best_hop(self, df):
        df_hop_score = df.groupby('n_hops')['answer_score'].mean().reset_index()
        bfs_best_hop = int(df_hop_score.sort_values(by=['answer_score', 'n_hops'], ascending=[False, True]).iloc[0]['n_hops'])
        return bfs_best_hop
    
    def get_dfs_eval(self, n_hop):
        dfs_queries_evaluation = []
        for id, row in self.df.iterrows():
            query = row['query']
            correct_source = row['correct_source']
            correct_type = row['correct_type']
            reference_retrieval = [row['reference_retrieval']] if ',' not in row['reference_retrieval'] else row['reference_retrieval'].split(', ')
            reference_generation = row['reference_generation']
            evaluation = self.get_evaluation(query, correct_source, correct_type, reference_retrieval, reference_generation, n_hop=n_hop, retriever_method='dfs')
            dfs_queries_evaluation.append(evaluation)
        df_dfs = pd.DataFrame(dfs_queries_evaluation)
        return df_dfs
    
    def get_weighted_eval(self, n_hop):
        weighted_queries_evaluation = []
        for id, row in self.df.iterrows():
            query = row['query']
            correct_source = row['correct_source']
            correct_type = row['correct_type']
            reference_retrieval = [row['reference_retrieval']] if ',' not in row['reference_retrieval'] else row['reference_retrieval'].split(', ')
            reference_generation = row['reference_generation']
            evaluation = self.get_evaluation(query, correct_source, correct_type, reference_retrieval, reference_generation, n_hop=n_hop, retriever_method='weighted')
            weighted_queries_evaluation.append(evaluation)
        df_weighted = pd.DataFrame(weighted_queries_evaluation)
        return df_weighted
    
    def get_best_method(self, df, best_hop):
        df_method_score = df[df['n_hops'] == best_hop].groupby('graphdb_retrieval_method')['answer_score'].mean().reset_index()
        best_method = df_method_score.sort_values(by=['answer_score'], ascending=[False]).iloc[0]['graphdb_retrieval_method']
        return best_method


    def get_full_eval(self):
        print('===================================================')
        print('BFS Retrieval')
        print('===================================================')
        df_bfs = self.get_bfs_eval()
        bfs_best_hop = self.get_best_hop(df_bfs)
        print('BFS best hop:', bfs_best_hop, '\n')
        
        print('===================================================')
        print('DFS Retrieval')
        print('===================================================')
        df_dfs = self.get_dfs_eval(bfs_best_hop)

        print('===================================================')
        print('Weighted Retrieval')
        print('===================================================/n')
        df_weighted = self.get_weighted_eval(bfs_best_hop)
        
        df_combined = pd.concat([df_bfs, df_dfs, df_weighted]).reset_index(drop=True)
        best_method = self.get_best_method(df_combined, bfs_best_hop)
        print('Best retrieval method:', best_method, '\n')

        return df_combined
