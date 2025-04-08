import os
import re
from dotenv import load_dotenv
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import openai
from weighted_vectordb_retriever import VectorDbRetriever, GraphDbRetriever, Reranker
from classify_query import QueryClassifierAgent
from summary_and_output import SummaryAgent, OutputAgent
from source_router import SourceRouterAgent


load_dotenv()

app = Flask(__name__)
app.secret_key = "REPLACE_WITH_SOME_RANDOM_SECRET_KEY"

@app.route("/", methods=["GET", "POST"])
def index():
    # Initialize chat history and step if not present
    if "chat_history" not in session:
        session["chat_history"] = []
    if "step" not in session:
        session["step"] = "start"


    intro_message = f"""I am a financial regulation checker chatbot. Currently, I have access to the following knowledge base:
    - **Banking Act 1970**
    - **MAS Fair Dealing Guidelines**
    - **MAS 626 Notice on Money Laundering Prevention**
    
    Feel free to ask questions related to any of these topics or explore the following sample questions:
    - "What are the key provisions of the Banking Act 1970?"
    - "Can you explain the MAS Fair Dealing Guidelines?"
    - "What does MAS 626 say about money laundering prevention?"
    - "How does the Banking Act 1970 impact financial institutions in Singapore?"
    - "What are the obligations for financial institutions under the MAS Fair Dealing Guidelines?"
    
    I'm here to help with your queries related to financial regulations. Ask away!
    """
    
    if request.method == "POST":
        # Depending on the current step, process the input accordingly
        if session["step"] == "start":
            # Get the initial query from the user
            user_query = request.form.get("user_query", "")
            # Append user's message to chat history
            chat_history = session["chat_history"]
            chat_history.append({"sender": "user", "message": user_query})
            session["chat_history"] = chat_history
            session["user_query"] = user_query

            # Determine data source 
            router = SourceRouterAgent()
            data_source = router.get_source(user_query)
            # print("Data Source:", data_source) 
            if 'ba' in data_source:
                data_source = 'ba'
            elif 'mas' in data_source:
                data_source = 'mas'
            else:
                chat_history.append({"sender": "bot", "message": intro_message})
                # Reset chat state to [START]
                session["chat_history"] = chat_history
                session["step"] = "start"
                return render_template("index.html", step="start")

            # Classify the query
            classifier_agent = QueryClassifierAgent()
            classification = classifier_agent.classify_query(user_query)
            # print("Classification:", classification)
            if 'factual' in classification.lower():
                classification = 'factual'
            elif 'reasoning' in classification.lower():
                classification = 'reasoning'
            else: 
                # assign misclassification as factual for simplicity
                classification = 'factual'
            session["classification"] = classification
            
            # Retrieve top-k chunks from VectorDB 
            vectorretriever = VectorDbRetriever(top_k=10)
            top_k_chunks = vectorretriever.get_top_k_chunks(user_query, data_source)
            # print("VectorDB Chunks:", top_k_chunks)

            # Append references using GraphDB 
            graphretriever = GraphDbRetriever(hops=1)
            appended_chunks = graphretriever.get_appended_chunks(top_k_chunks, data_source)
            # print("GraphDB Appended Chunks:", appended_chunks)

            # Rerank top_k appended chunks 
            reranker = Reranker(top_k=5)
            reranked_chunks = reranker.rerank(user_query, appended_chunks)
            # print("Reranker Chunks:", reranked_chunks)

            # Summarize chunks and tag with chunk ID
            summary_agent = SummaryAgent()
            summarized_chunks = []
            for chunk in reranked_chunks:
                id_pattern = re.match(r"\(([^)]+)\)", chunk)
                chunk_id = id_pattern.group(1)
                if data_source == 'ba':
                    chunk_id = chunk_id.replace('ba', 'Banking Act')
                elif data_source == 'mas':
                    chunk_id = chunk_id.replace('mas', 'MAS')

                before, sep, after = chunk_id.rpartition('-')
                chunk_id = before + ', ' + after
                chunk_id = chunk_id.replace('-', ' ').title()    

                summary = summary_agent.summarize_text(chunk, user_query)
                summary = summary.replace('\n', ' ')
                summarized_chunks.append(f'{chunk_id}: {summary}')
            final_chunks = '\n\n'.join(summarized_chunks)
            # print('Final Summarized Chunks:', final_chunks)

            output_agent = OutputAgent()
            final_answer = output_agent.output_response(data_source, classification, user_query, summarized_chunks=final_chunks)
            # print(f"Final Answer:\n{final_answer}")

            # Append bot's final answer and follow-up message to chat history
            chat_history = session["chat_history"]
            chat_history.append({"sender": "bot", "message": final_answer})
            chat_history.append({"sender": "bot", "message": "Can I help you with anything else?"})
            session["chat_history"] = chat_history

            # Reset step to "start" so conversation can continue
            session["step"] = "start"
            return render_template("index.html", step="start")

    return render_template("index.html", step=session["step"])
    
@app.route('/clear_chat_history', methods=['POST'])
def clear_chat_history():
    session['chat_history'] = []
    return jsonify({'message': 'Chat history cleared'})
    
if __name__ == "__main__":
    app.run(debug=True)
