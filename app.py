import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, session, redirect, url_for
import openai
from vectordb_retriever import VectorDbRetriever, GraphDbRetriever, Reranker
from dfs_vectordb_retriever import DFSRetriever
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

            # Classify the query
            classifier_agent = QueryClassifierAgent()
            classification = classifier_agent.classify_query(user_query)
            session["classification"] = classification

            if classification.lower() == "reasoning":
                # Append bot's request for additional context and change step
                chat_history.append({"sender": "bot", "message": "Could you please provide additional context?"})
                session["chat_history"] = chat_history
                session["step"] = "reasoning"
                return render_template("index.html", step="reasoning")
            elif classification.lower() == "factual":
                refined_query = user_query
            else:
                chat_history.append({"sender": "bot", "message": "Please enter a valid query so that I can assist you!"})
                session["chat_history"] = chat_history
                session["step"] = "start"
                return render_template("index.html", step="start")

        elif session["step"] == "reasoning":
            # Get additional context from the user
            user_additional_info = request.form.get("user_additional_info", "")
            chat_history = session["chat_history"]
            chat_history.append({"sender": "user", "message": user_additional_info})
            session["chat_history"] = chat_history
            user_query = session.get("user_query", "")
            refined_query = f"{user_query} Additional context: {user_additional_info}"
        
        # Determine Data Source using SourceRouterAgent
        router = SourceRouterAgent()
        data_source = router.get_source(refined_query)
        print("Chosen Data Source:", data_source) 
        
        # Run the retrieval pipeline using the refined_query
        vectorretriever = VectorDbRetriever(top_k=10)
        top_k_chunks = vectorretriever.get_top_k_chunks(refined_query, data_source)
        #To swap methods, just uncomment either of the 2 lines of code
        #graphretriever = GraphDbRetriever(top_k=10, hops=1)
        #appended_chunks = graphretriever.get_appended_chunks(top_k_chunks, data_source)
        dfsretriever = DFSRetriever(top_k=3, path_length=3)
        appended_chunks = dfsretriever.run_DFS(refined_query, top_k_chunks, data_source)
        reranker = Reranker(top_k=10)
        ranked_chunks = reranker.rerank(refined_query, appended_chunks)
        summary_agent = SummaryAgent()
        summarized_chunks = []
        for chunk in ranked_chunks:
            summary = summary_agent.summarize_text(chunk, refined_query)
            summarized_chunks.append(summary)
        output_agent = OutputAgent()
        final_answer = output_agent.output_response("factual", query=refined_query, summarized_chunks=summarized_chunks)

        # Append bot's final answer and follow-up message to chat history
        chat_history = session["chat_history"]
        chat_history.append({"sender": "bot", "message": final_answer})
        chat_history.append({"sender": "bot", "message": "Can I help you with anything else?"})
        session["chat_history"] = chat_history

        # Reset step to "start" so conversation can continue
        session["step"] = "start"
        return render_template("index.html", step="start")

    return render_template("index.html", step=session["step"])

if __name__ == "__main__":
    app.run(debug=True)