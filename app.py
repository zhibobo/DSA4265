import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, session, redirect, url_for
import openai
from vectordb_retriever import VectorDbRetriever, GraphDbRetriever, Reranker
from classify_query import QueryClassifierAgent
from summary_and_output import SummaryAgent, OutputAgent
from source_router import SourceRouterAgent

load_dotenv()

app = Flask(__name__)
app.secret_key = "REPLACE_WITH_SOME_RANDOM_SECRET_KEY"

# --------------------------
# Flask Route Integration
# --------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if "step" not in session:
        session["step"] = "start"

    if request.method == "GET":
        return render_template("index.html", step=session["step"], result=None)

    if request.method == "POST":
        # STEP 1: User submits initial query
        if session["step"] == "start":
            user_query = request.form.get("user_query", "")
            session["user_query"] = user_query

            # --- Classify the query first ---
            classifier_agent = QueryClassifierAgent()
            classification = classifier_agent.classify_query(user_query)
            session["classification"] = classification

            if classification.lower() == "reasoning":
                # If reasoning, immediately prompt for additional input
                session["step"] = "reasoning"
                return render_template("index.html", step="reasoning", result=None)
            elif classification.lower() == "factual":
                refined_query = user_query
            else:
                session["step"] = "start"
                return render_template("index.html", step="done", result="Error: Unrecognized classification.")

        # STEP 2: Handle additional input for reasoning queries
        elif session["step"] == "reasoning":
            user_additional_info = request.form.get("user_additional_info", "")
            user_query = session.get("user_query", "")
            refined_query = f"{user_query} Additional context: {user_additional_info}"
        
        # Determine Data Source using SourceRouterAgent
        router = SourceRouterAgent()
        data_source = router.get_source(refined_query)
        print("Chosen Data Source:", data_source) 
        
        # Run the retrieval pipeline using the refined_query
        vectorretriever = VectorDbRetriever(top_k=10)
        top_k_chunks = vectorretriever.get_top_k_chunks(refined_query, data_source)
        graphretriever = GraphDbRetriever(top_k=10, hops=1)
        appended_chunks = graphretriever.get_appended_chunks(top_k_chunks, data_source)
        reranker = Reranker(top_k=10)
        ranked_chunks = reranker.rerank(refined_query, appended_chunks)
        
        # Summarize each reranked chunk
        summary_agent = SummaryAgent()
        summarized_chunks = []
        for chunk, score in ranked_chunks:
            summary = summary_agent.summarize_text(chunk, refined_query)
            summarized_chunks.append(summary)
        
        # Output final answer
        output_agent = OutputAgent()
        final_answer = output_agent.output_response("factual", query = refined_query, summarized_chunks=summarized_chunks)
        session["step"] = "start"
        return render_template("index.html", step="done", result=final_answer)

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
