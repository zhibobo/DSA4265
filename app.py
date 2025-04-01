from flask import Flask, render_template, request, session, redirect, url_for

app = Flask(__name__)

app.secret_key = "replace_with_secret_key"

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Step 1: Display a form to capture the user's initial query.
    Step 2: If POST, classify the query as 'fact' or 'reasoning'.
    Step 3: If 'fact', show final answer. If 'reasoning', prompt for more info.
    """
   
    if "step" not in session:
        session["step"] = "start"
    
    if request.method == "GET":
        return render_template("index.html",
                               step=session["step"],
                               result=None)

    if request.method == "POST":
        # STEP 1: Handling the initial query
        if session["step"] == "start":
            user_query = request.form.get("user_query", "")
            
            # Store the user query in session so we can use it later if needed
            session["user_query"] = user_query
            
            # 1) Classify the query as 'fact' or 'reasoning'
            classification = classify_query(user_query)
            
            if classification == "fact":
                # 2) If fact, get the final answer
                answer = get_fact_answer(user_query)
                
                # Render the page with the final result
                # Reset session or set a new step to allow repeated queries
                session["step"] = "start"
                return render_template("index.html",
                                       step="done",
                                       result=answer)
            else:
                # 3) If reasoning, go to next step
                session["step"] = "reasoning"
                return render_template("index.html",
                                       step="reasoning",
                                       result=None)

        # STEP 2: Handling additional user input for reasoning queries
        elif session["step"] == "reasoning":
            user_additional_info = request.form.get("user_additional_info", "")
            user_query = session.get("user_query", "")
            
            # 1) Get the final answer using the query + additional info
            answer = get_reasoning_answer(user_query, user_additional_info)
            
            # 2) Show the final result
            session["step"] = "start"
            return render_template("index.html",
                                   step="done",
                                   result=answer)

    return redirect(url_for("index"))

def classify_query(query_text):
    """
    Placeholder function that determines if the user’s query
    is fact-based or requires reasoning. (to replace with agent)
    """
    lowered = query_text.lower()
    if "what" in lowered or "when" in lowered or "is" in lowered:
        return "fact"
    else:
        return "reasoning"

def get_fact_answer(query_text):
    """
    Placeholder for retrieving a fact-based answer.
    (to replace with agent)
    """
    return f"[Fact-based Answer]: The statement '{query_text}' appears valid under our regulations."

def get_reasoning_answer(query_text, additional_info):
    """
    Placeholder for reasoning-based answer. 
    (to replace with agent)
    """
    return (f"For the query '{query_text}', "
            f"considering your additional info '{additional_info}', "
            "this action is likely permissible under regulation X, Y, Z.")

if __name__ == "__main__":
    app.run(debug=True)
