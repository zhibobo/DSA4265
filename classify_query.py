import openai
import os
from dotenv import load_dotenv

load_dotenv()

#Agent to determine if query is 'Factual" or "Reasoning"
class QueryClassifierAgent:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=self.openai_key)


    def classify_query(self, query):
        """
        Takes in user query and classifies it by outputing "Factual" or "Reasoning". LLM used: gpt-4o-mini
        
        :query: User query
        :return: "Factual" or "Reasoning"
        """

        prompt = f"""
        Classify the following question as 'Factual' or 'Reasoning':
        
        {query}
        
        A factual question can be directly answered using retrieved context, while a reasoning question requires inference.
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "developer", 
                    "content": """You are an agent for financial regulations chatbot. You are to classify if the user's question is factual or reasoning.
                    Factual questions can be directly answered using retrieved context, while a reasoning question requires inference or deduction by 
                    analyzing relationships, applying logical steps, or drawing conclusions based on multiple pieces of evidence"""},
                    {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    agent = QueryClassifierAgent()
    
    #Test
    sample_query = "What is the limit on equity investments for banks in Singapore?"
    print(agent.classify_query(sample_query))