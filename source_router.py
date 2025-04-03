import os
from dotenv import load_dotenv
import openai

load_dotenv()

# Agent to determine the data source of the query
class SourceRouterAgent:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=self.openai_key)

    def get_source(self, query):
        """
        Takes in user query and get the most suitable data source to answer the query.
        
        :query: User query
        :return: "ba" or "mas"
        """

        prompt = f"""
        Choose the most appropriate data source to answer the query below. Classify it as either 'ba' or 'mas':

        Query: {query}

        - **'ba' (Banking Act 1970)**: Regulates the licensing, operation, and supervision of banks and financial institutions in Singapore. It covers bank licensing, capital adequacy, risk management, liquidity, deposit-taking, mergers, and advertising. The Act aims to ensure financial stability by protecting depositors and mitigating risks, with enforcement by the Monetary Authority of Singapore (MAS).

        - **'mas' (MAS Financial Regulation Documents)**: Includes two key documents:
        1. **Fair Dealing Guidelines (30 May 2024)** – Provides a framework for the ethical and legal use of copyrighted materials, ensuring compliance with copyright laws while permitting limited usage for education, research, news reporting, and criticism. It defines fair dealing criteria such as purpose, nature, and market impact.
        2. **MAS 626 Notice** – A directive issued by MAS to prevent money laundering and terrorist financing in financial institutions. It mandates compliance measures such as customer due diligence, transaction monitoring, and reporting obligations to mitigate financial crime risks.

        Output only the classification ('ba' or 'mas'), without additional text.
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "developer", 
                    "content": """You are an AI assistant that classifies queries based on relevant financial and regulatory documents. Your task is to determine whether a given query should be answered using the **Banking Act 1970 ('ba')** or **MAS Financial Regulation documents ('mas')**, which include the **Fair Dealing Guidelines (30 May 2024)** and **MAS 626 Notice on Money Laundering Prevention**."""},
                    {"role": "user", 
                     "content": prompt}]
        )
        return response.choices[0].message.content

if __name__ == "__main__":

    router = SourceRouterAgent()

    # Banking Act Example 
    ba_query = "What are the capital and liquidity requirements for banks in Singapore?"
    ba_source = router.get_source(ba_query)
    print('BA Example source:', ba_source)

    # MAS Example 
    mas_query = "How does fair dealing differ from copyright infringement?"
    mas_source = router.get_source(mas_query)
    print('MAS Example source:', mas_source)