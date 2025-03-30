import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
from ba_data_extraction.banking_act_preprocessor import BankingActPreprocessor

preprocessor = BankingActPreprocessor()
chunk_list = preprocessor.get_chunks()

load_dotenv()

#BASIC CONFIGURATION
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

#Agent to determine if query is 'Factual" or "Reasoning"
def classify_query(query):
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

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "developer", 
                   "content": """You are an agent for financial regulations chatbot. You are to classify if the user's question is factual or reasoning.
                   Factual questions can be directly answered using retrieved context, while a reasoning question requires inference or deduction by 
                   analyzing relationships, applying logical steps, or drawing conclusions based on multiple pieces of evidence"""},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

"""
#Test
query1 = "If a company's revenue has increased by 20% year-over-year, but its net profit has declined, what could be the possible reasons?"
print(f"The response for query 1 is: {classify_query(query1)}")

query2 = "What was the company's revenue and net profit last year?"
print(f"The response for query 1 is: {classify_query(query2)}")
"""
#Retrieve top chunk
"""
Defined dense_index in pinecone

results = dense_index.search(
    namespace="financial_docs_context", #defined by Hans
    query={
        "top_k": 1,
        "inputs": {
            'text': query
        }
    }
)
"""
#print(results)
# Print the results for multiple
#for hit in results['result']['hits']:
#    print(f"id: {hit['_id']}, score: {round(hit['_score'], 2)}, text: {hit['fields']['chunk_text']}, category: {hit['fields']['category']}")

sample_chunk = {'id': 'ba-1970-5.2a',
  'text': '5(2A) Sections 4A(3) and 4B(1), (2) and (3) apply, with the necessary modifications, to an advertisement made by a representative office mentioned in subsection (2)(a).',
  'metadata': {'part_id': '3',
   'part_title': 'Licensing of banks',
   'section_id': '5',
   'section_title': 'Use of word “bank”',
   'regulation': 'Banking Act 1970',
   'references': ['ba-1970-4a.3',
    'ba-1970-4b.1',
    'ba-1970-4b.2',
    'ba-1970-4b.3',
    'ba-1970-5.2']}}

def extract_metadata(chunk):
    """
    Takes in a chunk and outputs all other chunk that has been referenced in its text

    :chunk: top 1 similarity score chunk retrieved after query 
    :
    """
    return (chunk['metadata'])['references']

def retrieve_reference(reference):
    for chunk in chunk_list:
        if chunk['id'] == reference:
            return chunk['text']

def append_references(chunk):
    reference_list = extract_metadata(chunk)
    top_chunk_with_reference = chunk['text']
    for references in reference_list:
        top_chunk_with_reference += f"### {retrieve_reference(references)}"
    return top_chunk_with_reference

print(append_references(sample_chunk))