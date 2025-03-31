from dotenv import load_dotenv
import os
import json
from typing import List, Tuple
from dotenv import load_dotenv
import openai
from pinecone import Pinecone

load_dotenv()


class PineconeUploader:
    def __init__(self, index_name: str):
        load_dotenv()

        self.index_name = index_name
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

        self.pc = Pinecone(api_key=self.api_key)
        self.index = self._get_index()

        self.client = openai.OpenAI(api_key=self.openai_key)

    def _get_index(self):
        if self.index_name not in self.pc.list_indexes().names():
            raise ValueError(f"Index '{self.index_name}' does not exist.")
        return self.pc.Index(self.index_name)

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        embeds = [x.embedding for x in response.data]
        return embeds[0]


    def load_json(self, path: str) -> List[dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def prepare_vectors(self, data: List[dict]) -> List[Tuple[str, List[float], dict]]:
        vectors = []
        for item in data:
            vector_id = item["id"]
            text = item["text"]
            metadata = item["metadata"]
            metadata["text"] = text
            embedding = self.get_embedding(text)
            vectors.append((vector_id, embedding, metadata))
        return vectors

    def upload_vectors(self, vectors: List[Tuple[str, List[float], dict]], batch_size: int = 100):
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
            print(f"✅ Upserted batch {i // batch_size + 1} with {len(batch)} records.")

    def upload_json_to_index(self, json_path: str):
        print(f"📥 Loading data from {json_path}...")
        data = self.load_json(json_path)

        print("🔎 Generating embeddings and preparing vectors...")
        vectors = self.prepare_vectors(data)

        print(f"📤 Uploading {len(vectors)} vectors to Pinecone index '{self.index_name}'...")
        self.upload_vectors(vectors)
        print("✅ Upload complete.")

# Uncomment this code to upload
# if __name__ == "__main__":
#     uploader = PineconeUploader(index_name=os.getenv("PINECONE_INDEX_MAS_NAME"))
#     uploader.upload_json_to_index("../mas_data_extraction/mas_chunks.json")
#     uploader = PineconeUploader(index_name=os.getenv("PINECONE_INDEX_BA_NAME"))
#     uploader.upload_json_to_index("../ba_data_extraction/banking_act.json")
