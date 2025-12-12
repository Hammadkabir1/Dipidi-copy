import os
import logging
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class QdrantConnector:

    def __init__(self):
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.cluster_id = os.getenv("QDRANT_CLUSTER_ID")
        self.client = None
        self.embedding_model = None
        self.openai_client = None
        self.use_openai_embeddings = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"

        self.collections = {
            "messages": "dipidi_messages",
            "processed": "dipidi_processed",
            "intent": "dipidi_intent",
            "plans": "dipidi_plans",
            "summaries": "dipidi_summaries"
        }
        
    def connect(self) -> bool:
        try:
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
            )

            # Initialize embedding model(s)
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            if self.use_openai_embeddings:
                self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                logger.info("Using OpenAI embeddings for high-quality vectors")

            collections_info = self.client.get_collections()
            logger.info(f"Connected to Qdrant cluster: {self.cluster_id}")
            logger.info(f"Available collections: {[c.name for c in collections_info.collections]}")

            self._ensure_collections_exist()
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            return False
    
    def _ensure_collections_exist(self):
        try:
            existing_collections = {c.name for c in self.client.get_collections().collections}
            
            for collection_type, collection_name in self.collections.items():
                if collection_name not in existing_collections:
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                    )
                    logger.info(f"Created collection: {collection_name}")
                    
        except Exception as e:
            logger.error(f"Error ensuring collections exist: {e}")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI or local model"""
        try:
            if self.use_openai_embeddings and self.openai_client:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                return response.data[0].embedding
            else:
                if self.embedding_model is None:
                    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                return self.embedding_model.encode(text).tolist()
        except Exception as e:
            logger.warning(f"Embedding generation failed, using local fallback: {e}")
            if self.embedding_model is None:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            return self.embedding_model.encode(text).tolist()
    
    def _prepare_message_for_embedding(self, message: Dict[str, Any]) -> str:
        text_parts = []
        if message.get('content'):
            text_parts.append(str(message['content']))
        if message.get('author'):
            text_parts.append(f"Author: {message['author']}")
        if message.get('timestamp'):
            text_parts.append(f"Time: {message['timestamp']}")
        return " ".join(text_parts)
    
    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if self.client is None:
            if not self.connect():
                logger.error("Cannot retrieve messages: No database connection")
                return []
        
        try:
            collection_name = self.collections["messages"]
            
            result = self.client.scroll(
                collection_name=collection_name,
                limit=limit or 1000,
                with_payload=True,
                with_vectors=False
            )
            
            messages = []
            for point in result[0]:
                message = point.payload
                message['_id'] = str(point.id)
                messages.append(message)
            
            logger.info(f"Retrieved {len(messages)} messages from Qdrant")
            return messages
            
        except Exception as e:
            logger.error(f"Error retrieving messages: {e}")
            return []
    
    def _prepare_embedding_text(self, item: Dict[str, Any], collection_type: str) -> str:
        if collection_type == "processed" or collection_type == "messages":
            return self._prepare_message_for_embedding(item)
        elif collection_type == "intent":
            return f"Intent: {item.get('intent', '')} Message: {item.get('message_content', '')}"
        elif collection_type == "plans":
            return f"Plan: {item.get('plan_description', '')} Participants: {', '.join(item.get('participants', []))}"
        elif collection_type == "summaries":
            return f"Summary: {item.get('summary', '')} Topic: {item.get('topic', '')}"
        return str(item)

    def _insert_to_collection(self, data: List[Dict[str, Any]], collection_type: str, data_label: str) -> bool:
        if self.client is None:
            if not self.connect():
                logger.error(f"Cannot insert {data_label}: No database connection")
                return False

        try:
            if not data:
                return True

            collection_name = self.collections[collection_type]
            points = []

            for item in data:
                text_for_embedding = self._prepare_embedding_text(item, collection_type)
                embedding = self._generate_embedding(text_for_embedding)

                point_id = str(uuid.uuid4())
                payload = {k: v for k, v in item.items() if k != '_id'}

                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
                points.append(point)

            self.client.upsert(
                collection_name=collection_name,
                points=points
            )

            logger.info(f"Inserted {len(data)} {data_label}")
            return True

        except Exception as e:
            logger.error(f"Error inserting {data_label}: {e}")
            return False

    def insert_processed_messages(self, messages: List[Dict[str, Any]]) -> bool:
        return self._insert_to_collection(messages, "processed", "processed messages")

    def insert_intent_results(self, intent_data: List[Dict[str, Any]]) -> bool:
        return self._insert_to_collection(intent_data, "intent", "intent results")

    def insert_plans(self, plans: List[Dict[str, Any]]) -> bool:
        return self._insert_to_collection(plans, "plans", "plans")

    def insert_summaries(self, summaries: List[Dict[str, Any]]) -> bool:
        return self._insert_to_collection(summaries, "summaries", "summaries")

    def insert_messages(self, messages: List[Dict[str, Any]]) -> bool:
        return self._insert_to_collection(messages, "messages", "messages")

    def search_similar_messages(self, query_text: str, limit: int = 10, collection_type: str = "messages") -> List[Dict[str, Any]]:
        if self.client is None:
            if not self.connect():
                logger.error("Cannot search messages: No database connection")
                return []
        
        try:
            collection_name = self.collections.get(collection_type, self.collections["messages"])
            query_embedding = self._generate_embedding(query_text)
            
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True
            )
            
            results = []
            for hit in search_result:
                result = hit.payload
                result['_id'] = str(hit.id)
                result['similarity_score'] = hit.score
                results.append(result)
            
            logger.info(f"Found {len(results)} similar messages for query: {query_text[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar messages: {e}")
            return []

    def close(self):
        if self.client:
            self.client.close()
            logger.info("Qdrant connection closed")