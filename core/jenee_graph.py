
import os
import logging
from typing import List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class JeneeState(TypedDict, total=False):
    """State for Jenee RAG workflow"""
    query: str
    group_ids: List[str]  # User's accessible group IDs
    query_embedding: List[float]
    retrieved_docs: List[Dict[str, Any]]
    response: str
    error: str


def embed_query_node(state: JeneeState) -> JeneeState:
    """Embed the user's query using local model for speed"""
    logger.info("Embed Query Node - Starting")
    
    try:
        query = state.get("query", "")
        if not query:
            raise ValueError("No query provided")
        
        # Use local model for fast embedding
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode(query).tolist()
        
        state["query_embedding"] = embedding
        logger.info(f"Query embedded successfully: {len(embedding)} dimensions")
        
    except Exception as e:
        logger.error(f"Error in embed_query_node: {e}")
        state["error"] = str(e)
    
    return state


def retrieve_node(state: JeneeState) -> JeneeState:
    """Retrieve relevant summaries from Qdrant with group_id isolation"""
    logger.info("Retrieve Node - Starting")
    
    try:
        query_embedding = state.get("query_embedding")
        group_ids = state.get("group_ids", [])
        
        if not query_embedding:
            raise ValueError("No query embedding available")
        
        if not group_ids:
            logger.warning("No group_ids provided, user might not have access to any groups")
            state["retrieved_docs"] = []
            return state
        
        # Connect to Qdrant
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        
        # CRITICAL: Filter by group_id for data isolation
        search_filter = Filter(
            should=[
                FieldCondition(
                    key="group_id",
                    match=MatchValue(value=str(group_id))
                )
                for group_id in group_ids
            ]
        )
        
        # Search summaries collection
        search_results = client.search(
            collection_name="summaries",
            query_vector=query_embedding,
            limit=5,
            query_filter=search_filter,
            with_payload=True
        )
        
        # Extract documents
        retrieved_docs = []
        for hit in search_results:
            doc = hit.payload.copy()
            doc["_score"] = hit.score
            retrieved_docs.append(doc)
        
        state["retrieved_docs"] = retrieved_docs
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
    except Exception as e:
        logger.error(f"Error in retrieve_node: {e}")
        state["error"] = str(e)
        state["retrieved_docs"] = []
    
    return state


def generate_node(state: JeneeState) -> JeneeState:
    """Generate response using GPT-4o with retrieved context"""
    logger.info("Generate Node - Starting")
    
    try:
        query = state.get("query", "")
        retrieved_docs = state.get("retrieved_docs", [])
        
        if not retrieved_docs:
            state["response"] = (
                "I don't have enough information to answer that question. "
                "Try asking about conversations from groups you're a member of."
            )
            return state
        
        # Build context from retrieved docs
        context_parts = []
        for idx, doc in enumerate(retrieved_docs[:5], 1):
            summary = doc.get("summary", "")
            participants = ", ".join(doc.get("participants", []))
            timestamp = doc.get("first_timestamp", "")
            
            context_parts.append(
                f"[Conversation {idx}]\n"
                f"Participants: {participants}\n"
                f"Time: {timestamp}\n"
                f"Summary: {summary}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are Jenee, a helpful AI assistant for the Dipidi messaging app.
You help users understand their group chat conversations.

Context from user's conversations:
{context}

User question: {query}

Instructions:
1. Answer based ONLY on the provided context
2. Be specific and reference relevant details
3. If the context doesn't fully answer the question, acknowledge what you know and what's unclear
4. Keep responses natural and conversational
5. Mention which conversation or participants you're referencing

Answer:"""
        
        # Call GPT-4o
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content.strip()
        state["response"] = answer
        logger.info("Response generated successfully")
        
    except Exception as e:
        logger.error(f"Error in generate_node: {e}")
        state["error"] = str(e)
        state["response"] = "I encountered an error while generating the response. Please try again."
    
    return state


class JeneeGraph:
    """Main RAG system using LangGraph"""
    
    def __init__(self):
        self.graph = self._build_graph()
        logger.info("Jenee Graph initialized successfully")
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(JeneeState)
        
        # Add nodes
        workflow.add_node("embed_query", embed_query_node)
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("generate", generate_node)
        
        # Define flow
        workflow.set_entry_point("embed_query")
        workflow.add_edge("embed_query", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def query(self, query: str, group_ids: List[str]) -> Dict[str, Any]:
        """
        Execute RAG query
        
        Args:
            query: User's question
            group_ids: List of group IDs the user has access to
        
        Returns:
            Dict with 'response' and 'retrieved_docs'
        """
        logger.info(f"Processing query: {query[:100]}")
        
        initial_state: JeneeState = {
            "query": query,
            "group_ids": group_ids,
            "query_embedding": [],
            "retrieved_docs": [],
            "response": "",
            "error": ""
        }
        
        final_state = self.graph.invoke(initial_state)
        
        return {
            "response": final_state.get("response", ""),
            "retrieved_docs": final_state.get("retrieved_docs", []),
            "error": final_state.get("error", "")
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    jenee = JeneeGraph()
    
    # Test query
    result = jenee.query(
        query="What did we discuss about the weekend?",
        group_ids=["1", "2"]  # Example group IDs
    )
    
    print("Response:", result["response"])
    print(f"Retrieved {len(result['retrieved_docs'])} documents")
