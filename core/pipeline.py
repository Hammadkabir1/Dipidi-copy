import json
import os
import re
import logging
from typing import Dict, Any, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
import joblib
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DipidiState(TypedDict, total=False):
    raw_messages: List[Dict[str, Any]]
    processed_messages: List[Dict[str, Any]]
    potential_plans: List[Dict[str, Any]]
    discarded_messages: List[Dict[str, Any]]
    confirmed_plans: List[Dict[str, Any]]
    intent_results: List[Dict[str, Any]]
    extracted_plans: List[Dict[str, Any]]
    message_counter: int
    summarizer_batch: List[Dict[str, Any]]
    summaries: List[Dict[str, Any]]
    current_node: str
    pipeline_status: str
    error_message: str
    model_path: str


def message_ingest_node(state: DipidiState) -> DipidiState:
    logger.info("Message Ingest Node - Starting")

    try:
        raw_messages = state.get("raw_messages", [])
        if not raw_messages:
            raise ValueError("No raw messages provided")

        processed_messages: List[Dict[str, Any]] = []

        emoji_pattern = re.compile(
            "["                      
            "\U0001F600-\U0001F64F"  
            "\U0001F300-\U0001F5FF"  
            "\U0001F680-\U0001F6FF"  
            "\U0001F1E0-\U0001F1FF"  
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )

        system_patterns = [
            r"\buser .* joined\b",
            r"\buser .* left\b",
            r".* added .*",
            r".* removed .*",
            r"\bgroup created\b",
            r"\bgroup name changed\b",
        ]

        for message in raw_messages:
            text_original = message.get("text", "") or ""

            text_clean = text_original.lower()
            text_clean = re.sub(r"\s+", " ", text_clean)
            text_clean = emoji_pattern.sub(" ", text_clean)
            text_clean = re.sub(r"[^\w\s.,!?-]", " ", text_clean)
            text_clean = re.sub(r"\s+", " ", text_clean).strip()

            is_valid = True
            if not text_clean or len(text_clean.split()) < 3:
                is_valid = False

            if any(re.search(p, text_clean, flags=re.IGNORECASE) for p in system_patterns):
                is_valid = False

            processed_message = {
                "_id": message.get("_id", ""),
                "user": message.get("user", ""),
                "group_id": message.get("group_id", ""),
                "text": text_original,
                "timestamp": message.get("timestamp", ""),
                "text_clean": text_clean,
                "is_valid": is_valid,
            }

            if message.get("_id"):
                processed_message["_id"] = message["_id"]

            processed_messages.append(processed_message)

        state["processed_messages"] = processed_messages
        state["current_node"] = "message_ingest"
        state["pipeline_status"] = "ingest_complete"

        logger.info(f"Message Ingest completed: {len(processed_messages)} messages processed")

    except Exception as e:
        logger.error(f"Error in Message Ingest Node: {e}")
        state["pipeline_status"] = "error"
        state["error_message"] = str(e)

    return state


def lightweight_filter_node(state: DipidiState) -> DipidiState:
    logger.info("ML Filter Node - Starting")

    try:
        processed_messages = state.get("processed_messages", [])
        if not processed_messages:
            raise ValueError("No processed messages available")

        model_path = state.get("model_path", "training/plan_detection_model.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ML model not found at: {model_path}")

        try:
            model = joblib.load(model_path)
            logger.info("ML model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load ML model: {e}")

        threshold = 0.55
        potential_plans: List[Dict[str, Any]] = []
        discarded_messages: List[Dict[str, Any]] = []

        for message in processed_messages:
            final_message = {
                "_id": message.get("_id", ""),
                "user": message.get("user", ""),
                "group_id": message.get("group_id", ""),
                "text": message.get("text", ""),
                "timestamp": message.get("timestamp", ""),
            }

            if not message.get("is_valid", False):
                final_message["ml_confidence"] = 0.0
                final_message["tag"] = "invalid"
                discarded_messages.append(final_message)
                continue

            text = message["text_clean"]

            if len(text.split()) < 3:
                final_message["ml_confidence"] = 0.0
                final_message["tag"] = "noise"
                discarded_messages.append(final_message)
                continue

            if model:
                try:
                    probabilities = model.predict_proba([text])[0]
                    labels = list(model.classes_)
                    plan_idx = labels.index("plan") if "plan" in labels else 0
                    plan_prob = float(probabilities[plan_idx])

                    final_message["ml_confidence"] = plan_prob

                    if plan_prob >= threshold:
                        final_message["tag"] = "potential_plan"
                        potential_plans.append(final_message)
                    else:
                        final_message["tag"] = "noise"
                        discarded_messages.append(final_message)

                except Exception as e:
                    logger.error(f"ML prediction failed for message: {e}")
                    raise RuntimeError(f"ML model prediction failed: {e}")
            else:
                raise ValueError("ML model is required but not available")

        state["potential_plans"] = potential_plans
        state["discarded_messages"] = discarded_messages
        state["current_node"] = "lightweight_filter"
        state["pipeline_status"] = "filter_complete"

        logger.info(
            f"ML Filter completed: "
            f"{len(potential_plans)} potential plans, {len(discarded_messages)} discarded"
        )

    except Exception as e:
        logger.error(f"Error in ML Filter Node: {e}")
        state["pipeline_status"] = "error"
        state["error_message"] = str(e)

    return state


def intent_router_node(state: DipidiState) -> DipidiState:
    logger.info("Intent Router Node - Starting")
    
    try:
        potential_plans = state.get("potential_plans", [])
        all_messages = state.get("raw_messages", [])
        
        if not potential_plans:
            state["confirmed_plans"] = []
            state["intent_results"] = []
            state["extracted_plans"] = []
            state["current_node"] = "intent_router"
            state["pipeline_status"] = "router_complete"
            logger.info("Intent Router completed: 0 plans to verify")
            return state
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        confirmed_plans = []
        intent_results = []
        extracted_plans = []
        processed_groups = set()  
        
        plan_groups = {}
        for plan_msg in potential_plans:
            group_id = plan_msg.get("group_id", "unknown")
            if group_id not in plan_groups:
                plan_groups[group_id] = []
            plan_groups[group_id].append(plan_msg)
        
        for group_id, group_plans in plan_groups.items():
            if group_id in processed_groups:
                continue

            group_plans.sort(key=lambda x: x.get("timestamp", ""))

            group_msg_ids = [msg.get("_id") for msg in group_plans]

            start_idx = len(all_messages)
            end_idx = 0

            for msg in all_messages:
                if msg.get("_id") in group_msg_ids:
                    msg_idx = all_messages.index(msg)
                    start_idx = min(start_idx, msg_idx)
                    end_idx = max(end_idx, msg_idx)

            start_idx = max(0, start_idx - 10)
            end_idx = min(len(all_messages), end_idx + 10)
            context_messages = all_messages[start_idx:end_idx + 1]

            context_messages = [msg for msg in context_messages if msg.get("group_id") == group_id]
            
            context_text = "\n".join([
                f"{msg.get('user', 'Unknown')} ({msg.get('timestamp', '')}): {msg.get('text', '')}"
                for msg in context_messages
            ])
            
            prompt = f"""Analyze this group conversation and extract ONE consolidated plan if it exists.

Conversation:
{context_text}

Look for:
1. A concrete plan being discussed and agreed upon
2. Specific details about what, when, where
3. Participants who are involved/committed

If there IS a concrete plan being made, respond with JSON:
{{
    "has_plan": true,
    "plan_description": "Detailed description of what they're planning to do",
    "execution_date_time": "When the plan will be executed (date/time mentioned)",
    "location": "Where it will happen",
    "participants": ["list", "of", "people", "involved"],
    "discussion_period": "Time range when this plan was discussed",
    "key_messages": ["important messages that define the plan"]
}}

If there is NO concrete plan (just casual chat), respond:
{{
    "has_plan": false
}}"""

            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,
                    temperature=0
                )
                
                result_text = response.choices[0].message.content.strip()
                
                try:
                    import json
                    result = json.loads(result_text)
                    has_plan = result.get("has_plan", False)
                except:
                    has_plan = "true" in result_text.lower() and "has_plan" in result_text.lower()
                    result = {"has_plan": has_plan}

                for plan_msg in group_plans:
                    intent_data = {
                        "_id": plan_msg.get("_id"),
                        "user": plan_msg.get("user"),
                        "group_id": plan_msg.get("group_id"),
                        "text": plan_msg.get("text"),
                        "timestamp": plan_msg.get("timestamp"),
                        "ml_confidence": plan_msg.get("ml_confidence"),
                        "openai_result": "PLAN" if has_plan else "NOISE",
                        "context": context_text[:500] + "..." if len(context_text) > 500 else context_text
                    }
                    intent_results.append(intent_data)

                    if has_plan:
                        plan_msg["tag"] = "confirmed_plan"
                        confirmed_plans.append(plan_msg)
                    else:
                        plan_msg["tag"] = "false_positive"
                        state.setdefault("discarded_messages", []).append(plan_msg)

                if has_plan:
                    first_msg = min(group_plans, key=lambda x: x.get("timestamp", ""))
                    last_msg = max(group_plans, key=lambda x: x.get("timestamp", ""))

                    plan_description = result.get("plan_description", "")
                    if not plan_description:
                        key_messages = [msg.get("text", "") for msg in group_plans[:3]]
                        plan_description = f"Plan discussed in group {group_id}: " + " | ".join(key_messages)

                    participants = result.get("participants", [])
                    if not participants:
                        participants = list(set([msg.get("user", "") for msg in group_plans if msg.get("user")]))
                    
                    plan_data = {
                        "_id": f"plan_{group_id}_{first_msg.get('timestamp', '').replace(':', '').replace('-', '')}",
                        "group_id": group_id,
                        "plan_description": plan_description,
                        "execution_date_time": result.get("execution_date_time", "Not specified"),
                        "location": result.get("location", "Not specified"),
                        "participants": participants,
                        "discussion_start": first_msg.get("timestamp", ""),
                        "discussion_end": last_msg.get("timestamp", ""),
                        "key_messages": result.get("key_messages", [msg.get("text", "") for msg in group_plans]),
                        "related_message_ids": [msg.get("_id") for msg in group_plans],
                        "created_at": last_msg.get("timestamp", "")
                    }
                    extracted_plans.append(plan_data)
                    
            except Exception as e:
                logger.error(f"OpenAI API error for group {group_id}: {e}")
                for plan_msg in group_plans:
                    plan_msg["tag"] = "api_error"
                    state.setdefault("discarded_messages", []).append(plan_msg)
                    
                    intent_data = {
                        "_id": plan_msg.get("_id"),
                        "user": plan_msg.get("user"),
                        "group_id": plan_msg.get("group_id"),
                        "text": plan_msg.get("text"),
                        "timestamp": plan_msg.get("timestamp"),
                        "ml_confidence": plan_msg.get("ml_confidence"),
                        "openai_result": "ERROR",
                        "context": context_text[:500] + "..." if len(context_text) > 500 else context_text,
                        "error": str(e)
                    }
                    intent_results.append(intent_data)
            
            processed_groups.add(group_id)
        
        state["confirmed_plans"] = confirmed_plans
        state["intent_results"] = intent_results
        state["extracted_plans"] = extracted_plans
        state["current_node"] = "intent_router"
        state["pipeline_status"] = "router_complete"
        
        logger.info(f"Intent Router completed: {len(confirmed_plans)} confirmed plans, {len(extracted_plans)} consolidated plans extracted")
        
    except Exception as e:
        logger.error(f"Error in Intent Router Node: {e}")
        state["pipeline_status"] = "error"
        state["error_message"] = str(e)
    
    return state


def summarizer_node(state: DipidiState) -> DipidiState:
    logger.info("Summarizer Node - Starting")
    
    try:
        current_counter = state.get("message_counter", 0)
        batch = state.get("summarizer_batch", [])
        all_messages = state.get("raw_messages", [])

        batch.extend(all_messages)
        current_counter += len(all_messages)

        summaries = state.get("summaries", [])

        if len(batch) >= 20:
            messages_to_summarize = batch[:min(50, len(batch))]
            remaining_batch = batch[len(messages_to_summarize):]
            
            conversation_text = "\n".join([
                f"{msg.get('user', 'Unknown')} ({msg.get('timestamp', '')}): {msg.get('text', '')}"
                for msg in messages_to_summarize
            ])
            
            prompt = f"""Summarize this conversation focusing on key topics, decisions, and important information.

Conversation:
{conversation_text}

Provide a concise summary highlighting:
1. Main topics discussed
2. Key decisions or agreements
3. Important information shared
4. Notable participants and their contributions"""

            try:
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.3
                )
                
                summary_text = response.choices[0].message.content.strip()
                
                summary_data = {
                    "_id": f"summary_{len(summaries) + 1}",
                    "message_count": len(messages_to_summarize),
                    "start_timestamp": messages_to_summarize[0].get("timestamp", ""),
                    "end_timestamp": messages_to_summarize[-1].get("timestamp", ""),
                    "participants": list(set(msg.get("user", "Unknown") for msg in messages_to_summarize)),
                    "summary": summary_text,
                    "created_at": messages_to_summarize[-1].get("timestamp", "")
                }
                
                summaries.append(summary_data)
                logger.info(f"Generated summary for {len(messages_to_summarize)} messages")
                
                state["summarizer_batch"] = remaining_batch
                state["message_counter"] = len(remaining_batch)
                
            except Exception as e:
                logger.error(f"OpenAI API error in summarizer: {e}")
                state["summarizer_batch"] = batch
                state["message_counter"] = current_counter
        else:
            state["summarizer_batch"] = batch
            state["message_counter"] = current_counter
        
        state["summaries"] = summaries
        state["current_node"] = "summarizer"
        state["pipeline_status"] = "summarizer_complete"
        
        logger.info(f"Summarizer completed: {len(summaries)} summaries generated")
        
    except Exception as e:
        logger.error(f"Error in Summarizer Node: {e}")
        state["pipeline_status"] = "error"
        state["error_message"] = str(e)
    
    return state


class DipidiPipeline:

    def __init__(self, model_path: str = "training/plan_detection_model.pkl"):
        self.model_path = model_path
        self.graph = self._build_graph()

    def display_graph_structure(self):
        """Display the LangGraph's built-in graph structure"""
        try:
            # Get the graph representation from LangGraph
            print("\n" + "="*60)
            print("LANGGRAPH PIPELINE STRUCTURE")
            print("="*60)
            
            # Print the graph's string representation
            print(self.graph.get_graph().print_ascii())
            
            print("="*60 + "\n")
        except Exception as e:
            print(f"Could not display graph structure: {e}")
            # Fallback to basic info
            print("Graph nodes:", list(self.graph.get_graph().nodes.keys()))
            print("Graph edges:", list(self.graph.get_graph().edges))

    def _build_graph(self):
        workflow = StateGraph(DipidiState)
        
        workflow.add_node("message_ingest", message_ingest_node)
        workflow.add_node("lightweight_filter", lightweight_filter_node)
        workflow.add_node("intent_router", intent_router_node)
        workflow.add_node("summarizer", summarizer_node)
        
        workflow.set_entry_point("message_ingest")
        workflow.add_edge("message_ingest", "lightweight_filter")
        workflow.add_edge("lightweight_filter", "intent_router")
        workflow.add_edge("intent_router", "summarizer")
        workflow.add_edge("summarizer", END)
        
        return workflow.compile()

    def run(self, raw_messages: List[Dict[str, Any]]) -> DipidiState:
        logger.info("Starting Dipidi Pipeline")

        initial_state: DipidiState = {
            "raw_messages": raw_messages,
            "processed_messages": [],
            "potential_plans": [],
            "discarded_messages": [],
            "confirmed_plans": [],
            "intent_results": [],
            "extracted_plans": [],
            "message_counter": 0,
            "summarizer_batch": [],
            "summaries": [],
            "current_node": "",
            "pipeline_status": "starting",
            "error_message": "",
            "model_path": self.model_path,
        }

        final_state = self.graph.invoke(initial_state)
        logger.info(f"Pipeline completed: {final_state.get('pipeline_status', 'unknown')}")
        return final_state

    def run_from_file(self, file_path: str) -> DipidiState:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_messages = json.load(f)
            return self.run(raw_messages)
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return {
                "raw_messages": [],
                "processed_messages": [],
                "potential_plans": [],
                "discarded_messages": [],
                "confirmed_plans": [],
                "intent_results": [],
                "extracted_plans": [],
                "message_counter": 0,
                "summarizer_batch": [],
                "summaries": [],
                "current_node": "",
                "pipeline_status": "error",
                "error_message": str(e),
            }

    def save_graph_diagram(self, out_path: str = "pipeline_graph.mmd"):
        try:
            graph_obj = self.graph.get_graph()

            if hasattr(graph_obj, 'draw_mermaid'):
                mermaid_code = graph_obj.draw_mermaid()
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(mermaid_code)
                logger.info(f"Saved graph diagram to '{out_path}'")
                logger.info(f"Open {out_path} and copy content to https://mermaid.live for visualization")
                return True
            else:
                mermaid_code = """graph TD
    A[message_ingest] --> B[lightweight_filter]
    B --> C[intent_router]
    C --> D[summarizer]
    D --> E[END]"""

                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(mermaid_code)
                logger.info(f"Saved basic graph diagram to '{out_path}'")
                logger.info(f"Open {out_path} and copy content to https://mermaid.live for visualization")
                return True

        except Exception as e:
            logger.error(f"Could not save graph diagram: {e}")
            return False
