import streamlit as st
import streamlit.components.v1 as components
import sys
import os
import base64

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.jenee_graph import JeneeGraph
from datetime import datetime
from st_realtime_audio import realtime_audio_conversation

st.set_page_config(
    page_title="Jenee - AI Assistant",
    page_icon="💜",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0E0E0E;
    }
    
    [data-testid="stSidebar"] {
        background-color: #1A1A1A;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #E0E0E0 !important;
    }
    
    .header-container {
        background: linear-gradient(135deg, #6B46C1 0%, #9333EA 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(107, 70, 193, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1.5rem;
    }
    
    .header-logo {
        width: 80px;
        height: 80px;
    }
    
    .header-content {
        display: flex;
        flex-direction: column;
    }
   
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1rem;
        margin-top: 0.3rem;
    }
    
    .chat-container {
        background-color: #1A1A1A;
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 1rem;
        height: 600px;
        overflow-y: auto;
    }
    
    .user-message {
        background: linear-gradient(135deg, #6B46C1 0%, #7C3AED 100%);
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.75rem 0;
        margin-left: auto;
        max-width: 75%;
        box-shadow: 0 2px 8px rgba(107, 70, 193, 0.3);
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .assistant-message {
        background-color: #2A2A2A;
        color: #E0E0E0;
        padding: 1rem 1.25rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.75rem 0;
        margin-right: auto;
        max-width: 75%;
        border-left: 3px solid #9333EA;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .message-time {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.5);
        margin-top: 0.3rem;
    }
    
    .stTextInput input {
        background-color: #2A2A2A !important;
        color: #E0E0E0 !important;
        border: 1px solid #3A3A3A !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.95rem !important;
    }
    
    .stTextInput input:focus {
        border-color: #9333EA !important;
        box-shadow: 0 0 0 2px rgba(147, 51, 234, 0.2) !important;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #6B46C1 0%, #9333EA 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(107, 70, 193, 0.3);
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(107, 70, 193, 0.4);
    }
    
    .stMultiSelect [data-baseweb="select"] {
        background-color: #2A2A2A;
        border-radius: 12px;
    }
    
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #6B46C1;
    }
    
    .info-box {
        background-color: #2A2A2A;
        border-left: 4px solid #9333EA;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #E0E0E0;
        font-size: 0.9rem;
    }
    
    .welcome-box {
        background-color: #2A2A2A;
        border: 2px solid #3A3A3A;
        border-radius: 12px;
        padding: 3rem;
        text-align: center;
        margin-top: 3rem;
    }
    
    .welcome-title {
        color: #9333EA;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .welcome-text {
        color: #B0B0B0;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }
    
    .example-query {
        background-color: #1A1A1A;
        border: 1px solid #3A3A3A;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.5rem;
        display: inline-block;
        color: #9333EA;
        font-size: 0.85rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: #1A1A1A;
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #B0B0B0;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #2A2A2A;
        color: #E0E0E0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6B46C1 0%, #9333EA 100%);
        color: white !important;
    }
    
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1A1A1A;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #6B46C1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #7C3AED;
    }
    
    .block-container {
        padding-top: 2rem;
    }
    
    h1, h2, h3, p, label, span, div {
        color: #E0E0E0 !important;
    }
    
    .success-badge {
        background-color: rgba(107, 70, 193, 0.2);
        border: 1px solid #6B46C1;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        color: #9333EA;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'jenee' not in st.session_state:
    st.session_state.jenee = JeneeGraph()

if 'selected_groups' not in st.session_state:
    st.session_state.selected_groups = []

logo_path = os.path.join(os.path.dirname(__file__), "dipidi_logo.png")
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_data = base64.b64encode(f.read()).decode()
    
    st.markdown(f"""
    <div class="header-container">
        <img src="data:image/png;base64,{logo_data}" class="header-logo" alt="Dipidi Logo">
        <div class="header-content">
            <div class="header-title">Jenee</div>
            <div class="header-subtitle">Your AI assistant for group conversations</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="header-container">
        <div class="header-content">
            <div class="header-title">Jenee</div>
            <div class="header-subtitle">Your AI assistant for group conversations</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Group Selection")
    
    st.markdown("""
    <div class="info-box">
        Select the group IDs you want to search through. Each group represents a conversation from the dataset.
    </div>
    """, unsafe_allow_html=True)
    
    available_group_ids = [str(i) for i in range(1, 101)]
    
    selected_groups = st.multiselect(
        "Select Group IDs",
        options=available_group_ids,
        default=["1", "2", "3"],
        help="Choose which group conversations to search"
    )
    
    st.session_state.selected_groups = selected_groups
    
    if st.session_state.selected_groups:
        st.markdown(f"""
        <div class="success-badge">
            Searching across {len(st.session_state.selected_groups)} groups
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Please select at least one group")
    
    st.markdown("---")
    
    if st.button("Clear History"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("###  About Jenee")
    st.markdown("""
    <div class="info-box">
        Jenee helps you recall and understand your group conversations using AI-powered search and retrieval.
    </div>
    """, unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Chat", "Voice"])

with tab1:
    chat_container = st.container()
    
    with chat_container:
        if len(st.session_state.chat_history) == 0:
            st.markdown("""
            <div class="welcome-box">
                <div class="welcome-title">Welcome to Jenee</div>
                <div class="welcome-text">
                    Start by selecting group IDs from the sidebar, then ask me anything about those conversations.
                </div>
                <div>
                    <span class="example-query">"What did we discuss about the weekend?"</span>
                    <span class="example-query">"Who talked about movies?"</span>
                    <span class="example-query">"What plans were made?"</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for message in st.session_state.chat_history:
                role = message["role"]
                content = message["content"]
                timestamp = message.get("timestamp", "")
                
                if role == "user":
                    st.markdown(f"""
                    <div style="display: flex; justify-content: flex-end;">
                        <div class="user-message">
                            {content}
                            <div class="message-time">{timestamp}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="display: flex; justify-content: flex-start;">
                        <div class="assistant-message">
                            {content}
                            <div class="message-time">{timestamp}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    
    col_input, col_button = st.columns([5, 1])
    
    with col_input:
        user_input = st.text_input(
            "Message",
            placeholder="Type your question here...",
            label_visibility="collapsed",
            key="user_input"
        )
    
    with col_button:
        send_button = st.button("Send", use_container_width=True)
    
    if send_button and user_input and st.session_state.selected_groups:
        current_time = datetime.now().strftime("%I:%M %p")
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": current_time
        })
        
        with st.spinner("Jenee is thinking..."):
            result = st.session_state.jenee.query(
                query=user_input,
                group_ids=st.session_state.selected_groups
            )
            
            response = result.get("response", "I couldn't generate a response. Please try again.")
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": current_time
            })
        
        st.rerun()
    
    elif send_button and not st.session_state.selected_groups:
        st.error("Please select at least one group from the sidebar")

with tab2:
    st.markdown("""
    <style>
        /* Completely hide the component UI */
        iframe[title="st_realtime_audio.realtime_audio_conversation"] {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            width: 0 !important;
            position: absolute !important;
        }
        
        /* Hide any component containers */
        div[data-testid="stVerticalBlock"] > div:has(iframe[title="st_realtime_audio.realtime_audio_conversation"]) {
            display: none !important;
        }
        
        .voice-logo {
            width: 200px;
            height: 200px;
            margin: 3rem auto 2rem auto;
            display: block;
            transition: all 0.3s ease;
        }
        
        .voice-logo.idle {
            filter: drop-shadow(0 0 20px rgba(147, 51, 234, 0.3));
            animation: breathe 4s ease-in-out infinite;
        }
        
        .voice-logo.recording {
            filter: drop-shadow(0 0 40px rgba(147, 51, 234, 0.8));
            animation: pulse-recording 1s ease-in-out infinite;
        }
        
        .voice-logo.speaking {
            filter: drop-shadow(0 0 50px rgba(147, 51, 234, 1));
            animation: pulse-speaking 0.6s ease-in-out infinite;
        }
        
        @keyframes breathe {
            0%, 100% {
                transform: scale(1);
                filter: drop-shadow(0 0 20px rgba(147, 51, 234, 0.3));
            }
            50% {
                transform: scale(1.02);
                filter: drop-shadow(0 0 25px rgba(147, 51, 234, 0.4));
            }
        }
        
        @keyframes pulse-recording {
            0%, 100% {
                transform: scale(1);
                filter: drop-shadow(0 0 40px rgba(147, 51, 234, 0.8));
            }
            50% {
                transform: scale(1.08);
                filter: drop-shadow(0 0 60px rgba(147, 51, 234, 1));
            }
        }
        
        @keyframes pulse-speaking {
            0%, 100% {
                transform: scale(1);
                filter: drop-shadow(0 0 50px rgba(147, 51, 234, 1));
            }
            25% {
                transform: scale(1.12);
                filter: drop-shadow(0 0 70px rgba(147, 51, 234, 1));
            }
            50% {
                transform: scale(1.08);
                filter: drop-shadow(0 0 60px rgba(147, 51, 234, 0.9));
            }
            75% {
                transform: scale(1.15);
                filter: drop-shadow(0 0 75px rgba(147, 51, 234, 1));
            }
        }
    </style>
    """, unsafe_allow_html=True)
    
    if 'voice_active' not in st.session_state:
        st.session_state.voice_active = False
    
    if not st.session_state.selected_groups:
        st.warning(" Please select groups from the sidebar to enable voice chat")
    else:
        group_context = f"You have access to group conversations from groups: {', '.join(st.session_state.selected_groups)}. "
        group_context += "When relevant, reference information from these group conversations to provide helpful responses."
        
        instructions = f"""You are Jenee, a helpful AI assistant for the Dipidi messaging app. 
        
{group_context}

Be natural, conversational, and concise in your responses. Keep answers brief unless asked for details."""
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if openai_api_key:
            logo_class = "idle"
            conversation = None
            
            if st.session_state.voice_active:
                conversation = realtime_audio_conversation(
                    api_key=openai_api_key,
                    instructions=instructions,
                    voice="alloy",
                    temperature=0.8,
                    turn_detection_threshold=0.5,
                    auto_start=True,
                    key="jenee_voice_conversation"
                )
                
                if conversation.get('is_recording'):
                    logo_class = "recording"
                elif conversation.get('status') == 'speaking':
                    logo_class = "speaking"
            
            voice_logo_path = os.path.join(os.path.dirname(__file__), "jeene.png")
            if os.path.exists(voice_logo_path):
                with open(voice_logo_path, "rb") as f:
                    logo_data = base64.b64encode(f.read()).decode()
                st.markdown(f'<img src="data:image/png;base64,{logo_data}" class="voice-logo {logo_class}" alt="Jenee">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if not st.session_state.voice_active:
                    if st.button(" Start Call", use_container_width=True, type="primary"):
                        st.session_state.voice_active = True
                        st.rerun()
                else:
                    if st.button(" End Call", use_container_width=True, type="secondary"):
                        st.session_state.voice_active = False
                        st.rerun()
            
            if conversation:
                if conversation.get('error'):
                    st.error(conversation['error'])
                
                if conversation.get('transcript'):
                    st.markdown("---")
                    st.markdown("### Conversation")
                    for message in conversation['transcript']:
                        msg_type = message.get('type', '')
                        content = message.get('content', '')
                        
                        if msg_type == 'user':
                            st.chat_message("user").write(content)
                        elif msg_type == 'assistant':
                            st.chat_message("assistant").write(content)
        else:
            st.error("OPENAI_API_KEY not found in environment variables")



