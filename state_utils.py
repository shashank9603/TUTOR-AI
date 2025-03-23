import streamlit as st
import uuid
import logging
from datetime import datetime

logger = logging.getLogger("state_utils")

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    # Check if this is a new session
    is_new_session = "session_id" not in st.session_state
    
    # Basic state variables
    if "tutor_system" not in st.session_state:
        st.session_state.tutor_system = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = None
    if "current_topic" not in st.session_state:
        st.session_state.current_topic = None
    if "current_quiz" not in st.session_state:
        st.session_state.current_quiz = None
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    if "quiz_results" not in st.session_state:
        st.session_state.quiz_results = {}
    if "quiz_analysis" not in st.session_state:
        st.session_state.quiz_analysis = None
    if "learning_mode" not in st.session_state:
        st.session_state.learning_mode = "explore"  # explore, knowledge_graph, learning_path, study_mode, etc.
    if "topic_focus" not in st.session_state:
        st.session_state.topic_focus = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        logger.info(f"Created new session: {st.session_state.session_id}")
    if "study_notes" not in st.session_state:
        st.session_state.study_notes = {}
    if "study_start_time" not in st.session_state:
        st.session_state.study_start_time = datetime.now()
    if "study_duration" not in st.session_state:
        st.session_state.study_duration = 0
    if "last_action_time" not in st.session_state:
        st.session_state.last_action_time = datetime.now()
    
    # Update last action time
    st.session_state.last_action_time = datetime.now()
    
    # Return whether this is a new session
    return is_new_session

def save_session_state():
    """Save important session state variables that should persist."""
    if "tutor_system" in st.session_state and st.session_state.tutor_system:
        # Update session timestamp
        st.session_state.last_action_time = datetime.now()
        
        # Record study duration if applicable
        if "study_start_time" in st.session_state and "current_topic" in st.session_state and st.session_state.current_topic:
            end_time = datetime.now()
            duration_minutes = (end_time - st.session_state.study_start_time).total_seconds() / 60
            
            # Only record if significant time spent (more than 30 seconds)
            if duration_minutes > 0.5:
                topic = st.session_state.current_topic
                # Update student model with study session
                st.session_state.tutor_system.student_model.update_from_study_session(
                    topic=topic,
                    duration_minutes=duration_minutes
                )
                
                # Reset study start time
                st.session_state.study_start_time = datetime.now()
                
                # Accumulate for session
                if "study_duration" in st.session_state:
                    st.session_state.study_duration += duration_minutes
                    
                logger.info(f"Recorded study session: {topic}, {duration_minutes:.2f} minutes")
                
                # The session state automatically persists between Streamlit reruns

def check_session_timeout(timeout_minutes=60):
    """Check if session has timed out due to inactivity."""
    if "last_action_time" in st.session_state:
        elapsed = (datetime.now() - st.session_state.last_action_time).total_seconds() / 60
        
        if elapsed > timeout_minutes:
            # Session timed out
            st.warning(f"Your session was inactive for {elapsed:.1f} minutes and has been reset.")
            
            # Save any pending study time
            save_session_state()
            
            # Keep tutor system but reset transient state
            if "current_topic" in st.session_state:
                st.session_state.current_topic = None
            if "current_quiz" in st.session_state:
                st.session_state.current_quiz = None
            if "learning_mode" in st.session_state:
                st.session_state.learning_mode = "explore"
            
            # Reset last action time
            st.session_state.last_action_time = datetime.now()
            
            return True
            
    return False