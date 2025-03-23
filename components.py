import streamlit as st
import time
from datetime import datetime

def create_navigation_cards():
    """Create the four navigation cards similar to the dark-themed UI in the image."""
    st.markdown("## Learning Journey")
    
    # First row: Knowledge Graph and Learning Path
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown("### 🔍 Knowledge Graph")
            st.markdown("View topic relationships")
            if st.button("View Knowledge Graph", key="nav_kg_btn", use_container_width=True):
                st.session_state.learning_mode = "knowledge_graph"
                st.rerun()
    
    with col2:
        with st.container(border=True):
            st.markdown("### 🧭 Learning Path")
            st.markdown("Follow your personalized path")
            if st.button("View Learning Path", key="nav_lp_btn", use_container_width=True):
                st.session_state.learning_mode = "learning_path"
                st.rerun()
    
    # Second row: Study Mode and Quiz Mode
    col3, col4 = st.columns(2)
    
    with col3:
        with st.container(border=True):
            st.markdown("### 📚 Study Mode")
            st.markdown("Learn a specific topic")
            if st.button("Start Studying", key="nav_study_btn", use_container_width=True):
                st.session_state.learning_mode = "study_mode"
                st.rerun()
    
    with col4:
        with st.container(border=True):
            st.markdown("### ✏️ Quiz Mode")
            st.markdown("Test your knowledge")
            if st.button("Take a Quiz", key="nav_quiz_btn", use_container_width=True):
                st.session_state.learning_mode = "new_quiz"
                st.rerun()

def display_topic_card(topic, topic_type="topic", mastery=0.0, description="", is_active=False):
    """Display a card for a topic with mastery indicator."""
    # Create a container with a border if it's the active topic
    card = st.container(border=is_active)
    
    with card:
        # Get icon based on topic type
        icon = get_icon_for_topic_type(topic_type)
        
        # Display topic with icon
        st.markdown(f"### {icon} {topic}")
        
        # Show mastery progress bar
        st.progress(mastery, text=f"Mastery: {mastery*100:.1f}%")
        
        # Show description if available
        if description:
            with st.expander("Description"):
                st.markdown(description)
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Study", key=f"study_{topic.replace(' ', '_')}", use_container_width=True):
                st.session_state.current_topic = topic
                st.session_state.learning_mode = "study_mode"
                return "study"
        
        with col2:
            if st.button("Quiz", key=f"quiz_{topic.replace(' ', '_')}", use_container_width=True):
                st.session_state.topic_focus = topic
                st.session_state.learning_mode = "new_quiz"
                return "quiz"
    
    return None  # No action taken

def display_mastery_badge(mastery_percentage):
    """Display a colored badge based on mastery level."""
    if mastery_percentage >= 80:
        st.markdown(f'<span style="background-color: #4CAF50; color: white; padding: 3px 7px; border-radius: 3px;">Expert ({mastery_percentage:.1f}%)</span>', unsafe_allow_html=True)
    elif mastery_percentage >= 60:
        st.markdown(f'<span style="background-color: #8BC34A; color: white; padding: 3px 7px; border-radius: 3px;">Proficient ({mastery_percentage:.1f}%)</span>', unsafe_allow_html=True)
    elif mastery_percentage >= 40:
        st.markdown(f'<span style="background-color: #FFC107; color: black; padding: 3px 7px; border-radius: 3px;">Intermediate ({mastery_percentage:.1f}%)</span>', unsafe_allow_html=True)
    elif mastery_percentage >= 20:
        st.markdown(f'<span style="background-color: #FF9800; color: white; padding: 3px 7px; border-radius: 3px;">Basic ({mastery_percentage:.1f}%)</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span style="background-color: #F44336; color: white; padding: 3px 7px; border-radius: 3px;">Novice ({mastery_percentage:.1f}%)</span>', unsafe_allow_html=True)

def get_icon_for_topic_type(topic_type):
    """Return an appropriate icon for a topic type."""
    if topic_type == "fundamental":
        return "🔵"  # Blue circle
    elif topic_type == "derived":
        return "🟢"  # Green circle
    elif topic_type == "technique":
        return "🟠"  # Orange circle
    elif topic_type == "question":
        return "🔴"  # Red circle
    else:
        return "⚪"  # White circle

def countdown_timer(seconds, message="Time remaining"):
    """Display a countdown timer for timed quizzes or study sessions."""
    # Create a placeholder for the timer
    timer_placeholder = st.empty()
    
    # Calculate end time
    end_time = datetime.now().timestamp() + seconds
    
    # Update the timer every second
    while datetime.now().timestamp() < end_time:
        remaining = int(end_time - datetime.now().timestamp())
        minutes, secs = divmod(remaining, 60)
        timer_placeholder.markdown(f"**{message}:** {minutes:02d}:{secs:02d}")
        time.sleep(1)
    
    timer_placeholder.markdown("**Time's up!**")
    return True

def quiz_option_button(option_letter, option_text, selected=False, correct=None):
    """Display a styled quiz option button."""
    # Determine the style based on state
    if correct is True:
        style = "quiz-correct"
    elif correct is False:
        style = "quiz-incorrect"
    elif selected:
        style = "quiz-selected"
    else:
        style = "quiz-option"
    
    # Create the HTML for the button
    button_html = f"""
    <div class="{style}" style="padding: 10px; margin: 5px 0; border-radius: 5px;">
        <b>{option_letter}.</b> {option_text}
    </div>
    """
    
    st.markdown(button_html, unsafe_allow_html=True)

def display_error_message(message, show_details=False, details=None):
    """Display an error message with optional details in an expander."""
    error_container = st.container(border=True)
    with error_container:
        st.error(message)
        
        if show_details and details:
            with st.expander("Error Details"):
                st.code(details)
        
        if st.button("Retry"):
            return True
    
    return False

def display_success_message(message, auto_dismiss=False):
    """Display a success message with optional auto-dismiss."""
    success_container = st.container()
    with success_container:
        st.success(message)
        
        if auto_dismiss:
            # Auto-dismiss after 3 seconds
            time.sleep(3)
            success_container.empty()

def display_explanation_placeholder():
    """Create a placeholder for loading an explanation."""
    explanation_container = st.container(border=True)
    with explanation_container:
        col1, col2 = st.columns([1, 5])
        
        with col1:
            st.markdown("🔍")
        
        with col2:
            st.markdown("### Loading explanation...")
            st.text("Please wait while we generate the explanation.")
            
            # Show a progress bar
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress((i + 1) / 100)
                time.sleep(0.01)
            
            return explanation_container