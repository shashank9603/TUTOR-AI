<<<<<<< HEAD
import streamlit as st
import os
import logging
from datetime import datetime
import uuid

# Import core components
from core.knowledge_graph import KnowledgeGraph
from core.document_processor import DocumentProcessor
from core.quiz_generator import QuizGenerator
from core.learning_path import LearningPathManager
from core.student_model import StudentModel
from tutor_system import PersonalizedTutorSystem

# Import UI components
from ui.styling import apply_custom_css
from ui.components import create_navigation_cards
from ui.sidebar import display_sidebar, display_material_upload, display_learning_dashboard
from ui.knowledge_graph_ui import display_knowledge_graph
from ui.learning_path_ui import display_learning_path
from ui.study_mode_ui import display_study_mode
from ui.quiz_ui import display_new_quiz, display_quiz_mode, display_quiz_results
from utils.state_utils import initialize_session_state

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tutor_app")

def main():
    """Main application entry point."""
    # Configure page
    st.set_page_config(
        page_title="Tutor", 
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://ollama.com',
            'About': "Personalized KG-RAG Tutor System | A self-adaptive learning system"
        }
    )
    
    # Apply dark theme styling
    apply_custom_css()
    
    # Initialize session state, check if new session
    is_new_session = initialize_session_state()
    
    # Simplified header and description
    st.markdown("# Tutor")
    st.markdown("An adaptive learning system that creates personalized learning paths and quizzes")
    
    # Sidebar components
    display_sidebar()
    display_material_upload()
    display_learning_dashboard()
    
    # Main content based on session state
    if not st.session_state.tutor_system:
        # Welcome screen when not initialized
        st.info("👈 Please initialize the tutor system and upload materials in the sidebar to get started.")
        
        st.subheader("How This Tutor Works")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 1. Upload Materials")
            st.markdown("""
            - Upload your textbook PDFs
            - Provide a list of topics/questions
            - System processes and indexes content
            """)
        
        with col2:
            st.markdown("### 2. Learn Adaptively")
            st.markdown("""
            - System creates a knowledge graph
            - Generates personalized learning path
            - Provides explanations and examples
            """)
        
        with col3:
            st.markdown("### 3. Test & Improve")
            st.markdown("""
            - Take quizzes to test understanding
            - Get personalized feedback
            - System adapts to your strengths and weaknesses
            """)
            
        # Show developer info in expander
        with st.expander("Developer Information"):
            st.markdown("""
            ### System Architecture
            This application is structured in a modular way for easier debugging and maintenance:
            
            - **core/**: Contains the core functionality classes
            - **ui/**: Contains all the UI components
            - **utils/**: Contains utility functions
            - **tutor_system.py**: Main system class
            - **app.py**: Entry point for the Streamlit app
            
            ### Troubleshooting
            If you encounter issues:
            
            1. Ensure Ollama is running on port 11434
            2. Check that you have sufficient memory for PDF processing
            3. Try restarting the application if it becomes unresponsive
            4. Check the logs for detailed error information
            """)
    else:
        # If this is a new session but the system exists, return to the main screen
        if is_new_session and st.session_state.learning_mode != "explore":
            st.session_state.learning_mode = "explore"
            st.rerun()
        
        # Display the correct content based on mode
        if st.session_state.learning_mode == "explore":
            # Main navigation cards
            create_navigation_cards()
        elif st.session_state.learning_mode == "knowledge_graph":
            display_knowledge_graph()
        elif st.session_state.learning_mode == "learning_path":
            display_learning_path()
        elif st.session_state.learning_mode == "study_mode":
            display_study_mode()
        elif st.session_state.learning_mode == "new_quiz":
            display_new_quiz()
        elif st.session_state.learning_mode == "quiz_mode":
            display_quiz_mode()
        elif st.session_state.learning_mode == "quiz_results":
            display_quiz_results()
        else:
            # Fallback to main navigation
            logger.error(f"Unknown mode: {st.session_state.learning_mode}")
            st.error(f"Unknown mode: {st.session_state.learning_mode}")
            st.session_state.learning_mode = "explore"
            st.rerun()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Log error
        logger.error(f"Application error: {str(e)}", exc_info=True)
        
        # Show error in UI
        st.error(f"An error occurred: {str(e)}")
        
        # Show more details in expander
        with st.expander("Error Details"):
=======
import streamlit as st
import os
import logging
from datetime import datetime
import uuid

# Import core components
from core.knowledge_graph import KnowledgeGraph
from core.document_processor import DocumentProcessor
from core.quiz_generator import QuizGenerator
from core.learning_path import LearningPathManager
from core.student_model import StudentModel
from tutor_system import PersonalizedTutorSystem

# Import UI components
from ui.styling import apply_custom_css
from ui.components import create_navigation_cards
from ui.sidebar import display_sidebar, display_material_upload, display_learning_dashboard
from ui.knowledge_graph_ui import display_knowledge_graph
from ui.learning_path_ui import display_learning_path
from ui.study_mode_ui import display_study_mode
from ui.quiz_ui import display_new_quiz, display_quiz_mode, display_quiz_results
from utils.state_utils import initialize_session_state

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tutor_app")

def main():
    """Main application entry point."""
    # Configure page
    st.set_page_config(
        page_title="Tutor", 
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://ollama.com',
            'About': "Personalized KG-RAG Tutor System | A self-adaptive learning system"
        }
    )
    
    # Apply dark theme styling
    apply_custom_css()
    
    # Initialize session state, check if new session
    is_new_session = initialize_session_state()
    
    # Simplified header and description
    st.markdown("# Tutor")
    st.markdown("An adaptive learning system that creates personalized learning paths and quizzes")
    
    # Sidebar components
    display_sidebar()
    display_material_upload()
    display_learning_dashboard()
    
    # Main content based on session state
    if not st.session_state.tutor_system:
        # Welcome screen when not initialized
        st.info("👈 Please initialize the tutor system and upload materials in the sidebar to get started.")
        
        st.subheader("How This Tutor Works")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 1. Upload Materials")
            st.markdown("""
            - Upload your textbook PDFs
            - Provide a list of topics/questions
            - System processes and indexes content
            """)
        
        with col2:
            st.markdown("### 2. Learn Adaptively")
            st.markdown("""
            - System creates a knowledge graph
            - Generates personalized learning path
            - Provides explanations and examples
            """)
        
        with col3:
            st.markdown("### 3. Test & Improve")
            st.markdown("""
            - Take quizzes to test understanding
            - Get personalized feedback
            - System adapts to your strengths and weaknesses
            """)
            
        # Show developer info in expander
        with st.expander("Developer Information"):
            st.markdown("""
            ### System Architecture
            This application is structured in a modular way for easier debugging and maintenance:
            
            - **core/**: Contains the core functionality classes
            - **ui/**: Contains all the UI components
            - **utils/**: Contains utility functions
            - **tutor_system.py**: Main system class
            - **app.py**: Entry point for the Streamlit app
            
            ### Troubleshooting
            If you encounter issues:
            
            1. Ensure Ollama is running on port 11434
            2. Check that you have sufficient memory for PDF processing
            3. Try restarting the application if it becomes unresponsive
            4. Check the logs for detailed error information
            """)
    else:
        # If this is a new session but the system exists, return to the main screen
        if is_new_session and st.session_state.learning_mode != "explore":
            st.session_state.learning_mode = "explore"
            st.rerun()
        
        # Display the correct content based on mode
        if st.session_state.learning_mode == "explore":
            # Main navigation cards
            create_navigation_cards()
        elif st.session_state.learning_mode == "knowledge_graph":
            display_knowledge_graph()
        elif st.session_state.learning_mode == "learning_path":
            display_learning_path()
        elif st.session_state.learning_mode == "study_mode":
            display_study_mode()
        elif st.session_state.learning_mode == "new_quiz":
            display_new_quiz()
        elif st.session_state.learning_mode == "quiz_mode":
            display_quiz_mode()
        elif st.session_state.learning_mode == "quiz_results":
            display_quiz_results()
        else:
            # Fallback to main navigation
            logger.error(f"Unknown mode: {st.session_state.learning_mode}")
            st.error(f"Unknown mode: {st.session_state.learning_mode}")
            st.session_state.learning_mode = "explore"
            st.rerun()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Log error
        logger.error(f"Application error: {str(e)}", exc_info=True)
        
        # Show error in UI
        st.error(f"An error occurred: {str(e)}")
        
        # Show more details in expander
        with st.expander("Error Details"):
>>>>>>> 9e33a41 (Initial commit)
            st.exception(e)