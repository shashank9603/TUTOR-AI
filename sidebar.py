import streamlit as st
import os
import torch
import time
import logging
from datetime import datetime
import tempfile

logger = logging.getLogger("sidebar")

def display_sidebar():
    """Display the sidebar with configuration options."""
    with st.sidebar:
        st.header("⚙️ Setup & Configuration")
        
        # GPU Detection
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            try:
                gpu_info = torch.cuda.get_device_properties(0)
                st.success(f"GPU detected: {gpu_info.name} ({gpu_info.total_memory / 1024**3:.1f} GB)")
            except Exception as e:
                logger.error(f"Error getting GPU info: {str(e)}", exc_info=True)
                st.warning("GPU detected but unable to get details")
        else:
            st.warning("No GPU detected. Running in CPU mode.")
        
        # Model selection
        llm_model = st.selectbox(
            "LLM Model",
            options=["llama3.2:latest", "llama3:latest", "mistral:latest"],
            index=0,
            help="Choose the language model for text generation"
        )
        
        embedding_model = st.selectbox(
            "Embedding Model",
            options=[
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ],
            index=0,  # Use MiniLM by default as it's faster
            help="Choose the model for generating text embeddings"
        )
        
        use_gpu = st.checkbox("Use GPU Acceleration", value=gpu_available, 
                            help="Use GPU for faster processing (if available)")
        
        # Advanced settings in expander to save space
        with st.expander("Advanced Settings"):
            chunk_size = st.slider("Chunk Size", 100, 2000, 1000, 
                                 help="Size of text chunks for processing")
            chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200,
                                    help="Overlap between chunks to maintain context")
            
            # Debug options
            debug_mode = st.checkbox("Debug Mode", value=False,
                                   help="Enable detailed logging for troubleshooting")
            
            if debug_mode:
                import logging
                logging.getLogger().setLevel(logging.DEBUG)
                st.info("Debug mode enabled. Check the console for detailed logs.")
        
        # Initialize button
        if st.button("Initialize Tutor System", use_container_width=True):
            with st.spinner("Initializing tutor system..."):
                # Create a temporary directory if needed
                if "temp_dir" not in st.session_state:
                    st.session_state.temp_dir = tempfile.mkdtemp()
                    logger.info(f"Created temporary directory: {st.session_state.temp_dir}")
                
                # Import here to avoid circular imports
                from tutor_system import PersonalizedTutorSystem
                
                st.session_state.tutor_system = PersonalizedTutorSystem(
                    llm_model_name=llm_model,
                    embedding_model_name=embedding_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    use_gpu=use_gpu
                )
                
                # Check system status
                if st.session_state.tutor_system.initialized:
                    st.success(f"System initialized on {st.session_state.tutor_system.device}")
                    
                    # Check Ollama status
                    if not st.session_state.tutor_system.ollama_available:
                        st.error("⚠️ Ollama is not available. Please make sure Ollama is running on your machine.")
                        st.markdown("Get [Ollama](https://ollama.com)")
                else:
                    st.error("System initialization failed. Check errors below.")
                    if st.session_state.tutor_system.errors:
                        st.write("Errors:")
                        for error in st.session_state.tutor_system.errors[-3:]:  # Show last 3 errors
                            st.write(f"- {error}")


def display_material_upload():
    """Display the material upload section."""
    with st.sidebar:
        st.header("📄 Upload Learning Materials")
        
        # Use tabs for cleaner organization
        textbooks_tab, topics_tab = st.tabs(["Textbooks", "Topics & Questions"])
        
        with textbooks_tab:
            uploaded_textbooks = st.file_uploader(
                "Select textbook PDFs", 
                type="pdf", 
                accept_multiple_files=True,
                help="Upload PDF textbooks to be processed"
            )
            
            if uploaded_textbooks:
                st.text(f"{len(uploaded_textbooks)} files selected")
                
                if st.button("Process Textbooks", key="process_textbooks_btn", use_container_width=True):
                    if not st.session_state.tutor_system:
                        st.error("Please initialize the tutor system first.")
                    else:
                        # Check Ollama for topic extraction later
                        has_ollama = st.session_state.tutor_system.ollama_available
                        
                        with st.spinner("Processing textbooks..."):
                            processing_status = st.empty()
                            processing_status.info("Reading PDFs... This may take a moment.")
                            
                            success = st.session_state.tutor_system.process_textbooks(uploaded_textbooks)
                            
                            if success:
                                processing_status.success("✅ Textbooks processed successfully!")
                                
                                # Show auto-extraction option if Ollama is available
                                if has_ollama:
                                    st.markdown("### 🤖 Automatic Topic Extraction")
                                    st.info("No topics provided yet. Would you like to automatically extract topics from the textbooks?")
                                    if st.button("Generate Topics & Learning Plan", use_container_width=True):
                                        extract_status = st.empty()
                                        extract_status.info("Extracting topics... This may take a few minutes.")
                                        
                                        with st.spinner("Extracting topics and building learning plan..."):
                                            success, message = st.session_state.tutor_system.extract_and_build_plan_from_textbook()
                                            if success:
                                                extract_status.success(message)
                                                # Wait a moment before rerunning to show the success message
                                                time.sleep(1)
                                                st.rerun()
                                            else:
                                                extract_status.error(message)
                                else:
                                    st.warning("⚠️ Ollama is not available for automatic topic extraction. Please manually provide topics in the Topics tab.")
                            else:
                                processing_status.error("❌ Error processing textbooks.")
                                if st.session_state.tutor_system.errors:
                                    with st.expander("Show Errors"):
                                        for error in st.session_state.tutor_system.errors[-5:]:
                                            st.write(f"- {error}")
                                            
                                # Offer troubleshooting tips
                                with st.expander("Troubleshooting Tips"):
                                    st.markdown("""
                                    1. Make sure the PDFs are not password protected
                                    2. Try with smaller PDFs or reduce the chunk size in advanced settings
                                    3. Check that you have enough memory available
                                    4. Restart the application if issues persist
                                    """)
        
        with topics_tab:
            st.write("Upload a text file with topics/questions (one per line):")
            topic_file = st.file_uploader("Select topics file", type=["txt"])
            
            st.write("Or enter topics manually:")
            topic_text = st.text_area("Enter topics or questions (one per line)", 
                                    height=150,
                                    help="Enter one topic per line. Questions should end with '?'")
            
            if st.button("Process Topics", key="process_topics_btn", use_container_width=True):
                if not st.session_state.tutor_system:
                    st.error("Please initialize the tutor system first.")
                else:
                    # Check if Ollama is available
                    if not st.session_state.tutor_system.ollama_available:
                        st.warning("⚠️ Ollama is not available. Topic relationships cannot be identified automatically.")
                    
                    with st.spinner("Processing topics..."):
                        topics_status = st.empty()
                        topics_status.info("Processing topics...")
                        
                        success = st.session_state.tutor_system.process_topic_list(topic_file, topic_text)
                        if success:
                            topics_status.success("✅ Topics processed successfully!")
                            # Wait a moment before rerunning to show the success message
                            time.sleep(1)
                            st.rerun()
                        else:
                            # If no topics provided but we have textbooks, offer to build automatically
                            if st.session_state.tutor_system.doc_vector_store and st.session_state.tutor_system.ollama_available:
                                topics_status.warning("No topics provided. Would you like to generate topics from the textbook content?")
                                if st.button("Generate Topics Automatically", use_container_width=True):
                                    extract_status = st.empty()
                                    extract_status.info("Extracting topics...")
                                    
                                    with st.spinner("Extracting topics..."):
                                        success, message = st.session_state.tutor_system.extract_and_build_plan_from_textbook()
                                        if success:
                                            extract_status.success(message)
                                            # Wait a moment before rerunning
                                            time.sleep(1)
                                            st.rerun()
                                        else:
                                            extract_status.error(message)
                            else:
                                topics_status.error("❌ Please provide topics or upload a textbook first.")
                                
                            if st.session_state.tutor_system.errors:
                                with st.expander("Error Details"):
                                    for error in st.session_state.tutor_system.errors[-3:]:
                                        st.write(f"- {error}")


def display_learning_dashboard():
    """Display learning progress dashboard if system is initialized."""
    if not st.session_state.tutor_system:
        return
        
    with st.sidebar:
        st.header("📊 Learning Progress")
        
        # Topic counts
        mastered = len(st.session_state.tutor_system.student_model.topics_mastered)
        in_progress = len(st.session_state.tutor_system.student_model.topics_in_progress)
        to_learn = len(st.session_state.tutor_system.student_model.topics_to_learn)
        total = mastered + in_progress + to_learn
        
        # Progress bar
        if total > 0:
            progress = (mastered + in_progress * 0.5) / total
            st.progress(progress, text=f"Overall Progress: {progress*100:.1f}%")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Mastered", mastered)
            col2.metric("In Progress", in_progress)
            col3.metric("To Learn", to_learn)
            
            # Study statistics
            study_stats = st.session_state.tutor_system.student_model.get_study_statistics()
            
            if study_stats["total_study_time"] > 0:
                with st.expander("Study Statistics"):
                    st.markdown(f"**Total Study Time:** {study_stats['total_study_time']:.1f} minutes")
                    st.markdown(f"**Topics Studied:** {study_stats['topics_studied']}")
                    if study_stats["topics_studied"] > 0:
                        st.markdown(f"**Avg. Time Per Topic:** {study_stats['avg_time_per_topic']:.1f} minutes")
        
        # Current topic
        if "current_topic" in st.session_state and st.session_state.current_topic:
            current_topic = st.session_state.current_topic
            
            st.markdown(f"**Current Topic:** {current_topic}")
            
            # Show mastery for current topic
            mastery = st.session_state.tutor_system.student_model.get_topic_mastery(current_topic) * 100
            st.progress(mastery/100, text=f"Mastery: {mastery:.1f}%")
        
        # Recent quiz results
        if st.session_state.tutor_system.student_model.quiz_results:
            st.subheader("Recent Quiz Results")
            
            results = list(st.session_state.tutor_system.student_model.quiz_results.values())
            recent_results = results[-min(3, len(results)):]
            
            for i, result in enumerate(recent_results):
                score = result.get('score', 0)
                
                # Determine color based on score
                if score >= 80:
                    color = "#4CAF50"  # Green
                elif score >= 60:
                    color = "#8BC34A"  # Light Green
                elif score >= 40:
                    color = "#FFC107"  # Yellow
                else:
                    color = "#F44336"  # Red
                
                # Display score with color
                st.markdown(f"Quiz {len(results) - i}: <span style='color:{color};font-weight:bold;'>{score:.1f}%</span>", unsafe_allow_html=True)
        
        # System status
        with st.expander("System Status"):
            status = st.session_state.tutor_system.get_system_status()
            
            st.markdown(f"**Session ID:** {status['session_id']}")
            st.markdown(f"**Device:** {status['device']}")
            st.markdown(f"**Documents:** {status['documents_processed']}")
            st.markdown(f"**Topics:** {status['topics_in_graph']}")
            st.markdown(f"**Relationships:** {status['relationships_in_graph']}")
            
            if status["errors"]:
                st.markdown("**Recent Errors:**")
                for error in status["errors"][-3:]:
                    st.markdown(f"- {error}")
            
            # Add clear session button for debugging
            if st.button("Clear Session Data", key="clear_session"):
                # Reset only transient session data
                if "current_topic" in st.session_state:
                    st.session_state.current_topic = None
                if "current_quiz" in st.session_state:
                    st.session_state.current_quiz = None
                if "quiz_answers" in st.session_state:
                    st.session_state.quiz_answers = {}
                if "quiz_results" in st.session_state:
                    st.session_state.quiz_results = {}
                if "quiz_analysis" in st.session_state:
                    st.session_state.quiz_analysis = None
                
                st.success("Session data cleared!")
                st.rerun()