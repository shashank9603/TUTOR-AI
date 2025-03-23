import streamlit as st
import time
import logging
from datetime import datetime
import pandas as pd
import plotly.express as px

# Configure logging
logger = logging.getLogger("quiz_ui")

def display_new_quiz():
    """Display quiz generation options with improved error handling."""
    st.subheader("✏️ Generate New Quiz")
    
    # Back to main navigation button
    if st.button("← Back to Learning Journey", key="quiz_back_btn"):
        st.session_state.learning_mode = "explore"
        st.rerun()
    
    # Check if system is initialized
    if not st.session_state.tutor_system:
        st.warning("Please initialize the tutor system first.")
        return
    
    # Clear any old quiz data
    if 'current_quiz' in st.session_state:
        st.session_state.current_quiz = None
    if 'quiz_answers' in st.session_state:
        st.session_state.quiz_answers = {}
    if 'quiz_results' in st.session_state:
        st.session_state.quiz_results = {}
    if 'quiz_analysis' in st.session_state:
        st.session_state.quiz_analysis = None
    
    # Check Ollama availability
    if not st.session_state.tutor_system.ollama_available:
        st.warning("⚠️ Ollama LLM is required for quiz generation. Make sure Ollama is running.")
        st.markdown("""
        To install Ollama:
        1. Visit [ollama.com](https://ollama.com)
        2. Install and run the Ollama application
        3. Ensure it's running on port 11434
        """)
        return
    
    # Check if document store is available
    if not st.session_state.tutor_system.doc_vector_store:
        st.warning("Please upload textbooks first to generate quizzes.")
        if st.button("Go to Upload Section"):
            # This would need to navigate to the materials upload section
            st.rerun()
        return
    
    # Quiz generation form
    with st.form("quiz_generation_form"):
        st.markdown("### Quiz Settings")
        
        # Topic selection - with smarter defaults
        topic_options = ["Current learning position"]
        
        # Add current topic if set
        if "current_topic" in st.session_state and st.session_state.current_topic:
            topic_options.insert(0, st.session_state.current_topic)
        
        # Add focused topic if set
        if "topic_focus" in st.session_state and st.session_state.topic_focus and st.session_state.topic_focus != st.session_state.current_topic:
            topic_options.insert(0, st.session_state.topic_focus)
        
        # Add "All topics" option
        topic_options.append("All topics")
        
        # Add specific topics from knowledge graph
        if st.session_state.tutor_system.knowledge_graph.get_all_nodes():
            # Get top 10 topics by centrality (importance)
            try:
                centrality = st.session_state.tutor_system.knowledge_graph.calculate_centrality()
                top_topics = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                for topic, _ in top_topics:
                    if topic not in topic_options:
                        topic_options.append(topic)
            except Exception as e:
                logger.error(f"Error calculating centrality: {str(e)}")
                # If centrality calculation fails, just add all topics
                for topic in st.session_state.tutor_system.knowledge_graph.get_all_nodes():
                    if topic not in topic_options:
                        topic_options.append(topic)
        
        # Remove duplicates preserving order
        topic_options = list(dict.fromkeys(topic_options))
        
        # Topic selection
        selected_topic_option = st.selectbox(
            "Quiz Focus:", 
            topic_options,
            help="Choose what to focus on in this quiz",
            key="quiz_topic_select"
        )
        
        # Quiz parameters
        col1, col2 = st.columns(2)
        
        with col1:
            num_questions = st.slider(
                "Number of questions:", 
                3, 10, 5,
                help="More questions will take longer to generate",
                key="quiz_num_questions"
            )
        
        with col2:
            difficulty = st.select_slider(
                "Difficulty:", 
                options=["easy", "medium", "hard", "adaptive"], 
                value="adaptive",
                help="Adaptive difficulty adjusts based on your past performance",
                key="quiz_difficulty"
            )
        
        # Generate button
        submit_button = st.form_submit_button("Generate Quiz", use_container_width=True)
    
    if submit_button:
        quiz_generation_status = st.empty()
        quiz_generation_status.info("Generating quiz... This may take a moment.")
        
        progress_bar = st.progress(0, text="Preparing quiz...")
        
        try:
            # Determine actual topic
            actual_topic = None
            if selected_topic_option == "Current learning position":
                # Use current position
                learning_path = st.session_state.tutor_system.student_model.learning_path
                current_position = st.session_state.tutor_system.student_model.current_position
                
                if learning_path and current_position < len(learning_path):
                    actual_topic = learning_path[current_position]
                    quiz_generation_status.info(f"Generating quiz on current topic: {actual_topic}")
            elif selected_topic_option == "All topics":
                # No specific topic (comprehensive quiz)
                actual_topic = None
                quiz_generation_status.info("Generating comprehensive quiz on all topics")
            else:
                # Use selected topic
                actual_topic = selected_topic_option
                quiz_generation_status.info(f"Generating quiz on: {actual_topic}")
            
            # Update progress
            progress_bar.progress(0.2, text="Retrieving relevant content...")
            time.sleep(0.5)  # Small delay for visual feedback
            
            # Generate quiz with timeout handling
            start_time = time.time()
            max_generation_time = 60  # 60 seconds timeout
            
            # Clear any previous errors
            if hasattr(st.session_state.tutor_system, 'errors'):
                st.session_state.tutor_system.errors = []
            
            progress_bar.progress(0.4, text="Creating questions...")
            
            # Generate quiz
            st.session_state.current_quiz = st.session_state.tutor_system.generate_quiz(
                topic=actual_topic,
                num_questions=num_questions,
                difficulty=difficulty
            )
            
            generation_time = time.time() - start_time
            logger.info(f"Quiz generation took {generation_time:.1f} seconds")
            
            progress_bar.progress(0.8, text="Finalizing quiz...")
            time.sleep(0.5)  # Small delay for visual feedback
            
            # Check if we got questions
            if st.session_state.current_quiz and len(st.session_state.current_quiz) > 0:
                # Reset quiz answers
                st.session_state.quiz_answers = {}
                
                progress_bar.progress(1.0, text="Quiz ready!")
                quiz_generation_status.success(f"Quiz generated successfully with {len(st.session_state.current_quiz)} questions!")
                
                # Switch to quiz mode after a short delay
                time.sleep(1)
                st.session_state.learning_mode = "quiz_mode"
                st.rerun()
            else:
                quiz_generation_status.error("Failed to generate quiz questions.")
                progress_bar.empty()
                
                st.error("The system couldn't generate quiz questions. This might be due to:")
                st.markdown("""
                1. Insufficient learning material on the selected topic
                2. The Ollama service might be overloaded
                3. An internal processing error
                """)
                
                if st.session_state.tutor_system.errors:
                    with st.expander("Technical Details"):
                        for error in st.session_state.tutor_system.errors[-3:]:
                            st.write(f"- {error}")
                
                # Offer to try again
                if st.button("Try Again with Simpler Settings"):
                    st.session_state.topic_focus = None
                    st.rerun()
        except Exception as e:
            quiz_generation_status.error(f"Error generating quiz")
            progress_bar.empty()
            
            logger.error(f"Quiz generation error: {str(e)}", exc_info=True)
            
            st.error("An error occurred while generating the quiz.")
            with st.expander("Error Details"):
                st.write(str(e))
            
            # Offer to try again
            if st.button("Try Again"):
                st.rerun()


def display_quiz_mode():
    """Display active quiz questions and collect answers."""
    st.subheader("📝 Quiz")
    
    # Back to main navigation button
    if st.button("← Back to Learning Journey", key="quiz_mode_back_btn"):
        st.session_state.learning_mode = "explore"
        st.rerun()
    
    quiz_questions = st.session_state.current_quiz
    
    if not quiz_questions:
        st.warning("No quiz questions available.")
        
        # Provide a button to generate a new quiz
        if st.button("Generate New Quiz"):
            st.session_state.learning_mode = "new_quiz"
            st.rerun()
        return
    
    # Display quiz info
    quiz_info_container = st.container(border=True)
    with quiz_info_container:
        col1, col2 = st.columns(2)
        
        with col1:
            if "topic_focus" in st.session_state and st.session_state.topic_focus:
                st.markdown(f"**Topic Focus:** {st.session_state.topic_focus}")
            else:
                st.markdown("**Topic Focus:** Multiple topics")
        
        with col2:
            st.markdown(f"**Number of Questions:** {len(quiz_questions)}")
            
            # Show difficulty if available
            if quiz_questions and "difficulty" in quiz_questions[0]:
                st.markdown(f"**Difficulty:** {quiz_questions[0]['difficulty'].capitalize()}")
    
    # Initialize quiz answers if needed
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    
    # Show progress
    current_answers = len(st.session_state.quiz_answers)
    total_questions = len(quiz_questions)
    if current_answers > 0:
        st.progress(current_answers / total_questions, 
                  text=f"Progress: {current_answers}/{total_questions} questions answered")
    
    # Create a form to collect all answers at once
    with st.form("quiz_form"):
        # Display each question in its own container
        for i, question in enumerate(quiz_questions):
            question_container = st.container(border=True)
            
            with question_container:
                st.markdown(f"**Question {i+1}:** {question['question']}")
                
                # Show topic if available
                if "topic" in question:
                    st.caption(f"Topic: {question['topic']}")
                
                # Create radio buttons for options
                options = {opt["letter"]: opt["text"] for opt in question["options"]}
                selected_answer = st.radio(
                    f"Select answer for question {i+1}:",
                    options.keys(),
                    format_func=lambda x: f"{x}. {options[x]}",
                    key=f"quiz_q{i}"
                )
                
                # Store the answer in session state
                st.session_state.quiz_answers[i] = selected_answer
        
        # Submit button
        submit_col1, submit_col2 = st.columns([3, 1])
        with submit_col1:
            submit_button = st.form_submit_button("Submit Quiz", use_container_width=True)
        
        with submit_col2:
            cancel_button = st.form_submit_button("Cancel", use_container_width=True)
    
    # Process form submission
    if submit_button:
        # Check if all questions are answered
        if len(st.session_state.quiz_answers) < len(quiz_questions):
            st.warning(f"You've only answered {len(st.session_state.quiz_answers)} of {len(quiz_questions)} questions. Please answer all questions before submitting.")
        else:
            with st.spinner("Evaluating your answers..."):
                # Process answers
                results = {}
                for i, question in enumerate(quiz_questions):
                    selected = st.session_state.quiz_answers.get(i, "")
                    is_correct = selected == question["correct_answer"]
                    
                    results[i] = {
                        "question": question["question"],
                        "selected": selected,
                        "correct": is_correct,
                        "correct_answer": question["correct_answer"],
                        "explanation": question.get("explanation", ""),
                        "topic": question.get("topic", "General")
                    }
                
                # Store results
                st.session_state.quiz_results = results
                
                # Evaluate results if tutor system exists
                if st.session_state.tutor_system:
                    # Check Ollama for enhanced analysis
                    if st.session_state.tutor_system.ollama_available:
                        try:
                            analysis = st.session_state.tutor_system.evaluate_quiz_results(results)
                            st.session_state.quiz_analysis = analysis
                        except Exception as e:
                            logger.error(f"Error analyzing quiz: {str(e)}", exc_info=True)
                            # Create basic analysis without Ollama
                            correct_count = sum(1 for result in results.values() if result["correct"])
                            total_count = len(results)
                            score_percentage = (correct_count / total_count) * 100 if total_count > 0 else 0
                            st.session_state.quiz_analysis = {
                                "score": score_percentage,
                                "correct_count": correct_count,
                                "total_count": total_count
                            }
                    else:
                        # Create basic analysis without Ollama
                        correct_count = sum(1 for result in results.values() if result["correct"])
                        total_count = len(results)
                        score_percentage = (correct_count / total_count) * 100 if total_count > 0 else 0
                        st.session_state.quiz_analysis = {
                            "score": score_percentage,
                            "correct_count": correct_count,
                            "total_count": total_count
                        }
                
                # Change to results view
                st.session_state.learning_mode = "quiz_results"
                st.rerun()
    
    if cancel_button:
        if st.button("Confirm Cancellation", key="confirm_cancel"):
            st.session_state.current_quiz = None
            st.session_state.quiz_answers = {}
            st.session_state.learning_mode = "new_quiz"
            st.rerun()


def display_quiz_results():
    """Display quiz results and analysis."""
    st.subheader("📊 Quiz Results")
    
    # Back to main navigation button
    if st.button("← Back to Learning Journey", key="results_back_btn"):
        st.session_state.learning_mode = "explore"
        st.rerun()
    
    if "quiz_results" not in st.session_state or not st.session_state.quiz_results:
        st.warning("No quiz results available.")
        
        # Provide a button to generate a new quiz
        if st.button("Take a New Quiz"):
            st.session_state.learning_mode = "new_quiz"
            st.rerun()
        return
    
    # Get results and analysis
    results = st.session_state.quiz_results
    analysis = st.session_state.quiz_analysis if "quiz_analysis" in st.session_state else None
    
    # Display overall score
    if analysis and "score" in analysis:
        score = analysis["score"]
        correct_count = analysis.get("correct_count", sum(1 for r in results.values() if r["correct"]))
        total_count = analysis.get("total_count", len(results))
    else:
        correct_count = sum(1 for r in results.values() if r["correct"])
        total_count = len(results)
        score = (correct_count / total_count) * 100 if total_count > 0 else 0
    
    # Create a score card
    score_container = st.container(border=True)
    with score_container:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Display score as a big number
            st.markdown(f"# {score:.1f}%")
        
        with col2:
            st.markdown(f"### Score: {correct_count}/{total_count} correct")
            
            # Add a performance message based on score
            if score >= 90:
                st.success("Excellent! You've mastered this material.")
            elif score >= 70:
                st.success("Good job! You have a solid understanding.")
            elif score >= 50:
                st.info("You're making progress. Keep studying to improve.")
            else:
                st.warning("You need more practice with this material.")
    
    # Topic performance if available
    if analysis and "topic_results" in analysis and analysis["topic_results"]:
        st.subheader("Topic Performance")
        
        topic_results = analysis["topic_results"]
        
        # Create dataframe for visualization
        topic_df = pd.DataFrame([
            {
                "Topic": topic,
                "Correct": results["correct"],
                "Total": results["total"],
                "Percentage": (results["correct"] / results["total"]) * 100 if results["total"] > 0 else 0
            }
            for topic, results in topic_results.items()
        ])
        
        try:
            # Display as horizontal bar chart
            fig = px.bar(
                topic_df, 
                y="Topic", 
                x="Percentage",
                title="Topic Performance",
                labels={"Percentage": "Percentage Correct (%)"},
                color="Percentage",
                color_continuous_scale=["red", "yellow", "green"],
                range_color=[0, 100],
                height=max(100, len(topic_df) * 50 + 100),  # Dynamic height based on topics
                text=topic_df["Percentage"].apply(lambda x: f"{x:.0f}%")
            )
            
            fig.update_layout(
                yaxis={'categoryorder':'total ascending'},
                plot_bgcolor='rgba(17,17,17,1)',
                paper_bgcolor='rgba(17,17,17,1)',
                font=dict(color='white')
            )
            fig.update_traces(textposition='outside')
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}", exc_info=True)
            # Fallback to table if visualization fails
            st.table(topic_df)
    
    # Show strengths and weaknesses
    if analysis and ("weaknesses" in analysis or "strengths" in analysis):
        insights_container = st.container(border=True)
        with insights_container:
            st.subheader("Learning Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Areas to Improve")
                if "weaknesses" in analysis and analysis["weaknesses"]:
                    for weakness in analysis["weaknesses"]:
                        st.markdown(f"- {weakness}")
                        
                        # Add a study button for each weakness
                        if st.button(f"Study {weakness}", key=f"study_weakness_{weakness}"):
                            st.session_state.current_topic = weakness
                            st.session_state.learning_mode = "study_mode"
                            st.rerun()
                else:
                    st.markdown("*No specific weaknesses identified*")
            
            with col2:
                st.markdown("### Strengths")
                if "strengths" in analysis and analysis["strengths"]:
                    for strength in analysis["strengths"]:
                        st.markdown(f"- {strength}")
                else:
                    st.markdown("*No specific strengths identified*")
    
    # Display personalized feedback if available
    if analysis and "feedback" in analysis and analysis["feedback"]:
        st.subheader("Personalized Feedback")
        with st.container(border=True):
            st.markdown(analysis["feedback"])
    
    # Question review
    with st.expander("Question Review", expanded=True):
        # Display each question with feedback
        for i, result in sorted(results.items()):
            question_container = st.container(border=True)
            
            with question_container:
                # Question header with correct/incorrect indicator
                header_col1, header_col2 = st.columns([0.9, 0.1])
                
                with header_col1:
                    st.markdown(f"**Question {i+1}:** {result['question']}")
                
                with header_col2:
                    if result["correct"]:
                        st.markdown("✅")
                    else:
                        st.markdown("❌")
                
                # Selected and correct answers
                st.markdown(f"**Your answer:** {result['selected']}")
                
                if not result["correct"]:
                    st.markdown(f"**Correct answer:** {result['correct_answer']}")
                    
                    # Show explanation
                    if "explanation" in result and result["explanation"]:
                        st.markdown(f"**Explanation:** {result['explanation']}")
                
                # Show topic
                if "topic" in result and result["topic"]:
                    st.caption(f"Topic: {result['topic']}")
    
    # Next steps buttons
    st.subheader("Next Steps")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if "topic_focus" in st.session_state and st.session_state.topic_focus:
            # Continue studying the topic
            if st.button("Continue Studying", use_container_width=True):
                st.session_state.current_topic = st.session_state.topic_focus
                st.session_state.learning_mode = "study_mode"
                st.rerun()
        else:
            # Back to learning path
            if st.button("Back to Learning Path", use_container_width=True):
                st.session_state.learning_mode = "learning_path"
                st.rerun()
    
    with col2:
        # Take another quiz on the same topic
        focus_text = f" on {st.session_state.topic_focus}" if "topic_focus" in st.session_state and st.session_state.topic_focus else ""
        if st.button(f"Take Another Quiz{focus_text}", use_container_width=True):
            st.session_state.learning_mode = "new_quiz"
            st.rerun()
    
    with col3:
        # Review weak areas
        if analysis and "weaknesses" in analysis and analysis["weaknesses"]:
            # Get first weakness to study
            weakness = list(analysis["weaknesses"])[0]
            if st.button(f"Study Weak Area: {weakness[:15]}...", use_container_width=True):
                st.session_state.current_topic = weakness
                st.session_state.learning_mode = "study_mode"
                st.rerun()
        else:
            # Next topic in path
            if st.session_state.tutor_system.student_model.learning_path:
                current_pos = st.session_state.tutor_system.student_model.current_position
                if current_pos < len(st.session_state.tutor_system.student_model.learning_path) - 1:
                    next_topic = st.session_state.tutor_system.student_model.learning_path[current_pos + 1]
                    if st.button(f"Next Topic: {next_topic[:15]}...", use_container_width=True):
                        st.session_state.current_topic = next_topic
                        st.session_state.tutor_system.student_model.current_position = current_pos + 1
                        st.session_state.learning_mode = "study_mode"
                        st.rerun()