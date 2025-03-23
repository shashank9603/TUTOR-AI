import streamlit as st
import time
from datetime import datetime

def display_study_mode():
    """Display learning material for a topic with improved tracking."""
    st.subheader("📚 Study Mode")
    
    # Track study time
    if "study_start_time" not in st.session_state:
        st.session_state.study_start_time = datetime.now()
        st.session_state.study_duration = 0
    
    # Back to main navigation button
    if st.button("← Back to Learning Journey", key="study_back_btn"):
        # Calculate study duration
        end_time = datetime.now()
        duration_minutes = (end_time - st.session_state.study_start_time).total_seconds() / 60
        
        # Only record if there was a current topic and significant time spent
        if "current_topic" in st.session_state and st.session_state.current_topic and duration_minutes > 0.5:
            topic = st.session_state.current_topic
            # Update student model with study session
            st.session_state.tutor_system.student_model.update_from_study_session(
                topic=topic,
                duration_minutes=duration_minutes
            )
            # Record for the session
            st.session_state.study_duration += duration_minutes
        
        st.session_state.learning_mode = "explore"
        st.rerun()
    
    # Check Ollama availability
    if not st.session_state.tutor_system.ollama_available:
        st.warning("⚠️ Ollama LLM is required for generating learning materials. Make sure Ollama is running.")
        return
    
    # Check if document store is available
    if not st.session_state.tutor_system.doc_vector_store:
        st.warning("Please upload textbooks first to generate learning materials.")
        return
    
    # Get the current topic
    current_topic = st.session_state.current_topic
    
    if not current_topic:
        # Get the current topic from the learning path
        if st.session_state.tutor_system.student_model.learning_path:
            current_pos = st.session_state.tutor_system.student_model.current_position
            if current_pos < len(st.session_state.tutor_system.student_model.learning_path):
                current_topic = st.session_state.tutor_system.student_model.learning_path[current_pos]
    
    if not current_topic:
        # Show topic selection if no current topic
        _show_topic_selection()
        return
    
    # Set the current topic
    st.session_state.current_topic = current_topic
    
    # Display topic details
    st.subheader(f"📚 Learning: {current_topic}")
    
    # Show current mastery level
    mastery = st.session_state.tutor_system.student_model.get_topic_mastery(current_topic) * 100
    mastery_color = _get_mastery_color(mastery)
    
    # Display mastery progress bar
    st.markdown(f"**Current Mastery:** {mastery:.1f}%")
    st.progress(mastery/100, text=f"Mastery: {mastery:.1f}%")
    
    # Create tabs for different learning components
    explanation_tab, examples_tab, related_tab, notes_tab = st.tabs([
        "🔍 Explanation", 
        "📝 Examples", 
        "🔗 Related Topics",
        "📓 My Notes"
    ])
    
    with explanation_tab:
        # Get topic explanation with a loading placeholder
        explanation_placeholder = st.empty()
        explanation_placeholder.info("Generating explanation...")
        
        try:
            explanation = st.session_state.tutor_system.get_topic_explanation(current_topic)
            explanation_placeholder.markdown(explanation)
            
            # Show prerequisites if any
            prerequisites = st.session_state.tutor_system.knowledge_graph.get_prerequisites(current_topic)
            if prerequisites:
                with st.expander("Prerequisites"):
                    st.markdown("Before fully understanding this topic, you should be familiar with:")
                    for prereq in prerequisites:
                        st.markdown(f"- {prereq}")
                        if st.button(f"Study {prereq}", key=f"prereq_{prereq}"):
                            # Calculate current study duration before switching
                            if "study_start_time" in st.session_state:
                                end_time = datetime.now()
                                duration_minutes = (end_time - st.session_state.study_start_time).total_seconds() / 60
                                
                                # Update student model with partial study session
                                if duration_minutes > 0.5:
                                    st.session_state.tutor_system.student_model.update_from_study_session(
                                        topic=current_topic,
                                        duration_minutes=duration_minutes
                                    )
                            
                            # Update current topic and start time
                            st.session_state.current_topic = prereq
                            st.session_state.study_start_time = datetime.now()
                            st.rerun()
        except Exception as e:
            explanation_placeholder.error(f"Error generating explanation: {str(e)}")
    
    with examples_tab:
        # Get examples with a loading placeholder
        examples_placeholder = st.empty()
        examples_placeholder.info("Generating examples...")
        
        try:
            examples = st.session_state.tutor_system.get_topic_examples(current_topic)
            examples_placeholder.markdown(examples)
        except Exception as e:
            examples_placeholder.error(f"Error generating examples: {str(e)}")
    
    with related_tab:
        # Show related topics
        related = st.session_state.tutor_system.knowledge_graph.get_related_concepts(current_topic)
        if related:
            st.markdown("### Related Topics")
            st.markdown("These topics are related and might be interesting to explore next:")
            
            # Display in a grid for better organization
            cols = st.columns(min(3, len(related)))
            for i, topic in enumerate(related):
                with cols[i % len(cols)]:
                    # Skip if it's the current topic
                    if topic == current_topic:
                        continue
                        
                    # Create a card-like container
                    topic_card = st.container(border=True)
                    with topic_card:
                        st.markdown(f"#### {topic}")
                        # Add mastery indicator
                        topic_mastery = st.session_state.tutor_system.student_model.get_topic_mastery(topic) * 100
                        st.progress(topic_mastery/100, text=f"Mastery: {topic_mastery:.1f}%")
                        
                        if st.button("Study This", key=f"related_{topic}"):
                            # Calculate current study duration before switching
                            if "study_start_time" in st.session_state:
                                end_time = datetime.now()
                                duration_minutes = (end_time - st.session_state.study_start_time).total_seconds() / 60
                                
                                # Update student model with partial study session
                                if duration_minutes > 0.5:
                                    st.session_state.tutor_system.student_model.update_from_study_session(
                                        topic=current_topic,
                                        duration_minutes=duration_minutes
                                    )
                            
                            # Update current topic and start time
                            st.session_state.current_topic = topic
                            st.session_state.study_start_time = datetime.now()
                            st.rerun()
        else:
            st.info("No related topics found.")
    
    with notes_tab:
        # Initialize notes dictionary if it doesn't exist
        if "study_notes" not in st.session_state:
            st.session_state.study_notes = {}
        
        # Initialize notes for current topic if they don't exist
        if current_topic not in st.session_state.study_notes:
            st.session_state.study_notes[current_topic] = ""
        
        st.markdown("### My Notes")
        st.markdown("Take notes while studying to help remember key concepts:")
        
        # Notes text area
        notes = st.text_area(
            "Your notes:", 
            value=st.session_state.study_notes[current_topic],
            height=200,
            key=f"notes_{current_topic}"
        )
        
        # Save notes
        if st.button("Save Notes"):
            st.session_state.study_notes[current_topic] = notes
            st.success("Notes saved!")
    
    # Navigation and action buttons at the bottom
    st.divider()
    
    # Get position in learning path if applicable
    in_path = False
    path_position = -1
    
    if st.session_state.tutor_system.student_model.learning_path:
        try:
            path_position = st.session_state.tutor_system.student_model.learning_path.index(current_topic)
            in_path = True
        except ValueError:
            in_path = False
    
    # Suggestion for next steps based on mastery
    suggestions = st.session_state.tutor_system.path_manager.suggest_next_steps(
        current_topic, 
        st.session_state.tutor_system.student_model.mastery_levels
    )
    
    if suggestions:
        with st.container(border=True):
            st.markdown("### Suggested Next Steps")
            for suggestion in suggestions:
                if suggestion["type"] == "continue":
                    st.info(suggestion["reason"])
                elif suggestion["type"] == "next":
                    st.markdown(f"**Next Topic:** {suggestion['topic']}")
                    st.markdown(suggestion["reason"])
                elif suggestion["type"] == "related":
                    st.markdown(f"**Related Topic:** {suggestion['topic']}")
                    st.markdown(suggestion["reason"])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Back to learning path button
        if in_path:
            if st.button("⬅️ Back to Learning Path", use_container_width=True):
                # Save study time
                end_time = datetime.now()
                duration_minutes = (end_time - st.session_state.study_start_time).total_seconds() / 60
                
                # Update student model with study session
                if duration_minutes > 0.5:
                    st.session_state.tutor_system.student_model.update_from_study_session(
                        topic=current_topic,
                        duration_minutes=duration_minutes
                    )
                
                st.session_state.tutor_system.student_model.current_position = path_position
                st.session_state.learning_mode = "learning_path"
                st.rerun()
        else:
            if st.button("⬅️ Back to Learning Path", use_container_width=True):
                # Save study time
                end_time = datetime.now()
                duration_minutes = (end_time - st.session_state.study_start_time).total_seconds() / 60
                
                # Update student model with study session
                if duration_minutes > 0.5:
                    st.session_state.tutor_system.student_model.update_from_study_session(
                        topic=current_topic,
                        duration_minutes=duration_minutes
                    )
                
                st.session_state.learning_mode = "learning_path"
                st.rerun()
    
    with col2:
        # Test knowledge button
        if st.button("✅ Test My Knowledge", use_container_width=True):
            # Save study time
            end_time = datetime.now()
            duration_minutes = (end_time - st.session_state.study_start_time).total_seconds() / 60
            
            # Update student model with study session
            if duration_minutes > 0.5:
                st.session_state.tutor_system.student_model.update_from_study_session(
                    topic=current_topic,
                    duration_minutes=duration_minutes
                )
            
            st.session_state.topic_focus = current_topic
            st.session_state.learning_mode = "new_quiz"
            st.rerun()
    
    with col3:
        # Next topic button (if in path)
        if in_path and path_position < len(st.session_state.tutor_system.student_model.learning_path) - 1:
            next_topic = st.session_state.tutor_system.student_model.learning_path[path_position + 1]
            if st.button(f"➡️ Next: {next_topic[:15]}...", use_container_width=True):
                # Save study time
                end_time = datetime.now()
                duration_minutes = (end_time - st.session_state.study_start_time).total_seconds() / 60
                
                # Update student model with study session
                if duration_minutes > 0.5:
                    st.session_state.tutor_system.student_model.update_from_study_session(
                        topic=current_topic,
                        duration_minutes=duration_minutes
                    )
                
                st.session_state.current_topic = next_topic
                st.session_state.tutor_system.student_model.current_position = path_position + 1
                st.session_state.study_start_time = datetime.now()
                st.rerun()
        else:
            # Mark as complete button
            if st.button("✓ Mark as Complete", use_container_width=True):
                # Save study time
                end_time = datetime.now()
                duration_minutes = (end_time - st.session_state.study_start_time).total_seconds() / 60
                
                # Update student model with completion
                st.session_state.tutor_system.student_model.update_from_study_session(
                    topic=current_topic,
                    duration_minutes=duration_minutes,
                    completed=True
                )
                
                # Show confirmation
                st.success(f"'{current_topic}' marked as complete!")
                
                # Go back to learning path after a brief delay
                time.sleep(1)
                st.session_state.learning_mode = "learning_path"
                st.rerun()
    
    # Study time tracker
    current_duration = (datetime.now() - st.session_state.study_start_time).total_seconds() / 60
    total_duration = st.session_state.study_duration + current_duration
    
    # Only show if meaningful time has passed
    if total_duration > 0.5:
        st.caption(f"Study time: {total_duration:.1f} minutes")


def _show_topic_selection():
    """Show topic selection when no current topic is set."""
    st.subheader("📚 Select a Topic to Study")
    
    all_topics = st.session_state.tutor_system.knowledge_graph.get_all_nodes()
    if all_topics:
        # Add search functionality
        topic_search = st.text_input("Search for a topic:", key="study_topic_search")
        
        filtered_topics = all_topics
        if topic_search:
            filtered_topics = [t for t in all_topics if topic_search.lower() in t.lower()]
        
        if filtered_topics:
            # Show sorting options
            sort_option = st.radio(
                "Sort by:", 
                ["Alphabetical", "Mastery Level (Low to High)", "In Learning Path Order"],
                horizontal=True
            )
            
            # Sort topics according to selection
            if sort_option == "Alphabetical":
                sorted_topics = sorted(filtered_topics)
            elif sort_option == "Mastery Level (Low to High)":
                sorted_topics = sorted(
                    filtered_topics, 
                    key=lambda t: st.session_state.tutor_system.student_model.get_topic_mastery(t)
                )
            else:  # In Learning Path Order
                learning_path = st.session_state.tutor_system.student_model.learning_path
                # Put topics in learning path order, then alphabetical for others
                in_path = [t for t in learning_path if t in filtered_topics]
                not_in_path = [t for t in filtered_topics if t not in learning_path]
                sorted_topics = in_path + sorted(not_in_path)
            
            # Create a visually appealing topic selection
            st.write("")  # Add some spacing
            
            # Display topics in a grid with mastery indicators
            num_cols = 3
            rows = [sorted_topics[i:i+num_cols] for i in range(0, len(sorted_topics), num_cols)]
            
            for row in rows:
                cols = st.columns(num_cols)
                for i, topic in enumerate(row):
                    with cols[i]:
                        topic_card = st.container(border=True)
                        with topic_card:
                            st.markdown(f"**{topic}**")
                            
                            # Show mastery level
                            mastery = st.session_state.tutor_system.student_model.get_topic_mastery(topic) * 100
                            st.progress(mastery/100)
                            
                            # Is it in the learning path?
                            in_path = topic in st.session_state.tutor_system.student_model.learning_path
                            if in_path:
                                st.caption("📋 In learning path")
                            
                            if st.button("Study", key=f"select_{topic}"):
                                st.session_state.current_topic = topic
                                st.session_state.study_start_time = datetime.now()
                                st.rerun()
        else:
            st.info("No matching topics found.")
    else:
        st.info("No topics available. Please process your learning materials first.")


def _get_mastery_color(mastery_percentage: float) -> str:
    """Return a color based on mastery percentage."""
    if mastery_percentage < 30:
        return "red"
    elif mastery_percentage < 70:
        return "orange"
    else:
        return "green"