import streamlit as st
import pandas as pd
import plotly.express as px
import logging

# Configure logging
logger = logging.getLogger("learning_path_ui")

def display_learning_path():
    """Display the current learning path and navigation options."""
    st.subheader("🧭 Your Learning Path")
    
    # Back to main navigation button
    if st.button("← Back to Learning Journey", key="lp_back_btn"):
        st.session_state.learning_mode = "explore"
        st.rerun()
    
    # Get learning path and current position
    learning_path = st.session_state.tutor_system.student_model.learning_path
    current_position = st.session_state.tutor_system.student_model.current_position
    
    if learning_path:
        # Show path overview
        overview_container = st.container(border=True)
        
        with overview_container:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Show visual progress bar
                progress = (current_position + 1) / len(learning_path) if learning_path else 0
                st.progress(progress, text=f"Progress: {current_position + 1}/{len(learning_path)} topics")
            
            with col2:
                # Show current position
                if current_position < len(learning_path):
                    current_topic = learning_path[current_position]
                    st.info(f"Current topic: **{current_topic}**", icon="🔍")
        
        # Add tabs for different views
        path_tab, visualization_tab, recommendation_tab = st.tabs([
            "Learning Path", "Path Visualization", "Recommendations"
        ])
        
        with path_tab:
            _display_path_as_list(learning_path, current_position)
        
        with visualization_tab:
            _display_path_visualization(learning_path, current_position)
        
        with recommendation_tab:
            _display_recommendations()
        
        # Action buttons for current topic
        st.subheader("Current Topic")
        current_topic = learning_path[current_position] if current_position < len(learning_path) else None
        
        if current_topic:
            current_card = st.container(border=True)
            with current_card:
                st.markdown(f"### {current_topic}")
                
                # Show topic description if available
                topic_attrs = st.session_state.tutor_system.knowledge_graph.get_node_attributes(current_topic)
                if topic_attrs and 'description' in topic_attrs and topic_attrs['description']:
                    st.markdown(f"*{topic_attrs['description']}*")
                
                # Show mastery
                mastery = st.session_state.tutor_system.student_model.get_topic_mastery(current_topic) * 100
                st.progress(mastery/100, text=f"Mastery: {mastery:.1f}%")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Study Now", key="study_current", use_container_width=True):
                        st.session_state.current_topic = current_topic
                        st.session_state.learning_mode = "study_mode"
                        st.rerun()
                
                with col2:
                    if st.button("Take Quiz", key="quiz_current", use_container_width=True):
                        st.session_state.topic_focus = current_topic
                        st.session_state.learning_mode = "new_quiz"
                        st.rerun()
        
        # Path navigation buttons
        st.subheader("Path Navigation")
        nav_col1, nav_col2 = st.columns(2)
        
        with nav_col1:
            # Previous topic button (disabled if at beginning)
            prev_disabled = current_position <= 0
            if st.button("⬅️ Previous Topic", disabled=prev_disabled, use_container_width=True):
                if current_position > 0:
                    st.session_state.tutor_system.student_model.current_position = current_position - 1
                    st.rerun()
        
        with nav_col2:
            # Next topic button (disabled if at end)
            next_disabled = current_position >= len(learning_path) - 1
            if st.button("Next Topic ➡️", disabled=next_disabled, use_container_width=True):
                if current_position < len(learning_path) - 1:
                    st.session_state.tutor_system.student_model.current_position = current_position + 1
                    st.rerun()
        
        # Path management buttons
        management_col1, management_col2 = st.columns(2)
        
        with management_col1:
            # Recalculate learning path button
            if st.button("🔄 Recalculate Learning Path", use_container_width=True):
                recalculate_status = st.empty()
                recalculate_status.info("Recalculating learning path...")
                
                with st.spinner("Recalculating learning path..."):
                    new_path = st.session_state.tutor_system.path_manager.calculate_learning_path()
                    st.session_state.tutor_system.student_model.learning_path = new_path
                    st.session_state.tutor_system.student_model.current_position = 0
                    recalculate_status.success("Learning path recalculated!")
                    st.rerun()
        
        with management_col2:
            # Skip current topic button
            if st.button("⏭️ Skip Current Topic", use_container_width=True, disabled=next_disabled):
                if current_position < len(learning_path) - 1:
                    st.session_state.tutor_system.student_model.current_position = current_position + 1
                    st.warning(f"Skipped topic: {current_topic}")
                    st.rerun()
    else:
        st.info("No learning path available. Please upload textbooks and topics to generate a path.")
        
        # Offer to generate path from knowledge graph if it exists
        if st.session_state.tutor_system.knowledge_graph.get_all_nodes():
            st.markdown("### Generate a Learning Path")
            st.markdown("We can generate a learning path based on the topics in your knowledge graph.")
            
            if st.button("Generate Learning Path", use_container_width=True):
                generate_status = st.empty()
                generate_status.info("Generating learning path...")
                
                with st.spinner("Generating learning path..."):
                    new_path = st.session_state.tutor_system.path_manager.calculate_learning_path()
                    st.session_state.tutor_system.student_model.learning_path = new_path
                    st.session_state.tutor_system.student_model.current_position = 0
                    generate_status.success("Learning path generated!")
                    st.rerun()
        else:
            st.warning("Knowledge graph is empty. Please upload materials first.")
            
            if st.button("Go to Upload Section"):
                # This would navigate to the materials upload section
                st.session_state.current_tab = "upload"
                st.rerun()


def _display_path_as_list(learning_path, current_position):
    """Display the learning path as an interactive list with status indicators."""
    path_container = st.container()
    with path_container:
        # Group path items into sections of 5 for better organization
        sections = [learning_path[i:i+5] for i in range(0, len(learning_path), 5)]
        
        for section_idx, section in enumerate(sections):
            st.markdown(f"### Section {section_idx + 1}")
            
            for i, topic in enumerate(section):
                absolute_idx = section_idx * 5 + i
                
                # Create a card-like effect for each topic with border for current
                is_current = absolute_idx == current_position
                topic_row = st.container(border=is_current)
                
                with topic_row:
                    # Determine completion status
                    status_icon = "🟢" if topic in st.session_state.tutor_system.student_model.topics_mastered else \
                                "🟡" if topic in st.session_state.tutor_system.student_model.topics_in_progress else \
                                "⚪"
                    
                    # Mark current position
                    current_marker = "👉 " if absolute_idx == current_position else ""
                    
                    # Get mastery level
                    mastery = st.session_state.tutor_system.student_model.get_topic_mastery(topic) * 100
                    
                    col1, col2 = st.columns([0.8, 0.2])
                    with col1:
                        st.markdown(f"{status_icon} {current_marker}**{absolute_idx+1}. {topic}**")
                        st.progress(mastery/100, text=f"Mastery: {mastery:.1f}%")
                    
                    with col2:
                        # Add buttons to jump to topic
                        if st.button("Study", key=f"path_study_{absolute_idx}"):
                            st.session_state.current_topic = topic
                            st.session_state.learning_mode = "study_mode"
                            st.rerun()


def _display_path_visualization(learning_path, current_position):
    """Display a visual representation of the learning path."""
    if not learning_path:
        st.info("No learning path to visualize.")
        return
    
    try:
        # Create a DataFrame for visualization
        path_data = []
        
        for i, topic in enumerate(learning_path):
            # Get mastery and status
            mastery = st.session_state.tutor_system.student_model.get_topic_mastery(topic) * 100
            
            status = "Current" if i == current_position else \
                    "Mastered" if topic in st.session_state.tutor_system.student_model.topics_mastered else \
                    "In Progress" if topic in st.session_state.tutor_system.student_model.topics_in_progress else \
                    "To Learn"
            
            # Get topic type
            topic_attrs = st.session_state.tutor_system.knowledge_graph.get_node_attributes(topic)
            topic_type = topic_attrs.get("type", "unknown")
            
            path_data.append({
                "Position": i + 1,
                "Topic": topic,
                "Mastery": mastery,
                "Status": status,
                "Type": topic_type.capitalize()
            })
        
        df = pd.DataFrame(path_data)
        
        # Create visualization
        fig = px.scatter(df, x="Position", y="Mastery", size="Mastery", color="Status",
                       hover_name="Topic", hover_data=["Type"],
                       color_discrete_map={
                           "Current": "#1E88E5",  # Blue
                           "Mastered": "#4CAF50",  # Green
                           "In Progress": "#FFC107",  # Yellow
                           "To Learn": "#9E9E9E"   # Grey
                       },
                       title="Learning Path Visualization",
                       labels={"Position": "Path Position", "Mastery": "Mastery (%)"},
                       height=400)
        
        # Add a line connecting the points
        fig.add_scatter(x=df["Position"], y=df["Mastery"], mode='lines', 
                      line=dict(color='rgba(100, 100, 100, 0.4)', width=1),
                      hoverinfo='skip', showlegend=False)
        
        # Mark current position
        if current_position < len(learning_path):
            current_data = df.iloc[current_position]
            fig.add_scatter(x=[current_data["Position"]], y=[current_data["Mastery"]],
                          marker=dict(color='#1E88E5', size=16, line=dict(color='white', width=2)),
                          hoverinfo='skip', showlegend=False)
        
        # Update layout for dark theme
        fig.update_layout(
            plot_bgcolor='rgba(17,17,17,1)',
            paper_bgcolor='rgba(17,17,17,1)',
            font=dict(color='white'),
            xaxis=dict(gridcolor='rgba(80,80,80,0.2)'),
            yaxis=dict(gridcolor='rgba(80,80,80,0.2)'),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        with st.expander("Understanding the Visualization"):
            st.markdown("""
            - **X-axis**: Position in your learning path
            - **Y-axis**: Your mastery level for each topic (0-100%)
            - **Dot size**: Larger dots indicate higher mastery
            - **Dot color**: Indicates topic status (blue = current, green = mastered, yellow = in progress, grey = to learn)
            - **Hover**: Mouse over a dot to see the topic name and type
            """)
    
    except Exception as e:
        logger.error(f"Error displaying path visualization: {str(e)}", exc_info=True)
        st.error(f"Error displaying path visualization: {str(e)}")


def _display_recommendations():
    """Display learning recommendations based on current progress."""
    student_model = st.session_state.tutor_system.student_model
    
    # Get weaknesses
    weaknesses = list(student_model.weaknesses)
    
    # Get next recommended topic
    next_recommended = student_model.get_next_recommended_topic()
    
    # Get most viewed topics with low mastery
    most_viewed = student_model.get_most_viewed_topics(5)
    low_mastery_viewed = [t for t in most_viewed if t["mastery"] < 0.7]
    
    # Display recommendations
    st.subheader("Learning Recommendations")
    
    # Recommendations based on analysis
    if weaknesses:
        st.markdown("### Areas Needing Improvement")
        st.markdown("Based on your quiz results, you should focus on:")
        
        for weakness in weaknesses[:3]:  # Show top 3
            weakness_card = st.container(border=True)
            with weakness_card:
                st.markdown(f"#### {weakness}")
                
                # Show mastery
                mastery = student_model.get_topic_mastery(weakness) * 100
                st.progress(mastery/100, text=f"Mastery: {mastery:.1f}%")
                
                if st.button("Study This Topic", key=f"rec_study_{weakness}"):
                    st.session_state.current_topic = weakness
                    st.session_state.learning_mode = "study_mode"
                    st.rerun()
    
    if low_mastery_viewed:
        st.markdown("### Topics You've Viewed But Need More Practice")
        
        for topic_info in low_mastery_viewed[:3]:  # Show top 3
            topic = topic_info["topic"]
            topic_card = st.container(border=True)
            with topic_card:
                st.markdown(f"#### {topic}")
                st.markdown(f"You've viewed this topic {topic_info['views']} times")
                
                # Show mastery
                mastery = topic_info["mastery"] * 100
                st.progress(mastery/100, text=f"Mastery: {mastery:.1f}%")
                
                if st.button("Continue Studying", key=f"rec_continue_{topic}"):
                    st.session_state.current_topic = topic
                    st.session_state.learning_mode = "study_mode"
                    st.rerun()
    
    if next_recommended and next_recommended not in weaknesses and not any(t["topic"] == next_recommended for t in low_mastery_viewed):
        st.markdown("### Next Recommended Topic")
        
        next_card = st.container(border=True)
        with next_card:
            st.markdown(f"#### {next_recommended}")
            
            # Show mastery
            mastery = student_model.get_topic_mastery(next_recommended) * 100
            st.progress(mastery/100, text=f"Mastery: {mastery:.1f}%")
            
            if st.button("Study Next Topic", key="rec_next_topic"):
                st.session_state.current_topic = next_recommended
                st.session_state.learning_mode = "study_mode"
                st.rerun()
    
    # Check if learning path is complete
    if student_model.has_all_topics_mastered():
        st.success("🎉 Congratulations! You've mastered all topics in your learning path!")