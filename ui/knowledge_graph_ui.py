import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import logging

# Configure logging
logger = logging.getLogger("knowledge_graph_ui")

def display_knowledge_graph():
    """Display the knowledge graph visualization and details with improved UI."""
    st.subheader("🔍 Knowledge Graph Visualization")
    
    # Back to main navigation button
    if st.button("← Back to Learning Journey", key="kg_back_btn"):
        st.session_state.learning_mode = "explore"
        st.rerun()
    
    # Display knowledge graph if it has nodes
    graph = st.session_state.tutor_system.knowledge_graph
    nodes = graph.get_all_nodes()
    
    if nodes:
        # Add tab-based navigation for different views
        graph_tab, details_tab, stats_tab = st.tabs([
            "Graph View", 
            "Topic Details", 
            "Graph Statistics"
        ])
        
        with graph_tab:
            # Controls for graph visualization
            controls_container = st.container()
            with controls_container:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Filter options
                    filter_option = st.selectbox(
                        "Filter by type:",
                        ["All Topics", "Fundamental Concepts", "Derived Concepts", 
                         "Techniques/Methods", "Questions", "Mastered Topics", "Topics to Learn"],
                        key="kg_filter"
                    )
                
                with col2:
                    # Layout options
                    layout_option = st.selectbox(
                        "Graph layout:",
                        ["Spring", "Circular", "Hierarchical", "Radial"],
                        key="kg_layout"
                    )
            
            # Apply filters
            filtered_nodes = _filter_nodes(nodes, filter_option)
            
            if filtered_nodes:
                # Create the visualization
                try:
                    graph_container = st.container()
                    with graph_container:
                        fig = _create_graph_visualization(
                            graph, 
                            filtered_nodes,
                            layout_option
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show legend
                        legend_container = st.container(border=True)
                        with legend_container:
                            st.markdown("### Legend")
                            legend_col1, legend_col2, legend_col3, legend_col4 = st.columns(4)
                            
                            with legend_col1:
                                st.markdown("🔵 Fundamental Concept")
                            with legend_col2:
                                st.markdown("🟢 Derived Concept")
                            with legend_col3:
                                st.markdown("🟠 Technique/Method")
                            with legend_col4:
                                st.markdown("🔴 Question")
                                
                            st.caption("Node size indicates topic importance (centrality)")
                except Exception as e:
                    logger.error(f"Error visualizing graph: {str(e)}", exc_info=True)
                    st.error(f"Error visualizing graph: {str(e)}")
            else:
                st.info(f"No topics match the selected filter: {filter_option}")
        
        with details_tab:
            _display_topic_details(graph, nodes)
        
        with stats_tab:
            _display_graph_statistics(graph)
    else:
        st.info("Knowledge graph is empty. Please upload textbooks and topics to generate the graph.")
        
        # Add guidance for building the graph
        with st.expander("How to Build a Knowledge Graph"):
            st.markdown("""
            To build a knowledge graph:
            
            1. **Upload Textbooks**: Upload PDF files containing learning materials in the sidebar
            2. **Add Topics**: Either upload a list of topics or let the system extract them automatically
            3. **Process Materials**: Click the 'Process' buttons to analyze content and build relationships
            
            The system will create a knowledge graph showing how topics relate to each other, which helps create an optimal learning path.
            """)


def _filter_nodes(nodes, filter_option):
    """Filter nodes based on selected option."""
    if filter_option == "All Topics":
        return nodes
    
    # Get student model for mastery info
    student_model = st.session_state.tutor_system.student_model
    graph = st.session_state.tutor_system.knowledge_graph
    
    filtered = []
    for node in nodes:
        node_type = graph.get_node_attributes(node).get("type", "")
        
        if filter_option == "Fundamental Concepts" and node_type == "fundamental":
            filtered.append(node)
        elif filter_option == "Derived Concepts" and node_type == "derived":
            filtered.append(node)
        elif filter_option == "Techniques/Methods" and node_type == "technique":
            filtered.append(node)
        elif filter_option == "Questions" and node_type == "question":
            filtered.append(node)
        elif filter_option == "Mastered Topics" and node in student_model.topics_mastered:
            filtered.append(node)
        elif filter_option == "Topics to Learn" and node in student_model.topics_to_learn:
            filtered.append(node)
    
    return filtered


def _create_graph_visualization(graph, filtered_nodes, layout_option):
    """Create a Plotly visualization of the knowledge graph."""
    if not filtered_nodes:
        # Return empty figure if no nodes
        fig = go.Figure()
        fig.update_layout(
            title="Knowledge Graph (Empty)",
            showlegend=False,
            plot_bgcolor='rgba(17,17,17,1)',
            paper_bgcolor='rgba(17,17,17,1)',
            font=dict(color='white')
        )
        return fig
        
    try:
        # Create a subgraph with only the filtered nodes
        subgraph = graph.graph.subgraph(filtered_nodes)
        
        # Compute layout
        if layout_option == "Spring":
            pos = nx.spring_layout(subgraph, seed=42)
        elif layout_option == "Circular":
            pos = nx.circular_layout(subgraph)
        elif layout_option == "Hierarchical":
            # For hierarchical, first check if it's a DAG
            try:
                # If not a DAG, fall back to spring layout
                if not nx.is_directed_acyclic_graph(subgraph):
                    pos = nx.spring_layout(subgraph, seed=42)
                else:
                    pos = nx.multipartite_layout(subgraph, subset_key="layer")
            except:
                # Fall back to spring layout on error
                pos = nx.spring_layout(subgraph, seed=42)
        elif layout_option == "Radial":
            pos = nx.kamada_kawai_layout(subgraph)
        else:
            # Default to spring layout
            pos = nx.spring_layout(subgraph, seed=42)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Get edge type if available
            edge_type = subgraph.edges[edge].get("relationship", "")
            edge_text.extend([edge_type, edge_type, None])
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # Create node traces based on node type
        node_traces = {}
        
        # Define node types and colors
        node_types = {
            "fundamental": {"color": '#4287f5', "name": "Fundamental Concept"},
            "derived": {"color": '#42f5a7', "name": "Derived Concept"},
            "technique": {"color": '#f5a742', "name": "Technique/Method"},
            "question": {"color": '#f54242', "name": "Question"},
            "unknown": {"color": '#a9a9a9', "name": "Unknown Type"}
        }
        
        # Initialize traces for each node type
        for node_type, info in node_types.items():
            node_traces[node_type] = go.Scatter(
                x=[], y=[],
                mode='markers',
                hoverinfo='text',
                name=info["name"],
                marker=dict(
                    color=info["color"],
                    size=15,
                    line_width=2))
        
        # Get centrality for node sizing
        centrality = graph.calculate_centrality()
        
        # Get student model for mastery info
        student_model = st.session_state.tutor_system.student_model
        
        # Add nodes to appropriate traces
        for node in subgraph.nodes():
            x, y = pos[node]
            
            # Get node attributes
            attrs = graph.get_node_attributes(node)
            node_type = attrs.get('type', 'unknown')
            
            # Make sure we have a valid node type
            if node_type not in node_traces:
                node_type = 'unknown'
            
            # Add coordinates to the trace
            node_traces[node_type]['x'] = node_traces[node_type]['x'] + (x,)
            node_traces[node_type]['y'] = node_traces[node_type]['y'] + (y,)
            
            # Create hover text with node details
            hover_text = f"<b>{node}</b><br>"
            hover_text += f"Type: {attrs.get('type', 'unknown')}<br>"
            if 'description' in attrs and attrs['description']:
                # Limit description length
                desc = attrs['description']
                if len(desc) > 100:
                    desc = desc[:97] + "..."
                hover_text += f"Description: {desc}<br>"
            
            # Add mastery info if available
            mastery = student_model.get_topic_mastery(node) * 100
            hover_text += f"Mastery: {mastery:.1f}%<br>"
            
            hover_text += f"Centrality: {centrality.get(node, 0):.3f}"
            
            # Add to trace text
            node_traces[node_type]['text'] = node_traces[node_type]['text'] + (hover_text,)
            
            # Adjust node size based on centrality
            importance = centrality.get(node, 0) * 50
            size = 10 + importance  # Base size + importance
            
            # If this is the first node, initialize the size array
            if len(node_traces[node_type]['marker']['size']) == 1:
                node_traces[node_type]['marker']['size'] = [size]
            else:
                # Append to the size array
                node_traces[node_type]['marker']['size'] = node_traces[node_type]['marker']['size'] + (size,)
        
        # Create figure with all traces
        data = [edge_trace]
        for trace in node_traces.values():
            if len(trace['x']) > 0:  # Only add traces with nodes
                data.append(trace)
        
        fig = go.Figure(
            data=data,
            layout=go.Layout(
                title=f'Knowledge Graph ({len(filtered_nodes)} topics)',
                titlefont_size=16,
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                plot_bgcolor='rgba(17,17,17,1)',
                paper_bgcolor='rgba(17,17,17,1)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(50,50,50,0.6)"
                )
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating graph visualization: {str(e)}", exc_info=True)
        # Return error figure
        fig = go.Figure()
        fig.update_layout(
            title="Knowledge Graph (Error in Visualization)",
            annotations=[
                dict(
                    text=f"Error: {str(e)}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    font=dict(color='white')
                )
            ],
            plot_bgcolor='rgba(17,17,17,1)',
            paper_bgcolor='rgba(17,17,17,1)',
            font=dict(color='white')
        )
        return fig


def _display_topic_details(graph, all_topics):
    """Display details for an individual topic."""
    st.subheader("Topic Details")
    
    # Get all topics for selection
    all_topics.sort()  # Sort alphabetically for easier navigation
    
    # Add a search box for topics
    topic_search = st.text_input("Search for a topic:", key="kg_topic_search")
    
    filtered_topics = all_topics
    if topic_search:
        filtered_topics = [t for t in all_topics if topic_search.lower() in t.lower()]
        
        if not filtered_topics:
            st.info(f"No topics found matching '{topic_search}'")
    
    if filtered_topics:
        # Select a topic to view details
        selected_topic = st.selectbox(
            "Select a topic to view details:", 
            filtered_topics,
            key="kg_topic_select"
        )
        
        if selected_topic:
            # Get topic details
            topic_attrs = graph.get_node_attributes(selected_topic)
            
            # Create a card-like container for topic details
            topic_card = st.container(border=True)
            with topic_card:
                st.markdown(f"### {selected_topic}")
                
                # Show mastery progress
                if hasattr(st.session_state.tutor_system, 'student_model'):
                    mastery = st.session_state.tutor_system.student_model.get_topic_mastery(selected_topic) * 100
                    st.progress(mastery/100, text=f"Mastery: {mastery:.1f}%")
                
                # Show topic metadata
                if topic_attrs:
                    # Show in columns for better layout
                    meta_col1, meta_col2 = st.columns(2)
                    
                    with meta_col1:
                        st.markdown(f"**Type:** {topic_attrs.get('type', 'Unknown')}")
                        if 'source' in topic_attrs:
                            st.markdown(f"**Source:** {topic_attrs['source']}")
                    
                    with meta_col2:
                        if 'centrality' in topic_attrs:
                            st.markdown(f"**Importance:** {topic_attrs['centrality']:.3f}")
                        else:
                            # Calculate centrality on the fly
                            centrality = graph.calculate_centrality().get(selected_topic, 0)
                            st.markdown(f"**Importance:** {centrality:.3f}")
                
                # Show description
                if topic_attrs and 'description' in topic_attrs and topic_attrs['description']:
                    st.markdown("### Description")
                    st.markdown(topic_attrs['description'])
                
                # Get relationships
                prerequisites = graph.get_prerequisites(selected_topic)
                successors = list(graph.graph.successors(selected_topic))
                
                # Sort by name
                prerequisites.sort()
                successors.sort()
                
                # Remove topic from its own relationships
                if selected_topic in prerequisites:
                    prerequisites.remove(selected_topic)
                if selected_topic in successors:
                    successors.remove(selected_topic)
                
                # Show in columns
                rel_col1, rel_col2 = st.columns(2)
                
                with rel_col1:
                    st.markdown("### Prerequisites")
                    if prerequisites:
                        for prereq in prerequisites:
                            prereq_container = st.container(border=True)
                            with prereq_container:
                                st.markdown(f"#### {prereq}")
                                prereq_mastery = st.session_state.tutor_system.student_model.get_topic_mastery(prereq) * 100
                                st.progress(prereq_mastery/100, text=f"Mastery: {prereq_mastery:.1f}%")
                                
                                if st.button("Study", key=f"study_prereq_{prereq}"):
                                    st.session_state.current_topic = prereq
                                    st.session_state.learning_mode = "study_mode"
                                    st.rerun()
                    else:
                        st.markdown("*No prerequisites*")
                
                with rel_col2:
                    st.markdown("### Builds Toward")
                    if successors:
                        for succ in successors:
                            succ_container = st.container(border=True)
                            with succ_container:
                                st.markdown(f"#### {succ}")
                                succ_mastery = st.session_state.tutor_system.student_model.get_topic_mastery(succ) * 100
                                st.progress(succ_mastery/100, text=f"Mastery: {succ_mastery:.1f}%")
                                
                                if st.button("Study", key=f"study_succ_{succ}"):
                                    st.session_state.current_topic = succ
                                    st.session_state.learning_mode = "study_mode"
                                    st.rerun()
                    else:
                        st.markdown("*No dependent topics*")
            
            # Actions
            action_col1, action_col2 = st.columns(2)
            
            with action_col1:
                if st.button("Study This Topic", key=f"kg_study_{selected_topic}", use_container_width=True):
                    st.session_state.current_topic = selected_topic
                    st.session_state.learning_mode = "study_mode"
                    st.rerun()
            
            with action_col2:
                if st.button("Quiz on This Topic", key=f"kg_quiz_{selected_topic}", use_container_width=True):
                    st.session_state.topic_focus = selected_topic
                    st.session_state.learning_mode = "new_quiz"
                    st.rerun()


def _display_graph_statistics(graph):
    """Display statistics about the knowledge graph."""
    st.subheader("Knowledge Graph Statistics")
    
    # Get graph statistics
    num_nodes = len(graph.get_all_nodes())
    num_edges = len(graph.get_all_edges())
    
    # Calculate node type distribution
    node_types = {}
    for node in graph.get_all_nodes():
        node_type = graph.get_node_attributes(node).get("type", "unknown")
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    # Calculate centrality
    centrality = graph.calculate_centrality()
    
    # Most central nodes
    most_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Display overall stats
    st.markdown("### Overview")
    
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric("Topics", num_nodes)
    
    with metrics_col2:
        st.metric("Relationships", num_edges)
    
    with metrics_col3:
        # Calculate the graph's connectivity
        try:
            if num_nodes > 0:
                density = num_edges / (num_nodes * (num_nodes - 1))
                st.metric("Graph Density", f"{density*100:.2f}%")
            else:
                st.metric("Graph Density", "N/A")
        except:
            st.metric("Graph Density", "N/A")
    
    # Display node type distribution
    st.markdown("### Topic Types")
    
    if node_types:
        # Convert to DataFrame for visualization
        type_data = []
        for node_type, count in node_types.items():
            type_data.append({"Type": node_type.capitalize(), "Count": count})
        
        type_df = pd.DataFrame(type_data)
        st.bar_chart(type_df, x="Type", y="Count")
    
    # Display most central nodes
    st.markdown("### Most Important Topics")
    
    if most_central:
        # Convert to DataFrame for display
        central_data = []
        for node, central_value in most_central:
            central_data.append({
                "Topic": node,
                "Importance": central_value,
                "Type": graph.get_node_attributes(node).get("type", "unknown").capitalize()
            })
        
        central_df = pd.DataFrame(central_data)
        st.dataframe(central_df, hide_index=True, use_container_width=True)