import networkx as nx
import plotly.graph_objects as go
import logging
from typing import List, Dict, Any, Tuple, Set, Optional

# Configure logging
logger = logging.getLogger("knowledge_graph")

class KnowledgeGraph:
    """Knowledge Graph class to store and manipulate the subject's knowledge structure."""
    
    def __init__(self):
        """Initialize the knowledge graph using NetworkX."""
        self.graph = nx.DiGraph()
        logger.info("Knowledge graph initialized")
        
    def add_node(self, node_id: str, **attr):
        """Add a node to the knowledge graph."""
        if not node_id or not isinstance(node_id, str):
            logger.warning(f"Invalid node_id: {node_id}. Skipping.")
            return False
            
        try:
            # Only add if not exists or update attributes
            if node_id in self.graph.nodes:
                # Update attributes
                for key, value in attr.items():
                    self.graph.nodes[node_id][key] = value
                logger.debug(f"Updated node attributes: {node_id}")
            else:
                # Add new node
                self.graph.add_node(node_id, **attr)
                logger.debug(f"Added new node: {node_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding node {node_id}: {str(e)}", exc_info=True)
            return False
        
    def add_edge(self, source: str, target: str, **attr):
        """Add an edge to the knowledge graph."""
        if not source or not target:
            logger.warning(f"Invalid edge: {source} -> {target}. Skipping.")
            return False
            
        try:
            # Check if nodes exist, add them if they don't
            if source not in self.graph:
                self.add_node(source)
            if target not in self.graph:
                self.add_node(target)
                
            # Only add if not exists or update attributes
            if self.graph.has_edge(source, target):
                # Update attributes
                for key, value in attr.items():
                    self.graph.edges[source, target][key] = value
                logger.debug(f"Updated edge attributes: {source} -> {target}")
            else:
                # Add new edge
                self.graph.add_edge(source, target, **attr)
                logger.debug(f"Added new edge: {source} -> {target}")
            return True
        except Exception as e:
            logger.error(f"Error adding edge {source} -> {target}: {str(e)}", exc_info=True)
            return False
        
    def get_prerequisites(self, node_id: str) -> List[str]:
        """Get prerequisite concepts for a given node."""
        if node_id not in self.graph:
            return []
            
        try:
            predecessors = list(self.graph.predecessors(node_id))
            return predecessors
        except Exception as e:
            logger.error(f"Error getting prerequisites for {node_id}: {str(e)}", exc_info=True)
            return []
        
    def get_related_concepts(self, node_id: str) -> List[str]:
        """Get concepts related to the given node."""
        if node_id not in self.graph:
            return []
            
        try:
            predecessors = list(self.graph.predecessors(node_id))
            successors = list(self.graph.successors(node_id))
            return list(set(predecessors + successors))
        except Exception as e:
            logger.error(f"Error getting related concepts for {node_id}: {str(e)}", exc_info=True)
            return []
    
    def get_node_attributes(self, node_id: str) -> Dict:
        """Get all attributes of a node."""
        if node_id not in self.graph:
            return {}
            
        try:
            return dict(self.graph.nodes[node_id])
        except Exception as e:
            logger.error(f"Error getting attributes for {node_id}: {str(e)}", exc_info=True)
            return {}
    
    def set_node_attribute(self, node_id: str, attr_name: str, attr_value: Any) -> bool:
        """Set a specific attribute for a node."""
        if node_id not in self.graph:
            return False
            
        try:
            self.graph.nodes[node_id][attr_name] = attr_value
            return True
        except Exception as e:
            logger.error(f"Error setting attribute for {node_id}: {str(e)}", exc_info=True)
            return False
    
    def get_all_nodes(self) -> List[str]:
        """Get all nodes in the graph."""
        return list(self.graph.nodes)
    
    def get_all_edges(self) -> List[Tuple[str, str]]:
        """Get all edges in the graph."""
        return list(self.graph.edges)
    
    def calculate_centrality(self) -> Dict[str, float]:
        """Calculate centrality to identify important concepts."""
        if not self.graph.nodes:
            return {}
            
        try:
            # Try betweenness centrality first
            try:
                return nx.betweenness_centrality(self.graph)
            except:
                # Fall back to degree centrality if betweenness fails
                logger.warning("Betweenness centrality failed, using degree centrality")
                return nx.degree_centrality(self.graph)
        except Exception as e:
            logger.error(f"Error calculating centrality: {str(e)}", exc_info=True)
            return {node: 0.0 for node in self.graph.nodes}
    
    def get_node_count_by_type(self) -> Dict[str, int]:
        """Count nodes by type attribute."""
        type_counts = {}
        
        for node in self.graph.nodes:
            node_type = self.graph.nodes[node].get('type', 'unknown')
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        
        return type_counts
    
    def find_path(self, start_node: str, end_node: str) -> List[str]:
        """Find a path between two nodes if it exists."""
        if start_node not in self.graph or end_node not in self.graph:
            return []
            
        try:
            if nx.has_path(self.graph, start_node, end_node):
                return nx.shortest_path(self.graph, start_node, end_node)
            return []
        except Exception as e:
            logger.error(f"Error finding path from {start_node} to {end_node}: {str(e)}", exc_info=True)
            return []
    
    def export_to_json(self) -> Dict:
        """Export the knowledge graph to JSON format."""
        try:
            data = {
                "nodes": [],
                "links": []
            }
            
            # Add nodes
            for node_id in self.graph.nodes:
                node_data = {
                    "id": node_id,
                    **self.get_node_attributes(node_id)
                }
                data["nodes"].append(node_data)
            
            # Add edges
            for source, target in self.graph.edges:
                edge_data = {
                    "source": source,
                    "target": target,
                    **self.graph.edges[source, target]
                }
                data["links"].append(edge_data)
            
            return data
        except Exception as e:
            logger.error(f"Error exporting graph to JSON: {str(e)}", exc_info=True)
            return {"nodes": [], "links": []}
    
    def visualize(self) -> go.Figure:
        """Create a Plotly visualization of the knowledge graph."""
        if not self.graph.nodes:
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
            pos = nx.spring_layout(self.graph, seed=42)  # Fixed seed for consistency
            
            # Create edge traces
            edge_x = []
            edge_y = []
            
            for edge in self.graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines')
            
            # Create node traces
            node_x = []
            node_y = []
            node_text = []
            node_importance = []
            node_colors = []
            
            centrality = self.calculate_centrality()
            
            for node in self.graph.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # Get node type for coloring
                node_type = self.graph.nodes[node].get('type', 'unknown')
                if node_type == 'fundamental':
                    node_colors.append('#4287f5')  # blue
                elif node_type == 'derived':
                    node_colors.append('#42f5a7')  # green
                elif node_type == 'technique':
                    node_colors.append('#f5a742')  # orange
                elif node_type == 'question':
                    node_colors.append('#f54242')  # red
                else:
                    node_colors.append('#a9a9a9')  # gray
                
                # Create hover text with node details
                attrs = self.get_node_attributes(node)
                hover_text = f"<b>{node}</b><br>"
                hover_text += f"Type: {attrs.get('type', 'unknown')}<br>"
                if 'description' in attrs and attrs['description']:
                    hover_text += f"Description: {attrs['description']}<br>"
                hover_text += f"Centrality: {centrality.get(node, 0):.3f}"
                
                node_text.append(hover_text)
                node_importance.append(centrality.get(node, 0) * 100)
                
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    size=15,
                    colorbar=dict(
                        thickness=15,
                        title='Node Importance',
                        xanchor='left',
                        titleside='right',
                        titlefont=dict(color='white'),
                        tickfont=dict(color='white')
                    ),
                    color=node_importance,
                    line_width=2))
            
            # Create the figure
            fig = go.Figure(data=[edge_trace, node_trace],
                           layout=go.Layout(
                               title='Knowledge Graph Visualization',
                               titlefont_size=16,
                               showlegend=False,
                               hovermode='closest',
                               margin=dict(b=20,l=5,r=5,t=40),
                               plot_bgcolor='rgba(17,17,17,1)',
                               paper_bgcolor='rgba(17,17,17,1)',
                               font=dict(color='white'),
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating graph visualization: {str(e)}", exc_info=True)
            # Return empty figure on error
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