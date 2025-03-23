import logging
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional

# Configure logging
logger = logging.getLogger("learning_path")

class LearningPathManager:
    """Manages learning paths and student progress."""
    
    def __init__(self, knowledge_graph):
        """Initialize with a knowledge graph."""
        self.knowledge_graph = knowledge_graph
        self.errors = []
        self.all_source_topics = set()  # Keep track of topics extracted from documents
        
    def register_document_topic(self, topic: str) -> None:
        """Register a topic as coming from a document."""
        self.all_source_topics.add(topic)
        logger.info(f"Registered document topic: {topic}")
    
    def register_document_topics(self, topics: List[str]) -> None:
        """Register multiple topics as coming from documents."""
        for topic in topics:
            self.register_document_topic(topic)
    
    def calculate_learning_path(self, topics: List[str] = None, current_mastery: Dict[str, float] = None) -> List[str]:
        """Calculate an optimal learning path based on the knowledge graph and mastery."""
        try:
            # Get all topics if not provided
            all_topics = topics or self.knowledge_graph.get_all_nodes()
            
            # If the graph is empty, return empty path
            if not all_topics:
                logger.warning("No topics available for learning path")
                return []
            
            # Prioritize topics from documents
            doc_topics = [t for t in all_topics if t in self.all_source_topics]
            
            # If we have document topics, prioritize those
            if doc_topics:
                logger.info(f"Prioritizing {len(doc_topics)} topics from documents")
                primary_topics = doc_topics
            else:
                primary_topics = all_topics
            
            # Find starting points (nodes with no prerequisites)
            starting_points = []
            for node in primary_topics:
                if not list(self.knowledge_graph.graph.predecessors(node)):
                    starting_points.append(node)
            
            logger.info(f"Found {len(starting_points)} potential starting points")
            
            # If no clear starting points, use centrality to find important concepts
            if not starting_points:
                logger.info("No clear starting points, using centrality")
                centrality = self.knowledge_graph.calculate_centrality()
                # Sort by lowest centrality (likely to be foundational)
                sorted_by_centrality = sorted(centrality.items(), key=lambda x: x[1])
                if sorted_by_centrality:
                    starting_points = [sorted_by_centrality[0][0]]
            
            # Generate a path using topological sort if possible
            try:
                # Check if graph has cycles
                if nx.is_directed_acyclic_graph(self.knowledge_graph.graph):
                    logger.info("Graph is acyclic, using topological sort")
                    # Use topological sort for a linear learning sequence
                    path = list(nx.topological_sort(self.knowledge_graph.graph))
                    
                    # Filter to only include primary topics and their prerequisites
                    final_path = []
                    included = set()
                    
                    # First add all primary topics in topological order
                    for topic in path:
                        if topic in primary_topics and topic not in included:
                            final_path.append(topic)
                            included.add(topic)
                    
                    # Then ensure prerequisites are included
                    for topic in list(final_path):  # Use a copy to avoid modification during iteration
                        prereqs = self._get_all_prerequisites(topic)
                        for prereq in prereqs:
                            if prereq not in included:
                                # Find right position to insert (before the topic that needs it)
                                insert_pos = final_path.index(topic)
                                final_path.insert(insert_pos, prereq)
                                included.add(prereq)
                    
                    path = final_path
                else:
                    logger.info("Graph has cycles, using custom traversal")
                    # Fall back to a custom traversal approach
                    path = self._custom_learning_path_traversal(starting_points, primary_topics)
            except Exception as graph_error:
                logger.error(f"Graph traversal error: {str(graph_error)}", exc_info=True)
                self.errors.append(f"Graph traversal error: {str(graph_error)}")
                # Fall back to custom traversal if topological sort fails
                path = self._custom_learning_path_traversal(starting_points, primary_topics)
            
            # Adjust path based on mastery if provided
            if current_mastery:
                path = self._adjust_path_based_on_mastery(path, current_mastery)
            
            logger.info(f"Final learning path contains {len(path)} topics")
            return path
            
        except Exception as e:
            error_msg = f"Error calculating learning path: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return list(self.knowledge_graph.get_all_nodes())  # Fall back to simple list of all nodes
    
    def _get_all_prerequisites(self, topic: str) -> List[str]:
        """Get all prerequisites for a topic, including transitive prerequisites."""
        visited = set()
        prereqs = []
        
        def visit_prereq(t):
            if t in visited:
                return
            visited.add(t)
            
            direct_prereqs = list(self.knowledge_graph.graph.predecessors(t))
            for prereq in direct_prereqs:
                visit_prereq(prereq)
                if prereq not in prereqs:
                    prereqs.append(prereq)
        
        visit_prereq(topic)
        return prereqs
    
    def _custom_learning_path_traversal(self, starting_points: List[str], primary_topics: List[str]) -> List[str]:
        """Custom graph traversal to create a learning path."""
        path = []
        visited = set()
        
        def visit(node):
            if node in visited:
                return
            
            # Mark as visited
            visited.add(node)
            
            # Process prerequisites first (depth-first)
            for prereq in self.knowledge_graph.get_prerequisites(node):
                visit(prereq)
            
            # Add to path
            if node not in path:
                path.append(node)
            
            # Process related concepts
            for next_node in self.knowledge_graph.graph.successors(node):
                if next_node not in visited:
                    # Check if all prerequisites are visited
                    prerequisites = list(self.knowledge_graph.graph.predecessors(next_node))
                    if all(p in visited for p in prerequisites):
                        visit(next_node)
        
        # Start from each starting point
        for start in starting_points:
            visit(start)
        
        # Make sure all primary topics are included
        for topic in primary_topics:
            if topic not in visited:
                # Get prerequisites first
                prereqs = self._get_all_prerequisites(topic)
                for prereq in prereqs:
                    if prereq not in visited:
                        visit(prereq)
                
                # Now visit the topic itself
                visit(topic)
        
        # Add any remaining nodes
        for node in self.knowledge_graph.graph.nodes():
            if node not in visited:
                visit(node)
        
        return path
    
    def _adjust_path_based_on_mastery(self, path: List[str], mastery: Dict[str, float]) -> List[str]:
        """Adjust learning path based on current mastery levels."""
        if not path or not mastery:
            return path
            
        try:
            # Separate topics into buckets based on mastery
            low_mastery = []    # < 0.4
            medium_mastery = [] # 0.4 - 0.7
            high_mastery = []   # > 0.7
            
            for topic in path:
                m = mastery.get(topic, 0.0)
                if m < 0.4:
                    low_mastery.append(topic)
                elif m < 0.7:
                    medium_mastery.append(topic)
                else:
                    high_mastery.append(topic)
            
            logger.info(f"Mastery distribution: low={len(low_mastery)}, medium={len(medium_mastery)}, high={len(high_mastery)}")
            
            # Create a new path prioritizing low mastery topics
            # while respecting prerequisites
            new_path = []
            
            # First, add foundational topics that must come first (regardless of mastery)
            # These are topics with no prerequisites
            for topic in path:
                if not self.knowledge_graph.get_prerequisites(topic) and topic not in new_path:
                    new_path.append(topic)
            
            # Then add low mastery topics that don't depend on other low mastery topics
            for topic in low_mastery:
                if topic not in new_path:
                    # Check if all prerequisites are already in the new path
                    prerequisites = self.knowledge_graph.get_prerequisites(topic)
                    if all(p in new_path for p in prerequisites):
                        new_path.append(topic)
            
            # Add medium mastery topics
            for topic in medium_mastery:
                if topic not in new_path:
                    prerequisites = self.knowledge_graph.get_prerequisites(topic)
                    if all(p in new_path or p in high_mastery for p in prerequisites):
                        new_path.append(topic)
            
            # Add remaining topics from the original path that aren't in the new path yet
            for topic in path:
                if topic not in new_path:
                    new_path.append(topic)
            
            return new_path
            
        except Exception as e:
            error_msg = f"Error adjusting path based on mastery: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return path  # Return original path on error
    
    def update_learning_path_after_quiz(self, path: List[str], current_position: int, 
                                       topic_mastery: Dict[str, float], weaknesses: List[str]) -> Tuple[List[str], int]:
        """Update learning path based on quiz results."""
        if not path:
            return path, 0
            
        try:
            # Determine if current topic is mastered
            new_position = current_position
            if current_position < len(path):
                current_topic = path[current_position]
                if current_topic in topic_mastery and topic_mastery[current_topic] > 0.8:
                    # Topic mastered, move to next
                    logger.info(f"Topic mastered ({topic_mastery[current_topic]:.2f}), moving to next position")
                    new_position = min(current_position + 1, len(path) - 1)
            
            # Prioritize weaknesses in the learning path
            if weaknesses:
                logger.info(f"Prioritizing weaknesses in learning path: {weaknesses}")
                # Get current path
                new_path = path.copy()
                
                # Remove weaknesses from current position onwards
                for weakness in weaknesses:
                    if weakness in new_path[new_position:]:
                        new_path.remove(weakness)
                
                # Add weaknesses immediately after current position
                for weakness in reversed(weaknesses):  # Reverse to maintain order
                    # Get prerequisites for the weakness
                    prereqs = self._get_all_prerequisites(weakness)
                    
                    # Check if prerequisites are in the path before current position
                    missing_prereqs = [p for p in prereqs if p not in new_path[:new_position]]
                    
                    # Insert missing prerequisites first
                    for prereq in reversed(missing_prereqs):
                        new_path.insert(new_position + 1, prereq)
                    
                    # Then insert the weakness
                    new_path.insert(new_position + 1 + len(missing_prereqs), weakness)
                
                return new_path, new_position
            
            return path, new_position
            
        except Exception as e:
            error_msg = f"Error updating learning path: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return path, current_position  # Return original on error
    
    def suggest_next_steps(self, current_topic: str, mastery_levels: Dict[str, float]) -> List[Dict]:
        """Suggest next steps after studying a topic."""
        if not current_topic:
            return []
            
        try:
            suggestions = []
            
            # 1. Check if current topic is mastered
            current_mastery = mastery_levels.get(current_topic, 0.0)
            
            # 2. If not mastered, suggest to continue with this topic
            if current_mastery < 0.8:
                suggestions.append({
                    "type": "continue",
                    "topic": current_topic,
                    "reason": f"You've made progress ({current_mastery*100:.1f}%), but should continue studying this topic."
                })
            
            # 3. Get related topics
            related = self.knowledge_graph.get_related_concepts(current_topic)
            
            # 4. Filter to find topics that build on current topic
            builds_on = []
            for topic in related:
                if current_topic in self.knowledge_graph.get_prerequisites(topic):
                    builds_on.append(topic)
            
            # Add suggestion if we found topics that build on this one
            if builds_on:
                # Sort by mastery level (lowest first)
                sorted_topics = sorted(builds_on, key=lambda t: mastery_levels.get(t, 0.0))
                for topic in sorted_topics[:2]:  # Suggest up to 2
                    suggestions.append({
                        "type": "next",
                        "topic": topic,
                        "reason": f"This topic builds on {current_topic} and is a logical next step."
                    })
            
            # 5. Suggest related topics that aren't directly connected
            related_only = [r for r in related if r not in builds_on and r != current_topic]
            if related_only:
                # Sort by mastery level (lowest first)
                sorted_related = sorted(related_only, key=lambda t: mastery_levels.get(t, 0.0))
                for topic in sorted_related[:2]:  # Suggest up to 2
                    suggestions.append({
                        "type": "related",
                        "topic": topic,
                        "reason": f"This topic is related to {current_topic} and would complement your understanding."
                    })
            
            return suggestions
            
        except Exception as e:
            error_msg = f"Error suggesting next steps: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return []