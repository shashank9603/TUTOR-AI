from typing import Dict, Set, List, Optional
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger("student_model")

class StudentModel:
    """Tracks student progress and knowledge state."""
    
    def __init__(self):
        """Initialize the student model."""
        self.topics_mastered = set()
        self.topics_in_progress = set()
        self.topics_to_learn = set()
        self.quiz_results = {}
        self.learning_path = []
        self.current_position = 0
        self.weaknesses = set()
        self.strengths = set()
        self.mastery_levels = {}  # Maps topics to mastery scores (0-1)
        self.study_history = {}   # Maps topics to last study time and duration
        self.topic_views = {}     # Count of topic views to track engagement
        
    def update_from_quiz_results(self, quiz_analysis: Dict) -> None:
        """Update student model based on quiz results."""
        if not quiz_analysis:
            logger.warning("Attempted to update from empty quiz analysis")
            return
            
        logger.info(f"Updating student model from quiz results: {quiz_analysis.get('score')}%")
            
        # Update topic mastery levels
        if "topic_mastery" in quiz_analysis:
            for topic, mastery in quiz_analysis["topic_mastery"].items():
                self.mastery_levels[topic] = mastery
                
                # Update topic sets based on mastery
                if mastery > 0.8:  # Mastered
                    logger.info(f"Topic mastered: {topic} ({mastery:.2f})")
                    self.topics_mastered.add(topic)
                    self.topics_in_progress.discard(topic)
                    self.topics_to_learn.discard(topic)
                elif mastery > 0.4:  # In progress
                    logger.info(f"Topic in progress: {topic} ({mastery:.2f})")
                    self.topics_in_progress.add(topic)
                    self.topics_to_learn.discard(topic)
                    self.topics_mastered.discard(topic)
                else:  # Needs work
                    logger.info(f"Topic needs work: {topic} ({mastery:.2f})")
                    self.topics_to_learn.add(topic)
                    self.topics_in_progress.discard(topic)
                    self.topics_mastered.discard(topic)
        
        # Update weaknesses and strengths
        if "weaknesses" in quiz_analysis:
            self.weaknesses = set(quiz_analysis["weaknesses"])
            
        if "strengths" in quiz_analysis:
            self.strengths = set(quiz_analysis["strengths"])
        
        # Store quiz results
        quiz_id = f"quiz_{len(self.quiz_results) + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.quiz_results[quiz_id] = quiz_analysis
    
    def update_from_study_session(self, topic: str, duration_minutes: float, completed: bool = False) -> None:
        """
        Update student model based on a study session.
        
        Args:
            topic: The topic that was studied
            duration_minutes: Time spent studying in minutes
            completed: Whether the user marked the topic as completed
        """
        if not topic:
            return
            
        logger.info(f"Updating from study session: {topic}, duration: {duration_minutes}m, completed: {completed}")
            
        # Record study history
        now = datetime.now()
        if topic not in self.study_history:
            self.study_history[topic] = []
            
        self.study_history[topic].append({
            "timestamp": now,
            "duration_minutes": duration_minutes,
            "completed": completed
        })
        
        # Increase view count
        self.topic_views[topic] = self.topic_views.get(topic, 0) + 1
        
        # Update mastery level based on study session
        current_mastery = self.mastery_levels.get(topic, 0.0)
        
        if completed:
            # If marked as completed, high mastery
            new_mastery = max(current_mastery, 0.85)
            self.topics_mastered.add(topic)
            self.topics_in_progress.discard(topic)
            self.topics_to_learn.discard(topic)
        else:
            # Otherwise, increment mastery based on duration (diminishing returns)
            # The more you study, the more mastery improves, but with diminishing returns
            mastery_increment = min(0.2, 0.05 * (1 + duration_minutes / 20))
            new_mastery = min(0.8, current_mastery + mastery_increment)
            
            # Update topic sets
            if new_mastery > 0.7:
                self.topics_in_progress.add(topic)
                self.topics_to_learn.discard(topic)
            else:
                self.topics_to_learn.add(topic)
                self.topics_in_progress.discard(topic)
        
        # Update mastery level
        self.mastery_levels[topic] = new_mastery
        logger.info(f"Updated mastery for {topic}: {current_mastery:.2f} -> {new_mastery:.2f}")
    
    def mark_topic_completed(self, topic: str) -> None:
        """Mark a topic as completed (mastered)."""
        if not topic:
            return
            
        logger.info(f"Marking topic as completed: {topic}")
        self.topics_mastered.add(topic)
        self.topics_in_progress.discard(topic)
        self.topics_to_learn.discard(topic)
        self.mastery_levels[topic] = 1.0
        
        # Record completion event
        now = datetime.now()
        if topic not in self.study_history:
            self.study_history[topic] = []
            
        self.study_history[topic].append({
            "timestamp": now,
            "duration_minutes": 0,  # We don't know the duration
            "completed": True,
            "manually_marked": True
        })
    
    def get_topic_mastery(self, topic: str) -> float:
        """Get mastery level for a specific topic."""
        return self.mastery_levels.get(topic, 0.0)
    
    def get_overall_progress(self) -> float:
        """Calculate overall progress as percentage."""
        total_topics = len(self.topics_mastered) + len(self.topics_in_progress) + len(self.topics_to_learn)
        if total_topics == 0:
            return 0.0
            
        # Weight mastered topics as 1.0, in-progress as weighted by mastery level
        progress_sum = len(self.topics_mastered)
        
        # Add weighted progress for in-progress topics
        for topic in self.topics_in_progress:
            mastery = self.get_topic_mastery(topic)
            progress_sum += mastery
        
        # Calculate overall progress
        progress = progress_sum / total_topics
        return progress * 100  # Return as percentage
    
    def get_mastery_distribution(self) -> Dict[str, int]:
        """Get distribution of topics by mastery level."""
        return {
            "mastered": len(self.topics_mastered),
            "in_progress": len(self.topics_in_progress),
            "to_learn": len(self.topics_to_learn)
        }
    
    def get_study_statistics(self) -> Dict:
        """Get statistics about study patterns."""
        if not self.study_history:
            return {
                "total_study_time": 0,
                "topics_studied": 0,
                "avg_time_per_topic": 0
            }
        
        # Calculate total study time
        total_time = 0
        for topic, sessions in self.study_history.items():
            for session in sessions:
                total_time += session.get("duration_minutes", 0)
        
        # Calculate topics with active study time
        topics_with_time = sum(1 for topic, sessions in self.study_history.items() 
                             if any(session.get("duration_minutes", 0) > 0 for session in sessions))
        
        # Calculate average time per topic
        avg_time = total_time / topics_with_time if topics_with_time > 0 else 0
        
        return {
            "total_study_time": total_time,
            "topics_studied": topics_with_time,
            "avg_time_per_topic": avg_time
        }
    
    def get_most_viewed_topics(self, n: int = 5) -> List[Dict]:
        """Get the most frequently viewed topics."""
        # Sort topics by view count
        sorted_topics = sorted(self.topic_views.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N
        result = []
        for topic, views in sorted_topics[:n]:
            result.append({
                "topic": topic,
                "views": views,
                "mastery": self.get_topic_mastery(topic)
            })
            
        return result
    
    def has_all_topics_mastered(self) -> bool:
        """Check if all known topics have been mastered."""
        all_topics = self.topics_mastered | self.topics_in_progress | self.topics_to_learn
        return len(all_topics) > 0 and all(topic in self.topics_mastered for topic in all_topics)
    
    def get_next_recommended_topic(self) -> Optional[str]:
        """Get the next recommended topic to study."""
        # If there's a current learning path, use the current position
        if self.learning_path and self.current_position < len(self.learning_path):
            return self.learning_path[self.current_position]
        
        # Otherwise, prioritize:
        # 1. Weaknesses
        # 2. In-progress topics (sorted by lowest mastery)
        # 3. Topics to learn
        
        if self.weaknesses:
            return list(self.weaknesses)[0]
        
        if self.topics_in_progress:
            # Find topic with lowest mastery
            sorted_topics = sorted(self.topics_in_progress, 
                                  key=lambda t: self.get_topic_mastery(t))
            return sorted_topics[0]
        
        if self.topics_to_learn:
            return list(self.topics_to_learn)[0]
        
        return None