import time
import re
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Configure logging
logger = logging.getLogger("quiz_generator")

# Constants
MAX_RETRY_ATTEMPTS = 3

class QuizGenerator:
    """Handles quiz generation and evaluation."""
    
    def __init__(self, llm, doc_retriever=None):
        """Initialize with LLM and optional retriever."""
        self.llm = llm
        self.doc_retriever = doc_retriever
        self.errors = []
        
    def generate_quiz(self, topics: List[str], num_questions: int = 5, difficulty: str = 'adaptive', mastery_levels: Dict[str, float] = None) -> List[Dict]:
        """Generate a quiz based on the provided topics."""
        if not self.llm:
            error_msg = "No LLM available for quiz generation"
            self.errors.append(error_msg)
            logger.error(error_msg)
            return []
            
        if not topics:
            error_msg = "No topics provided for quiz generation"
            self.errors.append(error_msg)
            logger.error(error_msg)
            return []
            
        try:
            logger.info(f"Generating quiz for topics: {topics}, difficulty: {difficulty}")
            
            # Adjust difficulty based on mastery levels if provided and adaptive
            if difficulty == 'adaptive' and mastery_levels:
                avg_mastery = sum(mastery_levels.values()) / len(mastery_levels) if mastery_levels else 0
                
                if avg_mastery < 0.3:
                    difficulty = 'easy'
                elif avg_mastery < 0.7:
                    difficulty = 'medium'
                else:
                    difficulty = 'hard'
                
                logger.info(f"Adaptive difficulty set to: {difficulty} based on mastery: {avg_mastery}")
            
            # Get relevant context if retriever is available
            context = ""
            if self.doc_retriever:
                # Combine topics into a query
                query = " ".join(topics)
                logger.info(f"Retrieving context for query: {query}")
                
                # Retrieve relevant documents
                try:
                    docs = self.doc_retriever.invoke(query)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    logger.info(f"Retrieved context: {len(context)} characters")
                except Exception as retrieval_error:
                    error_msg = f"Error retrieving context: {str(retrieval_error)}"
                    self.errors.append(error_msg)
                    logger.error(error_msg)
                    # Continue without context
                    context = ""
            
            # Format topics for prompt
            topics_text = "\n".join([f"- {topic}" for topic in topics])
            
            # Create quiz prompt
            quiz_prompt = PromptTemplate(
                template=f"""
                Create a quiz with {num_questions} questions on the following topics:
                {topics_text}
                
                Difficulty level: {difficulty}
                
                Use the following context from learning materials to create highly relevant questions:
                {{context}}
                
                For each question, include:
                1. A clear question 
                2. Four answer choices (A, B, C, D)
                3. The correct answer
                4. A brief explanation of why it's correct
                5. The specific topic it relates to (from the list above)
                
                Create a mix of question types:
                - Factual recall questions
                - Conceptual understanding questions
                - Application/problem-solving questions
                
                Format each question as follows:
                
                Question: [question text]
                A. [option A]
                B. [option B]
                C. [option C]
                D. [option D]
                Correct Answer: [letter]
                Explanation: [explanation]
                Topic: [related topic]
                Difficulty: [easy/medium/hard]
                
                Make sure the questions are challenging but fair, and directly relevant to the topics.
                """,
                input_variables=["context"]
            )
            
            # Create quiz generation chain
            quiz_chain = LLMChain(
                llm=self.llm,
                prompt=quiz_prompt
            )
            
            # Generate questions with retry mechanism
            max_retries = MAX_RETRY_ATTEMPTS
            for attempt in range(max_retries):
                try:
                    logger.info(f"Quiz generation attempt {attempt+1}/{max_retries}")
                    result = quiz_chain.invoke({"context": context})
                    quiz_text = result["text"]
                    
                    logger.debug(f"Generated quiz text: {quiz_text[:200]}...")
                    
                    # Parse questions
                    questions = self._parse_quiz_questions(quiz_text)
                    logger.info(f"Parsed {len(questions)} questions from quiz text")
                    
                    # Validate questions
                    valid_questions = []
                    for q in questions:
                        if self._validate_question(q):
                            valid_questions.append(q)
                    
                    logger.info(f"{len(valid_questions)} of {len(questions)} questions are valid")
                    
                    # If we got at least one valid question, return them
                    if valid_questions:
                        return valid_questions
                        
                    # If no valid questions and more retries left, try again
                    if attempt < max_retries - 1:
                        logger.warning("No valid questions generated, retrying...")
                        continue
                        
                    # Last attempt failed to produce valid questions
                    error_msg = "Failed to generate valid quiz questions"
                    self.errors.append(error_msg)
                    logger.error(error_msg)
                    
                    # Fall back to a simpler prompt if we've failed to generate valid questions
                    return self._generate_fallback_quiz(topics, num_questions, difficulty)
                    
                except Exception as retry_error:
                    error_msg = f"Error in quiz generation attempt {attempt+1}: {str(retry_error)}"
                    logger.error(error_msg)
                    
                    if attempt == max_retries - 1:
                        self.errors.append(f"Failed to generate quiz after {max_retries} attempts: {str(retry_error)}")
                        # Try fallback method
                        return self._generate_fallback_quiz(topics, num_questions, difficulty)
                    
                    time.sleep(1)  # Wait before retrying
            
            return []
                
        except Exception as e:
            error_msg = f"Error generating quiz: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return []
    
    def _generate_fallback_quiz(self, topics: List[str], num_questions: int, difficulty: str) -> List[Dict]:
        """Generate a simpler quiz as a fallback when the main method fails."""
        logger.info("Attempting fallback quiz generation")
        
        try:
            # Simpler prompt with less complex formatting requirements
            simple_prompt = PromptTemplate(
                template=f"""
                Create {num_questions} simple quiz questions about these topics:
                {', '.join(topics)}
                
                Difficulty: {difficulty}
                
                Format:
                
                Question: [question]
                A. [option]
                B. [option]
                C. [option]
                D. [option]
                Correct Answer: [A/B/C/D]
                Explanation: [simple explanation]
                Topic: [topic]
                """,
                input_variables=[]
            )
            
            # Create simple quiz chain
            simple_chain = LLMChain(
                llm=self.llm,
                prompt=simple_prompt
            )
            
            # Try to generate
            for attempt in range(2):  # Just 2 attempts for fallback
                try:
                    result = simple_chain.invoke({})
                    quiz_text = result["text"]
                    
                    questions = self._parse_quiz_questions(quiz_text)
                    
                    # Be more lenient in validation for fallback
                    valid_questions = []
                    for q in questions:
                        if self._validate_fallback_question(q):
                            valid_questions.append(q)
                    
                    if valid_questions:
                        logger.info(f"Fallback generated {len(valid_questions)} valid questions")
                        return valid_questions
                        
                except Exception as fallback_error:
                    logger.error(f"Fallback generation error: {str(fallback_error)}")
                    
                    if attempt == 1:  # Last attempt
                        # Create minimal hardcoded quiz if everything else fails
                        return self._create_minimal_quiz(topics)
            
            return self._create_minimal_quiz(topics)
                
        except Exception as e:
            logger.error(f"Error in fallback quiz generation: {str(e)}", exc_info=True)
            return self._create_minimal_quiz(topics)
    
    def _create_minimal_quiz(self, topics: List[str]) -> List[Dict]:
        """Create a minimal quiz with basic questions when all else fails."""
        logger.info("Creating minimal hardcoded quiz")
        
        minimal_questions = []
        
        for i, topic in enumerate(topics[:3]):  # Use up to 3 topics
            # Create a very basic question for each topic
            question = {
                "question": f"Which of the following best describes '{topic}'?",
                "options": [
                    {"letter": "A", "text": f"A fundamental concept in {topic}"},
                    {"letter": "B", "text": f"An advanced application of {topic}"},
                    {"letter": "C", "text": f"A related field to {topic}"},
                    {"letter": "D", "text": f"None of the above"}
                ],
                "correct_answer": "A",
                "explanation": f"This is a basic definition question about {topic}.",
                "topic": topic,
                "difficulty": "medium"
            }
            minimal_questions.append(question)
        
        logger.info(f"Created {len(minimal_questions)} minimal questions")
        return minimal_questions
    
    def _parse_quiz_questions(self, quiz_text: str) -> List[Dict]:
        """Parse generated quiz text into structured question objects."""
        questions = []
        current_question = {}
        
        # Add debug logging for parsing
        logger.debug(f"Parsing quiz text of length {len(quiz_text)}")
        
        try:
            # Split by double newlines to separate questions
            question_blocks = re.split(r'\n\s*\n', quiz_text)
            logger.debug(f"Split into {len(question_blocks)} question blocks")
            
            for block in question_blocks:
                lines = block.strip().split('\n')
                
                # Check if this looks like a new question
                if lines and any(line.strip().startswith("Question:") for line in lines):
                    # Save previous question if it exists
                    if current_question and "question" in current_question and "options" in current_question:
                        questions.append(current_question.copy())
                    
                    # Start new question
                    current_question = {
                        "question": "",
                        "options": [],
                        "correct_answer": "",
                        "explanation": "",
                        "topic": "",
                        "difficulty": "medium"  # Default difficulty
                    }
                    
                    # Process lines
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Question
                        if line.startswith("Question:"):
                            current_question["question"] = line[9:].strip()
                        
                        # Option A, B, C, D
                        elif line.startswith("A.") or line.startswith("B.") or line.startswith("C.") or line.startswith("D."):
                            option_letter = line[0]
                            option_text = line[2:].strip()
                            current_question["options"].append({"letter": option_letter, "text": option_text})
                        
                        # Correct answer
                        elif line.startswith("Correct Answer:"):
                            current_question["correct_answer"] = line[15:].strip()
                        
                        # Explanation
                        elif line.startswith("Explanation:"):
                            current_question["explanation"] = line[12:].strip()
                        
                        # Topic
                        elif line.startswith("Topic:"):
                            current_question["topic"] = line[6:].strip()
                        
                        # Difficulty
                        elif line.startswith("Difficulty:"):
                            current_question["difficulty"] = line[11:].strip().lower()
            
            # Add the last question
            if current_question and "question" in current_question and "options" in current_question:
                questions.append(current_question.copy())
            
            logger.debug(f"Successfully parsed {len(questions)} questions")
            return questions
            
        except Exception as e:
            logger.error(f"Error parsing quiz questions: {str(e)}", exc_info=True)
            return []
    
    def _validate_question(self, question: Dict) -> bool:
        """Validate that a question has all required fields and is well-formed."""
        try:
            # Check required fields
            required_fields = ["question", "options", "correct_answer", "explanation", "topic"]
            for field in required_fields:
                if field not in question or not question[field]:
                    logger.debug(f"Question missing required field: {field}")
                    return False
            
            # Check options
            if len(question["options"]) != 4:
                logger.debug(f"Question has {len(question['options'])} options, not 4")
                return False
                
            # Check that correct answer is one of the options
            valid_answers = ["A", "B", "C", "D"]
            if question["correct_answer"] not in valid_answers:
                logger.debug(f"Question has invalid correct answer: {question['correct_answer']}")
                return False
                
            # Check that options have correct format
            for option in question["options"]:
                if "letter" not in option or "text" not in option:
                    logger.debug("Question has option with missing letter or text")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error validating question: {str(e)}")
            return False
    
    def _validate_fallback_question(self, question: Dict) -> bool:
        """More lenient validation for fallback questions."""
        try:
            # Make sure there's a question text
            if "question" not in question or not question["question"]:
                return False
            
            # Make sure there are at least 2 options
            if "options" not in question or len(question["options"]) < 2:
                return False
                
            # Make sure there's a correct answer that matches one of the options
            if "correct_answer" not in question:
                return False
                
            # Make sure we have a topic
            if "topic" not in question or not question["topic"]:
                return False
                
            return True
            
        except Exception:
            return False
    
    def evaluate_quiz_results(self, quiz_results: Dict) -> Dict:
        """Evaluate quiz results and provide analysis."""
        if not quiz_results:
            self.errors.append("No quiz results to analyze")
            return {}
            
        try:
            # Calculate overall score
            correct_count = sum(1 for result in quiz_results.values() if result["correct"])
            total_count = len(quiz_results)
            score_percentage = (correct_count / total_count) * 100 if total_count > 0 else 0
            
            # Group results by topic
            topic_results = defaultdict(lambda: {"correct": 0, "total": 0})
            
            for question_idx, result in quiz_results.items():
                topic = result["topic"]
                topic_results[topic]["total"] += 1
                if result["correct"]:
                    topic_results[topic]["correct"] += 1
            
            # Calculate topic mastery and identify weaknesses
            topic_mastery = {}
            weaknesses = []
            strengths = []
            
            for topic, results in topic_results.items():
                mastery = (results["correct"] / results["total"]) if results["total"] > 0 else 0
                topic_mastery[topic] = mastery
                
                # Categorize as weakness or strength
                if mastery < 0.6:  # Less than 60% correct
                    weaknesses.append(topic)
                elif mastery > 0.8:  # More than 80% correct
                    strengths.append(topic)
            
            # Create analysis information
            analysis = {
                "score": score_percentage,
                "correct_count": correct_count,
                "total_count": total_count,
                "topic_results": topic_results,
                "topic_mastery": topic_mastery,
                "weaknesses": weaknesses,
                "strengths": strengths
            }
            
            # Generate personalized feedback if LLM is available
            if self.llm:
                analysis["feedback"] = self._generate_feedback(
                    score_percentage, 
                    correct_count, 
                    total_count,
                    topic_results,
                    weaknesses,
                    strengths
                )
            
            return analysis
                
        except Exception as e:
            self.errors.append(f"Error analyzing quiz results: {str(e)}")
            logger.error(f"Quiz analysis error: {str(e)}", exc_info=True)
            return {
                "score": 0,
                "error": str(e)
            }
    
    def _generate_feedback(self, score, correct_count, total_count, topic_results, weaknesses, strengths):
        """Generate personalized feedback using LLM."""
        if not self.llm:
            return "Feedback generation not available."
            
        try:
            # Format topic performance for prompt
            topic_performance_text = "\n".join([
                f"- {topic}: {results['correct']}/{results['total']} correct ({(results['correct']/results['total'])*100:.1f}%)"
                for topic, results in topic_results.items()
            ])
            
            weaknesses_text = "\n".join([f"- {topic}" for topic in weaknesses]) if weaknesses else "None identified"
            strengths_text = "\n".join([f"- {topic}" for topic in strengths]) if strengths else "None identified"
            
            # Create feedback prompt
            feedback_prompt = PromptTemplate(
                template="""
                Analyze the following quiz results and provide personalized learning recommendations:
                
                Overall Score: {score}% ({correct_count}/{total_count})
                
                Topic Performance:
                {topic_performance}
                
                Identified Weaknesses:
                {weaknesses}
                
                Identified Strengths:
                {strengths}
                
                Please provide:
                1. A brief assessment of the student's understanding
                2. Specific areas that need improvement
                3. Recommended learning strategies for weak areas
                4. What to focus on next in the learning journey
                
                Make your analysis encouraging and constructive while being honest about areas for improvement.
                """,
                input_variables=["score", "correct_count", "total_count", "topic_performance", 
                                "weaknesses", "strengths"]
            )
            
            # Create feedback chain
            feedback_chain = LLMChain(
                llm=self.llm,
                prompt=feedback_prompt
            )
            
            # Generate feedback with retry mechanism
            max_retries = MAX_RETRY_ATTEMPTS
            for attempt in range(max_retries):
                try:
                    result = feedback_chain.invoke({
                        "score": f"{score:.1f}",
                        "correct_count": correct_count,
                        "total_count": total_count,
                        "topic_performance": topic_performance_text,
                        "weaknesses": weaknesses_text,
                        "strengths": strengths_text
                    })
                    
                    return result["text"]
                    
                except Exception as retry_error:
                    if attempt == max_retries - 1:
                        self.errors.append(f"Failed to generate feedback after {max_retries} attempts: {str(retry_error)}")
                        logger.error(f"Feedback generation error: {str(retry_error)}")
                    time.sleep(1)  # Wait before retrying
            
            return "Unable to generate personalized feedback at this time."
                
        except Exception as e:
            self.errors.append(f"Error generating feedback: {str(e)}")
            logger.error(f"Feedback generation error: {str(e)}")
            return "Error generating feedback. Please try again later."