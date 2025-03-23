<<<<<<< HEAD
import os
import tempfile
import shutil
import time
import uuid
import logging
import requests
from datetime import datetime
from typing import List, Dict, Tuple, Set, Optional, Union

import torch
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tutor_system")

# Constants
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
MAX_RETRY_ATTEMPTS = 3
OLLAMA_API_BASE_URL = "http://localhost:11434"

class PersonalizedTutorSystem:
    """Main class for the personalized tutor system with KG-RAG capabilities."""
    
    def __init__(self, 
                 llm_model_name="llama3.2:latest",
                 embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size=DEFAULT_CHUNK_SIZE,
                 chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                 use_gpu=True):
        """
        Initialize the Personalized Tutor system.
        
        Args:
            llm_model_name: The Ollama model for text generation
            embedding_model_name: The HuggingFace model for embeddings
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            use_gpu: Whether to use GPU acceleration
        """
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        
        # System state
        self.initialized = False
        self.ollama_available = False
        self.errors = []
        self.processing_times = {}
        self.documents_processed = []
        
        # Initialize components in a safe manner
        try:
            logger.info(f"Initializing tutor system with {llm_model_name} and {embedding_model_name}")
            
            # Initialize GPU settings
            self.use_gpu = self._initialize_gpu(use_gpu)
            self.device = "cuda" if self.use_gpu else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Initialize text splitter
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )
            logger.info(f"Text splitter initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")
            
            # Initialize embeddings model
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = self._initialize_embeddings(HuggingFaceEmbeddings)
            logger.info(f"Embeddings model initialized: {embedding_model_name}")
            
            # Initialize LLM
            self.ollama_available = self._check_ollama_connection()
            if self.ollama_available:
                from langchain_ollama import OllamaLLM
                from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
                self.llm = self._initialize_llm(OllamaLLM, StreamingStdOutCallbackHandler)
                logger.info(f"LLM initialized: {llm_model_name}")
            else:
                self.llm = None
                logger.warning("Ollama not available. LLM features will be disabled.")
            
            # Initialize vector stores
            self.doc_vector_store = None
            self.topic_vector_store = None
            
            # Import and initialize core components
            from core.knowledge_graph import KnowledgeGraph
            from core.document_processor import DocumentProcessor
            from core.quiz_generator import QuizGenerator
            from core.learning_path import LearningPathManager
            from core.student_model import StudentModel
            
            # Initialize core components
            self.knowledge_graph = KnowledgeGraph()
            self.document_processor = DocumentProcessor(self.embeddings, self.text_splitter, self.llm)
            self.quiz_generator = QuizGenerator(self.llm)
            self.path_manager = LearningPathManager(self.knowledge_graph)
            self.student_model = StudentModel()
            
            # Generate session ID
            self.session_id = str(uuid.uuid4())
            logger.info(f"Session ID: {self.session_id}")
            
            # Mark as initialized
            self.initialized = True
            logger.info("Tutor system initialization completed successfully")
            
        except Exception as e:
            error_msg = f"Initialization error: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
    
    def _initialize_gpu(self, use_gpu: bool) -> bool:
        """Initialize GPU settings and verify availability."""
        if not use_gpu:
            logger.info("GPU not requested, using CPU")
            return False
            
        try:
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                # Log GPU info
                gpu_info = torch.cuda.get_device_properties(0)
                logger.info(f"Using GPU: {gpu_info.name} ({gpu_info.total_memory / 1024**3:.1f} GB)")
                return True
            else:
                logger.info("GPU requested but not available. Using CPU instead.")
                return False
        except Exception as e:
            logger.warning(f"Error checking GPU availability: {str(e)}. Falling back to CPU.")
            return False
    
    def _initialize_embeddings(self, HuggingFaceEmbeddings):
        """Initialize embeddings model with fallback to CPU if needed."""
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": self.device}
            )
            logger.info(f"Embeddings initialized with model {self.embedding_model_name} on {self.device}")
            return embeddings
        except Exception as e:
            # Try falling back to CPU if there was an error
            if self.device == "cuda":
                logger.warning(f"Error initializing embeddings on GPU: {str(e)}. Falling back to CPU.")
                self.device = "cpu"
                self.use_gpu = False
                return HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                    model_kwargs={"device": "cpu"}
                )
            else:
                # If already on CPU, re-raise
                raise
    
    def _check_ollama_connection(self) -> bool:
        """Check if Ollama is running and available."""
        try:
            response = requests.get(f"{OLLAMA_API_BASE_URL}/api/version", timeout=2)
            if response.status_code == 200:
                logger.info("Ollama connection successful")
                return True
            else:
                logger.warning(f"Ollama returned status code {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"Error connecting to Ollama: {str(e)}")
            return False
    
    def _initialize_llm(self, OllamaLLM, StreamingStdOutCallbackHandler) -> Optional:
        """Initialize LLM with error handling."""
        if not self.ollama_available:
            return None
            
        try:
            callbacks = [StreamingStdOutCallbackHandler()]
            llm = OllamaLLM(model=self.llm_model_name, callbacks=callbacks)
            logger.info(f"LLM initialized with model {self.llm_model_name}")
            return llm
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            self.errors.append(f"LLM initialization error: {str(e)}")
            return None
    
    def process_textbooks(self, pdf_files) -> bool:
        """Process PDF textbooks and create a vector store."""
        if not pdf_files:
            self.errors.append("No PDF files provided")
            return False
            
        all_docs = []
        start_time = time.time()
        
        try:
            logger.info(f"Processing {len(pdf_files)} PDF files")
            
            # Create temporary directory for file storage
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Created temporary directory: {temp_dir}")
            
            # Process each PDF file
            for pdf_file in pdf_files:
                logger.info(f"Processing PDF: {pdf_file.name}")
                chunks, stats = self.document_processor.process_pdf(pdf_file, temp_dir)
                
                if chunks:
                    all_docs.extend(chunks)
                    self.processing_times[pdf_file.name] = stats
                    # Track processed documents
                    if pdf_file.name not in self.documents_processed:
                        self.documents_processed.append(pdf_file.name)
                else:
                    logger.warning(f"No chunks extracted from {pdf_file.name}")
            
            # Create vector store if we have documents
            if all_docs:
                logger.info(f"Creating vector store with {len(all_docs)} chunks")
                index_start_time = time.time()
                
                try:
                    # Create the vector store using FAISS
                    self.doc_vector_store = FAISS.from_documents(all_docs, self.embeddings)
                    
                    # Update the retriever in quiz generator
                    self.quiz_generator.doc_retriever = self.doc_vector_store.as_retriever(search_kwargs={"k": 5})
                    
                    # Record time
                    index_time = time.time() - index_start_time
                    self.processing_times["index_building"] = index_time
                    self.processing_times["total_time"] = time.time() - start_time
                    
                    logger.info(f"Vector store created with {len(all_docs)} chunks")
                    
                    # Try to extract topics from the content
                    if self.ollama_available and self.llm:
                        logger.info("Attempting to extract topics from textbooks")
                        try:
                            extracted_topics = self.document_processor.extract_topics_from_chunks(all_docs[:20])  # Limit to first 20 chunks
                            logger.info(f"Extracted {len(extracted_topics)} topics")
                            
                            # Add topics to knowledge graph
                            if extracted_topics:
                                for topic_info in extracted_topics:
                                    if "topic" in topic_info and topic_info["topic"]:
                                        # Add to knowledge graph
                                        self.knowledge_graph.add_node(
                                            topic_info["topic"],
                                            type=topic_info.get("type", "topic"),
                                            description=topic_info.get("description", ""),
                                            source="extracted"
                                        )
                                        
                                        # Register with path manager
                                        self.path_manager.register_document_topic(topic_info["topic"])
                            
                            # Extract relationships between topics
                            topic_names = [t["topic"] for t in extracted_topics if "topic" in t]
                            if len(topic_names) > 1:
                                relationships = self.document_processor.extract_relationships(topic_names)
                                logger.info(f"Extracted {len(relationships)} relationships")
                                
                                # Add relationships to knowledge graph
                                for rel in relationships:
                                    if "source" in rel and "target" in rel:
                                        self.knowledge_graph.add_edge(
                                            rel["source"], 
                                            rel["target"],
                                            relationship=rel.get("type", "related")
                                        )
                        except Exception as topic_error:
                            logger.error(f"Error extracting topics: {str(topic_error)}", exc_info=True)
                    
                    return True
                    
                except Exception as e:
                    error_msg = f"Error creating vector store: {str(e)}"
                    self.errors.append(error_msg)
                    logger.error(error_msg, exc_info=True)
                    return False
            else:
                self.errors.append("No content extracted from PDFs")
                return False
                
        except Exception as e:
            error_msg = f"Error processing textbooks: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return False
        finally:
            # Clean up temp directory
            if 'temp_dir' in locals():
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as cleanup_error:
                    logger.warning(f"Error cleaning up temporary directory: {str(cleanup_error)}")
    
    def process_topic_list(self, topic_file=None, topic_text=None) -> bool:
        """Process a list of topics/questions from a file or text input."""
        try:
            logger.info("Processing topic list")
            topics = []
            
            # Extract topics either from file or text input
            if topic_file:
                topic_content = topic_file.getvalue().decode("utf-8")
                # Split by new lines and filter empty lines
                raw_topics = [line.strip() for line in topic_content.split('\n') if line.strip()]
                logger.info(f"Extracted {len(raw_topics)} topics from file")
            elif topic_text:
                raw_topics = [line.strip() for line in topic_text.split('\n') if line.strip()]
                logger.info(f"Extracted {len(raw_topics)} topics from text input")
            else:
                # If neither file nor text provided
                self.errors.append("No topic input provided")
                return False
            
            # Process each topic
            for topic in raw_topics:
                # Check if it's a question (ends with ?)
                is_question = topic.endswith('?')
                
                # Create topic document
                doc = Document(
                    page_content=topic,
                    metadata={
                        "type": "question" if is_question else "topic",
                        "source": "topic_list",
                        "processed": False
                    }
                )
                topics.append(doc)
            
            if not topics:
                self.errors.append("No valid topics found")
                return False
                
            # Create vector store for topics
            self.topic_vector_store = FAISS.from_documents(topics, self.embeddings)
            logger.info(f"Created topic vector store with {len(topics)} topics")
            
            # Update student model with topics to learn
            topic_names = [t.page_content for t in topics]
            self.student_model.topics_to_learn = set(topic_names)
            
            # Build knowledge graph
            self._build_knowledge_graph_from_topics(topics)
            
            return True
                
        except Exception as e:
            error_msg = f"Error processing topics: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return False
    
    def extract_and_build_plan_from_textbook(self) -> Tuple[bool, str]:
        """
        Automatically extract topics from textbook content and build a learning plan
        when no specific topics/questions are provided.
        """
        if not self.doc_vector_store:
            return False, "No textbook content available. Please upload textbooks first."
            
        if not self.ollama_available:
            return False, "Ollama LLM is required for automatic topic extraction."
        
        try:
            logger.info("Extracting topics and building plan from textbook")
            
            # Extract documents from the vector store
            all_docs = []
            for metadata, doc in self.doc_vector_store.docstore._dict.items():
                all_docs.append(doc)
            
            logger.info(f"Retrieved {len(all_docs)} documents from vector store")
            
            # Extract topics from textbook content - only process a subset for efficiency
            sample_size = min(30, len(all_docs))
            sample_docs = all_docs[:sample_size]
            
            extracted_topics = self.document_processor.extract_topics_from_chunks(sample_docs)
            
            if not extracted_topics:
                logger.warning("Could not extract topics from textbook content")
                return False, "Could not extract topics from textbook content."
            
            logger.info(f"Extracted {len(extracted_topics)} topics")
            
            # Create topic documents
            topic_docs = []
            topic_names = []
            
            for topic_info in extracted_topics:
                if "topic" not in topic_info or not topic_info["topic"].strip():
                    continue
                    
                topic_name = topic_info["topic"].strip()
                topic_names.append(topic_name)
                
                topic_doc = Document(
                    page_content=topic_name,
                    metadata={
                        "type": "topic",
                        "source": "extracted",
                        "topic_type": topic_info.get("type", "unknown"),
                        "description": topic_info.get("description", ""),
                        "processed": False
                    }
                )
                topic_docs.append(topic_doc)
            
            # Add topics to student model
            self.student_model.topics_to_learn = set(topic_names)
            
            # Validate topics against textbook content if needed
            if self.doc_vector_store:
                validated_topics = self.document_processor.validate_topics_against_content(
                    topic_names, all_docs
                )
                logger.info(f"Validated {len(validated_topics)} topics against content")
                
                # Register validated topics with path manager
                self.path_manager.register_document_topics(validated_topics)
            
            # Build knowledge graph
            self._build_knowledge_graph_from_topics(topic_docs)
            
            # Create vector store for topics
            if topic_docs:
                self.topic_vector_store = FAISS.from_documents(topic_docs, self.embeddings)
                logger.info(f"Created topic vector store with {len(topic_docs)} topics")
            
            return True, f"Successfully extracted {len(topic_docs)} topics from textbook content and built a learning plan."
            
        except Exception as e:
            error_msg = f"Error building plan from textbook: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    def _build_knowledge_graph_from_topics(self, topics: List[Document]) -> bool:
        """Build initial knowledge graph structure from the provided topics."""
        try:
            logger.info(f"Building knowledge graph from {len(topics)} topics")
            
            # First, add all topics as nodes
            for topic in topics:
                content = topic.page_content
                is_question = content.endswith('?')
                
                # Determine topic type from metadata if available
                topic_type = "question" if is_question else topic.metadata.get("topic_type", "topic")
                
                # Add to knowledge graph
                self.knowledge_graph.add_node(
                    content,
                    type=topic_type,
                    source=topic.metadata.get("source", "topic_list"),
                    description=topic.metadata.get("description", ""),
                    mastery_level=0.0
                )
                
                # Register with path manager if from document
                if topic.metadata.get("source") == "extracted":
                    self.path_manager.register_document_topic(content)
            
            # If Ollama is available, identify relationships between topics
            if self.ollama_available and self.llm:
                logger.info("Identifying topic relationships using LLM")
                self._identify_topic_relationships(topics)
            
            # Calculate initial learning path
            learning_path = self.path_manager.calculate_learning_path()
            self.student_model.learning_path = learning_path
            self.student_model.current_position = 0
            
            logger.info(f"Built knowledge graph with {len(self.knowledge_graph.get_all_nodes())} nodes and {len(self.knowledge_graph.get_all_edges())} edges")
            logger.info(f"Generated learning path with {len(learning_path)} topics")
            
            return True
                
        except Exception as e:
            error_msg = f"Error building knowledge graph: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return False
    
    def _identify_topic_relationships(self, topics: List[Document]) -> None:
        """Identify relationships between topics using LLM."""
        if not self.llm:
            logger.warning("No LLM available for relationship identification")
            return
            
        try:
            topic_names = [t.page_content for t in topics]
            
            # Use document processor to extract relationships
            relationships = self.document_processor.extract_relationships(topic_names)
            
            # Add relationships to knowledge graph
            for rel in relationships:
                if "source" in rel and "target" in rel:
                    self.knowledge_graph.add_edge(
                        rel["source"], 
                        rel["target"],
                        relationship=rel.get("type", "related")
                    )
            
            logger.info(f"Added {len(relationships)} relationships to knowledge graph")
            
        except Exception as e:
            error_msg = f"Error identifying topic relationships: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
    
    def generate_quiz(self, topic: str = None, num_questions: int = 5, difficulty: str = 'adaptive') -> List[Dict]:
        """
        Generate a quiz based on a specific topic or the current learning position.
        """
        if not self.ollama_available or not self.llm:
            self.errors.append("Ollama LLM is required for quiz generation")
            return []
            
        try:
            logger.info(f"Generating quiz: topic={topic}, questions={num_questions}, difficulty={difficulty}")
            
            # Determine the focus topic(s)
            focus_topics = []
            
            if topic:
                # Focus on the specified topic
                focus_topics.append(topic)
                
                # Also include related topics
                related = self.knowledge_graph.get_related_concepts(topic)
                focus_topics.extend(related[:2])  # Add up to 2 related topics
                logger.info(f"Quiz focus: {topic} plus {len(related[:2])} related topics")
            else:
                # Use the current position in the learning path
                if self.student_model.learning_path:
                    current_pos = self.student_model.current_position
                    if current_pos < len(self.student_model.learning_path):
                        current_topic = self.student_model.learning_path[current_pos]
                        focus_topics.append(current_topic)
                        logger.info(f"Quiz focus: current topic in learning path - {current_topic}")
                        
                        # Look ahead in the path
                        next_position = min(current_pos + 1, len(self.student_model.learning_path) - 1)
                        if next_position != current_pos:
                            focus_topics.append(self.student_model.learning_path[next_position])
            
            # If still no topics, use topics to learn
            if not focus_topics and self.student_model.topics_to_learn:
                focus_topics = list(self.student_model.topics_to_learn)[:2]
                logger.info(f"Quiz focus: topics to learn - {focus_topics}")
            
            # If still no topics, use weaknesses
            if not focus_topics and self.student_model.weaknesses:
                focus_topics = list(self.student_model.weaknesses)[:2]
                logger.info(f"Quiz focus: weakness areas - {focus_topics}")
            
            # If still no topics, use all available topics
            if not focus_topics:
                focus_topics = self.knowledge_graph.get_all_nodes()[:3]
                logger.info(f"Quiz focus: random topics - {focus_topics}")
            
            # Ensure we have topics
            if not focus_topics:
                logger.error("No topics available for quiz generation")
                self.errors.append("No topics available for quiz generation")
                return []
            
            # Set up document retriever if available
            if self.doc_vector_store:
                logger.info("Setting up document retriever for quiz")
                self.quiz_generator.doc_retriever = self.doc_vector_store.as_retriever(search_kwargs={"k": 5})
            
            # Get mastery levels for adaptive difficulty
            mastery_levels = {
                topic: self.student_model.get_topic_mastery(topic)
                for topic in focus_topics
            }
            
            # Generate quiz
            quiz = self.quiz_generator.generate_quiz(
                topics=focus_topics,
                num_questions=num_questions,
                difficulty=difficulty,
                mastery_levels=mastery_levels
            )
            
            logger.info(f"Generated quiz with {len(quiz)} questions")
            return quiz
                
        except Exception as e:
            error_msg = f"Error generating quiz: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return []
    
    def evaluate_quiz_results(self, quiz_results: Dict) -> Dict:
        """
        Evaluate quiz results and update the student model.
        """
        if not quiz_results:
            logger.warning("No quiz results provided for evaluation")
            self.errors.append("No quiz results provided for evaluation")
            return {}
            
        try:
            logger.info(f"Evaluating quiz results with {len(quiz_results)} questions")
            
            # Use the quiz generator to evaluate results
            analysis = self.quiz_generator.evaluate_quiz_results(quiz_results)
            
            # Update student model with the results
            self.student_model.update_from_quiz_results(analysis)
            
            # Update learning path based on results
            if "topic_mastery" in analysis and "weaknesses" in analysis:
                new_path, new_position = self.path_manager.update_learning_path_after_quiz(
                    self.student_model.learning_path,
                    self.student_model.current_position,
                    analysis["topic_mastery"],
                    analysis["weaknesses"]
                )
                
                self.student_model.learning_path = new_path
                self.student_model.current_position = new_position
                
                logger.info(f"Updated learning path based on quiz results, new position: {new_position}")
            
            return analysis
                
        except Exception as e:
            error_msg = f"Error evaluating quiz results: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return {}
    
    def get_topic_explanation(self, topic: str) -> str:
        """
        Get a concise explanation of a topic.
        """
        if not self.doc_vector_store:
            logger.warning("No document store available for topic explanation")
            return "Please upload textbooks first to generate explanations."
            
        if not self.ollama_available or not self.llm:
            logger.warning("LLM not available for topic explanation")
            return "LLM is required for topic explanations."
        
        try:
            logger.info(f"Generating explanation for topic: {topic}")
            
            # Retrieve relevant context
            retriever = self.doc_vector_store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(topic)
            
            # Extract content
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate explanation
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            explanation_prompt = PromptTemplate(
                template="""
                Provide a clear and concise explanation of the following topic:
                
                Topic: {topic}
                
                Use the following context from textbooks:
                {context}
                
                Your explanation should:
                1. Define the topic clearly
                2. Explain its significance
                3. Provide a simple example
                4. Be understandable to someone learning this for the first time
                
                Aim for a response that's informative but not overwhelming.
                """,
                input_variables=["topic", "context"]
            )
            
            explanation_chain = LLMChain(
                llm=self.llm,
                prompt=explanation_prompt
            )
            
            # Try with retry mechanism
            max_retries = MAX_RETRY_ATTEMPTS
            for attempt in range(max_retries):
                try:
                    result = explanation_chain.invoke({
                        "topic": topic,
                        "context": context
                    })
                    
                    return result["text"]
                    
                except Exception as retry_error:
                    logger.error(f"Explanation attempt {attempt+1} failed: {str(retry_error)}")
                    if attempt == max_retries - 1:
                        self.errors.append(f"Failed to generate explanation after {max_retries} attempts: {str(retry_error)}")
                        return f"Error generating explanation: {str(retry_error)}"
                    time.sleep(1)  # Wait before retrying
            
            return "Unable to generate explanation at this time."
                
        except Exception as e:
            error_msg = f"Error generating explanation: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return f"Error generating explanation: {str(e)}"
    
    def get_topic_examples(self, topic: str) -> str:
        """
        Get examples for a specific topic.
        """
        if not self.doc_vector_store:
            logger.warning("No document store available for topic examples")
            return "Please upload textbooks first to generate examples."
            
        if not self.ollama_available or not self.llm:
            logger.warning("LLM not available for topic examples")
            return "LLM is required for generating examples."
        
        try:
            logger.info(f"Generating examples for topic: {topic}")
            
            # Retrieve relevant context
            retriever = self.doc_vector_store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(f"{topic} examples")
            
            # Extract content
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate examples
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            examples_prompt = PromptTemplate(
                template="""
                Provide clear and instructive examples for the following topic:
                
                Topic: {topic}
                
                Use the following context from textbooks if relevant:
                {context}
                
                Your response should include:
                1. At least 3 examples of increasing complexity
                2. A step-by-step walkthrough of each example
                3. Explanation of why each example illustrates the topic
                4. Variations or extensions of the examples to deepen understanding
                
                Make your examples practical and relevant to real-world applications when possible.
                """,
                input_variables=["topic", "context"]
            )
            
            examples_chain = LLMChain(
                llm=self.llm,
                prompt=examples_prompt
            )
            
            # Try with retry mechanism
            max_retries = MAX_RETRY_ATTEMPTS
            for attempt in range(max_retries):
                try:
                    result = examples_chain.invoke({
                        "topic": topic,
                        "context": context
                    })
                    
                    return result["text"]
                    
                except Exception as retry_error:
                    logger.error(f"Examples generation attempt {attempt+1} failed: {str(retry_error)}")
                    if attempt == max_retries - 1:
                        self.errors.append(f"Failed to generate examples after {max_retries} attempts: {str(retry_error)}")
                        return f"Error generating examples: {str(retry_error)}"
                    time.sleep(1)  # Wait before retrying
            
            return "Unable to generate examples at this time."
                
        except Exception as e:
            error_msg = f"Error generating examples: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return f"Error generating examples: {str(e)}"
    
    def get_system_status(self) -> Dict:
        """Get the current system status and statistics."""
        return {
            "initialized": self.initialized,
            "ollama_available": self.ollama_available,
            "device": self.device,
            "documents_processed": len(self.documents_processed),
            "documents_list": self.documents_processed,
            "topics_extracted": len(self.document_processor.get_all_extracted_topics()),
            "topics_in_graph": len(self.knowledge_graph.get_all_nodes()),
            "relationships_in_graph": len(self.knowledge_graph.get_all_edges()),
            "student_progress": self.student_model.get_overall_progress(),
            "mastery_distribution": self.student_model.get_mastery_distribution(),
            "session_id": self.session_id,
            "errors": self.errors[-10:] if self.errors else []  # Show last 10 errors
=======
import os
import tempfile
import shutil
import time
import uuid
import logging
import requests
from datetime import datetime
from typing import List, Dict, Tuple, Set, Optional, Union

import torch
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tutor_system")

# Constants
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
MAX_RETRY_ATTEMPTS = 3
OLLAMA_API_BASE_URL = "http://localhost:11434"

class PersonalizedTutorSystem:
    """Main class for the personalized tutor system with KG-RAG capabilities."""
    
    def __init__(self, 
                 llm_model_name="llama3.2:latest",
                 embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size=DEFAULT_CHUNK_SIZE,
                 chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                 use_gpu=True):
        """
        Initialize the Personalized Tutor system.
        
        Args:
            llm_model_name: The Ollama model for text generation
            embedding_model_name: The HuggingFace model for embeddings
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            use_gpu: Whether to use GPU acceleration
        """
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        
        # System state
        self.initialized = False
        self.ollama_available = False
        self.errors = []
        self.processing_times = {}
        self.documents_processed = []
        
        # Initialize components in a safe manner
        try:
            logger.info(f"Initializing tutor system with {llm_model_name} and {embedding_model_name}")
            
            # Initialize GPU settings
            self.use_gpu = self._initialize_gpu(use_gpu)
            self.device = "cuda" if self.use_gpu else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Initialize text splitter
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )
            logger.info(f"Text splitter initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")
            
            # Initialize embeddings model
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = self._initialize_embeddings(HuggingFaceEmbeddings)
            logger.info(f"Embeddings model initialized: {embedding_model_name}")
            
            # Initialize LLM
            self.ollama_available = self._check_ollama_connection()
            if self.ollama_available:
                from langchain_ollama import OllamaLLM
                from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
                self.llm = self._initialize_llm(OllamaLLM, StreamingStdOutCallbackHandler)
                logger.info(f"LLM initialized: {llm_model_name}")
            else:
                self.llm = None
                logger.warning("Ollama not available. LLM features will be disabled.")
            
            # Initialize vector stores
            self.doc_vector_store = None
            self.topic_vector_store = None
            
            # Import and initialize core components
            from core.knowledge_graph import KnowledgeGraph
            from core.document_processor import DocumentProcessor
            from core.quiz_generator import QuizGenerator
            from core.learning_path import LearningPathManager
            from core.student_model import StudentModel
            
            # Initialize core components
            self.knowledge_graph = KnowledgeGraph()
            self.document_processor = DocumentProcessor(self.embeddings, self.text_splitter, self.llm)
            self.quiz_generator = QuizGenerator(self.llm)
            self.path_manager = LearningPathManager(self.knowledge_graph)
            self.student_model = StudentModel()
            
            # Generate session ID
            self.session_id = str(uuid.uuid4())
            logger.info(f"Session ID: {self.session_id}")
            
            # Mark as initialized
            self.initialized = True
            logger.info("Tutor system initialization completed successfully")
            
        except Exception as e:
            error_msg = f"Initialization error: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
    
    def _initialize_gpu(self, use_gpu: bool) -> bool:
        """Initialize GPU settings and verify availability."""
        if not use_gpu:
            logger.info("GPU not requested, using CPU")
            return False
            
        try:
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                # Log GPU info
                gpu_info = torch.cuda.get_device_properties(0)
                logger.info(f"Using GPU: {gpu_info.name} ({gpu_info.total_memory / 1024**3:.1f} GB)")
                return True
            else:
                logger.info("GPU requested but not available. Using CPU instead.")
                return False
        except Exception as e:
            logger.warning(f"Error checking GPU availability: {str(e)}. Falling back to CPU.")
            return False
    
    def _initialize_embeddings(self, HuggingFaceEmbeddings):
        """Initialize embeddings model with fallback to CPU if needed."""
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": self.device}
            )
            logger.info(f"Embeddings initialized with model {self.embedding_model_name} on {self.device}")
            return embeddings
        except Exception as e:
            # Try falling back to CPU if there was an error
            if self.device == "cuda":
                logger.warning(f"Error initializing embeddings on GPU: {str(e)}. Falling back to CPU.")
                self.device = "cpu"
                self.use_gpu = False
                return HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                    model_kwargs={"device": "cpu"}
                )
            else:
                # If already on CPU, re-raise
                raise
    
    def _check_ollama_connection(self) -> bool:
        """Check if Ollama is running and available."""
        try:
            response = requests.get(f"{OLLAMA_API_BASE_URL}/api/version", timeout=2)
            if response.status_code == 200:
                logger.info("Ollama connection successful")
                return True
            else:
                logger.warning(f"Ollama returned status code {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"Error connecting to Ollama: {str(e)}")
            return False
    
    def _initialize_llm(self, OllamaLLM, StreamingStdOutCallbackHandler) -> Optional:
        """Initialize LLM with error handling."""
        if not self.ollama_available:
            return None
            
        try:
            callbacks = [StreamingStdOutCallbackHandler()]
            llm = OllamaLLM(model=self.llm_model_name, callbacks=callbacks)
            logger.info(f"LLM initialized with model {self.llm_model_name}")
            return llm
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            self.errors.append(f"LLM initialization error: {str(e)}")
            return None
    
    def process_textbooks(self, pdf_files) -> bool:
        """Process PDF textbooks and create a vector store."""
        if not pdf_files:
            self.errors.append("No PDF files provided")
            return False
            
        all_docs = []
        start_time = time.time()
        
        try:
            logger.info(f"Processing {len(pdf_files)} PDF files")
            
            # Create temporary directory for file storage
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Created temporary directory: {temp_dir}")
            
            # Process each PDF file
            for pdf_file in pdf_files:
                logger.info(f"Processing PDF: {pdf_file.name}")
                chunks, stats = self.document_processor.process_pdf(pdf_file, temp_dir)
                
                if chunks:
                    all_docs.extend(chunks)
                    self.processing_times[pdf_file.name] = stats
                    # Track processed documents
                    if pdf_file.name not in self.documents_processed:
                        self.documents_processed.append(pdf_file.name)
                else:
                    logger.warning(f"No chunks extracted from {pdf_file.name}")
            
            # Create vector store if we have documents
            if all_docs:
                logger.info(f"Creating vector store with {len(all_docs)} chunks")
                index_start_time = time.time()
                
                try:
                    # Create the vector store using FAISS
                    self.doc_vector_store = FAISS.from_documents(all_docs, self.embeddings)
                    
                    # Update the retriever in quiz generator
                    self.quiz_generator.doc_retriever = self.doc_vector_store.as_retriever(search_kwargs={"k": 5})
                    
                    # Record time
                    index_time = time.time() - index_start_time
                    self.processing_times["index_building"] = index_time
                    self.processing_times["total_time"] = time.time() - start_time
                    
                    logger.info(f"Vector store created with {len(all_docs)} chunks")
                    
                    # Try to extract topics from the content
                    if self.ollama_available and self.llm:
                        logger.info("Attempting to extract topics from textbooks")
                        try:
                            extracted_topics = self.document_processor.extract_topics_from_chunks(all_docs[:20])  # Limit to first 20 chunks
                            logger.info(f"Extracted {len(extracted_topics)} topics")
                            
                            # Add topics to knowledge graph
                            if extracted_topics:
                                for topic_info in extracted_topics:
                                    if "topic" in topic_info and topic_info["topic"]:
                                        # Add to knowledge graph
                                        self.knowledge_graph.add_node(
                                            topic_info["topic"],
                                            type=topic_info.get("type", "topic"),
                                            description=topic_info.get("description", ""),
                                            source="extracted"
                                        )
                                        
                                        # Register with path manager
                                        self.path_manager.register_document_topic(topic_info["topic"])
                            
                            # Extract relationships between topics
                            topic_names = [t["topic"] for t in extracted_topics if "topic" in t]
                            if len(topic_names) > 1:
                                relationships = self.document_processor.extract_relationships(topic_names)
                                logger.info(f"Extracted {len(relationships)} relationships")
                                
                                # Add relationships to knowledge graph
                                for rel in relationships:
                                    if "source" in rel and "target" in rel:
                                        self.knowledge_graph.add_edge(
                                            rel["source"], 
                                            rel["target"],
                                            relationship=rel.get("type", "related")
                                        )
                        except Exception as topic_error:
                            logger.error(f"Error extracting topics: {str(topic_error)}", exc_info=True)
                    
                    return True
                    
                except Exception as e:
                    error_msg = f"Error creating vector store: {str(e)}"
                    self.errors.append(error_msg)
                    logger.error(error_msg, exc_info=True)
                    return False
            else:
                self.errors.append("No content extracted from PDFs")
                return False
                
        except Exception as e:
            error_msg = f"Error processing textbooks: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return False
        finally:
            # Clean up temp directory
            if 'temp_dir' in locals():
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as cleanup_error:
                    logger.warning(f"Error cleaning up temporary directory: {str(cleanup_error)}")
    
    def process_topic_list(self, topic_file=None, topic_text=None) -> bool:
        """Process a list of topics/questions from a file or text input."""
        try:
            logger.info("Processing topic list")
            topics = []
            
            # Extract topics either from file or text input
            if topic_file:
                topic_content = topic_file.getvalue().decode("utf-8")
                # Split by new lines and filter empty lines
                raw_topics = [line.strip() for line in topic_content.split('\n') if line.strip()]
                logger.info(f"Extracted {len(raw_topics)} topics from file")
            elif topic_text:
                raw_topics = [line.strip() for line in topic_text.split('\n') if line.strip()]
                logger.info(f"Extracted {len(raw_topics)} topics from text input")
            else:
                # If neither file nor text provided
                self.errors.append("No topic input provided")
                return False
            
            # Process each topic
            for topic in raw_topics:
                # Check if it's a question (ends with ?)
                is_question = topic.endswith('?')
                
                # Create topic document
                doc = Document(
                    page_content=topic,
                    metadata={
                        "type": "question" if is_question else "topic",
                        "source": "topic_list",
                        "processed": False
                    }
                )
                topics.append(doc)
            
            if not topics:
                self.errors.append("No valid topics found")
                return False
                
            # Create vector store for topics
            self.topic_vector_store = FAISS.from_documents(topics, self.embeddings)
            logger.info(f"Created topic vector store with {len(topics)} topics")
            
            # Update student model with topics to learn
            topic_names = [t.page_content for t in topics]
            self.student_model.topics_to_learn = set(topic_names)
            
            # Build knowledge graph
            self._build_knowledge_graph_from_topics(topics)
            
            return True
                
        except Exception as e:
            error_msg = f"Error processing topics: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return False
    
    def extract_and_build_plan_from_textbook(self) -> Tuple[bool, str]:
        """
        Automatically extract topics from textbook content and build a learning plan
        when no specific topics/questions are provided.
        """
        if not self.doc_vector_store:
            return False, "No textbook content available. Please upload textbooks first."
            
        if not self.ollama_available:
            return False, "Ollama LLM is required for automatic topic extraction."
        
        try:
            logger.info("Extracting topics and building plan from textbook")
            
            # Extract documents from the vector store
            all_docs = []
            for metadata, doc in self.doc_vector_store.docstore._dict.items():
                all_docs.append(doc)
            
            logger.info(f"Retrieved {len(all_docs)} documents from vector store")
            
            # Extract topics from textbook content - only process a subset for efficiency
            sample_size = min(30, len(all_docs))
            sample_docs = all_docs[:sample_size]
            
            extracted_topics = self.document_processor.extract_topics_from_chunks(sample_docs)
            
            if not extracted_topics:
                logger.warning("Could not extract topics from textbook content")
                return False, "Could not extract topics from textbook content."
            
            logger.info(f"Extracted {len(extracted_topics)} topics")
            
            # Create topic documents
            topic_docs = []
            topic_names = []
            
            for topic_info in extracted_topics:
                if "topic" not in topic_info or not topic_info["topic"].strip():
                    continue
                    
                topic_name = topic_info["topic"].strip()
                topic_names.append(topic_name)
                
                topic_doc = Document(
                    page_content=topic_name,
                    metadata={
                        "type": "topic",
                        "source": "extracted",
                        "topic_type": topic_info.get("type", "unknown"),
                        "description": topic_info.get("description", ""),
                        "processed": False
                    }
                )
                topic_docs.append(topic_doc)
            
            # Add topics to student model
            self.student_model.topics_to_learn = set(topic_names)
            
            # Validate topics against textbook content if needed
            if self.doc_vector_store:
                validated_topics = self.document_processor.validate_topics_against_content(
                    topic_names, all_docs
                )
                logger.info(f"Validated {len(validated_topics)} topics against content")
                
                # Register validated topics with path manager
                self.path_manager.register_document_topics(validated_topics)
            
            # Build knowledge graph
            self._build_knowledge_graph_from_topics(topic_docs)
            
            # Create vector store for topics
            if topic_docs:
                self.topic_vector_store = FAISS.from_documents(topic_docs, self.embeddings)
                logger.info(f"Created topic vector store with {len(topic_docs)} topics")
            
            return True, f"Successfully extracted {len(topic_docs)} topics from textbook content and built a learning plan."
            
        except Exception as e:
            error_msg = f"Error building plan from textbook: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    def _build_knowledge_graph_from_topics(self, topics: List[Document]) -> bool:
        """Build initial knowledge graph structure from the provided topics."""
        try:
            logger.info(f"Building knowledge graph from {len(topics)} topics")
            
            # First, add all topics as nodes
            for topic in topics:
                content = topic.page_content
                is_question = content.endswith('?')
                
                # Determine topic type from metadata if available
                topic_type = "question" if is_question else topic.metadata.get("topic_type", "topic")
                
                # Add to knowledge graph
                self.knowledge_graph.add_node(
                    content,
                    type=topic_type,
                    source=topic.metadata.get("source", "topic_list"),
                    description=topic.metadata.get("description", ""),
                    mastery_level=0.0
                )
                
                # Register with path manager if from document
                if topic.metadata.get("source") == "extracted":
                    self.path_manager.register_document_topic(content)
            
            # If Ollama is available, identify relationships between topics
            if self.ollama_available and self.llm:
                logger.info("Identifying topic relationships using LLM")
                self._identify_topic_relationships(topics)
            
            # Calculate initial learning path
            learning_path = self.path_manager.calculate_learning_path()
            self.student_model.learning_path = learning_path
            self.student_model.current_position = 0
            
            logger.info(f"Built knowledge graph with {len(self.knowledge_graph.get_all_nodes())} nodes and {len(self.knowledge_graph.get_all_edges())} edges")
            logger.info(f"Generated learning path with {len(learning_path)} topics")
            
            return True
                
        except Exception as e:
            error_msg = f"Error building knowledge graph: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return False
    
    def _identify_topic_relationships(self, topics: List[Document]) -> None:
        """Identify relationships between topics using LLM."""
        if not self.llm:
            logger.warning("No LLM available for relationship identification")
            return
            
        try:
            topic_names = [t.page_content for t in topics]
            
            # Use document processor to extract relationships
            relationships = self.document_processor.extract_relationships(topic_names)
            
            # Add relationships to knowledge graph
            for rel in relationships:
                if "source" in rel and "target" in rel:
                    self.knowledge_graph.add_edge(
                        rel["source"], 
                        rel["target"],
                        relationship=rel.get("type", "related")
                    )
            
            logger.info(f"Added {len(relationships)} relationships to knowledge graph")
            
        except Exception as e:
            error_msg = f"Error identifying topic relationships: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
    
    def generate_quiz(self, topic: str = None, num_questions: int = 5, difficulty: str = 'adaptive') -> List[Dict]:
        """
        Generate a quiz based on a specific topic or the current learning position.
        """
        if not self.ollama_available or not self.llm:
            self.errors.append("Ollama LLM is required for quiz generation")
            return []
            
        try:
            logger.info(f"Generating quiz: topic={topic}, questions={num_questions}, difficulty={difficulty}")
            
            # Determine the focus topic(s)
            focus_topics = []
            
            if topic:
                # Focus on the specified topic
                focus_topics.append(topic)
                
                # Also include related topics
                related = self.knowledge_graph.get_related_concepts(topic)
                focus_topics.extend(related[:2])  # Add up to 2 related topics
                logger.info(f"Quiz focus: {topic} plus {len(related[:2])} related topics")
            else:
                # Use the current position in the learning path
                if self.student_model.learning_path:
                    current_pos = self.student_model.current_position
                    if current_pos < len(self.student_model.learning_path):
                        current_topic = self.student_model.learning_path[current_pos]
                        focus_topics.append(current_topic)
                        logger.info(f"Quiz focus: current topic in learning path - {current_topic}")
                        
                        # Look ahead in the path
                        next_position = min(current_pos + 1, len(self.student_model.learning_path) - 1)
                        if next_position != current_pos:
                            focus_topics.append(self.student_model.learning_path[next_position])
            
            # If still no topics, use topics to learn
            if not focus_topics and self.student_model.topics_to_learn:
                focus_topics = list(self.student_model.topics_to_learn)[:2]
                logger.info(f"Quiz focus: topics to learn - {focus_topics}")
            
            # If still no topics, use weaknesses
            if not focus_topics and self.student_model.weaknesses:
                focus_topics = list(self.student_model.weaknesses)[:2]
                logger.info(f"Quiz focus: weakness areas - {focus_topics}")
            
            # If still no topics, use all available topics
            if not focus_topics:
                focus_topics = self.knowledge_graph.get_all_nodes()[:3]
                logger.info(f"Quiz focus: random topics - {focus_topics}")
            
            # Ensure we have topics
            if not focus_topics:
                logger.error("No topics available for quiz generation")
                self.errors.append("No topics available for quiz generation")
                return []
            
            # Set up document retriever if available
            if self.doc_vector_store:
                logger.info("Setting up document retriever for quiz")
                self.quiz_generator.doc_retriever = self.doc_vector_store.as_retriever(search_kwargs={"k": 5})
            
            # Get mastery levels for adaptive difficulty
            mastery_levels = {
                topic: self.student_model.get_topic_mastery(topic)
                for topic in focus_topics
            }
            
            # Generate quiz
            quiz = self.quiz_generator.generate_quiz(
                topics=focus_topics,
                num_questions=num_questions,
                difficulty=difficulty,
                mastery_levels=mastery_levels
            )
            
            logger.info(f"Generated quiz with {len(quiz)} questions")
            return quiz
                
        except Exception as e:
            error_msg = f"Error generating quiz: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return []
    
    def evaluate_quiz_results(self, quiz_results: Dict) -> Dict:
        """
        Evaluate quiz results and update the student model.
        """
        if not quiz_results:
            logger.warning("No quiz results provided for evaluation")
            self.errors.append("No quiz results provided for evaluation")
            return {}
            
        try:
            logger.info(f"Evaluating quiz results with {len(quiz_results)} questions")
            
            # Use the quiz generator to evaluate results
            analysis = self.quiz_generator.evaluate_quiz_results(quiz_results)
            
            # Update student model with the results
            self.student_model.update_from_quiz_results(analysis)
            
            # Update learning path based on results
            if "topic_mastery" in analysis and "weaknesses" in analysis:
                new_path, new_position = self.path_manager.update_learning_path_after_quiz(
                    self.student_model.learning_path,
                    self.student_model.current_position,
                    analysis["topic_mastery"],
                    analysis["weaknesses"]
                )
                
                self.student_model.learning_path = new_path
                self.student_model.current_position = new_position
                
                logger.info(f"Updated learning path based on quiz results, new position: {new_position}")
            
            return analysis
                
        except Exception as e:
            error_msg = f"Error evaluating quiz results: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return {}
    
    def get_topic_explanation(self, topic: str) -> str:
        """
        Get a concise explanation of a topic.
        """
        if not self.doc_vector_store:
            logger.warning("No document store available for topic explanation")
            return "Please upload textbooks first to generate explanations."
            
        if not self.ollama_available or not self.llm:
            logger.warning("LLM not available for topic explanation")
            return "LLM is required for topic explanations."
        
        try:
            logger.info(f"Generating explanation for topic: {topic}")
            
            # Retrieve relevant context
            retriever = self.doc_vector_store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(topic)
            
            # Extract content
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate explanation
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            explanation_prompt = PromptTemplate(
                template="""
                Provide a clear and concise explanation of the following topic:
                
                Topic: {topic}
                
                Use the following context from textbooks:
                {context}
                
                Your explanation should:
                1. Define the topic clearly
                2. Explain its significance
                3. Provide a simple example
                4. Be understandable to someone learning this for the first time
                
                Aim for a response that's informative but not overwhelming.
                """,
                input_variables=["topic", "context"]
            )
            
            explanation_chain = LLMChain(
                llm=self.llm,
                prompt=explanation_prompt
            )
            
            # Try with retry mechanism
            max_retries = MAX_RETRY_ATTEMPTS
            for attempt in range(max_retries):
                try:
                    result = explanation_chain.invoke({
                        "topic": topic,
                        "context": context
                    })
                    
                    return result["text"]
                    
                except Exception as retry_error:
                    logger.error(f"Explanation attempt {attempt+1} failed: {str(retry_error)}")
                    if attempt == max_retries - 1:
                        self.errors.append(f"Failed to generate explanation after {max_retries} attempts: {str(retry_error)}")
                        return f"Error generating explanation: {str(retry_error)}"
                    time.sleep(1)  # Wait before retrying
            
            return "Unable to generate explanation at this time."
                
        except Exception as e:
            error_msg = f"Error generating explanation: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return f"Error generating explanation: {str(e)}"
    
    def get_topic_examples(self, topic: str) -> str:
        """
        Get examples for a specific topic.
        """
        if not self.doc_vector_store:
            logger.warning("No document store available for topic examples")
            return "Please upload textbooks first to generate examples."
            
        if not self.ollama_available or not self.llm:
            logger.warning("LLM not available for topic examples")
            return "LLM is required for generating examples."
        
        try:
            logger.info(f"Generating examples for topic: {topic}")
            
            # Retrieve relevant context
            retriever = self.doc_vector_store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(f"{topic} examples")
            
            # Extract content
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate examples
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            examples_prompt = PromptTemplate(
                template="""
                Provide clear and instructive examples for the following topic:
                
                Topic: {topic}
                
                Use the following context from textbooks if relevant:
                {context}
                
                Your response should include:
                1. At least 3 examples of increasing complexity
                2. A step-by-step walkthrough of each example
                3. Explanation of why each example illustrates the topic
                4. Variations or extensions of the examples to deepen understanding
                
                Make your examples practical and relevant to real-world applications when possible.
                """,
                input_variables=["topic", "context"]
            )
            
            examples_chain = LLMChain(
                llm=self.llm,
                prompt=examples_prompt
            )
            
            # Try with retry mechanism
            max_retries = MAX_RETRY_ATTEMPTS
            for attempt in range(max_retries):
                try:
                    result = examples_chain.invoke({
                        "topic": topic,
                        "context": context
                    })
                    
                    return result["text"]
                    
                except Exception as retry_error:
                    logger.error(f"Examples generation attempt {attempt+1} failed: {str(retry_error)}")
                    if attempt == max_retries - 1:
                        self.errors.append(f"Failed to generate examples after {max_retries} attempts: {str(retry_error)}")
                        return f"Error generating examples: {str(retry_error)}"
                    time.sleep(1)  # Wait before retrying
            
            return "Unable to generate examples at this time."
                
        except Exception as e:
            error_msg = f"Error generating examples: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return f"Error generating examples: {str(e)}"
    
    def get_system_status(self) -> Dict:
        """Get the current system status and statistics."""
        return {
            "initialized": self.initialized,
            "ollama_available": self.ollama_available,
            "device": self.device,
            "documents_processed": len(self.documents_processed),
            "documents_list": self.documents_processed,
            "topics_extracted": len(self.document_processor.get_all_extracted_topics()),
            "topics_in_graph": len(self.knowledge_graph.get_all_nodes()),
            "relationships_in_graph": len(self.knowledge_graph.get_all_edges()),
            "student_progress": self.student_model.get_overall_progress(),
            "mastery_distribution": self.student_model.get_mastery_distribution(),
            "session_id": self.session_id,
            "errors": self.errors[-10:] if self.errors else []  # Show last 10 errors
>>>>>>> 9e33a41 (Initial commit)
        }