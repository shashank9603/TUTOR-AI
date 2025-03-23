import os
import time
import re
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Set, Optional, Any

import PyPDF2
from langchain.docstore.document import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Configure logging
logger = logging.getLogger("document_processor")

# Constants
MAX_RETRY_ATTEMPTS = 3

class DocumentProcessor:
    """Handles document processing and topic extraction."""
    
    def __init__(self, embedding_model, text_splitter, llm=None):
        """Initialize with required components."""
        self.embedding_model = embedding_model
        self.text_splitter = text_splitter
        self.llm = llm
        self.processing_stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "topics_extracted": 0,
            "processing_time": 0
        }
        self.errors = []
        self.extracted_topics = set()  # Keep track of all extracted topics
        
    def process_pdf(self, pdf_file, temp_dir: str) -> Tuple[List[Document], Dict]:
        """Process a single PDF file."""
        stats = {"chunks": 0, "time": 0, "filename": pdf_file.name}
        docs = []
        
        try:
            start_time = time.time()
            
            # Save uploaded file to temp directory
            pdf_path = os.path.join(temp_dir, pdf_file.name)
            with open(pdf_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            
            logger.info(f"Processing PDF: {pdf_file.name}, saved to {pdf_path}")
            
            # Extract text from PDF
            text = ""
            with open(pdf_path, "rb") as f:
                try:
                    pdf = PyPDF2.PdfReader(f)
                    logger.info(f"PDF has {len(pdf.pages)} pages")
                    
                    for page_num in range(len(pdf.pages)):
                        page = pdf.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                except Exception as pdf_error:
                    error_msg = f"Error reading PDF {pdf_file.name}: {str(pdf_error)}"
                    self.errors.append(error_msg)
                    logger.error(error_msg, exc_info=True)
                    return [], stats
            
            # Check if we actually extracted any content
            if not text or len(text.strip()) < 100:
                error_msg = f"Insufficient text extracted from {pdf_file.name} (only {len(text)} characters)"
                self.errors.append(error_msg)
                logger.error(error_msg)
                return [], stats
                
            logger.info(f"Extracted {len(text)} characters from PDF")
            
            # Create document with metadata
            base_doc = Document(
                page_content=text, 
                metadata={
                    "source": pdf_file.name, 
                    "type": "textbook",
                    "date_processed": datetime.now().isoformat()
                }
            )
            
            # Split document into chunks
            split_docs = self.text_splitter.split_documents([base_doc])
            logger.info(f"Split into {len(split_docs)} chunks")
            
            # Update stats
            stats["chunks"] = len(split_docs)
            stats["time"] = time.time() - start_time
            
            self.processing_stats["chunks_created"] += len(split_docs)
            self.processing_stats["documents_processed"] += 1
            
            return split_docs, stats
            
        except Exception as e:
            error_msg = f"Error processing {pdf_file.name}: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return [], stats
    
    def extract_topics_from_chunks(self, chunks: List[Document]) -> List[Dict]:
        """Extract potential topics from document chunks using LLM."""
        if not self.llm:
            error_msg = "No LLM available for topic extraction"
            self.errors.append(error_msg)
            logger.error(error_msg)
            return []
            
        start_time = time.time()
        all_extracted_topics = []
        
        try:
            logger.info(f"Extracting topics from {len(chunks)} chunks")
            
            # Process chunks in batches to avoid overloading the LLM
            batch_size = 5
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                combined_text = "\n\n".join([chunk.page_content for chunk in batch])
                
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                
                # Create topic extraction prompt
                extraction_prompt = PromptTemplate(
                    template="""
                    Extract key educational topics, concepts, and terms from the following text. 
                    Focus on extracting clear, well-defined topics that would appear in a textbook's index or table of contents.
                    
                    For each topic, identify if it's:
                    - A fundamental concept (basic building block of the subject)
                    - A derived concept (builds on fundamental concepts)
                    - A technique/method (a process or approach)
                    
                    Text to analyze:
                    {text}
                    
                    Return ONLY topics that are clearly defined in the text. DO NOT invent topics that aren't explicitly covered.
                    
                    Return the results in the following format:
                    Topic: [topic name]
                    Type: [fundamental/derived/technique]
                    Description: [brief description]
                    
                    Extract 5-10 topics if possible. Focus on quality over quantity - only extract clear, well-defined topics.
                    """,
                    input_variables=["text"]
                )
                
                extraction_chain = LLMChain(
                    llm=self.llm,
                    prompt=extraction_prompt
                )
                
                # Extract topics with retry mechanism
                max_retries = MAX_RETRY_ATTEMPTS
                for attempt in range(max_retries):
                    try:
                        result = extraction_chain.invoke({"text": combined_text})
                        extracted_text = result["text"]
                        
                        # Parse the results
                        topics = self._parse_extracted_topics(extracted_text)
                        
                        # Check if we got valid topics
                        if topics:
                            logger.info(f"Extracted {len(topics)} topics from batch {i//batch_size + 1}")
                            all_extracted_topics.extend(topics)
                            break
                        else:
                            logger.warning(f"No valid topics extracted, attempt {attempt+1}/{max_retries}")
                            if attempt == max_retries - 1:
                                logger.error("Failed to extract topics after all attempts")
                    except Exception as retry_error:
                        error_msg = f"Error in extraction attempt {attempt+1}: {str(retry_error)}"
                        logger.error(error_msg)
                        
                        if attempt == max_retries - 1:
                            self.errors.append(f"Failed to extract topics after {max_retries} attempts: {str(retry_error)}")
                        time.sleep(1)  # Wait before retrying
            
            # Remove duplicates while preserving order
            cleaned_topics = []
            seen_topics = set()
            
            for topic in all_extracted_topics:
                topic_name = topic.get("topic", "").strip().lower()
                if topic_name and topic_name not in seen_topics:
                    seen_topics.add(topic_name)
                    cleaned_topics.append(topic)
                    # Add to our global extracted topics set
                    self.extracted_topics.add(topic.get("topic", ""))
            
            logger.info(f"Total topics extracted: {len(cleaned_topics)} (after deduplication)")
            
            self.processing_stats["topics_extracted"] += len(cleaned_topics)
            self.processing_stats["processing_time"] += time.time() - start_time
            
            return cleaned_topics
                
        except Exception as e:
            error_msg = f"Error in topic extraction: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return []
    
    def _parse_extracted_topics(self, text: str) -> List[Dict]:
        """Parse the LLM output into structured topic data."""
        topics = []
        current_topic = {}
        
        try:
            # Split by double newlines or numbered entries to handle different formats
            lines = text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check for new topic entry
                if line.startswith("Topic:") or re.match(r'^\d+\.?\s+Topic:', line):
                    # Clean up numbered format
                    if re.match(r'^\d+\.?\s+Topic:', line):
                        line = re.sub(r'^\d+\.?\s+Topic:', 'Topic:', line)
                    
                    # Save previous topic if it exists
                    if current_topic and "topic" in current_topic:
                        topics.append(current_topic.copy())
                    
                    # Start new topic
                    current_topic = {"topic": line[6:].strip()}
                
                # Type
                elif line.startswith("Type:") or re.match(r'^\d+\.?\s+Type:', line):
                    if re.match(r'^\d+\.?\s+Type:', line):
                        line = re.sub(r'^\d+\.?\s+Type:', 'Type:', line)
                    
                    if current_topic:
                        type_value = line[5:].strip().lower()
                        
                        # Normalize type values
                        if "fundamental" in type_value:
                            current_topic["type"] = "fundamental"
                        elif "derived" in type_value:
                            current_topic["type"] = "derived"
                        elif "technique" in type_value or "method" in type_value:
                            current_topic["type"] = "technique"
                        else:
                            current_topic["type"] = "topic"  # Default type
                
                # Description
                elif line.startswith("Description:") or re.match(r'^\d+\.?\s+Description:', line):
                    if re.match(r'^\d+\.?\s+Description:', line):
                        line = re.sub(r'^\d+\.?\s+Description:', 'Description:', line)
                    
                    if current_topic:
                        current_topic["description"] = line[12:].strip()
            
            # Add the last topic
            if current_topic and "topic" in current_topic:
                topics.append(current_topic.copy())
                
            # Make sure all topics have required fields
            validated_topics = []
            for topic in topics:
                if not topic.get("topic"):
                    continue
                    
                # Ensure type exists
                if "type" not in topic:
                    topic["type"] = "topic"
                    
                # Ensure description exists
                if "description" not in topic or not topic["description"]:
                    topic["description"] = f"A concept related to {topic['topic']}"
                
                validated_topics.append(topic)
                
            return validated_topics
            
        except Exception as e:
            error_msg = f"Error parsing extracted topics: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.errors.append(error_msg)
            return []
    
    def validate_topics_against_content(self, topics: List[str], documents: List[Document]) -> List[str]:
        """Check which topics are actually present in the documents."""
        # Join all document content for searching
        all_content = " ".join([doc.page_content.lower() for doc in documents])
        
        # Check each topic
        validated_topics = []
        for topic in topics:
            # Convert to lowercase for case-insensitive search
            topic_lower = topic.lower()
            
            # Check if topic appears in the content
            if topic_lower in all_content:
                validated_topics.append(topic)
            else:
                # Try to find by removing plurals, etc.
                variations = [
                    topic_lower + "s",  # Plural
                    topic_lower[:-1] if topic_lower.endswith("s") else topic_lower,  # Singular
                    topic_lower + "ing",  # Gerund form
                    topic_lower + "ed"   # Past tense
                ]
                
                if any(var in all_content for var in variations):
                    validated_topics.append(topic)
        
        return validated_topics
    
    def get_all_extracted_topics(self) -> List[str]:
        """Get a list of all topics that have been extracted."""
        return list(self.extracted_topics)
    
    def extract_relationships(self, topics: List[str]) -> List[Dict]:
        """Extract relationships between topics."""
        if not self.llm or not topics:
            return []
            
        relationships = []
        
        try:
            # Create batches of topics for processing
            batch_size = 10
            batches = [topics[i:i+batch_size] for i in range(0, len(topics), batch_size)]
            
            for batch in batches:
                # Create relationship extraction prompt
                relationship_prompt = PromptTemplate(
                    template="""
                    Analyze the following educational topics and identify the relationships between them.
                    
                    Topics:
                    {topics}
                    
                    For each pair that has a clear relationship, indicate:
                    1. If one topic is a prerequisite for understanding the other (use format: Topic A -> Topic B)
                    2. The direction of dependency (which topic should be learned first)
                    
                    Only include pairs with clear prerequisite relationships. Don't invent relationships if they aren't obvious.
                    
                    Return your analysis in the following format:
                    Topic A -> Topic B (meaning Topic A is a prerequisite for Topic B)
                    
                    If two topics are closely related but neither is clearly a prerequisite, you can note:
                    Topic A <-> Topic B (meaning they're related but neither is a prerequisite)
                    """,
                    input_variables=["topics"]
                )
                
                relationship_chain = LLMChain(
                    llm=self.llm,
                    prompt=relationship_prompt
                )
                
                # Extract relationships with retry
                topics_text = "\n".join(batch)
                
                for attempt in range(MAX_RETRY_ATTEMPTS):
                    try:
                        result = relationship_chain.invoke({"topics": topics_text})
                        relationship_text = result["text"]
                        
                        # Parse the relationships
                        batch_relationships = self._parse_relationships(relationship_text)
                        relationships.extend(batch_relationships)
                        break
                    except Exception as e:
                        logger.error(f"Error extracting relationships (attempt {attempt+1}): {str(e)}")
                        if attempt == MAX_RETRY_ATTEMPTS - 1:
                            self.errors.append(f"Failed to extract relationships: {str(e)}")
                        time.sleep(1)
            
            return relationships
            
        except Exception as e:
            error_msg = f"Error extracting relationships: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return []
    
    def _parse_relationships(self, text: str) -> List[Dict]:
        """Parse relationship text into structured data."""
        relationships = []
        
        try:
            lines = text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for directional relationship (prerequisite)
                if "->" in line:
                    parts = line.split("->")
                    if len(parts) == 2:
                        source = parts[0].strip()
                        target = parts[1].strip()
                        
                        # Clean up any parenthetical comments
                        source = re.sub(r'\s*\(.*?\)\s*$', '', source)
                        target = re.sub(r'\s*\(.*?\)\s*$', '', target)
                        
                        relationships.append({
                            "source": source,
                            "target": target,
                            "type": "prerequisite"
                        })
                
                # Check for bidirectional relationship (related)
                elif "<->" in line:
                    parts = line.split("<->")
                    if len(parts) == 2:
                        topic1 = parts[0].strip()
                        topic2 = parts[1].strip()
                        
                        # Clean up any parenthetical comments
                        topic1 = re.sub(r'\s*\(.*?\)\s*$', '', topic1)
                        topic2 = re.sub(r'\s*\(.*?\)\s*$', '', topic2)
                        
                        relationships.append({
                            "source": topic1,
                            "target": topic2,
                            "type": "related"
                        })
                        
                        relationships.append({
                            "source": topic2,
                            "target": topic1,
                            "type": "related"
                        })
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error parsing relationships: {str(e)}", exc_info=True)
            return []