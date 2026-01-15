# agents/session_aware_rag_agent.py
"""
RAG Agent with Excel/CSV Analytics Support
Can answer analytical questions about structured data in documents
"""
'''
from typing import List, Dict, Any, Optional
from pathlib import Path
import shutil

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from parsers.document_parser import DocumentParser
from tools.language_detector import LanguageDetector
from config import settings
from utils.logger import logger, log_section


class SessionAwareRAGAgent:
    """
    RAG Agent with Analytics Support for Excel/CSV
    Can answer percentage, distribution, and aggregation questions
    """
    
    def __init__(self, base_vector_store_path: str = "./data/vector_stores"):
        self.base_vector_store_path = Path(base_vector_store_path)
        self.base_vector_store_path.mkdir(parents=True, exist_ok=True)
        
        self.parser = DocumentParser()
        self.language_detector = LanguageDetector()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        
        logger.info("📄 Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.LLM_MODEL,
            temperature=0.1
        )
        
        self.session_stores: Dict[str, Dict[str, Any]] = {}
        
        logger.info("🤖 RAG Agent with Analytics initialized")
    
    def _get_session_path(self, conversation_id: str) -> Path:
        return self.base_vector_store_path / conversation_id
    
    def _select_prompt_template(self, question: str, has_structured_data: bool = False) -> str:
        """
        Select prompt based on question type
        Special handling for analytical queries on Excel/CSV
        """
        question_lower = question.lower()
        
        # ANALYTICS PROMPT for Excel/CSV data
        if has_structured_data and any(kw in question_lower for kw in [
            'percentage', 'percent', 'distribution', 'breakdown', 'share',
            'how many', 'count', 'total', 'average', 'ratio', 'proportion'
        ]):
            return """You are a data analyst extracting insights from structured data (Excel/CSV files).

DOCUMENT CONTEXT (Contains data tables):
{context}

QUESTION: {question}

INSTRUCTIONS FOR ANALYTICAL QUERIES:
1. Look for ALL relevant data in the context (rows, columns, values)
2. COUNT occurrences if asked for distribution or percentages
3. CALCULATE percentages: (count of category / total count) × 100
4. Show your calculation clearly
5. Present results in a clean, organized format

OUTPUT FORMAT FOR PERCENTAGES/DISTRIBUTIONS:
**[Category Name] Distribution:**
• Category A: X items (XX%)
• Category B: Y items (YY%)
• Category C: Z items (ZZ%)
**Total:** N items (100%)

**Calculation:**
- Total items: [count from data]
- Category A: [count] ÷ [total] × 100 = XX%
- Category B: [count] ÷ [total] × 100 = YY%

IMPORTANT:
- Extract ALL rows/data from the context
- Count accurately
- Show percentages with calculations
- If data is incomplete in context, mention it

Your analytical answer:"""
        
        # Performance review prompt
        elif any(kw in question_lower for kw in ['performance', 'review', 'rating', 'evaluation']):
            return """You are an HR analyst summarizing performance reviews.

CONTEXT: {context}
QUESTION: {question}

Format clearly with sections, bullet points, and key metrics in **bold**.

Your answer:"""
        
        # Default business prompt
        else:
            return """You are a business analyst extracting insights from documents.

CONTEXT: {context}
QUESTION: {question}

PROVIDE A CLEAR ANSWER:
• Extract relevant information
• Use bullet points for lists
• Put key facts in **bold**
• Be specific with numbers and names
• Organize logically

Your structured answer:"""
    
    def _create_qa_chain(self, vectorstore, question: str = None, has_structured_data: bool = False):
        """Create QA chain with appropriate prompt"""
        
        if question:
            template = self._select_prompt_template(question, has_structured_data)
        else:
            template = """You are an analyst providing clear insights.

CONTEXT: {context}
QUESTION: {question}

Provide a well-structured answer with bullet points and clear formatting.

Your answer:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 8}  # More context for analytics
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain
    
    def _has_structured_data(self, conversation_id: str) -> bool:
        """Check if session has Excel/CSV files"""
        if conversation_id in self.session_stores:
            loaded_files = self.session_stores[conversation_id].get("loaded_files", [])
            return any(f.lower().endswith(('.xlsx', '.xls', '.csv')) for f in loaded_files)
        return False
    
    def get_or_create_session(self, conversation_id: str) -> Dict[str, Any]:
        """Get or create session"""
        if conversation_id not in self.session_stores:
            session_path = self._get_session_path(conversation_id)
            
            if session_path.exists() and (session_path / "index.faiss").exists():
                try:
                    logger.info(f"📂 Loading session: {conversation_id}")
                    vectorstore = FAISS.load_local(
                        str(session_path),
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    
                    self.session_stores[conversation_id] = {
                        "vectorstore": vectorstore,
                        "qa_chain": None,
                        "loaded_files": [],
                        "total_chunks": vectorstore.index.ntotal
                    }
                    
                    logger.info(f"✅ Loaded: {vectorstore.index.ntotal} vectors")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Load error: {e}")
                    self.session_stores[conversation_id] = {
                        "vectorstore": None,
                        "qa_chain": None,
                        "loaded_files": [],
                        "total_chunks": 0
                    }
            else:
                logger.info(f"🆕 New session: {conversation_id}")
                self.session_stores[conversation_id] = {
                    "vectorstore": None,
                    "qa_chain": None,
                    "loaded_files": [],
                    "total_chunks": 0
                }
        
        return self.session_stores[conversation_id]
    
    def load_documents(self, conversation_id: str, file_paths: List[str]) -> Dict[str, Any]:
        """Load documents"""
        log_section(f"LOADING DOCUMENTS: {conversation_id}")
        
        session = self.get_or_create_session(conversation_id)
        
        all_documents = []
        loaded_files_info = []
        
        for file_path in file_paths:
            if not Path(file_path).exists():
                logger.error(f"❌ Not found: {file_path}")
                continue
            
            ext = Path(file_path).suffix.lower()
            filename = Path(file_path).name
            
            try:
                logger.info(f"📄 Processing: {filename}")
                
                if ext == '.xlsx':
                    docs = self.parser.parse_excel(file_path)
                elif ext == '.csv':
                    docs = self.parser.parse_csv(file_path)
                elif ext == '.pdf':
                    docs = self.parser.parse_pdf(file_path)
                elif ext == '.docx':
                    docs = self.parser.parse_docx(file_path)
                elif ext == '.txt':
                    docs = self.parser.parse_txt(file_path)
                else:
                    logger.warning(f"⚠️ Unsupported: {ext}")
                    continue
                
                all_documents.extend(docs)
                loaded_files_info.append({
                    "filename": filename,
                    "format": ext,
                    "chunks": len(docs)
                })
                session["loaded_files"].append(filename)
                
                logger.info(f"✅ Parsed {len(docs)} chunks from {filename}")
                
            except Exception as e:
                logger.error(f"❌ Parse error {filename}: {e}")
        
        if not all_documents:
            return {
                "success": False,
                "message": "No documents loaded",
                "conversation_id": conversation_id,
                "files_loaded": []
            }
        
        logger.info(f"✂️ Splitting chunks...")
        split_docs = self.text_splitter.split_documents(all_documents)
        logger.info(f"✅ Created {len(split_docs)} chunks")
        
        if session["vectorstore"] is None:
            logger.info(f"🔢 Creating vector store...")
            session["vectorstore"] = FAISS.from_documents(split_docs, self.embeddings)
        else:
            logger.info(f"🔢 Adding to vector store...")
            session["vectorstore"].add_documents(split_docs)
        
        total_vectors = session["vectorstore"].index.ntotal
        session["total_chunks"] = total_vectors
        
        session_path = self._get_session_path(conversation_id)
        session_path.mkdir(parents=True, exist_ok=True)
        session["vectorstore"].save_local(str(session_path))
        logger.info(f"💾 Saved: {total_vectors} vectors")
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "files_loaded": loaded_files_info,
            "total_chunks": len(split_docs),
            "total_vectors": total_vectors,
            "message": "Documents indexed"
        }
    
    def query(self, conversation_id: str, question: str, language: str = None) -> Dict[str, Any]:
        """Query with analytics support"""
        session = self.get_or_create_session(conversation_id)
        
        if not session["vectorstore"]:
            return {
                "success": False,
                "error": "No documents loaded",
                "answer": "❌ **No documents in this chat.**\n\nUpload documents first.",
                "conversation_id": conversation_id
            }
        
        log_section(f"QUERYING: {conversation_id}")
        logger.info(f"❓ Question: {question}")
        
        try:
            if not language:
                language = self.language_detector.detect_language(question)
            
            # Check if has structured data
            has_structured = self._has_structured_data(conversation_id)
            logger.info(f"📊 Has structured data: {has_structured}")
            
            # Create QA chain with appropriate prompt
            qa_chain = self._create_qa_chain(
                session["vectorstore"], 
                question,
                has_structured
            )
            
            # Execute query
            result = qa_chain.invoke({"query": question})
            
            # Extract sources
            sources = []
            for doc in result.get("source_documents", []):
                source_info = {
                    "source": doc.metadata.get("source", "unknown"),
                    "type": doc.metadata.get("type", "unknown"),
                    "preview": doc.page_content[:150] + "..."
                }
                
                if "page" in doc.metadata:
                    source_info["page"] = doc.metadata["page"]
                if "sheet" in doc.metadata:
                    source_info["sheet"] = doc.metadata["sheet"]
                if "row" in doc.metadata:
                    source_info["row"] = doc.metadata["row"]
                
                sources.append(source_info)
            
            answer = result.get("result", "No answer").strip()
            
            # Add source footer
            if sources and "**Source" not in answer:
                source_files = list(set([s["source"] for s in sources]))
                if len(source_files) == 1:
                    answer += f"\n\n---\n📄 **Source:** {source_files[0]}"
                else:
                    answer += f"\n\n---\n📄 **Sources:** {', '.join(source_files)}"
            
            logger.info(f"✅ Generated answer ({len(sources)} sources)")
            
            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "language": language,
                "source_count": len(sources),
                "conversation_id": conversation_id
            }
        
        except Exception as e:
            logger.error(f"❌ Query error: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": f"❌ **Error:** {str(e)}",
                "conversation_id": conversation_id
            }
    
    def get_session_status(self, conversation_id: str) -> Dict[str, Any]:
        """Get session status"""
        session = self.get_or_create_session(conversation_id)
        
        return {
            "conversation_id": conversation_id,
            "status": "active" if session["vectorstore"] else "idle",
            "loaded_files": session["loaded_files"],
            "total_documents": len(session["loaded_files"]),
            "total_vectors": session["total_chunks"],
            "vectorstore_ready": session["vectorstore"] is not None
        }
    
    def delete_session(self, conversation_id: str):
        """Delete session"""
        try:
            if conversation_id in self.session_stores:
                del self.session_stores[conversation_id]
            
            session_path = self._get_session_path(conversation_id)
            if session_path.exists():
                shutil.rmtree(session_path)
                logger.info(f"🗑️ Deleted: {conversation_id}")
        except Exception as e:
            logger.error(f"❌ Delete error: {e}")
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        return [self.get_session_status(cid) for cid in self.session_stores.keys()]
    
    def get_global_stats(self) -> Dict[str, Any]:
        total_sessions = len(self.session_stores)
        total_vectors = sum(s["total_chunks"] for s in self.session_stores.values())
        active_sessions = sum(1 for s in self.session_stores.values() if s["vectorstore"])
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "total_vectors": total_vectors,
            "base_path": str(self.base_vector_store_path)
        }'''


# agents/session_rag.py
"""
Session-Aware RAG Agent with Hybrid CSV Support
Handles both unstructured documents AND structured CSV/Excel files
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import shutil

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from parsers.document_parser import DocumentParser
from parsers.hybrid_csv_processor import HybridCSVProcessor
from database.db_manager import DatabaseManager
from tools.language_detector import LanguageDetector
from config import settings, ENABLE_HYBRID_CSV
from utils.logger import logger, log_section


class SessionAwareRAGAgent:
    """
    RAG Agent with Hybrid CSV/Excel Support
    - Regular docs → RAG only
    - CSV/Excel → SQL + RAG (hybrid)
    """
    
    def __init__(self, base_vector_store_path: str = "./data/vector_stores",
                 db_manager: DatabaseManager = None):
        self.base_vector_store_path = Path(base_vector_store_path)
        self.base_vector_store_path.mkdir(parents=True, exist_ok=True)
        
        self.parser = DocumentParser()
        self.language_detector = LanguageDetector()
        
        # Initialize hybrid CSV processor if enabled
        if ENABLE_HYBRID_CSV and db_manager:
            self.csv_processor = HybridCSVProcessor(db_manager)
            logger.info("✅ Hybrid CSV processing enabled")
        else:
            self.csv_processor = None
            logger.info("ℹ️  Hybrid CSV processing disabled")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        
        logger.info("📄 Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.LLM_MODEL,
            temperature=0.1
        )
        
        self.session_stores: Dict[str, Dict[str, Any]] = {}
        
        logger.info("🤖 Session-Aware RAG Agent initialized")
    
    def _get_session_path(self, conversation_id: str) -> Path:
        return self.base_vector_store_path / conversation_id
    
    def _is_structured_file(self, filename: str) -> bool:
        """Check if file is CSV/Excel"""
        return filename.lower().endswith(('.csv', '.xlsx', '.xls'))
    
    def get_or_create_session(self, conversation_id: str) -> Dict[str, Any]:
        """Get or create session with CSV metadata tracking"""
        if conversation_id not in self.session_stores:
            session_path = self._get_session_path(conversation_id)
            
            if session_path.exists() and (session_path / "index.faiss").exists():
                try:
                    logger.info(f"📂 Loading session: {conversation_id}")
                    vectorstore = FAISS.load_local(
                        str(session_path),
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    
                    self.session_stores[conversation_id] = {
                        "vectorstore": vectorstore,
                        "qa_chain": None,
                        "loaded_files": [],
                        "csv_tables": [],  # NEW: Track CSV tables
                        "total_chunks": vectorstore.index.ntotal
                    }
                    
                    logger.info(f"✅ Loaded: {vectorstore.index.ntotal} vectors")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Load error: {e}")
                    self.session_stores[conversation_id] = self._create_empty_session()
            else:
                logger.info(f"🆕 New session: {conversation_id}")
                self.session_stores[conversation_id] = self._create_empty_session()
        
        return self.session_stores[conversation_id]
    
    def _create_empty_session(self) -> Dict[str, Any]:
        """Create empty session structure"""
        return {
            "vectorstore": None,
            "qa_chain": None,
            "loaded_files": [],
            "csv_tables": [],
            "total_chunks": 0
        }
    
    def load_documents(self, conversation_id: str, file_paths: List[str]) -> Dict[str, Any]:
        """
        Load documents with hybrid CSV support
        CSV/Excel → SQL + RAG
        Other docs → RAG only
        """
        log_section(f"LOADING DOCUMENTS: {conversation_id}")
        
        session = self.get_or_create_session(conversation_id)
        
        all_documents = []
        loaded_files_info = []
        csv_tables = []
        
        for file_path in file_paths:
            if not Path(file_path).exists():
                logger.error(f"❌ Not found: {file_path}")
                continue
            
            filename = Path(file_path).name
            ext = Path(file_path).suffix.lower()
            
            try:
                # Check if structured data (CSV/Excel)
                if self._is_structured_file(filename) and self.csv_processor:
                    logger.info(f"📊 Processing as HYBRID: {filename}")
                    
                    # Process with hybrid approach
                    result = self.csv_processor.process_file(file_path, conversation_id)
                    
                    # Add RAG documents
                    all_documents.extend(result['rag_documents'])
                    
                    # Track CSV table
                    csv_tables.append({
                        "filename": filename,
                        "table_name": result['sql_table'],
                        "metadata": result['metadata']
                    })
                    
                    loaded_files_info.append({
                        "filename": filename,
                        "format": ext,
                        "type": "hybrid",
                        "sql_table": result['sql_table'],
                        "rag_chunks": len(result['rag_documents']),
                        "capabilities": result['capabilities']
                    })
                    
                    session["loaded_files"].append(filename)
                    session["csv_tables"].append(result['sql_table'])
                    
                    logger.info(f"""
✅ HYBRID Load Complete: {filename}
   📊 SQL: {result['sql_table']}
   📄 RAG: {len(result['rag_documents'])} chunks
                    """)
                
                else:
                    # Regular document processing (RAG only)
                    logger.info(f"📄 Processing as DOCUMENT: {filename}")
                    
                    if ext == '.pdf':
                        docs = self.parser.parse_pdf(file_path)
                    elif ext == '.docx':
                        docs = self.parser.parse_docx(file_path)
                    elif ext == '.txt':
                        docs = self.parser.parse_txt(file_path)
                    elif ext in ['.csv', '.xlsx', '.xls']:
                        # Fallback if hybrid disabled
                        if ext == '.csv':
                            docs = self.parser.parse_csv(file_path)
                        else:
                            docs = self.parser.parse_excel(file_path)
                    else:
                        logger.warning(f"⚠️ Unsupported: {ext}")
                        continue
                    
                    all_documents.extend(docs)
                    loaded_files_info.append({
                        "filename": filename,
                        "format": ext,
                        "type": "document",
                        "chunks": len(docs)
                    })
                    session["loaded_files"].append(filename)
                    
                    logger.info(f"✅ Parsed {len(docs)} chunks from {filename}")
                
            except Exception as e:
                logger.error(f"❌ Parse error {filename}: {e}")
        
        if not all_documents:
            return {
                "success": False,
                "message": "No documents loaded",
                "conversation_id": conversation_id,
                "files_loaded": []
            }
        
        # Process RAG documents
        logger.info(f"✂️ Splitting chunks...")
        split_docs = self.text_splitter.split_documents(all_documents)
        logger.info(f"✅ Created {len(split_docs)} chunks")
        
        if session["vectorstore"] is None:
            logger.info(f"🔢 Creating vector store...")
            session["vectorstore"] = FAISS.from_documents(split_docs, self.embeddings)
        else:
            logger.info(f"🔢 Adding to vector store...")
            session["vectorstore"].add_documents(split_docs)
        
        total_vectors = session["vectorstore"].index.ntotal
        session["total_chunks"] = total_vectors
        
        # Save
        session_path = self._get_session_path(conversation_id)
        session_path.mkdir(parents=True, exist_ok=True)
        session["vectorstore"].save_local(str(session_path))
        logger.info(f"💾 Saved: {total_vectors} vectors")
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "files_loaded": loaded_files_info,
            "csv_tables": csv_tables,
            "total_chunks": len(split_docs),
            "total_vectors": total_vectors,
            "message": f"Loaded {len(loaded_files_info)} files ({len(csv_tables)} hybrid CSV/Excel)"
        }
    
    def has_csv_data(self, conversation_id: str) -> bool:
        """Check if session has CSV/Excel data"""
        session = self.get_or_create_session(conversation_id)
        return len(session.get("csv_tables", [])) > 0
    
    def get_csv_tables(self, conversation_id: str) -> List[str]:
        """Get list of CSV tables for session"""
        session = self.get_or_create_session(conversation_id)
        return session.get("csv_tables", [])
    
    def query(self, conversation_id: str, question: str, language: str = None) -> Dict[str, Any]:
        """Query with analytics support (same as before)"""
        session = self.get_or_create_session(conversation_id)
        
        if not session["vectorstore"]:
            return {
                "success": False,
                "error": "No documents loaded",
                "answer": "❌ **No documents in this chat.**\n\nUpload documents first.",
                "conversation_id": conversation_id
            }
        
        log_section(f"QUERYING: {conversation_id}")
        logger.info(f"❓ Question: {question}")
        
        try:
            if not language:
                language = self.language_detector.detect_language(question)
            
            # Check if has structured data
            has_structured = self.has_csv_data(conversation_id)
            logger.info(f"📊 Has CSV data: {has_structured}")
            
            # Create QA chain
            qa_chain = self._create_qa_chain(
                session["vectorstore"], 
                question,
                has_structured
            )
            
            # Execute query
            result = qa_chain.invoke({"query": question})
            
            # Extract sources
            sources = []
            for doc in result.get("source_documents", []):
                source_info = {
                    "source": doc.metadata.get("source", "unknown"),
                    "type": doc.metadata.get("type", "unknown"),
                    "preview": doc.page_content[:150] + "..."
                }
                
                if "page" in doc.metadata:
                    source_info["page"] = doc.metadata["page"]
                if "sheet" in doc.metadata:
                    source_info["sheet"] = doc.metadata["sheet"]
                if "row" in doc.metadata:
                    source_info["row"] = doc.metadata["row"]
                
                sources.append(source_info)
            
            answer = result.get("result", "No answer").strip()
            
            # Add source footer
            if sources and "**Source" not in answer:
                source_files = list(set([s["source"] for s in sources]))
                if len(source_files) == 1:
                    answer += f"\n\n---\n📄 **Source:** {source_files[0]}"
                else:
                    answer += f"\n\n---\n📄 **Sources:** {', '.join(source_files)}"
            
            logger.info(f"✅ Generated answer ({len(sources)} sources)")
            
            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "language": language,
                "source_count": len(sources),
                "conversation_id": conversation_id
            }
        
        except Exception as e:
            logger.error(f"❌ Query error: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": f"❌ **Error:** {str(e)}",
                "conversation_id": conversation_id
            }
    
    def _create_qa_chain(self, vectorstore, question: str = None, has_structured_data: bool = False):
        """Create QA chain (same as before)"""
        
        if question:
            template = self._select_prompt_template(question, has_structured_data)
        else:
            template = """You are an analyst providing clear insights.

CONTEXT: {context}
QUESTION: {question}

Provide a well-structured answer with bullet points and clear formatting.

Your answer:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 8}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain
    
    def _select_prompt_template(self, question: str, has_structured_data: bool = False) -> str:
        """Select appropriate prompt (same as before)"""
        question_lower = question.lower()
        
        if has_structured_data and any(kw in question_lower for kw in [
            'percentage', 'percent', 'distribution', 'breakdown', 'share',
            'how many', 'count', 'total', 'average', 'ratio', 'proportion'
        ]):
            return """You are a data analyst extracting insights from structured data (Excel/CSV files).

DOCUMENT CONTEXT (Contains data tables):
{context}

QUESTION: {question}

INSTRUCTIONS FOR ANALYTICAL QUERIES:
1. Look for ALL relevant data in the context (rows, columns, values)
2. COUNT occurrences if asked for distribution or percentages
3. CALCULATE percentages: (count of category / total count) × 100
4. Show your calculation clearly
5. Present results in a clean, organized format

OUTPUT FORMAT FOR PERCENTAGES/DISTRIBUTIONS:
**[Category Name] Distribution:**
• Category A: X items (XX%)
• Category B: Y items (YY%)
• Category C: Z items (ZZ%)
**Total:** N items (100%)

**Calculation:**
- Total items: [count from data]
- Category A: [count] ÷ [total] × 100 = XX%
- Category B: [count] ÷ [total] × 100 = YY%

IMPORTANT:
- Extract ALL rows/data from the context
- Count accurately
- Show percentages with calculations
- If data is incomplete in context, mention it

Your analytical answer:"""
        
        elif any(kw in question_lower for kw in ['performance', 'review', 'rating', 'evaluation']):
            return """You are an HR analyst summarizing performance reviews.

CONTEXT: {context}
QUESTION: {question}

Format clearly with sections, bullet points, and key metrics in **bold**.

Your answer:"""
        
        else:
            return """You are a business analyst extracting insights from documents.

CONTEXT: {context}
QUESTION: {question}

PROVIDE A CLEAR ANSWER:
• Extract relevant information
• Use bullet points for lists
• Put key facts in **bold**
• Be specific with numbers and names
• Organize logically

Your structured answer:"""
    
    def get_session_status(self, conversation_id: str) -> Dict[str, Any]:
        """Get session status with CSV info"""
        session = self.get_or_create_session(conversation_id)
        
        return {
            "conversation_id": conversation_id,
            "status": "active" if session["vectorstore"] else "idle",
            "loaded_files": session["loaded_files"],
            "csv_tables": session.get("csv_tables", []),
            "has_csv_data": len(session.get("csv_tables", [])) > 0,
            "total_documents": len(session["loaded_files"]),
            "total_vectors": session["total_chunks"],
            "vectorstore_ready": session["vectorstore"] is not None
        }
    
    # ... rest of methods remain the same ...
    
    def delete_session(self, conversation_id: str):
        """Delete session"""
        try:
            if conversation_id in self.session_stores:
                del self.session_stores[conversation_id]
            
            session_path = self._get_session_path(conversation_id)
            if session_path.exists():
                shutil.rmtree(session_path)
                logger.info(f"🗑️ Deleted: {conversation_id}")
        except Exception as e:
            logger.error(f"❌ Delete error: {e}")
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        return [self.get_session_status(cid) for cid in self.session_stores.keys()]
    
    def get_global_stats(self) -> Dict[str, Any]:
        total_sessions = len(self.session_stores)
        total_vectors = sum(s["total_chunks"] for s in self.session_stores.values())
        active_sessions = sum(1 for s in self.session_stores.values() if s["vectorstore"])
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "total_vectors": total_vectors,
            "base_path": str(self.base_vector_store_path)
        }