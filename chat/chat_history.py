# chat/chat_history.py

import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid
from pathlib import Path
from utils.logger import logger

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("⚠️ Redis not installed, using SQLite only")


class ChatHistoryManager:
    """
    Manage chat history with optional Redis cache
    
    Features:
    - Session management
    - Message persistence
    - Conversation threads
    - Search history
    - Analytics
    """
    
    def __init__(
        self,
        db_path: str = "./data/chat_history.db",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        cache_ttl: int = 86400  # 24 hours
    ):
        self.db_path = db_path
        self.cache_ttl = cache_ttl
        
        # Initialize SQLite database
        self._init_database()
        
        # Initialize Redis if available
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    decode_responses=True,
                    socket_connect_timeout=2
                )
                self.redis_client.ping()
                logger.info("✅ Redis connected")
            except Exception as e:
                logger.warning(f"⚠️ Redis unavailable: {e}, using SQLite only")
                self.redis_client = None
        
        logger.info("💬 Chat History Manager initialized")
    
    def _init_database(self):
        """Initialize SQLite database"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                routing TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
            )
        """)
        
        # Indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_user 
            ON conversations(user_id, updated_at DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation 
            ON messages(conversation_id, created_at ASC)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info("✅ SQLite database initialized")
    
    def create_conversation(self, user_id: str, title: str = "New Chat") -> str:
        """Create new conversation"""
        conversation_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO conversations (conversation_id, user_id, title)
            VALUES (?, ?, ?)
        """, (conversation_id, user_id, title))
        
        conn.commit()
        conn.close()
        
        # Cache in Redis
        if self.redis_client:
            cache_key = f"conv:{conversation_id}"
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps({
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "title": title,
                    "messages": []
                })
            )
        
        logger.info(f"🆕 Created conversation: {conversation_id}")
        return conversation_id
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        routing: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Add message to conversation"""
        message_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO messages 
            (message_id, conversation_id, role, content, routing, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            message_id,
            conversation_id,
            role,
            content,
            routing,
            json.dumps(metadata or {})
        ))
        
        # Update conversation timestamp
        cursor.execute("""
            UPDATE conversations 
            SET updated_at = CURRENT_TIMESTAMP,
                title = CASE 
                    WHEN title = 'New Chat' THEN substr(?, 1, 50)
                    ELSE title 
                END
            WHERE conversation_id = ?
        """, (content, conversation_id))
        
        conn.commit()
        conn.close()
        
        # Update Redis cache
        if self.redis_client:
            cache_key = f"conv:{conversation_id}"
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    data = json.loads(cached)
                    data["messages"].append({
                        "message_id": message_id,
                        "role": role,
                        "content": content,
                        "routing": routing,
                        "metadata": metadata,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Keep last 100 messages in cache
                    if len(data["messages"]) > 100:
                        data["messages"] = data["messages"][-100:]
                    
                    self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(data))
            except Exception as e:
                logger.warning(f"⚠️ Redis cache update failed: {e}")
        
        logger.info(f"💾 Message saved: {message_id}")
        return message_id
    
    def get_conversation(
        self,
        conversation_id: str,
        limit: int = 50
    ) -> Dict[str, Any]:
        """Get conversation with messages"""
        
        # Try Redis first
        if self.redis_client:
            cache_key = f"conv:{conversation_id}"
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    data = json.loads(cached)
                    logger.info(f"📖 Retrieved from Redis cache")
                    return data
            except Exception as e:
                logger.warning(f"⚠️ Redis read failed: {e}")
        
        # Fallback to SQLite
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get conversation
        cursor.execute("""
            SELECT * FROM conversations WHERE conversation_id = ?
        """, (conversation_id,))
        
        conv = cursor.fetchone()
        if not conv:
            conn.close()
            return None
        
        # Get messages
        cursor.execute("""
            SELECT * FROM messages 
            WHERE conversation_id = ?
            ORDER BY created_at ASC
            LIMIT ?
        """, (conversation_id, limit))
        
        messages = []
        for row in cursor.fetchall():
            messages.append({
                "message_id": row["message_id"],
                "role": row["role"],
                "content": row["content"],
                "routing": row["routing"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "timestamp": row["created_at"]
            })
        
        conn.close()
        
        result = {
            "conversation_id": conv["conversation_id"],
            "user_id": conv["user_id"],
            "title": conv["title"],
            "messages": messages,
            "created_at": conv["created_at"],
            "updated_at": conv["updated_at"]
        }
        
        # Update Redis cache
        if self.redis_client:
            try:
                cache_key = f"conv:{conversation_id}"
                self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(result, default=str))
            except Exception as e:
                logger.warning(f"⚠️ Redis cache failed: {e}")
        
        logger.info(f"📖 Retrieved from SQLite: {len(messages)} messages")
        return result
    
    def get_user_conversations(
        self,
        user_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get all conversations for user"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                c.*,
                COUNT(m.message_id) as message_count
            FROM conversations c
            LEFT JOIN messages m ON c.conversation_id = m.conversation_id
            WHERE c.user_id = ?
            GROUP BY c.conversation_id
            ORDER BY c.updated_at DESC
            LIMIT ?
        """, (user_id, limit))
        
        conversations = []
        for row in cursor.fetchall():
            conversations.append({
                "conversation_id": row["conversation_id"],
                "title": row["title"],
                "message_count": row["message_count"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            })
        
        conn.close()
        
        return conversations
    
    def delete_conversation(self, conversation_id: str):
        """Delete conversation and messages"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        cursor.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
        
        conn.commit()
        conn.close()
        
        # Clear Redis cache
        if self.redis_client:
            try:
                self.redis_client.delete(f"conv:{conversation_id}")
            except Exception as e:
                logger.warning(f"⚠️ Redis delete failed: {e}")
        
        logger.info(f"🗑️ Deleted conversation: {conversation_id}")
    
    def search_conversations(
        self,
        user_id: str,
        query: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT c.*
            FROM conversations c
            JOIN messages m ON c.conversation_id = m.conversation_id
            WHERE c.user_id = ? AND m.content LIKE ?
            ORDER BY c.updated_at DESC
            LIMIT ?
        """, (user_id, f"%{query}%", limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "conversation_id": row["conversation_id"],
                "title": row["title"],
                "updated_at": row["updated_at"]
            })
        
        conn.close()
        
        return results
    
    def get_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get user analytics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total conversations
        cursor.execute("""
            SELECT COUNT(*) FROM conversations WHERE user_id = ?
        """, (user_id,))
        total_conversations = cursor.fetchone()[0]
        
        # Total messages
        cursor.execute("""
            SELECT COUNT(*) 
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.conversation_id
            WHERE c.user_id = ?
        """, (user_id,))
        total_messages = cursor.fetchone()[0]
        
        # Routing breakdown
        cursor.execute("""
            SELECT routing, COUNT(*) as count
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.conversation_id
            WHERE c.user_id = ? AND routing IS NOT NULL
            GROUP BY routing
        """, (user_id,))
        
        routing_stats = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "routing_breakdown": routing_stats
        }