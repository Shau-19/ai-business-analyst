'''Rotes the queries to SQL , RAG or HYBRID vased on usecases , like if precise calculations
then SQL, if deep understanding then RAG , if both then Hybrid'''

import re
from typing import Literal
from utils.logger import logger


QueryRoute = Literal["SQL", "RAG", "HYBRID"]


class CSVQueryRouter:
    '''Router for determining how to handle CSV queries - SQL, RAG, or HYBRID'''
    
    # Patterns that indicate need for SQL (precise calculations)
    SQL_PATTERNS = [
        # Counting & Aggregations
        r'\bhow many\b', r'\bcount\b', r'\btotal\b', r'\bsum\b',
        r'\baverage\b', r'\bmean\b', r'\bmedian\b', r'\bmode\b',
        
        # Percentages & Ratios
        r'\bpercentage\b', r'\bpercent\b', r'\bratio\b', r'\bproportion\b',
        r'\b%\b', r'\bper\b.*\bcent\b',
        
        # Comparisons
        r'\bgreater than\b', r'\bless than\b', r'\bmore than\b',
        r'\bbetween\b', r'\bexceeds\b', r'\bbelow\b', r'\babove\b',
        
        # Rankings
        r'\btop\s+\d+\b', r'\bbottom\s+\d+\b', r'\bhighest\b', r'\blowest\b',
        r'\bmax\b', r'\bmin\b', r'\bmaximum\b', r'\bminimum\b',
        
        # Operations
        r'\bcalculate\b', r'\bcompute\b', r'\bdistribution\b',
        r'\bgroup by\b', r'\bfilter\b', r'\bsort by\b'
    ]
    
    # Patterns that indicate need for RAG (semantic understanding)
    RAG_PATTERNS = [
        # Summaries & Overviews
        r'\bsummariz[e|ing]\b', r'\boverview\b', r'\bmain\s+(points|findings|takeaways)\b',
        r'\bkey\s+(insights|findings|points)\b',
        
        # Explanations
        r'\bexplain\b', r'\bdescribe\b', r'\btell me about\b', r'\bwhat is\b',
        r'\bwhat are\b', r'\bwhat does\b',
        
        # Patterns & Trends
        r'\btrend[s]?\b', r'\bpattern[s]?\b', r'\binsight[s]?\b',
        r'\bobservation[s]?\b', r'\bfinding[s]?\b',
        
        # Analysis requests
        r'\banalyze\b', r'\banalysis\b', r'\binterpret\b',
        r'\bsuggest[s]?\b', r'\bindicate[s]?\b', r'\bimpl(y|ies)\b',
        
        # Comparisons (qualitative)
        r'\bcompare.*\b(narrative|story|context)\b',
        r'\bsimilarit(y|ies)\b', r'\bdifference[s]?\b.*\b(why|how)\b'
    ]
    
    # Patterns that need BOTH SQL and RAG
    HYBRID_PATTERNS = [
        # Causation & Reasoning
        r'\bwhy\s+(did|is|are|does|do)\b', r'\bwhat caused\b', r'\breason for\b',
        r'\bwhat led to\b', r'\bhow come\b',
        
        # Contextualized numbers
        r'\bexplain.*\b(drop|increase|decrease|change|difference)\b',
        r'\bwhy.*\b\d+', r'\bhow.*\b\d+.*\b(compare|differ)\b',
        
        # Complex analysis
        r'\bcompare and explain\b', r'\banalyze and\b', 
        r'\bboth.*and.*why\b', r'\bnot only.*but also\b'
    ]
    
    def __init__(self):
        logger.info("🧭 CSV Query Router initialized")
    
    def route(self, question: str, has_csv_data: bool = False) -> QueryRoute:
        '''Routes the query to SQL, RAG, or HYBRID based on patterns'''
        if not has_csv_data:
            return "RAG"  # No CSV data, use normal RAG
        
        question_lower = question.lower()
        
        # Check hybrid first (most complex)
        if self._matches_hybrid(question_lower):
            logger.info(f"🔀 CSV Route: HYBRID (needs calculation + context)")
            return "HYBRID"
        
        # Score SQL vs RAG patterns
        sql_score = self._count_matches(question_lower, self.SQL_PATTERNS)
        rag_score = self._count_matches(question_lower, self.RAG_PATTERNS)
        
        logger.info(f"🧭 Routing scores - SQL: {sql_score}, RAG: {rag_score}")
        
        if sql_score > rag_score:
            logger.info(f"📊 CSV Route: SQL (precise calculations needed)")
            return "SQL"
        elif rag_score > sql_score:
            logger.info(f"📄 CSV Route: RAG (semantic understanding needed)")
            return "RAG"
        else:
            # Tie or both zero - default to SQL for structured data
            logger.info(f"📊 CSV Route: SQL (default for structured data)")
            return "SQL"
    
    def _matches_hybrid(self, question: str) -> bool:
        """Check if question needs hybrid approach"""
        return any(re.search(pattern, question) for pattern in self.HYBRID_PATTERNS)
    
    def _count_matches(self, question: str, patterns: list) -> int:
        """Count how many patterns match the question"""
        return sum(1 for pattern in patterns if re.search(pattern, question))
    
    def explain_route(self, question: str, route: QueryRoute) -> str:
        """Explain why a particular route was chosen"""
        
        explanations = {
            "SQL": "This question requires precise calculations or aggregations, best handled by SQL queries.",
            "RAG": "This question needs semantic understanding or contextual analysis, best handled by document search.",
            "HYBRID": "This question requires both precise calculations AND contextual understanding, using hybrid approach."
        }
        
        return explanations.get(route, "Unknown routing decision")

