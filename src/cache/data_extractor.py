from typing import List, Optional
from rapidfuzz import process, fuzz
import re
from src.utils.logger import logger

class DataExtractor:
    """
    Handles text normalization and fuzzy matching logic.
    Used before semantic caching to catch near-duplicate queries.
    """

    @staticmethod
    def normalize_query(query: str) -> str:
        """
        Normalize query by removing special characters, excessive spaces, and lowering case.
        Example: "  What is   AI?? " -> "what is ai"
        """
        if not query:
            return ""
        
        # Lowercase
        query = query.lower()
        
        # Remove special chars (keep alphanumeric and spaces)
        query = re.sub(r'[^a-z0-9\s]', '', query)
        
        # Collapse multiple spaces
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query

    @staticmethod
    def find_similar_query(query: str, existing_queries: List[str], threshold: float = 90.0) -> Optional[str]:
        """
        Find a query in the existing list that matches the input query with a score above threshold.
        Uses RapidFuzz's token_sort_ratio for robust matching.
        """
        if not existing_queries:
            return None

        normalized_input = DataExtractor.normalize_query(query)
        
        # RapidFuzz process.extractOne returns (match, score, index)
        result = process.extractOne(
            normalized_input, 
            existing_queries, 
            scorer=fuzz.token_sort_ratio,
            processor=DataExtractor.normalize_query
        )

        if result:
            match, score, _ = result
            if score >= threshold:
                logger.debug(f"Fuzzy match found: '{query}' ~= '{match}' (Score: {score})")
                return match
        
        return None

data_extractor = DataExtractor()
