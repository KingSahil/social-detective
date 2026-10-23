import re
from collections import Counter
from typing import List

from app.search import Candidate

class ContextManager:
    """
    Analyzes OSINT metadata from background crowd profiles
    to extract, score, and rank high-value context tags.
    """
    
    def __init__(self):
        # Common stop words to ignore during tag extraction
        self.stop_words = {
            "the", "and", "for", "with", "from", "that", "this", "is", "are", 
            "was", "were", "has", "have", "had", "not", "but", "what", "all",
            "just", "like", "their", "about", "out", "your", "can", "get",
            "who", "when", "where", "why", "how", "photo", "image", "post",
            "profile", "user", "instagram", "twitter", "linkedin", "facebook",
            "tiktok", "youtube", "social", "media", "verified", "identity",
            "https", "http", "com", "org", "net", "www"
        }
        
    def _clean_text(self, text: str) -> str:
        """Removes punctuation and normalizes spacing."""
        text = re.sub(r'[^\w\s#@]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def extract_tags(self, candidates: List[Candidate]) -> List[str]:
        """
        Parses raw OSINT data from background crowd profiles
        to extract potential high-value tags.
        """
        raw_tags = []
        for candidate in candidates:
            if candidate.title:
                # Extract hashtags
                hashtags = re.findall(r'#\w+', candidate.title)
                raw_tags.extend([h.lower() for h in hashtags])
                
                # Extract handles
                handles = re.findall(r'@\w+', candidate.title)
                raw_tags.extend([h.lower() for h in handles])
                
                # Extract capitalized words (potential proper nouns like universities, companies)
                cleaned = self._clean_text(candidate.title)
                words = cleaned.split()
                for word in words:
                    # Keep alphanumeric words that start with capital letters (Proper Nouns)
                    if len(word) > 3 and word.istitle() and word.lower() not in self.stop_words:
                        raw_tags.append(word.lower())
                        
            if candidate.domain:
                # Add domain names as potential context (e.g. stanford.edu)
                domain_parts = candidate.domain.split('.')
                if len(domain_parts) >= 2:
                    main_domain = domain_parts[-2].lower()
                    if main_domain not in self.stop_words and len(main_domain) > 3:
                        raw_tags.append(main_domain)
                        
        return raw_tags

    def score_and_rank_tags(self, raw_tags: List[str]) -> List[str]:
        """
        Scores tags based on frequency and specific high-value patterns
        (like hashtags or handles). Returns a ranked list of unique tags.
        """
        # Count frequencies
        counter = Counter(raw_tags)
        
        # Apply weighting
        scored_tags = {}
        for tag, count in counter.items():
            score = count
            # Boost score for hashtags and handles
            if tag.startswith('#') or tag.startswith('@'):
                score *= 2.0
            
            # Penalize very long or very short tags slightly
            if len(tag) < 4:
                score *= 0.5
            elif len(tag) > 15:
                score *= 0.8
                
            scored_tags[tag] = score
            
        # Sort by score descending
        ranked = sorted(scored_tags.items(), key=lambda x: x[1], reverse=True)
        return [tag for tag, score in ranked]

    def auto_select(self, candidates: List[Candidate], top_k: int = 5, memory_tags: List[str] = None) -> List[str]:
        """
        Parses candidates and returns the top K most potent tags.
        Prioritizes memory_tags if provided.
        """
        memory_tags = memory_tags or []
        formatted_memory_tags = [f"[MEMORY] {tag}" for tag in memory_tags]
        
        raw_tags = self.extract_tags(candidates)
        ranked_tags = self.score_and_rank_tags(raw_tags)
        
        # Filter out scraped tags that are already in memory_tags
        filtered_ranked = [t for t in ranked_tags if t not in memory_tags]
        
        final_tags = formatted_memory_tags + filtered_ranked
        return final_tags[:top_k]

    def process_and_suggest(self, candidates: List[Candidate], memory_tags: List[str] = None) -> List[str]:
        """
        Full pipeline: Takes candidates and memory_tags, extracts tags, ranks them,
        and returns the full ranked list for manual selection in UI.
        """
        memory_tags = memory_tags or []
        formatted_memory_tags = [f"[MEMORY] {tag}" for tag in memory_tags]
        
        raw_tags = self.extract_tags(candidates)
        ranked_tags = self.score_and_rank_tags(raw_tags)
        
        # Filter out scraped tags that are already in memory_tags
        filtered_ranked = [t for t in ranked_tags if t not in memory_tags]
        
        return formatted_memory_tags + filtered_ranked
