# aspect_extractor.py (revised)
import json
import spacy
from nltk import ngrams

class AspectExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        with open("aspect_candidates.json") as f:
            self.aspect_candidates = json.load(f)
    
    def extract_aspects(self, sentence, domain):
        """Extract non-overlapping aspects, prioritizing longer n-grams."""
        doc = self.nlp(sentence)
        tokens = [token.text.lower() for token in doc]
        aspects_found = set()
        covered_indices = set()  # Track indices already part of an aspect
        
        # Check n-grams from longest (3-grams) to shortest (1-grams)
        for n in range(3, 0, -1):
            for i in range(len(tokens) - n + 1):
                # Skip if any token in this n-gram is already covered
                if any(idx in covered_indices for idx in range(i, i + n)):
                    continue
                
                candidate = " ".join(tokens[i:i + n])
                if candidate in self.aspect_candidates.get(domain, []):
                    aspects_found.add(candidate)
                    # Mark these indices as covered
                    covered_indices.update(range(i, i + n))
        
        return list(aspects_found)
