import json
import spacy
from nltk import ngrams

class AspectExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        with open("aspect_candidates.json") as f:
            self.aspect_candidates = json.load(f)
    
    def extract_aspects(self, sentence, domain):
        """Extract aspects using training data terms and n-gram matching"""
        doc = self.nlp(sentence)
        tokens = [token.text.lower() for token in doc]
        aspects_found = set()
        
        # Check 1-grams, 2-grams, and 3-grams
        for n in [1, 2, 3]:
            for gram in ngrams(tokens, n):
                candidate = " ".join(gram)
                if candidate in self.aspect_candidates.get(domain, []):
                    aspects_found.add(candidate)
        
        return list(aspects_found)