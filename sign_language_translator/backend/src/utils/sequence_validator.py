import numpy as np
from collections import defaultdict
import pickle
from pathlib import Path

class SequenceValidator:
    def __init__(self, n_gram=3):
        self.n_gram = n_gram
        self.sequence_counts = defaultdict(int)
        self.total_sequences = 0
        self.min_probability = 0.01  # Minimum probability for a sequence to be considered valid

    def train(self, sequences):
        """Train the validator on valid sign sequences"""
        for sequence in sequences:
            # Convert sequence to string for easier handling
            seq_str = '-'.join(map(str, sequence))
            
            # Update n-gram counts
            for i in range(len(sequence) - self.n_gram + 1):
                n_gram = seq_str[i:i+self.n_gram]
                self.sequence_counts[n_gram] += 1
                self.total_sequences += 1

    def validate_sequence(self, sequence):
        """Validate a sequence of signs"""
        if len(sequence) < self.n_gram:
            return True  # Too short to validate
            
        seq_str = '-'.join(map(str, sequence))
        total_score = 0
        
        # Check each n-gram in the sequence
        for i in range(len(sequence) - self.n_gram + 1):
            n_gram = seq_str[i:i+self.n_gram]
            probability = self.sequence_counts.get(n_gram, 0) / self.total_sequences
            total_score += probability
            
        # Calculate average probability
        avg_probability = total_score / (len(sequence) - self.n_gram + 1)
        
        return avg_probability >= self.min_probability

    def suggest_correction(self, sequence):
        """Suggest corrections for invalid sequences"""
        if len(sequence) < self.n_gram:
            return sequence
            
        corrections = []
        seq_str = '-'.join(map(str, sequence))
        
        for i in range(len(sequence) - self.n_gram + 1):
            n_gram = seq_str[i:i+self.n_gram]
            if self.sequence_counts.get(n_gram, 0) / self.total_sequences < self.min_probability:
                # Find similar valid n-grams
                similar_ngrams = self._find_similar_ngrams(n_gram)
                if similar_ngrams:
                    corrections.append((i, similar_ngrams[0]))
                    
        return self._apply_corrections(sequence, corrections)

    def _find_similar_ngrams(self, n_gram):
        """Find similar valid n-grams"""
        similar = []
        for valid_ngram, count in self.sequence_counts.items():
            if count / self.total_sequences >= self.min_probability:
                # Simple similarity measure (can be improved)
                similarity = sum(1 for a, b in zip(n_gram, valid_ngram) if a == b) / len(n_gram)
                if similarity >= 0.7:  # 70% similarity threshold
                    similar.append((valid_ngram, count))
                    
        # Sort by count and return top matches
        similar.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in similar[:3]]

    def _apply_corrections(self, sequence, corrections):
        """Apply suggested corrections to the sequence"""
        if not corrections:
            return sequence
            
        corrected = sequence.copy()
        for pos, n_gram in corrections:
            # Convert n-gram back to numbers
            signs = list(map(int, n_gram.split('-')))
            corrected[pos:pos+len(signs)] = signs
            
        return corrected

    def save(self, path):
        """Save the validator state"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'n_gram': self.n_gram,
                'sequence_counts': dict(self.sequence_counts),
                'total_sequences': self.total_sequences,
                'min_probability': self.min_probability
            }, f)

    def load(self, path):
        """Load the validator state"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.n_gram = data['n_gram']
            self.sequence_counts = defaultdict(int, data['sequence_counts'])
            self.total_sequences = data['total_sequences']
            self.min_probability = data['min_probability'] 