import re

class AnonSentence(object):
    def __init__(self, tok, lemma, pos, concepts=None, categories=None, map_info=None,
                 isLower=False, dep=None):
        self.tokText = tok
        self.lemma = lemma
        self.pos = pos
        # it's the answer sequence
        if isLower:
            self.tokText = self.tokText.lower()
            self.lemma = self.lemma.lower()
        self.tok = re.split("\\s+", self.tokText)
        self.lemma = re.split("\\s+", self.lemma)
        self.pos = re.split("\\s+", self.pos)
        self.concepts = concepts
        self.categories = categories
        self.map_info = map_info
        self.tree = dep
        self.length = len(self.tok)

        self.idx = 0

        self.index_convered = False

    def get_length(self):
        return self.length

    def get_max_word_len(self):
        max_word_len = 0
        for word in self.tok:
            max_word_len = max(max_word_len, len(word))
        return max_word_len

    def get_char_len(self):
        return [len(word) for word in self.tok]

    def convert2index(self, word_vocab, char_vocab, POS_vocab, max_char_per_word=-1):
        if self.index_convered:
            return

        if word_vocab is not None:
            self.word_idx_seq = word_vocab.to_index_sequence(self.tokText)
            self.lemma_idx_seq = word_vocab.to_index_sequence_for_list(self.lemma)

        if char_vocab is not None:
            self.char_idx_matrix = char_vocab.to_character_matrix(self.tokText, max_char_per_word=max_char_per_word)

        if POS_vocab is not None:
            self.POS_idx_seq = POS_vocab.to_index_sequence_for_list(self.pos)

        self.index_convered = True

