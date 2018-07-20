from collections import defaultdict


class Vocabulary:
    def __init__(self):
        self.pos_tags = []
        # pos tag -> index в self.pos_tags
        # dict{pos tag:index in pos_tags}
        self.pos_tags_indices = {}
        self.size = 0
        self.wordforms_lemma_gramvalue_to_idx = defaultdict(int)
        self.idx_to_wordforms_lemma_gramvalue = None

    def build(self, data, pos_tags):
        self.pos_tags = pos_tags

        # item[0] - word form
        # item[1] - lemma
        # item[2:] - pos and additional grammatical tags
        for item in data:
            word = '|'.join(map(str, item))            
            self.wordforms_lemma_gramvalue_to_idx[word] = len(self.wordforms_lemma_gramvalue_to_idx)

        self.idx_to_wordforms_lemma_gramvalue = {index: word for word, index in self.wordforms_lemma_gramvalue_to_idx.items()}
        self.size = len(self.wordforms_lemma_gramvalue_to_idx)
        self.pos_tags_indices = dict(zip(self.pos_tags, range(len(self.pos_tags))))










