import re
from tools.file_processing import read_file


class Parser:
    """
    Parse word forms, lemmas and part of speech (POS) tags
    """

    def __init__(self):

        # group_pattern searches groups of word forms,
        # lemmas and part of speech (POS) tags in the file - word form[lemma/POS tags]
        # for example, знає[знати/verb:imperf:pres:s:3]
        self.group_pattern = "\w+['-]?\w*\[\w+['-]?\w*\/[\w+[:&]*]*\]"

        # word_pattern searches words and digits in a group
        self.word_pattern = "[&]*\w+['-]?\w*"

    def execute(self, index2word_set, path_to_file):
        """
        :param path_to_file: path to the LanguageTool API NLP UK output file
        :return: list of list of words
        """

        data = read_file(path_to_file)

        groups = re.findall(self.group_pattern, data)

        wordlist = []

        for group in groups:
            words = re.findall(self.word_pattern, group)
            if 'null' not in words \
                    and 'SENT_END' not in words \
                    and 'unknown' not in words \
                    and words[1].strip(' ').lower() in index2word_set:
                wordlist.append(words)
        return wordlist
