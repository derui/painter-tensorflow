import re


class Vocabulary(object):
    def __init__(self, vocab={}, unknown_token="<UNK>"):
        self._unknown_token = unknown_token
        self._vocab_index = {unknown_token: 0}
        self._freq = {}
        self._freeze = False

    def __len__(self):
        return len(self._vocab_index)

    def retrieve(self):
        """Retrieve vocabularies with index and frequency.
        """
        for tag in self._vocab_index:
            index = self._vocab_index[tag]
            freq = 0
            if tag != self._unknown_token:
                freq = self._freq[tag]
            yield (tag, index, freq)

    def freeze(self, freeze=True):
        self._freeze = freeze

    def trim(self, min_frequency):
        """trimming vocabulary for minimum frequency
        """
        self._freq = sorted(self._freq.items(), key=lambda x: x[0])
        self._freq = sorted(self._freq, key=lambda x: x[1], reverse=True)

        self._vocab_index = {self._unknown_token: 0}
        idx = 1

        for tag, count in self._freq:
            if count <= min_frequency:
                break

            self._vocab_index[tag] = idx
            idx += 1

        self._freq = dict(self._freq[:idx - 1])

    def get(self, tag):
        """Returns word's id in the vocabulary

        If tag is new, return new id for it.
        """
        if tag not in self._vocab_index:
            if self._freeze:
                return 0

            self._vocab_index[tag] = len(self._vocab_index)

        return self._vocab_index[tag]

    def append(self, tag):

        tag_id = self.get(tag)
        if tag_id <= 0:
            return

        if tag not in self._freq:
            self._freq[tag] = 0
        self._freq[tag] += 1

    def write(self, out_file):

        with open(out_file, "w") as f:
            vocab = []
            for k in self._vocab_index:
                index = self._vocab_index[k]
                freq = 0
                if k in self._freq:
                    freq = self._freq[k]
                vocab.append("{}\t{}\t{}\n".format(index, k, freq))

            f.writelines(vocab)

    def load(self, in_file):

        with open(in_file) as f:
            lines = f.readlines()
            lines = list(map(lambda x: x.strip().split('\t'), lines))

            self._vocab_index = {v[1]: int(v[0]) for v in lines}
            self._freq = {v[1]: int(v[2]) for v in lines}


def is_unreliable_tag(tag):
    """check unreliable tags from tag set
    """

    MATCHERS = [lambda x: x == "..." or x == "commentary_request"]

    for matcher in MATCHERS:
        if matcher(tag):
            return True

    return False


def normalize(tag):
    tag = tag.strip()
    tags = re.findall("(.+)_\((.+)\)$", tag)

    if len(tags) > 0:
        tags = list(tags[0])
    else:
        tags = [tag]

    return list(filter(lambda x: not is_unreliable_tag(x), tags))
