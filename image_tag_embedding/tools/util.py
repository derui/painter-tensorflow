import re


class Vocabulary(object):
    def __init__(self):
        self._vocab_index = {}

    def vocab_size(self):
        return len(self._vocab_index.keys())

    def as_vocab_index(self):
        return self._vocab_index.copy()

    def append(self, tag):
        if is_unreliable_tag(tag):
            return

        tags = normalize(tag)

        for tag in tags:
            if tag not in self._vocab_index:
                self._vocab_index[tag] = {
                    'index': len(self._vocab_index),
                    'freq': 1
                }
            else:
                self._vocab_index[tag]['freq'] += 1

    def write(self, out_file):

        with open(out_file, "w") as f:
            vocab = []
            for k in self._vocab_index:
                data = self._vocab_index[k]
                vocab.append("{}\t{}\t{}\n".format(data['index'], k, data['freq']))

            f.writelines(vocab)

    def load(self, in_file):

        with open(in_file) as f:
            lines = f.readlines()
            lines = list(map(lambda x: x.strip().split('\t'), lines))

            self._vocab_index = {v[1]: {'index': int(v[0]),
                                        'freq': int(v[2])} for v in lines}


def is_unreliable_tag(tag):
    """check unreliable tags from tag set
    """

    MATCHERS = [
        lambda x: x == "...",
        lambda x: not re.match("^[a-zA-Z]", x)
    ]

    for matcher in MATCHERS:
        if matcher(tag):
            return True

    return False


def normalize(tag):
    tag = tag.strip()
    tags = re.findall("(.+)_\((.+)\)$", tag)

    if len(tags) > 0:
        return list(tags[0])

    return [tag]
