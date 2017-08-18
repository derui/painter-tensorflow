import re


class Vocabulary(object):
    def __init__(self, vocab={}):
        self._vocab_index = vocab

    def vocab_size(self):
        return len(self._vocab_index.keys())

    def as_vocab_index(self):
        return self._vocab_index.copy()

    def is_contains(self, tag):
        if is_unreliable_tag(tag):
            return False
        tag = normalize(tag)

        ret = filter(lambda x: x, map(lambda x: x in self._vocab_index, tag))
        return len(list(ret)) != 0

    def filter(self, f):
        ret = {}
        for k in self._vocab_index.keys():
            if f(k, self._vocab_index[k]):
                ret[k] = self._vocab_index[k].copy()
                ret[k]['index'] = len(ret) - 1

        return Vocabulary(ret)

    def mapping(self, tags):
        ret = []

        for tag in filter(lambda x: not is_unreliable_tag(x), tags):
            tag = normalize(tag)
            for v in tag:
                if v in self._vocab_index:
                    ret.append(self._vocab_index[v]['index'])

        return ret

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
        lambda x: x == "..." or x == "commentary_request",
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
