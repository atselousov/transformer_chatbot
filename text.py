import spacy


class SpacyLowerTokenizer:
    def __init__(self):
        self.tokenizer = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])

    def __call__(self, string):
        return [t.text.lower() for t in self.tokenizer(string)]


class BPEVocab:
    we = '</w>'

    pad_token = '<pad>'
    bos_token = '<s>'
    eos_token = '</s>'
    talker1_token = '<t1>'
    talker2_token = '<t2>'

    @staticmethod
    def from_files(vocab_path, codes_path, *args, **kwargs):
        with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
            vocab = [t.strip() for t in vocab_file.readlines()]

        with open(codes_path, 'r', encoding='utf-8') as codes_file:
            codes = [c.strip() for c in codes_file.readlines()]

            if codes[0].startswith('#version'):
                codes = codes[1:]

            codes = [c.split() for c in codes if c]

        return BPEVocab(vocab, codes, *args, **kwargs)

    @staticmethod
    def get_pairs(string):
        if len(string) < 2:
            return set()

        return set(zip(string[:-1], string[1:]))

    def __init__(self, vocab, codes, tokenizer=SpacyLowerTokenizer()):
        #TODO: add check for special tokens
        vocab = [BPEVocab.pad_token, BPEVocab.bos_token, BPEVocab.eos_token, BPEVocab.talker1_token, BPEVocab.talker2_token] + vocab

        self.token2id = {t: i for i, t in enumerate(vocab)}
        self.id2token = {i: t for i, t in enumerate(vocab)}
        self.bpe_ranks = dict(zip(codes, range(len(codes))))
        self.tokenizer = tokenizer
        self.cache = {}

    def __len__(self):
        return len(self.token2id)

    @property
    def n_special_tokens(self):
        return 5

    @property
    def pad_id(self):
        return self.token2id[BPEVocab.pad_token]

    @property
    def bos_id(self):
        return self.token2id[BPEVocab.bos_token]

    @property
    def eos_id(self):
        return self.token2id[BPEVocab.eos_token]

    @property
    def talker1_id(self):
        return self.token2id[BPEVocab.talker1_token]

    @property
    def talker2_id(self):
        return self.token2id[BPEVocab.talker2_token]

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = tuple(token[:-1]) + (token[-1] + BPEVocab.we,)
        pairs = BPEVocab.get_pairs(word)

        if not pairs:
            return token + BPEVocab.we

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)

            if len(word) == 1:
                break
            else:
                pairs = BPEVocab.get_pairs(word)

        self.cache[token] = word

        return word

    def string2ids(self, string, add_bos=False, add_eos=False):
        tokens = self.tokenizer(string)
        bpe_tokens = sum([self._bpe(t) for t in tokens])
        ids = [self.token2id[t] for t in bpe_tokens]
        
        if add_bos:
            ids = [self.bos_id] + ids
            
        if add_eos:
            ids = ids + [self.eos_id]

        return ids


    def ids2string(self, ids, remove_bos=False, remove_eos=False):
        bpe_tokens = [self.id2token[id] for id in ids]
        
        if bpe_tokens[0] == self.bos_id and remove_bos:
            bpe_tokens = bpe_tokens[1:]
            
        if bpe_tokens[-1] == self.eos_id and remove_eos:
            bpe_tokens = bpe_tokens[:-1]

        return ''.join(bpe_tokens).replace(BPEVocab.we, ' ')
