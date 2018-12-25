#  transformer_chatbot
#  Copyright (C) 2018 Golovanov, Tselousov
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import language_check
from mosestokenizer import *
from collections import deque
from nltk import ngrams
from nltk.corpus import wordnet
import nltk
import difflib
import random
from .retrieval import RetrievalBot
import re
import numpy as np
from itertools import combinations


SPELL_EXCEPTIONS = ['lol']
STANDARD_ANSWERS = ['do you wanna talk about something else? ',
                    'tell me something about yourself.',
                    'it is interesting. how is it outside?',
                    'do you like walking outside?',
                    'cats... do you like cats!',
                    'how do you spend your free time?',
                    'how do you usually spend your weekend?',
                    'i think you are interesting person. tell me something about yourself.']


def syntax_fix(text):
    def _i_replace(text):
        text = text.split()
        for i in range(len(text)):
            if text[i] == 'i':
                text[i] = 'I'
            if text[i] == 'i\'m':
                text[i] = 'I\'m'

        text = ' '.join(text)

        return text

    tool = language_check.LanguageTool('en-US')

    matches = tool.check(text)
    matches = [m for m in matches if text[m.fromx:m.tox].lower() not in SPELL_EXCEPTIONS]

    return _i_replace(language_check.correct(text, matches))


def detokenize(text):
    text = text.split(' ')
    text[0] = text[0].title()

    with MosesDetokenizer('en') as detokenize:
        text = detokenize(text)

    text = syntax_fix(text)

    return text


class ReplyChecker:
    def __init__(self, max_len=10, theshold=0.8, correct_generative=True, split_into_sentences=True):
        self._replies = deque([], maxlen=max_len)
        self._theshold = theshold
        self._retrieval = RetrievalBot()
        self._info = None
        self._max_len = max_len

        self._correct_generative = correct_generative
        self._split_into_sentences = split_into_sentences

        self._reset_prob()

    def _reset_prob(self):
        self._def_prob = np.ones(len(STANDARD_ANSWERS)) / len(STANDARD_ANSWERS)

    def _ratio(self, seq1, seq2):
        # todo: only works good for same sequences
        return difflib.SequenceMatcher(None, seq1, seq2).ratio()

    def _sentence_max_coincidence_drop(self, reply):
        history = sum([re.split(r' *[\?\.\!][\'"\)\]]* *', r) for r in self._replies], [])

        split_reply = re.split(r' *[\?\.\!][\'"\)\]]* *', reply)
        punc = list(re.finditer(r' *[\?\.\!][\'"\)\]]* *', reply))

        # ratio = 0
        drop = []

        for i, r in enumerate(split_reply):
            for h in history:
                if h and r:
                    ratio = self._ratio(r, h)
                    if ratio > self._theshold:
                        drop.append(i)

        drop = sorted(set(drop), reverse=True)
        for d in drop:
            split_reply.pop(d)
            punc.pop(d)

        original_text = ''

        for s, m in zip(split_reply, punc):
            original_text += s + m.group()
        if len(split_reply) > len(punc):
            original_text += split_reply[-1]

        return original_text.strip()

    def _max_coincidence(self, reply):
        if not self._replies:
            return None, reply

        if self._split_into_sentences:
            reply = self._sentence_max_coincidence_drop(reply)
            if not reply:
                return 1.0, reply

        mc = max(self._replies, key=lambda x: self._ratio(x, reply))

        ratio = self._ratio(mc, reply)

        return ratio, reply

    def _replase_reply(self, reply, request, info):
        dialog = 2 * ['None'] + [request]
        res = self._retrieval.generate_question(dialog, info)
        if res is None:
            if self._info is None:
                self._info = self._retrieval.get_reply_info(info)

            if not self._info:
                idx = np.random.choice(range(len(STANDARD_ANSWERS)), p=self._def_prob)
                self._def_prob[idx] = 0

                if np.sum(self._def_prob) == 0:
                    self._reset_prob()
                else:
                    self._def_prob /= np.sum(self._def_prob)

                return STANDARD_ANSWERS[idx]

            res = random.choice(list(self._info.keys()))
            del self._info[res]

        return res

    @staticmethod
    def _correct_repeated_sentences(text):
        split_text = re.split(r' *[\?\.\!][\'"\)\]]* *', text)
        matches = list(re.finditer(r' *[\?\.\!][\'"\)\]]* *', text))

        drop = []
        for i, j in combinations(range(len(split_text)), 2):
            if split_text[j] and split_text[j] in split_text[i]:
                drop.append(j)
        drop = set(drop)
        drop = sorted(drop, reverse=True)

        for d in drop:
            split_text.pop(d)
            matches.pop(d)

        original_text = ''

        for s, m in zip(split_text, matches):
            original_text += s + m.group()
        if len(split_text) > len(matches):
            original_text += split_text[-1]
        return original_text

    def check_reply(self, reply, request, info):
        log = [reply]
        log_names = ['IN: ', 'RL: ', 'RS: ']

        try:
            if self._correct_generative:
                reply = ReplyChecker._correct_repeated_sentences(reply)

            ratio, reply = self._max_coincidence(reply)
            log.append(reply)
            if ratio is not None:
                # ratio = self._ratio(mc, reply)

                if ratio > self._theshold:
                    reply = self._replase_reply(reply, request, info)
                    log.append(reply)

        except Exception as e:
            print('ERROR: ', e)
            reply = log[0]

        # print('[' + ' | '.join([n + str(v) for n, v in zip(log_names, log) ]) + ']')
        self._replies.append(reply)

        return reply

    def clean(self):
        self._info = None
        self._replies = deque([], maxlen=self._max_len)
        self._reset_prob()


def get_syn(seq):
    seq = seq.replace('i ', 'I ')
    seq = nltk.pos_tag(nltk.word_tokenize(seq))

    synonyms = {}

    for w, s_p in seq:
        if len(w) < 3:
            continue
        if s_p not in ['VBP', 'NN', 'NNS']:
            continue

        pos = wordnet.VERB if s_p == 'VBP' else wordnet.NOUN

        s = wordnet.synsets(w, pos=pos)
        for word in s:
            for l in word.lemma_names():
                if l != w:
                    synonyms[l.replace('_', ' ')] = w
            break

    if not synonyms:
        return None

    key = random.choice(list(synonyms.keys()))
    return synonyms[key], key


def equal_phrases(phrases):
    matches = {' am ': '\'m ',
               ' are  ': '\'re ',
               ' have ': '\'ve ',
               ' has ': '\'s ',
               'do not': 'don\'t',
               'does not': 'doesn\'t'
               }

    replasments = []

    for ph in phrases:
        a = ph
        for o, r in matches.items():
            if o in a:
                a = a.replace(o, r)
                break
            if r in a:
                a = a.replace(r, o)
                break

        if a == ph:
            # todo: find synonims
            syn = get_syn(a)
            if syn is None:
                a = a.split(' ')
                a[-2], a[-1] = a[-1], a[-2]
                a = ' '.join(a)
            else:
                a = a.replace(syn[0], syn[1])

        replasments.append(a)

    return replasments


def ngram_replaser(info, reply, n=3):
    if info is None:
        return reply

    org_reply = reply

    info = re.split(r' *[\?\.\!][\'"\)\]]* *', info.strip().lower())
    reply = re.split(r' *[\?\.\!][\'"\)\]]* *', reply.strip().lower())

    info = sum([list(ngrams(i.split(), n=n)) for i in info if i], [])
    reply = sum([list(ngrams(r.split(), n=n)) for r in reply if r], [])

    phrases = []

    for i in info:
        for r in reply:
            if i == r:
                phrases.append(' '.join(r))

    replasments = equal_phrases(phrases)

    for o, r in zip(phrases, replasments):
        org_reply = org_reply.replace(o, r)

    return org_reply
