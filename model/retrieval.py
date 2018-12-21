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

import random
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import re
from tqdm import tqdm
import difflib

from .sentiment import get_mood


DIALOG_SIZE = 3


def make_documents(file, index_name):
    for i, line in enumerate(tqdm(file.readlines())):
        try:
            info, d1, d2, d3, response = line.strip().split('\t')
        except:
            continue
        # todo: tooks a lot of time
        sentiment = get_mood(d3)

        source = {'info': info,
                  'd1': d1,
                  'd2': d2,
                  'd3': d3,
                  'sentiment': sentiment,
                  'response': response}

        doc = {
            '_op_type': 'create',
            '_index': index_name,
            '_type': 'dialog',
            '_source': source,
            '_id': i
        }

        yield (doc)


class RetrievalBot:
    INDEX_NAME = 'dialogs'

    def __init__(self, update_index=False, raw_index_path=None):
        # todo: set host
        self.es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

        if not self.es.ping():
            raise ValueError('Connection to retrieval server is failed.')

        if update_index:
            assert raw_index_path is not None

            self.es.indices.delete(index=self.INDEX_NAME, ignore=[400, 404])

            with open(raw_index_path, 'r') as file:
                bulk(self.es, make_documents(file, self.INDEX_NAME))

    def _match_data(self, weights, dialog, info, use_sentiment, num_matches):

        names = ['d1', 'd2', 'd3']

        assert isinstance(dialog, list)

        dialog = DIALOG_SIZE * ['None'] + dialog
        dialog = dialog[-DIALOG_SIZE:]

        assert len(dialog) == len(weights) - 2
        assert len(dialog) == len(names)

        def get_fuzzy(v):
            # todo: sometimes work better | usually give worst results
            # return {"query": v, "fuzziness": "auto", "operator":  "and"}
            return v

        request = sum([w * [{'match': {n: v}}] for n, w, v in zip(names, weights, dialog)], [])

        if info is not None:
            assert isinstance(info, str)
            request += weights[-2] * [{'match': {'info': get_fuzzy(info)}}]

        if use_sentiment:
            sentiment = get_mood(dialog[-1])
            request += weights[-1] * [{'match': {'sentiment': sentiment}}]

        res = self.es.search(index=self.INDEX_NAME,
                             body={"size": num_matches, 'query': {'bool': {'must': request}}})

        return res

    def get_response(self, dialog, info=None, use_sentiment=False, num_matches=10, return_all=False):
        # d1, d2, d3, info, sentiment
        weights = [1, 1, 3, 1, 1]
        res = self._match_data(weights, dialog, info, use_sentiment, num_matches)
        res = res['hits']

        if res['total'] == 0:
            return None

        max_score = res['max_score']
        res = res['hits']

        responses = []

        for h in res:
            if h['_score'] == max_score or return_all:
                responses.append(h['_source']['response'])

        if return_all:
            return list(set(responses))
        else:
            return random.choice(responses)

    def generate_question(self, dialog, info=None, use_sentiment=False, num_matches=20, only_with_qwords=True,
                          return_list=False):
        # d1, d2, d3, info, sentiment
        weights = [0, 0, 1, 1, 1]
        res = self._match_data(weights, dialog, info, use_sentiment, num_matches)
        res = res['hits']

        if res['total'] == 0:
            return None

        res = res['hits']

        responses = []

        for h in res:
            if '?' in h['_source']['response']:
                responses.append(h['_source']['response'])

        res = None

        if responses:
            q_words = ['who', 'whom', 'whos' 'when', 'what', 'why', 'which', 'where', 'how']

            responses = responses[:5]
            res = []
            for seq in responses:

                seq = re.split(r' *[;,\.\!][\'"\)\]]* *', seq)
                seq = [s for s in seq if '?' in s]

                for s in seq:
                    q_idx = s.find('?')

                    if only_with_qwords and not any([q in s for q in q_words]):
                        continue

                    res.append(s[:q_idx + 1])

            for i in range(len(res)):
                matches = re.finditer(' | '.join(q_words), res[i])
                for m in matches:
                    res[i] = res[i][m.start(0):]
                    break

            if res:
                if not return_list:
                    res = random.choice(res).strip()
            else:
                res = None

        return res

    def get_reply_info(self, info, num_matches=5, threshold=0.5):
        if info is None:
            return {}

        info = re.split(r' *[\?\.\!][\'"\)\]]* *', info)
        info = [i for i in info if i]

        replies = {}

        for i in info:
            request = [{'match': {'info': i}}]
            request += [{'match': {'response': i}}]

            res = self.es.search(index=self.INDEX_NAME,
                                 body={"size": num_matches, 'query': {'bool': {'must': request}}})
            res = res['hits']['hits']
            for r in res:
                local_reply = r['_source']['response']
                local_reply = re.split(r' *[\?\.\!][\'"\)\]]* *', local_reply)
                local_reply = max(local_reply, key=lambda x: difflib.SequenceMatcher(None, x, i).ratio())
                replies[local_reply] = max(difflib.SequenceMatcher(None, local_reply, i).ratio(),
                                           replies.get(local_reply, 0))

        replies = {k: v for k, v in replies.items() if v > threshold}

        return replies