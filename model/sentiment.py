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

from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import emoji
import re


def get_sentiments(text):
    l_pos = "pos"
    l_neg = "neg"
    l_neut = "neu"

    vader_analyzer = SentimentIntensityAnalyzer()
    sentiments = vader_analyzer.polarity_scores(text)

    return {key: sentiments[key] for key in [l_pos, l_neg, l_neut]}


def get_mood(text):
    sentiments = get_sentiments(text)

    return max(sentiments, key=sentiments.get)


def pick_emoji(text):
    smiles = {
        "pos": [":grinning:", ":smiley:", ":smile:", ":grin:", ":wink:", ":slightly_smiling_face:"],
        "neg": [":worried:", ":slightly_frowning_face:", ":white_frowning_face:", ":fearful:", ":cold_sweat:",
                ":cry:"]

    }

    l_pos = "pos"
    l_neg = "neg"
    l_neut = "neu"

    sentiments = get_sentiments(text)

    if sum(sentiments.values()) != 1:
        return ''

    label = np.random.choice(list(sentiments.keys()),
                             p=list(sentiments.values()))

    if label == l_neut:
        return ''

    # prob = sentiments[label]
    # if np.random.uniform(0, 1) > prob:
    #     return ''
    return emoji.emojize(np.random.choice(smiles[label]), use_aliases=True)


def clean_emoji(text):
    text = emoji.get_emoji_regexp().sub(r'', text)
    text = re.sub(r'[\^:)(]', '', text)
    return text.strip()
