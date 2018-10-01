from parlai.core.params import ParlaiParser
from parlai.scripts.interactive import interactive

if __name__ == '__main__':
    parser = ParlaiParser(add_model_args=True)

    parser.set_defaults(model='agent:TransformerAgent',
                        sample=True,
                        wild_mode=False,
                        replace_repeat=True,
                        replace_ngram=True,
                        detokenize=True,
                        emoji_prob=0.3,
                        add_questions=0.4,
                        clean_emoji=True,
                        check_grammar=True,
                        correct_generative=True,
                        split_into_sentences=True,

                        max_seq_len=256,
                        beam_size=3,
                        annealing_topk=None,
                        annealing=0.6,
                        length_penalty=0.7)

    opt = parser.parse_args()
    interactive(opt)
