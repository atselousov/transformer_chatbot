from projects.convai2.eval_hits import setup_args, eval_hits


if __name__ == '__main__':
    parser = setup_args()

    parser.set_defaults(model='agent:TransformerAgent',
                        batchsize=10,
                        rank_candidates=True,
                        sample=False,
                        wild_mode=False,
                        replace_repeat=True,
                        replace_ngram=False,
                        detokenize=False,
                        emoji_prob=0,
                        add_questions=0,
                        clean_emoji=True,
                        check_grammar=True,
                        correct_generative=True,
                        split_into_sentences=True,
                        max_seq_len=256,
                        beam_size=3,
                        annealing_topk=None,
                        annealing=0,
                        length_penalty=0.6)
    opt = parser.parse_args()
    eval_hits(opt, print_parser=parser)

