from convai_world import ConvAIWorld
from parlai.core.params import ParlaiParser
from agent import TransformerAgent


def main():
    parser = ParlaiParser(True, True)
    parser.set_defaults(batchsize=10,
                        sample=True,
                        wild_mode=True,
                        replace_repeat=True,
                        replace_ngram=True,
                        detokenize=True,
                        emoji_prob=0.3,
                        add_questions=0.3,
                        clean_emoji=True,
                        check_grammar=True,
                        correct_generative=True,
                        split_into_sentences=True,
                        max_seq_len=256,
                        beam_size=5,
                        annealing_topk=None,
                        annealing=0.6,
                        length_penalty=0.2)

    ConvAIWorld.add_cmdline_args(parser)
    TransformerAgent.add_cmdline_args(parser)
    opt = parser.parse_args()

    agent = TransformerAgent(opt)
    world = ConvAIWorld(opt, [agent])

    while True:
        try:
            world.parley()
        except Exception as e:
            print('Exception: {}'.format(e))


if __name__ == '__main__':
    main()

