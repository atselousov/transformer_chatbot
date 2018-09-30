from convai_world import ConvAIWorld
from parlai.core.params import ParlaiParser
from agent import TransformerAgent


def main():
    parser = ParlaiParser(True, True)
    parser.set_defaults(batchsize=10,
                        wild_mode=True)

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

