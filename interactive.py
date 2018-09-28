from parlai.core.params import ParlaiParser
from parlai.scripts.interactive import interactive

if __name__ == '__main__':
    parser = ParlaiParser(add_model_args=True)

    parser.set_defaults(model='agent:TransformerAgent')

    opt = parser.parse_args()
    interactive(opt)
