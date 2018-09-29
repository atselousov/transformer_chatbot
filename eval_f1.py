from projects.convai2.eval_f1 import setup_args, eval_f1


if __name__ == '__main__':
    parser = setup_args()

    parser.set_defaults(model='agent:TransformerAgent',
                        batchsize=10,
                        sample=False)
    opt = parser.parse_args()
    eval_f1(opt, print_parser=parser)

