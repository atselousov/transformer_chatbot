from projects.convai2.eval_hits import setup_args, eval_hits


if __name__ == '__main__':
    parser = setup_args()

    parser.set_defaults(model='agent:TransformerAgent',
                        rank_candidates=True,
                        batchsize=10)
    opt = parser.parse_args()
    eval_hits(opt, print_parser=parser)

