from projects.convai2.eval_f1 import setup_args, eval_hits


if __name__ == '__main__':
    parser = setup_args()

    parser.set_defaults(model='agent:TransformerAgent',
                        rank_candidates=True,
                        batch_sort=True,
                        batchsize=8)
    opt = parser.parse_args(print_args=False)
    eval_hits(opt, print_parser=parser)

