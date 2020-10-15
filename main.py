"""import modules"""

from process import *
import argparse


def main():
    bg = args.build_network
    evaluate = args.evaluate
    eval_iter = args.eval_iter
    pred_iter = args.pred_iter
    other_pred = args.other_pred

    print("args received: ", args)
    # --------------------------------------------------------
    # --------------------Configurations----------------------
    # --------------------------------------------------------

    # path for content embedding
    content_emb_path = os.path.abspath('data/embedding/sentence_embedding/sentence_embedding.pkl')

    # path for constructed network
    original_G_path = os.path.abspath('data/classifier/original_G.txt')

    # --------------------------------------------------------
    # ------------------------Execution-----------------------
    # --------------------------------------------------------
    # only construct network
    if bg:
        build_graph(bg=bg)
        return

    # evaluate model performance
    if evaluate:
        model_eval(original_G_path=original_G_path, content_emb_path=content_emb_path, model_iter=eval_iter)

    # perform predictions using others
    elif other_pred:
        model_pred_alt(original_G_path=original_G_path, model_iter=pred_iter)

    # perform prediction using CrossNELP
    else:
        model_pred(original_G_path=original_G_path, content_emb_path=content_emb_path, model_iter=pred_iter)
    print("\n-----------------END--------------------")


parser = argparse.ArgumentParser()
parser.add_argument('--build_network', type=bool,
                    help="if set to True, the model will stop once the network is built; " +
                         "if set to False, the model will continue to either evaluating performance or " +
                         "making predictions. Default: False",
                    default=False)

parser.add_argument("--evaluate", type=bool,
                    help="if set to True, the model evaluates the performance and performs comparison; " +
                         "if set to False, the model makes prediction. Default: False",
                    default=False)

parser.add_argument("--eval_iter", type=int, help="the number of runs while evaluating the performance. " +
                                                  "Default: 30",
                    default=30)

parser.add_argument("--pred_iter", type=int, help="the number of runs while making predictions. Default: 5",
                    default=5)

parser.add_argument("--other_pred", type=bool, help="get predictions of others. Default: 5",
                    default=False)

args = parser.parse_args()

main()
