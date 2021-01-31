"""import modules"""

from process import *
from utils import build_graph_alt
import argparse


def main():
    bg = args.build_network
    evaluate = args.evaluate
    eval_iter = args.eval_iter
    pred_iter = args.pred_iter
    other_pred = args.other_pred
    leave_one_out = args.leave_one_out

    print("args: ", args)

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

    # perform predictions using other embedding model(s)
    elif other_pred:
        model_pred_alt(original_G_path=original_G_path, model_iter=pred_iter)

    # perform prediction using IMSP
    else:
        # full graph prediction
        if not leave_one_out:
            model_pred(original_G_path=original_G_path, content_emb_path=content_emb_path, model_iter=pred_iter)
        # leave-out-out infection prediction for evaluation purposes
        else:
            ext_names = build_graph_alt(graph_path=original_G_path)
            for i in range(len(ext_names)):
                print("leave one out: ", ext_names[i])
                model_pred(original_G_path=os.path.abspath('data/classifier/original_G_' + ext_names[i] + '.txt'),
                           content_emb_path=content_emb_path, model_iter=pred_iter)

    print("\n-----------------END--------------------")


parser = argparse.ArgumentParser()
parser.add_argument('--build_network', type=bool,
                    help="if set to True, the model will stop once the network is built; " +
                         "if set to False, the model will continue to either evaluating performance or " +
                         "making predictions. Default: False",
                    default=False)

parser.add_argument("--evaluate", type=bool,
                    help="if set to True, the model evaluates the link prediction performance and performs comparison; " +
                         "if set to False, the model makes prediction. Default: False",
                    default=False)

parser.add_argument("--eval_iter", type=int, help="the number of runs while evaluating the performance. " +
                                                  "Default: 30",
                    default=30)

parser.add_argument("--pred_iter", type=int, help="the number of runs while making predictions. Default: 10",
                    default=10)

parser.add_argument("--other_pred", type=bool, help="get predictions of others. Default: false",
                    default=False)

parser.add_argument("--leave_one_out", type=bool,
                    help="leave-out-out for infections prediction using IMSP. Default: false", default=False)

args = parser.parse_args()

# main()

if __name__ == '__main__':
    main()
