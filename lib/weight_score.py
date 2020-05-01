


def weight_score(scores, group_predict):
    n_cluster = len(set(group_predict))
    w_score = 0
    n_inst = len(group_predict)
    if len(scores) != n_cluster:
        print("Errors in weight score")
        quit()
    for i in range(n_cluster):
        w_score += scores[i] * len(group_predict[group_predict == i]) / float(n_inst)
    print("WEIGHTSCORE", w_score)
    return w_score