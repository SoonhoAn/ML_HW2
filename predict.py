import time
from utils import *


def k_nearest_neighbors(array, similarity):
    nearest_neighbor = {}
    similarity_dict = {}
    for i in range(len(array)):
        profile = array[i]
        similarity_array = similarity[profile].toarray()[0]
        profile_topk = np.argsort(similarity_array)[-501:]
        profile_topk = np.flip(profile_topk)
        similarity_array = np.take(similarity_array, profile_topk)
        nearest_neighbor[profile_topk[0]] = profile_topk[:]
        similarity_dict[profile_topk[0]] = similarity_array[1:]
        print("\rKNN Read : %d lines" % i, end=" ")
    print(" ")
    return nearest_neighbor, similarity_dict


# The function below is way much slower than the modified version, so it's abandoned.
"""
def naive_predict_score(
        K,  # K nearest neighbor
        profile_list,  # Memory-based CF: =user, Item-based CF: =movie
        objective_list,  # Memory-based CF: =movie, Item-based CF: =user
        nearest_neighbor,
        training_sparse,  # Memory-based CF: =training.X, Item-based CF: =training.X.transpose()
        training_obj,  # Memory-based CF: =movie, Item-based CF: =user
        training_score):

    np_predicted = []
    for i in range(len(objective_list)):
        exception = False
        profile = profile_list[i]
        objective = objective_list[i]
        nearest = nearest_neighbor.get(profile, None)
        if nearest is None:
            exception = True
        predicted = 0.0
        if not exception:
            for k in range(K):
                neighbor = nearest[k + 1]
                predicted += training_sparse[neighbor].toarray()[0][objective]
            predicted /= K
            predicted += 3
        elif exception:
            obj_where = np.argwhere(training_obj == objective)
            predicted = np.take(training_score, obj_where)
            predicted = np.mean(predicted)
        predicted = round(predicted)
        np_predicted.append(predicted)
        print("\rRead : %d lines" % i, end=" ")
    np_predicted = np.array(np_predicted)
    return np_predicted
"""


def predict_score(
        K,  # K nearest neighbor
        eval_route,
        div,  # development or test
        based,
        measure,
        profile_list,  # Memory-based CF: =user, Item-based CF: =movie
        objective_list,  # Memory-based CF: =movie, Item-based CF: =user
        nearest_neighbor,
        training_sparse,  # Memory-based CF: =training.X, Item-based CF: =training.X.transpose()
        training_obj,  # Memory-based CF: =movie, Item-based CF: =user
        training_score):
    fmt = "%f"
    t1 = time.time()
    np_predicted = []
    for i in range(len(objective_list)):
        # exception: True when the user(movie was) never rated (by) any movie(user), so there's no similarity exist
        exception = False
        profile = profile_list[i]
        objective = objective_list[i]
        nearest = nearest_neighbor.get(profile, None)
        if nearest is None:
            exception = True
        predicted = 0.0
        if not exception:
            nearest = nearest[1:K+1]
            nearest = nearest.tolist()
            # predicted = training_sparse[nearest].toarray()[:, objective]
            predicted = training_sparse[nearest, objective].toarray().ravel()
            predicted = np.sum(predicted)
            predicted /= K
            predicted += 3
        elif exception:
            if np.count_nonzero(training_obj == objective) == 0:
                predicted = 3
            else:
                obj_where = np.argwhere(training_obj == objective)
                predicted = np.take(training_score, obj_where)
                predicted = np.mean(predicted)
        # predicted = round(predicted)  # Making the output to be integer.(deprecated)
        np_predicted.append(predicted)
        print("\rPrediction Read : %d lines" % i, end=" ")
    np_predicted = np.array(np_predicted)
    print("K = %d finished" % K)
    f_label = open("./%s/%s_predictions_%s_%s_k%d.txt" % (eval_route, div, based, measure, K), "w")
    np.savetxt(f_label, np_predicted, fmt="%s" % fmt)
    f_label.close()
    t2 = time.time() - t1
    print("Time taken for predicting %s_%s_k%d : %f seconds" % (based, measure, K, t2))
    return np_predicted


def predict_weighted_score(
        K,  # K nearest neighbor
        eval_route,
        div,  # development or test
        based,
        measure,
        profile_list,  # Memory-based CF: =user, Item-based CF: =movie
        objective_list,  # Memory-based CF: =movie, Item-based CF: =user
        nearest_neighbor,
        training_sparse,  # Memory-based CF: =training.X, Item-based CF: =training.X.transpose()
        training_obj,  # Memory-based CF: =movie, Item-based CF: =user
        training_score,
        nearest_similarity):  # Memory-based CF: =movie, Item-based CF: =user
    fmt = "%f"
    t1 = time.time()
    np_predicted = []
    for i in range(len(objective_list)):
        # exception: True when the user never rated any movie, so there's no similarity exist
        exception = False
        profile = profile_list[i]
        objective = objective_list[i]
        nearest = nearest_neighbor.get(profile, None)
        if nearest is None:
            exception = True
        predicted = 0.0
        if not exception:
            nearest = nearest[1:K+1]
            weight = nearest_similarity.get(profile)[:K]
            if np.sum(weight) == 0:
                weight = np.ones(K)
            nearest = nearest.tolist()
            predicted = training_sparse[nearest, objective].toarray().ravel()
            predicted = np.dot(predicted, weight)
            predicted /= np.sum(weight)  # Absolute value?
            predicted += 3
        elif exception:
            if np.count_nonzero(training_obj == objective) == 0:
                predicted = 3
            else:
                obj_where = np.argwhere(training_obj == objective)
                predicted = np.take(training_score, obj_where)
                predicted = np.mean(predicted)
        # predicted = round(predicted)  # Making the output to be integer.(deprecated)
        np_predicted.append(predicted)
        print("\rPrediction Read : %d lines" % i, end=" ")
    np_predicted = np.array(np_predicted)
    print("K = %d finished" % K)
    f_label = open("./%s/%s_predictions_%s_weighted_%s_k%d.txt" % (eval_route, div, based, measure, K), "w")
    np.savetxt(f_label, np_predicted, fmt="%s" % fmt)
    f_label.close()
    t2 = time.time() - t1
    print("Time taken for predicting %s_weighted_%s_k%d : %f seconds" % (based, measure, K, t2))
    return np_predicted

