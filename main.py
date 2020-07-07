from scipy import sparse
from predict import *


def main():
    print("test")
    eval_route = "eval"
    training_path = "./data/train.csv"
    dev_path = "./data/dev.csv"
    # dev_query_path = "./data/dev.queries"
    test_path = "./data/test.csv"
    dev, test = "dev", "test"

    training = load_review_data_matrix(training_path)
    training_mov, training_u, training_score = load_raw_review_data(training_path)

    print("1. Corpus Exploration (10)")
    print("1.1 Basic statistics (5)")
    print("The total number of movies in training set : %d" % len(training.mov_set))
    print("The total number of users in training set : %d" % len(training.user_set))
    print("The number of times any movie was rated '1' : %d" % np.count_nonzero(training_score == 1))
    print("The number of times any movie was rated '3' : %d" % np.count_nonzero(training_score == 3))
    print("The number of times any movie was rated '5' : %d" % np.count_nonzero(training_score == 5))
    print("The average movie rating across all users and movies : %f" % np.mean(training_score))

    print("For user ID 4321")
    print("The number of movies rated : %d" % np.count_nonzero(training_u == 4321))
    u_4321_where = np.argwhere(training_u == 4321)
    u_4321 = np.take(training_score, u_4321_where)
    print("The number of times the user gave a '1' rating : %d" % np.count_nonzero(u_4321 == 1))
    print("The number of times the user gave a '3' rating : %d" % np.count_nonzero(u_4321 == 3))
    print("The number of times the user gave a '5' rating : %d" % np.count_nonzero(u_4321 == 5))
    print("The average movie rating for this user : %f" % np.mean(u_4321))

    print("For movie ID 3")
    print("The number of users rated this movie: %d" % np.count_nonzero(training_mov == 3))
    mov_3_where = np.argwhere(training_mov == 3)
    mov_3 = np.take(training_score, mov_3_where)
    print("The number of times the user gave a '1' rating : %d" % np.count_nonzero(mov_3 == 1))
    print("The number of times the user gave a '3' rating : %d" % np.count_nonzero(mov_3 == 3))
    print("The number of times the user gave a '5' rating : %d" % np.count_nonzero(mov_3 == 5))
    print("The average movie rating for this movie : %f" % np.mean(mov_3))

    print("1.2 Nearest Neighbors (5)")
    user_user_dot = training.X.dot(training.X.transpose())
    u_4321_top5 = np.argsort(user_user_dot[4321].toarray()[0])[-6:]
    u_4321_top5 = np.flip(u_4321_top5)
    print("Top 5 NNs of user 4321 in terms of dot product similarity : ", u_4321_top5)

    training_rowsum = np.sqrt(training.X.power(2).sum(axis=1).A.ravel())
    training_row_norm = sparse.diags(np.divide(1, training_rowsum, out=np.zeros_like(training_rowsum),
                                               where=training_rowsum != 0))
    training_row_normalized = training_row_norm.dot(training.X)

    user_user_cos = training_row_normalized.dot(training_row_normalized.transpose())
    u_4321_top5_cos = np.argsort(user_user_cos[4321].toarray()[0])[-6:]
    u_4321_top5_cos = np.flip(u_4321_top5_cos)
    print("Top 5 NNs of user 4321 in terms of cosine similarity : ", u_4321_top5_cos)

    training = load_review_data_matrix(training_path, matrix_func=csc_matrix)
    movie_movie_dot = training.X.transpose().dot(training.X)
    mov_3_top5 = np.argsort(movie_movie_dot[3].toarray()[0])[-6:]
    mov_3_top5 = np.flip(mov_3_top5)
    print("Top 5 NNs of movie 3 in terms of dot product similarity : ", mov_3_top5)

    training_colsum = np.sqrt(training.X.transpose().power(2).sum(axis=1).A.ravel())
    training_col_norm = sparse.diags(np.divide(1, training_colsum, out=np.zeros_like(training_colsum),
                                               where=training_colsum != 0))
    training_col_normalized = training_col_norm.dot(training.X.transpose())

    movie_movie_cos = training_col_normalized.dot(training_col_normalized.transpose())
    mov_3_top5_cos = np.argsort(movie_movie_cos[3].toarray()[0])[-6:]
    mov_3_top5_cos = np.flip(mov_3_top5_cos)
    print("Top 5 NNs of movie 3 in terms of cosine similarity : ", mov_3_top5_cos)

    # #####################################   Tutorial is Over ##################################### #

    del training_rowsum, training_row_norm, training_row_normalized
    del training_colsum, training_col_norm, training_col_normalized

    # dev_query = load_query_data(dev_query_path)
    dev_mov, dev_u = load_raw_test_data(dev_path)
    test_mov, test_u = load_raw_test_data(test_path)

    dev_user = set(dev_u)
    dev_user = np.array(list(dev_user))
    dev_movie = set(dev_mov)
    dev_movie = np.array(list(dev_movie))

    test_user = set(test_u)
    test_user = np.array(list(test_user))
    test_movie = set(test_mov)
    test_movie = np.array(list(test_movie))

    t1 = time.time()
    dot_user_nearest_neighbor, dot_user_nearest_similarity = k_nearest_neighbors(dev_user, user_user_dot)
    t2 = time.time() - t1
    print("Time taken for user dot KNN : %f seconds" % t2)
    t1 = time.time()
    cos_user_nearest_neighbor, cos_user_nearest_similarity = k_nearest_neighbors(dev_user, user_user_cos)
    t2 = time.time() - t1
    print("Time taken for user cos KNN : %f seconds" % t2)

    t1 = time.time()
    dot_movie_nearest_neighbor, dot_movie_nearest_similarity = k_nearest_neighbors(dev_movie, movie_movie_dot)
    t2 = time.time() - t1
    print("Time taken for movie dot KNN : %f seconds" % t2)
    t1 = time.time()
    cos_movie_nearest_neighbor, cos_movie_nearest_similarity = k_nearest_neighbors(dev_movie, movie_movie_cos)
    t2 = time.time() - t1
    print("Time taken for movie cos KNN : %f seconds" % t2)

    training = load_review_data_matrix(training_path)

    print("USER-USER")
    # Using Memory-based CF with dot product similarity, K=10 because it gave lowest RMSE in development set.
    predicted = predict_score(10, eval_route, dev, "user", "dot", dev_u, dev_mov,
                              dot_user_nearest_neighbor, training.X, training_mov, training_score)
    f_label = open("dev_predictions.txt", "w")
    np.savetxt(f_label, predicted, fmt="%f")
    f_label.close()

    predicted = predict_score(10, eval_route, test, "user", "dot", test_u, test_mov,
                              dot_user_nearest_neighbor, training.X, training_mov, training_score)
    f_label = open("test_predictions.txt", "w")
    np.savetxt(f_label, predicted, fmt="%f")
    f_label.close()

    predict_score(100, eval_route, dev, "user", "dot",
                  dev_u, dev_mov, dot_user_nearest_neighbor, training.X, training_mov, training_score)
    predict_score(500, eval_route, "user", "dot",
                  dev_u, dev_mov, dot_user_nearest_neighbor, training.X, training_mov, training_score)

    predict_score(10, eval_route, dev, "user", "cos",
                  dev_u, dev_mov, cos_user_nearest_neighbor, training.X, training_mov, training_score)
    predict_score(100, eval_route, dev, "user", "cos",
                  dev_u, dev_mov, cos_user_nearest_neighbor, training.X, training_mov, training_score)
    predict_score(500, eval_route, dev, "user", "cos",
                  dev_u, dev_mov, cos_user_nearest_neighbor, training.X, training_mov, training_score)

    predict_weighted_score(10, eval_route, dev, "user", "cos", dev_u, dev_mov, cos_user_nearest_neighbor,
                           training.X, training_mov, training_score, cos_user_nearest_similarity)
    predict_weighted_score(100, eval_route, dev, "user", "cos", dev_u, dev_mov, cos_user_nearest_neighbor,
                           training.X, training_mov, training_score, cos_user_nearest_similarity)
    predict_weighted_score(500, eval_route, dev, "user", "cos", dev_u, dev_mov, cos_user_nearest_neighbor,
                           training.X, training_mov, training_score, cos_user_nearest_similarity)

    print("MOVIE-MOVIE")
    predict_score(10, eval_route, dev, "movie", "dot",
                  dev_mov, dev_u, dot_movie_nearest_neighbor, training.X.transpose(), training_u, training_score)
    predict_score(100, eval_route, dev, "movie", "dot",
                  dev_mov, dev_u, dot_movie_nearest_neighbor, training.X.transpose(), training_u, training_score)
    predict_score(500, eval_route, dev, "movie", "dot",
                  dev_mov, dev_u, dot_movie_nearest_neighbor, training.X.transpose(), training_u, training_score)

    predict_score(10, eval_route, dev, "movie", "cos",
                  dev_mov, dev_u, cos_movie_nearest_neighbor, training.X.transpose(), training_u, training_score)
    predict_score(100, eval_route, dev, "movie", "cos",
                  dev_mov, dev_u, cos_movie_nearest_neighbor, training.X.transpose(), training_u, training_score)
    predict_score(500, eval_route, dev, "movie", "cos",
                  dev_mov, dev_u, cos_movie_nearest_neighbor, training.X.transpose(), training_u, training_score)

    predict_weighted_score(10, eval_route, dev, "movie", "cos", dev_mov, dev_u, cos_movie_nearest_neighbor,
                           training.X.transpose(), training_u, training_score, cos_movie_nearest_similarity)
    predict_weighted_score(100, eval_route, dev, "movie", "cos", dev_mov, dev_u, cos_movie_nearest_neighbor,
                           training.X.transpose(), training_u, training_score, cos_movie_nearest_similarity)
    predict_weighted_score(500, eval_route, dev, "movie", "cos", dev_mov, dev_u, cos_movie_nearest_neighbor,
                           training.X.transpose(), training_u, training_score, cos_movie_nearest_similarity)

    del user_user_dot, user_user_cos, movie_movie_dot, movie_movie_cos
    del dot_user_nearest_neighbor, dot_user_nearest_similarity, dot_movie_nearest_neighbor, dot_movie_nearest_similarity
    del cos_user_nearest_neighbor, cos_user_nearest_similarity, cos_movie_nearest_neighbor, cos_movie_nearest_similarity

    print("PCC")
    t1 = time.time()
    centerized_training = training.X.toarray()
    centerized_training = (centerized_training.transpose() - np.average(centerized_training, axis=1)).transpose()

    centerized_rowsum = np.sqrt(np.square(centerized_training).sum(axis=1))
    standardized_training = sparse.diags(np.divide(1, centerized_rowsum, out=np.zeros_like(centerized_rowsum),
                                                   where=centerized_rowsum != 0))
    standardized_training = standardized_training.dot(centerized_training)

    user_user_pcc = np.dot(standardized_training, standardized_training.transpose())
    t2 = time.time() - t1
    print("Time taken for user_user_pcc : %f seconds" % t2)
    del centerized_training, centerized_rowsum, standardized_training

    t1 = time.time()
    user_user_pcc = csr_matrix(user_user_pcc)
    pcc_user_nearest_neighbor, pcc_user_nearest_similarity = k_nearest_neighbors(dev_user, user_user_pcc)
    t2 = time.time() - t1
    print("Time taken for user pcc KNN : %f seconds" % t2)

    predict_score(10, eval_route, dev, "user", "PCC",
                  dev_u, dev_mov, pcc_user_nearest_neighbor, training.X, training_mov, training_score)
    predict_score(100, eval_route, dev, "user", "PCC",
                  dev_u, dev_mov, pcc_user_nearest_neighbor, training.X, training_mov, training_score)
    predict_score(500, eval_route, dev, "user", "PCC",
                  dev_u, dev_mov, pcc_user_nearest_neighbor, training.X, training_mov, training_score)

    predict_weighted_score(10, eval_route, dev, "user", "PCC", dev_u, dev_mov, pcc_user_nearest_neighbor,
                           training.X, training_mov, training_score, pcc_user_nearest_similarity)
    predict_weighted_score(100, eval_route, dev, "user", "PCC", dev_u, dev_mov, pcc_user_nearest_neighbor,
                           training.X, training_mov, training_score, pcc_user_nearest_similarity)
    predict_weighted_score(500, eval_route, dev, "user", "PCC", dev_u, dev_mov, pcc_user_nearest_neighbor,
                           training.X, training_mov, training_score, pcc_user_nearest_similarity)

    del user_user_pcc, pcc_user_nearest_neighbor, pcc_user_nearest_similarity

    print("PCC")
    t1 = time.time()
    centerized_training = training.X.toarray()
    centerized_training = (centerized_training - np.average(centerized_training, axis=0)).transpose()

    centerized_colsum = np.sqrt(np.square(centerized_training).sum(axis=1))
    standardized_training = sparse.diags(np.divide(1, centerized_colsum, out=np.zeros_like(centerized_colsum),
                                                   where=centerized_colsum != 0))
    standardized_training = standardized_training.dot(centerized_training)

    movie_movie_pcc = np.dot(standardized_training, standardized_training.transpose())

    t2 = time.time() - t1
    print("Time taken for movie_pcc : %f seconds" % t2)
    del centerized_training, centerized_colsum, standardized_training

    t1 = time.time()
    movie_movie_pcc = csr_matrix(movie_movie_pcc)
    pcc_movie_nearest_neighbor, pcc_movie_nearest_similarity = k_nearest_neighbors(dev_movie, movie_movie_pcc)
    t2 = time.time() - t1
    print("Time taken for movie pcc KNN : %f seconds" % t2)

    predict_score(10, eval_route, dev, "movie", "PCC",
                  dev_mov, dev_u, pcc_movie_nearest_neighbor, training.X.transpose(), training_u, training_score)
    predict_score(100, eval_route, dev, "movie", "PCC",
                  dev_mov, dev_u, pcc_movie_nearest_neighbor, training.X.transpose(), training_u, training_score)
    predict_score(500, eval_route, dev, "movie", "PCC",
                  dev_mov, dev_u, pcc_movie_nearest_neighbor, training.X.transpose(), training_u, training_score)

    predict_weighted_score(10, eval_route, dev, "movie", "PCC", dev_mov, dev_u, pcc_movie_nearest_neighbor,
                           training.X.transpose(), training_u, training_score, pcc_movie_nearest_similarity)
    predict_weighted_score(100, eval_route, dev, "movie", "PCC", dev_mov, dev_u, pcc_movie_nearest_neighbor,
                           training.X.transpose(), training_u, training_score, pcc_movie_nearest_similarity)
    predict_weighted_score(500, eval_route, dev, "movie", "PCC", dev_mov, dev_u, pcc_movie_nearest_neighbor,
                           training.X.transpose(), training_u, training_score, pcc_movie_nearest_similarity)

    print("end")


if __name__ == '__main__':
    main()
