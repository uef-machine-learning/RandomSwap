#!/usr/bin/env Rscript
# Written by Tomi Leppanen for Clustering Methods course on UEF, spring 2017

# Run like this (on Linux):
# chmod +x random_swap.R
# ./random_swap.R -d s1.txt -K 15 -g s1-cb.txt -i 100 \
#      -c clusters.log -p partition.log -s run.log
# -d (or --dataset) defines the dataset to use
# -K (or --clusters) defines number of clusters
# -t (or --ground_truth) defines ground truth file to use (optional)
# -i (or --iterations) defines how many iterations K-means is run (default 5000)
# -c (or --clusters) is filename to save clusters to (optional)
# -p (or --partition) is filename to save partition to (optional)
# 
# This can also be used inside R interpreter with source command.
# In that case call random_swap_clustering(X, K, n_iteratons, ground_truth),
# where X is data set as a matrix of row vectors,
# K is number of clusters,
# n_iterations is number of RS iterations (optional, default 5000) and
# ground_truth is ground truth clusters similarly to X for CI calculations
# (optional, CI is not calculated, if this is omitted).

# This code calculates nMSE.

# If used as a script, the code requires Xmisc package to parse command line
# arguments, which will be installed if it is not found unless there is no
# permission to install packages in which case message "unable to install
# packages" is shown. In that case user can install it manually to user's own
# directory like this:
#
# Run R
# Command: install.packages('Xmisc')
# Answer yes to using personal library and yes to creating personal library, if
# asked. Also choose a suitable mirror.
# Exit R, e.g. with command: q()
# No need to save the workspace image, answer no

optimal_partition <- function(X, C) {
    if (is.null(dim(X))) # Workaround if X has only one vector
        X <- array(X, c(1, length(X)))
    distances <- array(0, c(dim(X)[1], dim(C)[1]))
    for (i in 1:dim(C)[1]) {
        distances[, i] <- colSums((t(X)-C[i, ])^2)
    }
    return(apply(distances, 1, which.min))
}

optimal_centroids <- function(X, P, k) {
    C <- array(0, c(k, dim(X)[2]))
    for (i in 1:k) {
        n <- sum(P == i)
        if (n > 1)
            C[i, ] <- colSums(X[P == i, ])/n
        else if (n == 1)
            C[i, ] <- X[P == i, ]
    }
    return(C)
}

kmeans <- function(X, C, P_initial, max_iterations=1000000) {
    C_prev <- array(0, dim(C))
    iterations_left <- max_iterations
    while (C != C_prev && iterations_left > 0) {
        C_prev <- C
        P <- optimal_partition(X, C)
        C <- optimal_centroids(X, P, dim(C)[1])
        iterations_left <- iterations_left - 1
    }
    return(list("C"=C, "P"=P))
}

random_centroids <- function(X, k) {
    return(X[sample(1:dim(X)[1], k), ])
}

random_swap <- function(X, C) {
    C_new <- array(rep(C), dim(C))
    j <- sample(1:dim(C)[1], 1)
    new <- X[sample(1:dim(X)[1], 1), ]
    # Pick a new vector if `new` is already in C
    while (any(apply(C, 1, "==", new)))
		    new <- X[sample(1:dim(X)[1], 1), ]
    C_new[j, ] <- new
    return(list("C"=C_new, "j"=j))
}

local_repartition <- function(X, C, P, j) {
    P_new <- rep(P)
    if (!any(P_new == j))
        return(P_new)
    P_new[P_new == j] <- optimal_partition(X[P_new == j, ], C)
    for (i in 1:dim(C)[1]) {
        if (i == j || !any(P_new == i))
            next;
        P_test <- optimal_partition(X[P_new == i, ], rbind(C[j, ], C[i, ]))
        P_new[P_new == i][P_test == 0] = j
    }
    return(P_new)
}

mse <- function(X, C, P) {
    return(mean(unlist((X-C[P, ])^2)))
}

centroid_index <- function(A, B) {
    A_nearest <- optimal_partition(A, B)
    B_nearest <- optimal_partition(B, A)
    A_orphan <- hist(B_nearest, dim(A)[1], plot=FALSE)$counts == 0
    B_orphan <- hist(A_nearest, dim(B)[1], plot=FALSE)$counts == 0
    return(max(c(sum(A_orphan), sum(B_orphan))))
}

random_swap_clustering <- function(X, k, T=5000, ground_truth=NA) {
    C <- random_centroids(X, k)
    P <- optimal_partition(X, C)
    mse_best <- mse(X, C, P)
    for (i in 1:T) {
        t <- random_swap(X, C); C_new <- t$C; j <- t$j
        P_new <- local_repartition(X, C_new, P, j)
        t <- kmeans(X, C_new, P_new, 2); C_new <- t$C; P_new <- t$P
        mse_new <- mse(X, C_new, P_new)
        if (mse_new < mse_best) {
            C <- C_new; P <- P_new; mse_best <- mse_new
            if (length(ground_truth) > 1 || !is.na(ground_truth)) {
                ci <- centroid_index(C, ground_truth)
                message(sprintf("it: %d, mse: %f, ci: %d", i, mse_best, ci))
            } else {
                message(sprintf("it: %d, mse: %f", i, mse_best))
            }
        }
    }
    return(list("C"=C, "P"=P))
}

main <- function() {
    if (!require(Xmisc, warn.conflicts=FALSE, quietly=TRUE)) {
        install.packages('Xmisc')
        require(Xmisc)
    }

    parser <- ArgumentParser$new()
    parser$add_description('Random Swap implemented in R')
    parser$add_argument('-d', '--dataset', type='character', required=TRUE,
                        help='Dataset text file')
    parser$add_argument('-K', '--clusters', type='integer', required=TRUE,
                        help="Number of clusters")
    parser$add_argument('-i', '--iterations', type='integer', default=5000)
    parser$add_argument('-t', '--ground_truth', default=NA, type='character',
help="File to read ground truth from, used for calculating centroid index")
    parser$add_argument('-c', '--centroids', default=NA, type='character',
            help="File to save centroids to")
    parser$add_argument('-p', '--partition', default=NA, type='character',
            help="File to save partition to")
    parser$add_argument('-s', '--csv', default=NA, type='character',
            help="File to save statistics in CSV format")
    args <- parser$get_args()

    data <- as.matrix(read.table(args$dataset))
    if (!is.na(args$ground_truth))
        gt <- as.matrix(read.table(args$ground_truth))
    else
        gt = NA

    temp <- random_swap_clustering(data, args$clusters, args$iterations, gt)
    C <- temp$C; P <- temp$P

    message("Finished clustering")
    mse_final <- mse(data, C, P)
    if (!is.na(args$ground_truth)) {
        ci <- centroid_index(C, gt)
        message(sprintf("Final results: mse=%f, ci=%d", mse_final, ci))
    } else {
        ci <- "N/A"
        message(sprintf("Final results: mse=%f", mse_final))
    }

    if (!is.na(args$centroids))
        write(C, args$centroids, ncolumns=ncol(C))

    if (!is.na(args$partition))
        write(P, args$partition, ncolumns=1)

    if (!is.na(args$csv)) {
        if (!file.exists(args$csv)) {
            output = file(args$csv, 'w')
            write('K,MSE,CI', output)
        } else
            output = file(args$csv, 'a')
        write(paste(c(args$clusters, mse_final, ci), collapse=','), output)
    }
}

if (!interactive())
    main()