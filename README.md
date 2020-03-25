# Parallel Online K - Nearest Neighbour Search using a Progressive k-d Tree

### Authors
- Kaushal Bhogale
- Rahul John Roy

## Introduction and Motivation
The k-nearest neighbor (KNN) algorithm is a fundamental classification and regression method for machine learning. KNN works by finding the k-nearest points to a given query point in the feature space. 

With the advent of sophisticated data collection mechanisms, machine learning on large datasets has become very important. An existing parallel KNN algorithm is PANDA[1], which uses k-d trees. This work assumes data is assumed to be already available, and the entire k-d tree is built in a single shot.

In contrast to the one-shot algorithm, an online algorithm allows adding new points to trees even after the trees are built. This benefits interactive systems in that analysts do not have to wait until all data is loaded. Rather, the data is split into mini-batches, inserted into the k-d tree incrementally. But, as more points are inserted to a k-d tree, the tree can become unbalanced, deteriorating the query time. Thus, we propose a tree balancing method to address this issue.

## Problem Definition
The aim of the project is to develop an parallel online k-Nearest Neighbour search algorithm, which works on the principle of progressively growing k-d trees. The algorithm provides a balancing algorithm for k-d trees, which **minimizes the amount of communication latencies required between processes.**

## Project Goals
- Implement the existing parallel KNN algorithm.
- Design the online version of the same.
- Experiment for different input distributions.
- Test for various arrival frequencies and bandwidth of data.
- Provide empirical justification for scenarios in which the online algorithm will be useful.

## References
M. M. A. Patwary et al., “PANDA: Extreme Scale Parallel KNearest Neighbor on Distributed Architectures,” in IEEE Parallel and Distributed Processing Symposium, 2016, pp. 494–503
