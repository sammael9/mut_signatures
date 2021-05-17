from sklearn.metrics.pairwise import paired_distances
import numpy as np
import random


def fitness_random():
    return random.uniform(0, 1)


def fitness_manhattan_similarity_sum(catalog_matrix, exposure_data, original_data):
    new_matrix = np.dot(np.array(catalog_matrix), np.array(exposure_data).transpose())
    similarity = paired_distances(new_matrix.reshape(-1, 1), np.array(original_data).reshape(-1, 1), metric='manhattan')
    return - np.sum(similarity)


def fitness_manhattan_similarity_avg(catalog_matrix, exposure_data, original_data):
    new_matrix = np.dot(np.array(catalog_matrix), np.array(exposure_data).transpose())
    similarity = paired_distances(new_matrix.reshape(-1, 1), np.array(original_data).reshape(-1, 1), metric='manhattan')
    return - np.average(similarity)


def fitness_euclidean_similarity(catalog_matrix, exposure_data, original_data):
    new_matrix = np.dot(np.array(catalog_matrix), np.array(exposure_data).transpose())
    similarity = paired_distances(np.array(new_matrix).transpose(), np.array(original_data).transpose(), metric='euclidean')
    return - np.average(similarity)
