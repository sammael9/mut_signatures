import logging
import os
import sys
from sklearn.metrics.pairwise import paired_distances
import numpy as np
from src import fireworks, fitness
from sklearn.decomposition import NMF

import pandas as pd

from src.watcher import Watcher

logging.basicConfig(filename='example.log', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info("Hi")

# Read data

mutational_matrix = pd.read_csv(r'../resources/mut_mat.csv')
reconstructed_matrix = pd.read_csv(r'../resources/reconstructed.csv')
cosmic_catalog = pd.read_csv(r'../resources/cosmic_mutations.csv', delimiter='\t')
contribution_matrix = pd.read_csv(r'../resources/contribution.csv')

mutational_matrix = mutational_matrix.sort_values(by='Somatic Mutation Type', axis=0)
reconstructed_matrix = reconstructed_matrix.sort_values(by='Somatic Mutation Type', axis=0)
cosmic_catalog = cosmic_catalog.sort_values(by='Somatic Mutation Type', axis=0)
cosmic_catalog = cosmic_catalog.loc[:, ~cosmic_catalog.columns.str.contains('^Unnamed')]

chosen = ['colon1', 'colon2', 'colon3', 'intestine1', 'intestine2', 'intestine3', 'liver1', 'liver2', 'liver3']

my_input = mutational_matrix[chosen].values
my_catalog = cosmic_catalog.loc[:, cosmic_catalog.columns.str.contains('^Signature')].values
my_reconstructed = reconstructed_matrix[chosen].values
my_contribution = contribution_matrix[chosen].values

signeR_matrix = pd.read_csv(r'../resources/21_breast_cancers.mutations.txt', delimiter='\t')
signeR_contribution = pd.read_csv(r'../resources/exposures_signeR.csv')
signeR_signatures = pd.read_csv(r'../resources/signatures_signeR.csv')
signeR_matrix = signeR_matrix.loc[:, ~signeR_matrix.columns.str.contains('^Unnamed')].values
signeR_contribution = signeR_contribution.loc[:, ~signeR_contribution.columns.str.contains('^Unnamed')].values
signeR_signatures = signeR_signatures.loc[:, ~signeR_signatures.columns.str.contains('^Unnamed')].values
signeR_reconstructed = np.dot(signeR_signatures, signeR_contribution)
signeR_reconstructed = signeR_reconstructed.transpose()
signeR_similarity = paired_distances(signeR_reconstructed.reshape(-1, 1), signeR_matrix.reshape(-1, 1),
                                     metric='manhattan')
signeR_similariry_avg_score = np.average(signeR_similarity)
signeR_similariry_sum_score = np.sum(signeR_similarity)

# Initialize fireworks display for the first input column

diameter_max = 100
diameter_min = 15

my_display = fireworks.Display(firework_count=15, dimensions_exp_1=5, dimensions_exp_2=21, dimensions_sig_1=96,
                               dimensions_sig_2=5, dimension_exp_min=0, dimension_exp_max=1000, dimension_sig_min=0,
                               dimension_sig_max=0.5, diameter_exp_min=15, diameter_exp_max=100, spark_min=10,
                               spark_max=50, gaussian_spark_count=10,
                               fitness_function=fitness.fitness_manhattan_similarity_sum, catalog=my_catalog,
                               mutational_data=signeR_matrix, spark_dimension_count=12, dimension_limit=False)
my_display.create_fireworks(None)
my_display.showtime()

watcher = Watcher(iterations=15, threshold=100, starting_iteration=100, fw_display=my_display)

watcher.iterate(for_range=range(0, 300), reduction=0.98)

logging.info("Finished iterating. Comparing solutions.")

old_reconstructed = np.dot(np.array(signeR_signatures), np.array(signeR_contribution))
new_reconstructed = np.dot(np.array(my_display.best_spark.position_signature).transpose(),
                           np.array(my_display.best_spark.position_exposure).transpose())

FW_result_manhattan = np.average(paired_distances(
    np.array(new_reconstructed).reshape(-1, 1), np.array(signeR_matrix).reshape(-1, 1), metric='manhattan'))
SR_result_manhattan = np.average(paired_distances(
    np.array(old_reconstructed).transpose().reshape(-1, 1), np.array(signeR_matrix).reshape(-1, 1), metric='manhattan'))
FW_result_cosine = np.average(paired_distances(
    np.array(new_reconstructed), np.array(signeR_matrix), metric='cosine'))
SR_result_cosine = np.average(paired_distances(
    np.array(old_reconstructed).transpose(), np.array(signeR_matrix), metric='cosine'))
FW_result_euclidean = np.average(paired_distances(
    np.array(new_reconstructed), np.array(signeR_matrix), metric='euclidean'))
SR_result_euclidean = np.average(paired_distances(
    np.array(old_reconstructed).transpose(), np.array(signeR_matrix), metric='euclidean'))

logging.info("The End.")
