import logging
import os
import sys
from sklearn.metrics.pairwise import paired_distances
import numpy as np
from src import fireworks, fitness

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

chosen = 'intestine2'

my_input = mutational_matrix[chosen].values
my_catalog = cosmic_catalog.loc[:, cosmic_catalog.columns.str.contains('^Signature')].values
my_reconstructed = reconstructed_matrix[chosen].values
my_contribution = contribution_matrix[chosen].values

# Initialize fireworks display for the first input column

diameter_max = 100
diameter_min = 15

myDisplay = fireworks.Display(firework_count=15, dimensions=30, dimension_min=0, dimension_max=1000,
                              diameter_min=15, diameter_max=100, spark_min=10, spark_max=50, gaussian_spark_count=10,
                              fitness_function=fitness.fitness_manhattan_similarity_sum, catalog=my_catalog,
                              mutational_data=my_input, spark_dimension_count=10, dimension_limit=False)
myDisplay.create_fireworks(None)
myDisplay.showtime()

watcher = Watcher(iterations=15, threshold=5, starting_iteration=10, fw_display=myDisplay)

watcher.iterate(for_range=range(0, 500), reduction=0.99)

logging.info("Finished iterating. Comparing solutions.")

new_reconstructed = np.dot(np.array(my_catalog), np.array(myDisplay.best_spark.position))

FW_result = np.average(paired_distances(
    np.array(new_reconstructed).reshape(-1, 1), np.array(my_input).reshape(-1, 1), metric='manhattan'))
MP_result = np.average(paired_distances(
    np.array(my_reconstructed).reshape(-1, 1), np.array(my_input).reshape(-1, 1), metric='manhattan'))

logging.info("The End.")

