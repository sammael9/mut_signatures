import json
import logging
import sys
from sklearn.metrics.pairwise import paired_distances
import numpy as np
from src import fireworks, fitness

import pandas as pd

from src.gradient import PGD
from src.watcher import Watcher

logging.basicConfig(filename='example.log', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# Read data
with open('../resources/settings.json') as f:
    settings = json.load(f)

mutational_matrix = pd.read_csv(r'../resources/WGS_PCAWG.96.csv', delimiter=',')
#reconstructed_matrix = pd.read_csv(r'../resources/reconstructed.csv')
cosmic_catalog = pd.read_csv(r'../resources/PCAWG_signatures.csv', delimiter=',')
#contribution_matrix = pd.read_csv(r'../resources/contribution.csv')

#mutational_matrix = mutational_matrix.sort_values(by='Mutation Type', axis=0)
mutational_matrix = mutational_matrix.loc[:, ~mutational_matrix.columns.str.contains('^Unnamed')]
mutational_matrix = mutational_matrix.sort_values(by=['Mutation type', 'Trinucleotide'], axis=0, ascending=[True, True])
#reconstructed_matrix = reconstructed_matrix.sort_values(by='Somatic Mutation Type', axis=0)
cosmic_catalog = cosmic_catalog.sort_values(by='Mutation Type', axis=0)
cosmic_catalog = cosmic_catalog.loc[:, ~cosmic_catalog.columns.str.contains('^Unnamed')]

#my_input = mutational_matrix[settings['sample']].values
my_input = mutational_matrix.drop(['Mutation type','Trinucleotide'], axis=1).values
my_catalog = cosmic_catalog.loc[:, cosmic_catalog.columns.str.contains('^SBS')].values

my_expression = np.dot(np.array(my_catalog).transpose(), np.array(my_input)).transpose()
my_reconstructed = np.dot(np.array(my_catalog), np.array(my_expression).transpose())

MP_result_manhattan = np.average(paired_distances(
    np.array(my_reconstructed).reshape(-1, 1), np.array(my_input).reshape(-1, 1), metric='manhattan'))
MP_result_euclidean = np.average(paired_distances(
    np.array(my_reconstructed).transpose(), np.array(my_input).transpose(), metric='euclidean'))
MP_result_cosine = np.average(paired_distances(
    np.array(my_reconstructed).transpose(), np.array(my_input).transpose(), metric='cosine'))

print("MP_result_manhattan: " + str(MP_result_manhattan))
print("MP_result_euclidean: " + str(MP_result_euclidean))
print("MP_result_cosine: " + str(MP_result_cosine))

#my_reconstructed = reconstructed_matrix[settings['sample2']].values
#my_contribution = contribution_matrix[settings['sample2']].values

if settings['solution_type'] == 'fireworks':
    # Initialize fireworks display for the first input column

    fitness_function = None
    if settings['fitness_function'] == 'manhattan_sum':
        fitness_function = fitness.fitness_manhattan_similarity_sum
    elif settings['fitness_function'] == 'manhattan_avg':
        fitness_function = fitness.fitness_manhattan_similarity_avg
    elif settings['fitness_function'] == 'euclidean':
        fitness_function = fitness.fitness_euclidean_similarity
    elif settings['fitness_function'] == 'cosine':
        fitness_function = fitness.fitness_cosine_similarity
    else:
        logging.error("Failed to initialize fitness function - not selected or invalids")
        exit(1)

    dimension_limit = False
    if settings['dimension_limit'] == "True":
        dimension_limit = True

    myDisplay = fireworks.Display(firework_count=settings['fireworks_count'],
                                  dimensions=settings['dimensions'],
                                  dimensions2=len(settings['sample']),
                                  dimension_min=settings['dimension_min'],
                                  dimension_max=settings['dimension_max'],
                                  diameter_min=settings['diameter_min'],
                                  diameter_max=settings['diameter_max'],
                                  spark_min=settings['spark_min'],
                                  spark_max=settings['spark_max'],
                                  gaussian_spark_count=settings['gaussian_spark_count'],
                                  fitness_function=fitness_function,
                                  catalog=my_catalog,
                                  mutational_data=my_input,
                                  spark_dimension_count=settings["spark_dimension_count"],
                                  dimension_limit=dimension_limit)

    myDisplay.create_fireworks(None)
    myDisplay.showtime()

    watcher = Watcher(iterations=settings['watcher_iterations'], threshold=settings['watcher_threshold'],
                      starting_iteration=settings['watcher_starting_iteration'], fw_display=myDisplay)

    watcher.iterate(for_range=range(0, settings['watcher_range']), reduction=settings['watcher_reduction'])

    logging.info("Finished iterating. Comparing solutions.")

    new_reconstructed = np.dot(np.array(my_catalog), np.array(myDisplay.best_spark.position).transpose())

    FW_result_manhattan = np.average(paired_distances(
        np.array(new_reconstructed).reshape(-1, 1), np.array(my_input).reshape(-1, 1), metric='manhattan'))
    MP_result_manhattan = np.average(paired_distances(
        np.array(my_reconstructed).reshape(-1, 1), np.array(my_input).reshape(-1, 1), metric='manhattan'))
    FW_result_euclidean = np.average(paired_distances(
        np.array(new_reconstructed).transpose(), np.array(my_input).transpose(), metric='euclidean'))
    MP_result_euclidean = np.average(paired_distances(
        np.array(my_reconstructed).transpose(), np.array(my_input).transpose(), metric='euclidean'))
    FW_result_cosine = np.average(paired_distances(
        np.array(new_reconstructed).transpose(), np.array(my_input).transpose(), metric='cosine'))
    MP_result_cosine = np.average(paired_distances(
        np.array(my_reconstructed).transpose(), np.array(my_input).transpose(), metric='cosine'))

    print("FW_result_manhattan: " + str(FW_result_manhattan))
    print("MP_result_manhattan: " + str(MP_result_manhattan))
    print("FW_result_euclidean: " + str(FW_result_euclidean))
    print("MP_result_euclidean: " + str(MP_result_euclidean))
    print("FW_result_cosine: " + str(FW_result_cosine))
    print("MP_result_cosine: " + str(MP_result_cosine))

elif settings['solution_type'] == 'gradient':
    myPGD = PGD(dimensions=settings['dimensions'],
                dimensions2=my_input.shape[1],
                dimension_min=settings['dimension_min'],
                dimension_max=settings['dimension_max'],
                alpha=settings['alpha'],
                catalog=None,
                mutational_data=my_input)

    expressions, catalog = np.array(myPGD.run_de_novo(settings['iterations']))
    new_reconstructed = np.dot(catalog, expressions.transpose())

    logging.info("Finished iterating. Comparing solutions.")

    print("catalog")
    print(catalog)
    print("expressions")
    print(expressions.transpose())

    FW_result_manhattan = np.average(paired_distances(
        np.array(new_reconstructed).reshape(-1, 1), np.array(my_input).reshape(-1, 1), metric='manhattan'))
    MP_result_manhattan = np.average(paired_distances(
        np.array(my_reconstructed).reshape(-1, 1), np.array(my_input).reshape(-1, 1), metric='manhattan'))
    FW_result_euclidean = np.average(paired_distances(
        np.array(new_reconstructed).transpose(), np.array(my_input).transpose(), metric='euclidean'))
    MP_result_euclidean = np.average(paired_distances(
        np.array(my_reconstructed).transpose(), np.array(my_input).transpose(), metric='euclidean'))
    FW_result_cosine = np.average(paired_distances(
        np.array(new_reconstructed).transpose(), np.array(my_input).transpose(), metric='cosine'))
    MP_result_cosine = np.average(paired_distances(
        np.array(my_reconstructed).transpose(), np.array(my_input).transpose(), metric='cosine'))

    print("PGA_result_manhattan: " + str(FW_result_manhattan))
    print("MP_result_manhattan: " + str(MP_result_manhattan))
    print("PGA_result_euclidean: " + str(FW_result_euclidean))
    print("MP_result_euclidean: " + str(MP_result_euclidean))
    print("PGA_result_cosine: " + str(FW_result_cosine))
    print("MP_result_cosine: " + str(MP_result_cosine))

    np.savetxt(r'../resources/output_catalog_2780.csv', catalog, delimiter=",")
    np.savetxt(r'../resources/output_expressions_2780.csv', expressions.transpose(), delimiter=",")

logging.info("The End.")

