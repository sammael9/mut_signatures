import json
import logging
import sys
from sklearn.metrics.pairwise import paired_distances
import numpy as np
from src import fireworks, fitness

import pandas as pd

from src.gradient import PGD
from src.watcher import Watcher

# This script is an example how to use the FA and PGD algorithms

logging.basicConfig(filename='example.log', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# First we load some parameters from a json file, this can be done in script by passing the args manually or we can
# have them stored like this

with open('../resources/settings.json') as f:
    settings = json.load(f)

# Now we need to load the input data. This can be up to 3 matrices (mutational matrix, catalog matrix and opportunity)
# although only 1 is required for de novo (mutational) and 2 for fitting (mutational + catalog). We must assure that
# the input data is aligned, i.e. depending on how the mutational type is described (it can be one field with values
# like A[C>T]A or it can be two columns, one for mutation type other for context). We must sort them, in order for the
# algorithm to properly function. The output will have the mutational signatures and types in the same order as in
# the input. In this example we also load a reconstructed matrix from a different algorithm to compare to.

mutational_matrix = pd.read_csv(r'../resources/mut_mat.csv', delimiter=',')
reconstructed_matrix = pd.read_csv(r'../resources/reconstructed.csv', delimiter=",")
cosmic_catalog = pd.read_csv(r'../resources/cancer_signatures.csv', delimiter=',')

# Here we keep preparing the data, sorting etc.

mutational_matrix = mutational_matrix.loc[:, ~mutational_matrix.columns.str.contains('^Unnamed')]
mutational_matrix = mutational_matrix.sort_values(by='Somatic Mutation Type', axis=0, ascending=True)
reconstructed_matrix = reconstructed_matrix.loc[:, ~reconstructed_matrix.columns.str.contains('^Unnamed')]
reconstructed_matrix = reconstructed_matrix.sort_values(by='Somatic Mutation Type', axis=0, ascending=True)
cosmic_catalog = cosmic_catalog.loc[:, ~cosmic_catalog.columns.str.contains('^Unnamed')]
cosmic_catalog = cosmic_catalog.sort_values(by='Somatic Mutation Type', axis=0, ascending=True)

# Finally, we only need the actual values in form of 2d arrays / matrices

my_input = mutational_matrix[settings['sample']].values
my_catalog = cosmic_catalog.loc[:, cosmic_catalog.columns.str.contains('^Signature')].values
my_reconstructed = reconstructed_matrix[settings['sample']].values

# Here we also calculate the avergae manhattan, euclidean and cosine distances for the reconstructed data samples,
# we use MP as shortcut for MutationalPatterns software which produced it

MP_result_manhattan = np.average(paired_distances(
    np.array(my_reconstructed).reshape(-1, 1), np.array(my_input).reshape(-1, 1), metric='manhattan'))
MP_result_euclidean = np.average(paired_distances(
    np.array(my_reconstructed).transpose(), np.array(my_input).transpose(), metric='euclidean'))
MP_result_cosine = np.average(paired_distances(
    np.array(my_reconstructed).transpose(), np.array(my_input).transpose(), metric='cosine'))

print("MP_result_manhattan: " + str(MP_result_manhattan))
print("MP_result_euclidean: " + str(MP_result_euclidean))
print("MP_result_cosine: " + str(MP_result_cosine))

if settings['solution_type'] == 'fireworks':

    # Initialize fireworks display. 

    fitness_function = None
    if settings['fitness_function'] == 'manhattan_sum':
        fitness_function = fitness.fitness_manhattan_similarity_sum
    elif settings['fitness_function'] == 'manhattan_avg':
        fitness_function = fitness.fitness_manhattan_similarity_avg
    elif settings['fitness_function'] == 'euclidean':
        fitness_function = fitness.fitness_euclidean_similarity
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

    # Now we could iterate one iteration after another, but we are using a Watcher class to handle this.
    # It will stop iterating once the iteration count exceeds a set amount or there is no improvement over the course
    # of several iterations. To run it, we first make an initialized state using showtime() function and then let
    # the watcher handle the rest.
    myDisplay.showtime()

    watcher = Watcher(iterations=settings['watcher_iterations'], threshold=settings['watcher_threshold'],
                      starting_iteration=settings['watcher_starting_iteration'], fw_display=myDisplay)

    watcher.iterate(for_range=range(0, settings['watcher_range']), reduction=settings['watcher_reduction'])

    logging.info("Finished iterating. Comparing solutions.")

    # Now we reconstruct the initial matrix from the best solution we obtained compare it

    new_reconstructed = np.dot(np.array(my_catalog), np.array(myDisplay.best_spark.position).transpose())

    FW_result_manhattan = np.average(paired_distances(
        np.array(new_reconstructed).reshape(-1, 1), np.array(my_input).reshape(-1, 1), metric='manhattan'))
    FW_result_euclidean = np.average(paired_distances(
        np.array(new_reconstructed).transpose(), np.array(my_input).transpose(), metric='euclidean'))
    FW_result_cosine = np.average(paired_distances(
        np.array(new_reconstructed).transpose(), np.array(my_input).transpose(), metric='cosine'))

    print("FW_result_manhattan_distance: " + str(FW_result_manhattan))
    print("MP_result_manhattan_distance: " + str(MP_result_manhattan))
    print("FW_result_euclidean_distance: " + str(FW_result_euclidean))
    print("MP_result_euclidean_distance: " + str(MP_result_euclidean))
    print("FW_result_cosine_distance: " + str(FW_result_cosine))
    print("MP_result_cosine_distance: " + str(MP_result_cosine))

    # We save the data and results

    np.savetxt(r'../resources/output_fitted_exposures_FW.csv', myDisplay.best_spark.position, delimiter=",")

    results = {"FW_result_manhattan_distance": FW_result_manhattan, "FW_result_euclidean_distance": FW_result_euclidean,
               "FW_result_cosine_distance": FW_result_cosine, "settings": settings}

    with open(r'../resources/output_distances_FW.json', 'w') as fp:
        json.dump(results, fp)

elif settings['solution_type'] == 'gradient':

    # Initialize projected gradient descent. Here we can do it for both de novo and fitting.

    myPGD1 = PGD(dimensions=settings['dimensions'],
                 dimensions2=my_input.shape[1],
                 dimension_min=settings['dimension_min'],
                 dimension_max=settings['dimension_max'],
                 alpha=settings['alpha'],
                 beta=settings['beta'],
                 catalog=None,
                 mutational_data=my_input)

    # We run the algorithm and collect the output data, reconstruct the matrix.

    expressions, catalog = np.array(myPGD1.run_de_novo(settings['iterations']))

    new_reconstructed_de_novo = np.dot(catalog, expressions.transpose())

    # Silimar for the fitting treatment.

    myPGD2 = PGD(dimensions=settings['dimensions'],
                 dimensions2=my_input.shape[1],
                 dimension_min=settings['dimension_min'],
                 dimension_max=settings['dimension_max'],
                 alpha=settings['alpha'],
                 beta=settings['beta'],
                 catalog=my_catalog,
                 mutational_data=my_input)

    fitted = np.array(myPGD2.run_fitting(settings['iterations']))

    new_reconstructed_fitting = np.dot(my_catalog, fitted.transpose())

    logging.info("Finished iterating. Comparing solutions.")

    # We calculate the distances

    FW_result_manhattan_de_novo = np.average(paired_distances(
        np.array(new_reconstructed_de_novo).reshape(-1, 1), np.array(my_input).reshape(-1, 1), metric='manhattan'))
    FW_result_euclidean_de_novo = np.average(paired_distances(
        np.array(new_reconstructed_de_novo).transpose(), np.array(my_input).transpose(), metric='euclidean'))
    FW_result_cosine_de_novo = np.average(paired_distances(
        np.array(new_reconstructed_de_novo).transpose(), np.array(my_input).transpose(), metric='cosine'))

    FW_result_manhattan_fit = np.average(paired_distances(
        np.array(new_reconstructed_fitting).reshape(-1, 1), np.array(my_input).reshape(-1, 1), metric='manhattan'))
    FW_result_euclidean_fit = np.average(paired_distances(
        np.array(new_reconstructed_fitting).transpose(), np.array(my_input).transpose(), metric='euclidean'))
    FW_result_cosine_fit = np.average(paired_distances(
        np.array(new_reconstructed_fitting).transpose(), np.array(my_input).transpose(), metric='cosine'))

    print("PGD_result_manhattan_de_novo_distances: " + str(FW_result_manhattan_de_novo))
    print("PGD_result_manhattan_fit_distances: " + str(FW_result_manhattan_fit))
    print("MP_result_manhattan: " + str(MP_result_manhattan))
    print("PGD_result_euclidean_de_novo_distances: " + str(FW_result_euclidean_de_novo))
    print("PGD_result_euclidean_fit_distances: " + str(FW_result_euclidean_fit))
    print("MP_result_euclidean: " + str(MP_result_euclidean))
    print("PGD_result_cosine_de_novo_distances: " + str(FW_result_cosine_de_novo))
    print("PGD_result_cosine_fit_distances: " + str(FW_result_cosine_fit))
    print("MP_result_cosine: " + str(MP_result_cosine))

    # We store the data and the results

    np.savetxt(r'../resources/output_catalog_PGD.csv', catalog, delimiter=",")
    np.savetxt(r'../resources/output_exposures_PGD.csv', expressions.transpose(), delimiter=",")
    np.savetxt(r'../resources/output_fitted_exposures_PGD.csv', expressions.transpose(), delimiter=",")

    results = {"PGD_result_manhattan_de_novo_distances": FW_result_manhattan_de_novo,
               "PGD_result_euclidean_de_novo_distances": FW_result_euclidean_de_novo,
               "PGD_result_cosine_de_novo_distances": FW_result_cosine_de_novo, "settings": settings}

    with open(r'../resources/output_distances_FW_de_novo.json', 'w') as fp:
        json.dump(results, fp)

    results2 = {"PGD_result_manhattan_fit_distances": FW_result_manhattan_fit,
                "PGD_result_euclidean_fit_distances": FW_result_euclidean_fit,
                "PGD_result_cosine_fit_distances": FW_result_cosine_fit, "settings": settings}

    with open(r'../resources/output_distances_FW_fit.json', 'w') as fp:
        json.dump(results2, fp)

logging.info("Done.")
