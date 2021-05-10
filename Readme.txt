Bc. Filip Varga - AIS ID 79794
MUTATIONAL SIGNATURES ESTIMATION TOOL USING MODIFIED FIREWORK ALGORITHM


To run the project, execute main.py in source folder using a python 3.6 with all loaded libraries in requirements.txt.

Logs will be saved to example.log file in src folder.

In the project, you need to set and assign which file you are analyzing.
The default file with 9 samples is mut_mat.csv.
Default catalog is cancer_signatures or cosmic_mutations csv files.
Reconstructed output from the MutationalPatterns software is reconstructed.csv - expression matrix is in contribution.csv.
Second input file is 21_breast_cancers.mutations.txt, also csv format.

Settings.json serve to configure the algorithm.

{
  "sample": ["colon1"],
  "fireworks_count": 15,
  "dimensions": 30,
  "dimension_min": 0,
  "dimension_max": 1000,
  "diameter_min": 15,
  "diameter_max": 100,
  "spark_min": 10,
  "spark_max": 50,
  "gaussian_spark_count": 10,
  "fitness_function": "cosine",
  "spark_dimension_count": 12,
  "dimension_limit": "False",
  "watcher_iterations": 15,
  "watcher_threshold": 0.001,
  "watcher_starting_iteration": 10,
  "watcher_range": 250,
  "watcher_reduction": 0.98
}

sample - list of column names of samples to analyze
fireworks_count - number of fireworks in iteration
dimensions - number of signatures (must correspond to signature count in your catalog)
dimension_min - minimum value for a dimension, should be 0 for NMF
dimension_max - maximum value for a dimension, by default we use 1000 based on outputs from MutationalPatterns software
spark_min - number of sparks generated at maximum diameter
spark_max - number of sparks generated at minimum diameter
gaussian_spark_count - number of gaussian sparks generated after firework sparks are generated and evaluated
fitness_function - fitness function, can be manhattan_avg, manhattan_sum, euclidean or cosine, best performance on manhattan_sum and euclidean
spark_dimension_count - the limit of dimensions in which a spark can exist (actual limit will be random between this number and halfway between this number and number of dimensions)
dimension_limit - activates dimension limiting (spark_dimension_count has no meaning without it)
watcher_iterations - shows the evaluated dimension interval by the watcher, the higher the number, the more strict it is
watcher_threshold - threshold for the min-max difference of the interval, needs to be set individually for each fitness function (ideally around 1 for manhattan_sum, 0.001, cosine, 0.01 manhattan_avg and 0.1 for euclidean)
watcher_starting_iteration - at which iteration watcher starts checking the progress
watcher_range - max number of iterations, 100-1000 expected
watcher_reduction - alfa reduction of min and max diameters per iteration, ideally number between 0.95 - 0.995

Results for tests are in results_testing_vector.json and results_testing_matrix.json.
We evaluated manhattan_avg, euclidean_avg and cosine_avg for each.

MP - mutationalpatterns (state-of-the-art software)
FW - our algorithm

The application prints these results upon finishing.

Overview of project structure:

ROOT folder
Readme.txt - this file
results_testing_matrix/vector.json - results from my tests
requirements.txt - required libraries, IDEs such as pycharm should install automatically
data_analysis - some basic analysis done in jupyter notebook

SRC folder
__init__.py - just python stuff
fireworks.py - the framework of the algorithm with classes Display, Spark and Firework
fitness.py - various fitness functions
main.py - the main script that loads data and runs the algorithm, outputs result to console
watcher.py - helper class to monitor the progress of the algorithm, iterate it and stop it when needed

RESOURCES folder
21_breast_cancers.mutations.txt - an extra sample input file with 21 samples
21_breast_cancers.opportunity.txt - additional input to 21_breast_cancers (divides the matrix elementwise)
cancer_signatures.csv - cosmic_mutations.csv - various formats of the COSMIC catalog with 30 signatures
contribution.csv - MutationalPatterns example output of Expression matrix
mut_mat.csv - main input file we used in evaluations with 9 samples
reconstructed.csv - dot product of COSMIC catalog and contribution.csv
settings.json - settings for the algorithm