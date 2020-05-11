import logging
import random
from math import pi
from operator import attrgetter as atg


class Display:

    def __init__(self, firework_count, dimensions_exp_1, dimensions_exp_2, dimensions_sig_1, dimensions_sig_2,
                 dimension_exp_min, dimension_exp_max, dimension_sig_min, dimension_sig_max, diameter_exp_min,
                 diameter_exp_max, spark_min, spark_max, gaussian_spark_count,
                 fitness_function, catalog, mutational_data, spark_dimension_count, dimension_limit):
        self.firework_count = firework_count
        self.dimensions_exp_1 = dimensions_exp_1
        self.dimensions_exp_2 = dimensions_exp_2
        self.dimensions_sig_1 = dimensions_sig_1
        self.dimensions_sig_2 = dimensions_sig_2
        self.dimension_exp_min = dimension_exp_min
        self.dimension_exp_max = dimension_exp_max
        self.dimension_sig_min = dimension_sig_min
        self.dimension_sig_max = dimension_sig_max
        self.diameter_exp_min = diameter_exp_min
        self.diameter_exp_max = diameter_exp_max
        self.diameter_ratio = 0.001
        self.spark_min = spark_min
        self.spark_max = spark_max
        self.gaussian_spark_count = gaussian_spark_count
        self.fitness_function = fitness_function
        self.catalog = catalog
        self.mutational_data = mutational_data
        logging.info("Created new FireworksDisplay")
        logging.info("Firework Count: " + str(firework_count))
        logging.info("Dimensions: " + str(dimensions_exp_1))
        logging.info("Dimension min max: " + str(dimension_exp_min) + " " + str(dimension_exp_max))
        logging.info("Firework Diameter min max " + str(diameter_exp_min) + " " + str(diameter_exp_max))
        logging.info("Spark min max " + str(spark_min) + " " + str(spark_max))
        logging.info("Gaussian Spark Count: " + str(gaussian_spark_count))
        self.history = []
        self.fireworks = []
        self.iteration = 0
        self.best_solution = None
        self.best_spark = None
        self.gaussian_sparks = []
        self.spark_dimension_count = spark_dimension_count
        self.dimension_limit = dimension_limit

    def create_fireworks(self, selected_positions):
        self.fireworks = []
        self.gaussian_sparks = []
        self.best_solution = None
        logging.info("Instantiating fireworks")
        if selected_positions is None:
            for i in range(0, self.firework_count):
                logging.info("Firework " + str(i + 1) + " / " + str(self.firework_count))
                diameter = random.uniform(self.diameter_exp_min, self.diameter_exp_max)
                spark_count = self.count_sparks(diameter)
                firework_dimensions_exp = []
                for _ in range(0, self.dimensions_exp_2):
                    firework_dimension_list = []
                    for _ in range(0, self.dimensions_exp_1):
                        firework_dimension_list.append(random.uniform(self.dimension_exp_min, self.dimension_exp_max))
                    firework_dimensions_exp.append(firework_dimension_list)
                firework_dimensions_sig = []
                for _ in range(0, self.dimensions_sig_2):
                    firework_dimension_list = []
                    for _ in range(0, self.dimensions_sig_1):
                        firework_dimension_list.append(random.uniform(self.dimension_sig_min, self.dimension_sig_max))
                    firework_dimensions_sig.append(firework_dimension_list)
                self.fireworks.append(
                    Firework(diameter, diameter * self.diameter_ratio, spark_count, firework_dimensions_exp,
                             firework_dimensions_sig, self.dimension_exp_max, self.dimension_exp_min,
                             self.dimension_sig_max, self.dimension_sig_min, self.spark_dimension_count,
                             self.dimension_limit))
        else:
            for index, spark in enumerate(selected_positions):
                logging.info("Firework " + str(index + 1) + " / " + str(self.firework_count))
                diameter = random.uniform(self.diameter_exp_min, self.diameter_exp_max)
                spark_count = self.count_sparks(diameter)
                self.fireworks.append(
                    Firework(diameter, diameter * self.diameter_ratio, spark_count, spark.position_exposure,
                             spark.position_signature, self.dimension_exp_max, self.dimension_exp_min,
                             self.dimension_sig_max, self.dimension_sig_min, self.spark_dimension_count,
                             self.dimension_limit))

    def showtime(self):
        self.iteration += 1
        logging.info("Showtime, iteration no. " + str(self.iteration))
        for firework in self.fireworks:
            firework.explode()
        logging.info("Finding best solutions.")
        self.evaluate(self.fitness_function)
        self.find_best_solution()
        to_select = self.fireworks.copy()
        to_select.remove(self.best_solution)
        selected_solutions = random.choices(to_select, k=self.gaussian_spark_count)
        logging.info("Generating " + str(self.gaussian_spark_count) + " Gaussian sparks.")
        for selected_solution in selected_solutions:
            self.gaussian_sparks.append(self.generate_gaussian_spark(selected_solution.best_spark))
        self.evaluate_gaussian(self.fitness_function)
        logging.info("Showtime over.")

    def prepare_new_iteration(self, random_selection, new_diameter_max, new_diameter_min):
        self.diameter_exp_max = new_diameter_max
        self.diameter_exp_min = new_diameter_min
        position_list = list(map(lambda x: x.best_spark, self.fireworks))
        position_list = position_list + self.gaussian_sparks
        sorted_list = sorted(position_list, key=lambda spark: spark.fitness, reverse=True)
        logging.info("Current best solution has fitness: " + str(sorted_list[0].fitness))
        self.best_spark = sorted_list[0]
        if random_selection:
            return random.sample(sorted_list, self.firework_count)
        else:
            return sorted_list[0:self.firework_count]

    def generate_gaussian_spark(self, spark):
        all_positions_exposure = []
        for index1, dimension_list in enumerate(spark.position_exposure):
            positions = []
            for index2, dimension in enumerate(dimension_list):
                positions.append(
                    random.uniform(dimension, self.best_solution.best_spark.position_exposure[index1][index2]))
            if self.dimension_limit:
                to_keep_indexes = random.sample(range(0, len(positions)), random.randint(
                    len(positions) - self.spark_dimension_count, len(positions) - self.spark_dimension_count * 0.5))
                for index in to_keep_indexes:
                    positions[index] = 0
            all_positions_exposure.append(positions)
        all_positions_signature = []
        for index1, dimension_list in enumerate(spark.position_signature):
            positions = []
            for index2, dimension in enumerate(dimension_list):
                positions.append(
                    random.uniform(dimension, self.best_solution.best_spark.position_signature[index1][index2]))
            if self.dimension_limit:
                to_keep_indexes = random.sample(range(0, len(positions)), random.randint(
                    len(positions) - self.spark_dimension_count, len(positions) - self.spark_dimension_count * 0.5))
                for index in to_keep_indexes:
                    positions[index] = 0
            all_positions_signature.append(positions)
        return Spark(all_positions_exposure, all_positions_signature)

    def count_sparks(self, diameter):
        max_area = self.diameter_exp_max * self.diameter_exp_max * pi
        min_area = self.diameter_exp_min * self.diameter_exp_min * pi
        cur_area = diameter * diameter * pi
        spark_count = int(
            round(self.spark_max - cur_area / ((max_area - min_area) / (self.spark_max - self.spark_min))))
        return spark_count

    def evaluate(self, fitness_function):
        for firework in self.fireworks:
            for spark in firework.sparks:
                spark.fitness = fitness_function(spark.position_signature, spark.position_exposure,
                                                 self.mutational_data)
            firework.find_best_spark()

    def evaluate_gaussian(self, fitness_function):
        for spark in self.gaussian_sparks:
            spark.fitness = fitness_function(spark.position_signature, spark.position_exposure, self.mutational_data)

    def archive(self):
        self.history.append(self.fireworks)
        logging.info("Archiving current state. History size: " + str(len(self.history)))

    def find_best_solution(self):
        self.best_solution = max(self.fireworks, key=lambda firework: firework.best_spark.fitness)


class Firework:

    def __init__(self, diameter_exp, diameter_sig, spark_count, dimensions_exposure, dimensions_signature,
                 dimension_exp_max,
                 dimension_exp_min, dimension_sig_max, dimension_sig_min, spark_dimension_count,
                 dimension_limit):
        logging.info("Created firework")
        logging.info("Diameter " + str(diameter_exp))
        logging.info("Spark count " + str(spark_count))
        logging.info("Position exposure " + str(dimensions_exposure))
        logging.info("Position signature " + str(dimensions_signature))
        self.diameter_exp = diameter_exp
        self.diameter_sig = diameter_sig
        self.spark_count = spark_count
        self.dimensions_exposure = dimensions_exposure
        self.dimensions_signature = dimensions_signature
        self.dimension_exp_max = dimension_exp_max
        self.dimension_exp_min = dimension_exp_min
        self.dimension_sig_max = dimension_sig_max
        self.dimension_sig_min = dimension_sig_min
        self.best_spark = None
        self.best_spark_fitness = -1
        self.sparks = []
        self.spark_dimension_count = spark_dimension_count
        self.dimension_limit = dimension_limit

    def explode(self):
        for i in range(0, self.spark_count):
            all_positions_exposure = []
            for dimension_list in self.dimensions_exposure:
                position = []
                for dimension in dimension_list:
                    if dimension + self.diameter_exp > self.dimension_exp_max:
                        if dimension - self.diameter_exp < self.dimension_exp_min:
                            position.append(random.uniform(self.dimension_exp_min, self.dimension_exp_max))
                        else:
                            position.append(random.uniform(dimension - self.diameter_exp, self.dimension_exp_max))
                    else:
                        if dimension - self.diameter_exp < self.dimension_exp_min:
                            position.append(random.uniform(self.dimension_exp_min, dimension + self.diameter_exp))
                        else:
                            position.append(
                                random.uniform(dimension - self.diameter_exp, dimension + self.diameter_exp))
                if self.dimension_limit:
                    to_keep_indexes = random.sample(range(0, len(position)), random.randint(
                        len(position) - self.spark_dimension_count, len(position) - self.spark_dimension_count * 0.5))
                    for index in to_keep_indexes:
                        position[index] = 0
                all_positions_exposure.append(position)
            all_positions_signature = []
            for dimension_list in self.dimensions_signature:
                position = []
                for dimension in dimension_list:
                    if dimension + self.diameter_sig > self.dimension_sig_max:
                        if dimension - self.diameter_sig < self.dimension_sig_min:
                            position.append(random.uniform(self.dimension_sig_min, self.dimension_sig_max))
                        else:
                            position.append(random.uniform(dimension - self.diameter_sig, self.dimension_sig_max))
                    else:
                        if dimension - self.diameter_sig < self.dimension_sig_min:
                            position.append(random.uniform(self.dimension_sig_min, dimension + self.diameter_sig))
                        else:
                            position.append(
                                random.uniform(dimension - self.diameter_sig, dimension + self.diameter_sig))
                if self.dimension_limit:
                    to_keep_indexes = random.sample(range(0, len(position)), random.randint(
                        len(position) - self.spark_dimension_count, len(position) - self.spark_dimension_count * 0.5))
                    for index in to_keep_indexes:
                        position[index] = 0
                all_positions_signature.append(position)
            self.sparks.append(Spark(all_positions_exposure, all_positions_signature))

    def find_best_spark(self):
        self.best_spark = max(self.sparks, key=atg('fitness'))
        self.best_spark_fitness = self.best_spark.fitness
        return self.best_spark


class Spark:

    def __init__(self, position_exposure, position_signature):
        self.fitness = -1000000
        self.position_exposure = []
        self.position_signature = []
        self.position_exposure = position_exposure
        self.position_signature = position_signature
