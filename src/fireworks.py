import logging
import random
from math import pi
from operator import attrgetter as atg
import numpy as np


class Display:

    def __init__(self, firework_count, dimensions, dimension_min, dimension_max, diameter_min, diameter_max, spark_min,
                 spark_max, gaussian_spark_count, fitness_function, catalog, mutational_data, spark_dimension_count,
                 dimension_limit):
        self.firework_count = firework_count
        self.dimensions = dimensions
        self.dimension_min = dimension_min
        self.dimension_max = dimension_max
        self.diameter_min = diameter_min
        self.diameter_max = diameter_max
        self.spark_min = spark_min
        self.spark_max = spark_max
        self.gaussian_spark_count = gaussian_spark_count
        self.fitness_function = fitness_function
        self.catalog = catalog
        self.mutational_data = mutational_data
        logging.info("Created new FireworksDisplay")
        logging.info("Firework Count: " + str(firework_count))
        logging.info("Dimensions: " + str(dimensions))
        logging.info("Dimension min max: " + str(dimension_min) + " " + str(dimension_max))
        logging.info("Firework Diameter min max " + str(diameter_min) + " " + str(diameter_max))
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
                diameter = random.uniform(self.diameter_min, self.diameter_max)
                spark_count = self.count_sparks(diameter)
                firework_dimensions = []
                for _ in range(0, self.dimensions):
                    firework_dimensions.append(random.uniform(self.dimension_min, self.dimension_max))
                self.fireworks.append(Firework(diameter, spark_count, firework_dimensions, self.dimension_max,
                                               self.dimension_min, self.spark_dimension_count, self.dimension_limit))
        else:
            for index, spark in enumerate(selected_positions):
                logging.info("Firework " + str(index + 1) + " / " + str(self.firework_count))
                diameter = random.uniform(self.diameter_min, self.diameter_max)
                spark_count = self.count_sparks(diameter)
                self.fireworks.append(Firework(diameter, spark_count, spark.position, self.dimension_max,
                                               self.dimension_min, self.spark_dimension_count, self.dimension_limit))

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
        self.diameter_max = new_diameter_max
        self.diameter_min = new_diameter_min
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
        positions = []
        for index, dimension in enumerate(spark.position):
            positions.append(random.uniform(dimension, self.best_solution.best_spark.position[index]))
        if self.dimension_limit:
            to_keep_indexes = random.sample(range(0, len(positions)), random.randint(
                len(positions) - self.spark_dimension_count, len(positions) - self.spark_dimension_count * 0.5))
            for index in to_keep_indexes:
                positions[index] = 0
        return Spark(positions)

    def count_sparks(self, diameter):
        max_area = self.diameter_max * self.diameter_max * pi
        min_area = self.diameter_min * self.diameter_min * pi
        cur_area = diameter * diameter * pi
        spark_count = int(
            round(self.spark_max - cur_area / ((max_area - min_area) / (self.spark_max - self.spark_min))))
        return spark_count

    def evaluate(self, fitness_function):
        for firework in self.fireworks:
            for spark in firework.sparks:
                spark.fitness = fitness_function(self.catalog, spark.position, self.mutational_data)
            firework.find_best_spark()

    def evaluate_gaussian(self, fitness_function):
        for spark in self.gaussian_sparks:
            spark.fitness = fitness_function(self.catalog, spark.position, self.mutational_data)

    def archive(self):
        self.history.append(self.fireworks)
        logging.info("Archiving current state. History size: " + str(len(self.history)))

    def find_best_solution(self):
        self.best_solution = max(self.fireworks, key=lambda firework: firework.best_spark.fitness)


class Firework:

    def __init__(self, diameter, spark_count, dimensions, dimension_max, dimension_min, spark_dimension_count,
                 dimension_limit):
        logging.info("Created firework")
        logging.info("Diameter " + str(diameter))
        logging.info("Spark count " + str(spark_count))
        logging.info("Position " + str(dimensions))
        self.diameter = diameter
        self.spark_count = spark_count
        self.dimensions = dimensions
        self.dimension_max = dimension_max
        self.dimension_min = dimension_min
        self.best_spark = None
        self.best_spark_fitness = -1
        self.sparks = []
        self.spark_dimension_count = spark_dimension_count
        self.dimension_limit = dimension_limit

    def explode(self):
        for i in range(0, self.spark_count):
            position = []
            for dimension in self.dimensions:
                if dimension + self.diameter > self.dimension_max:
                    if dimension - self.diameter < self.dimension_min:
                        position.append(random.uniform(self.dimension_min, self.dimension_max))
                    else:
                        position.append(random.uniform(dimension - self.diameter, self.dimension_max))
                else:
                    if dimension - self.diameter < self.dimension_min:
                        position.append(random.uniform(self.dimension_min, dimension + self.diameter))
                    else:
                        position.append(random.uniform(dimension - self.diameter, dimension + self.diameter))
            if self.dimension_limit:
                to_keep_indexes = random.sample(range(0, len(position)), random.randint(
                    len(position) - self.spark_dimension_count, len(position) - self.spark_dimension_count * 0.5))
                for index in to_keep_indexes:
                    position[index] = 0
            self.sparks.append(Spark(position))

    def find_best_spark(self):
        self.best_spark = max(self.sparks, key=atg('fitness'))
        self.best_spark_fitness = self.best_spark.fitness
        return self.best_spark


class Spark:

    def __init__(self, position):
        self.fitness = -1000000
        self.position = []
        # logging.info("Created Spark at " + str(position))
        self.position = position
