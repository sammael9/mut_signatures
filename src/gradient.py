import logging
import random

import numpy as np


class PGD:

    def __init__(self, catalog, mutational_data, dimensions, dimensions2, dimension_min, dimension_max, alpha, beta):
        self.dimensions2 = dimensions2
        self.dimensions = dimensions
        self.dimension_max = dimension_max
        self.dimension_min = dimension_min
        self.mutational_data = np.array(mutational_data)
        if catalog is None:
            cat_tmp = []
            for _ in range(0, self.mutational_data.shape[0]):
                cat_tmp_dim = []
                for _ in range(0, self.dimensions):
                    cat_tmp_dim.append(random.uniform(0, 0.1))
                cat_tmp.append(cat_tmp_dim)
            self.catalog = np.array(cat_tmp)
        else:
            self.catalog = np.array(catalog)
        self.iteration = 0
        tmp_xpr = []
        for _ in range(0, self.dimensions2):
            expression_dimension = []
            for _ in range(0, self.dimensions):
                expression_dimension.append(0)
            tmp_xpr.append(expression_dimension)
        self.exposures = np.array(tmp_xpr)
        tmp = []
        for _ in range(0, self.dimensions2):
            tmp_dimension = []
            for _ in range(0, self.dimensions):
                tmp_dimension.append(random.uniform(0, 100))
            tmp.append(tmp_dimension)
        self.residuals = np.array(tmp)
        self.result = np.array(tmp)
        self.alpha = alpha
        self.beta = beta
        logging.debug("Initialized catalog")
        logging.debug(self.catalog)
        logging.debug("Initialized exposures")
        logging.debug(self.exposures)

    def run_de_novo(self, iterations):
        while self.iteration < iterations:
            logging.debug("Iteration no. " + str(self.iteration))
            logging.debug("Catalog matrix")
            logging.debug(self.catalog)
            logging.debug("Exposure matrix")
            logging.debug(self.exposures)
            self.result = np.dot(self.catalog, self.exposures.transpose())
            self.residuals = self.mutational_data - self.result

            self.exposures = self.exposures + (
                        self.alpha * np.dot(self.catalog.transpose(), self.residuals)).transpose()

            self.result = np.dot(self.catalog, self.exposures.transpose())
            self.residuals = self.mutational_data - self.result

            self.catalog = abs(self.catalog * (1 + self.beta * np.dot(self.residuals, self.exposures)))

            np.clip(self.exposures, a_min=0, a_max=self.dimension_max, out=self.exposures)
            np.clip(self.catalog, a_min=0, a_max=1.0, out=self.catalog)

            self.iteration = self.iteration + 1

        return self.exposures, self.catalog

    def run_fitting(self, iterations):
        while self.iteration < iterations:
            self.iteration = self.iteration + 1
            logging.debug("Iteration no. " + str(self.iteration))
            logging.debug("Exposure matrix")
            logging.debug(self.exposures)
            self.result = np.dot(self.catalog, self.exposures.transpose())
            self.residuals = self.mutational_data - self.result

            self.exposures = self.exposures + (self.alpha * np.dot(self.catalog.transpose(),
                                                                   self.residuals)).transpose()

            np.clip(self.exposures, a_min=0, a_max=self.dimension_max, out=self.exposures)

        return self.exposures
