import logging
import random

import numpy as np


class PGD:

    def __init__(self, catalog, mutational_data, dimensions, dimensions2, dimension_min, dimension_max, alpha):
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
        self.expressions = np.array(tmp_xpr)
        tmp = []
        for _ in range(0, self.dimensions2):
            tmp_dimension = []
            for _ in range(0, self.dimensions):
                tmp_dimension.append(random.uniform(0, 100))
            tmp.append(tmp_dimension)
        self.residuals = np.array(tmp)
        self.result = np.array(tmp)
        self.alpha = alpha

        print("catalog")
        print(self.catalog)
        print("expressions")
        print(self.expressions)

    def run_de_novo(self, iterations):
        while self.iteration < iterations:
            logging.info("Iteration no. " + str(self.iteration))

            self.result = np.dot(self.catalog, self.expressions.transpose())
            self.residuals = self.mutational_data - self.result

            self.expressions = self.expressions + (self.alpha * np.dot(self.catalog.transpose(), self.residuals)).transpose()

            self.result = np.dot(self.catalog, self.expressions.transpose())
            self.residuals = self.mutational_data - self.result

            self.catalog = abs(self.catalog * (1 + (self.alpha / 5000000000) * np.dot(self.residuals, self.expressions)))

            np.clip(self.expressions, a_min=0, a_max=1000000, out=self.expressions)
            np.clip(self.catalog, a_min=0, a_max=1.0, out=self.catalog)

           # for i in range(0, self.dimensions2):
           #     smallest = self.expressions[i].argsort()[:40]
           #     self.expressions[i][smallest] = 0

       #     if self.iteration == 10000:
       #

       #np.savetxt(r'../resources/output_catalog_' + str(self.iteration) + '.csv', self.catalog, delimiter=",")

            self.iteration = self.iteration + 1

        return self.expressions, self.catalog

    def run_fitting(self, iterations):
        while self.iteration < iterations:
            self.iteration = self.iteration + 1
            logging.info("Iteration no. " + str(self.iteration))
            self.result = np.dot(self.catalog, self.expressions.transpose())
            self.residuals = self.mutational_data - self.result

            self.expressions = self.expressions + (self.alpha * np.dot(self.catalog.transpose(),
                                                                       self.residuals)).transpose()

            np.clip(self.expressions, a_min=0, a_max=1000000, out=self.expressions)

            for i in range(0, self.dimensions2):
                smallest = self.expressions[i].argsort()[:50]
                self.expressions[i][smallest] = 0

        return self.expressions
