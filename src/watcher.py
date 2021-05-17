import logging


class Watcher:
    def __init__(self, iterations, threshold, starting_iteration, fw_display):
        self.iterations = iterations
        self.threshold = threshold
        self.starting_iteration = starting_iteration
        self.fw_display = fw_display

    def iterate(self, for_range, reduction):
        fitness_list = []
        for i in for_range:
            diameter_max = self.fw_display.diameter_max * reduction
            diameter_min = self.fw_display.diameter_min * reduction
            new_starting_positions = self.fw_display.prepare_new_iteration(False, diameter_max, diameter_min)
            self.fw_display.create_fireworks(new_starting_positions)
            self.fw_display.showtime()
            logging.debug("Completed iteration no. " + str(i))
            if i >= self.starting_iteration:
                fitness_list.append(self.fw_display.best_spark.fitness)
            if len(fitness_list) == self.iterations:
                if max(fitness_list) - min(fitness_list) < self.threshold:
                    logging.debug("No significant improvement for the last " + str(self.iterations) + " iterations.")
                    logging.debug("Stopping at fitness " + str(self.fw_display.best_spark.fitness)
                                 + " at iteration no. " + str(i))
                    break
                else:
                    fitness_list.pop(0)
