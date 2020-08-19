from . import dists
import csv

class Searcher:
    def __init__(self,dbPath):
        # store the database path
        self.dbPath = dbPath

    def search(self, queryFeatures, numResults=10):
        # initialize the results dictionary
        results = {}
        # open the database for reading
        with open(self.dbPath) as f:
            # initialize the CSV reader
            reader = csv.reader(f)

            # loop over the rows in the index
            for row in reader:
                # parse out the image ID and features, then compute the chi-squared
                # distance between the features in our database and the query features
                features = [float(x) for x in row[1:]]
                d = dists.chi2_distance(features,queryFeatures)

                results[row[0]] = d

        # sort the results, so that the smaller distances (i.e. more relevant images)
        # are at the front of the list
        results = sorted([(v, k) for (k, v) in results.items()])

        # return the results
        return results[:numResults]

