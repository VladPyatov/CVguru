from __future__ import print_function
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import datetime
import h5py


class Vocabulary:
    def __init__(self, dbPath, verbose=True):
        # store the verbosity setting ans database path
        self.dbPath = dbPath
        self.verbose = verbose

    def fit(self, numClusters, samplePercent, randomState = None):
        # open the database and grab the total number of features
        db = h5py.File(self.dbPath, mode='r')
        totalFeatures = db["features"].shape[0]

        # determine the number of features to sample, generate the indexes of the
        # sample, sorting them in the ascending order to speed up access time from the
        # HDF5 database
        sampleSize = int(np.ceil(samplePercent * totalFeatures))
        idxs = np.random.choice(np.arange(0, totalFeatures), sampleSize, replace=False)
        idxs.sort()
        data = []
        self._debug("starting sampling...")

        # loop over the randomly sampled indexes and accumulate the features to cluster
        for i in idxs:
            data.append(db["features"][i][2:])

        # cluster the data
        self._debug(f"sampled {len(idxs):,} features from a population of {totalFeatures:,}")
        self._debug(f"clustering with k={numClusters:,}")
        clt = MiniBatchKMeans(n_clusters=numClusters, random_state=randomState)
        clt.fit(data)
        self._debug(f"cluster shape: {clt.cluster_centers_.shape}")

        # close the database
        db.close()

        # return the cluster centroids
        return clt.cluster_centers_

    def _debug(self, msg, msgType="[INFO]"):
        # check to see the message should be printed
        if self.verbose:
            print(f"{msgType} {msg} - {datetime.datetime.now()}")


