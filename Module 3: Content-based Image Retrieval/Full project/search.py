from __future__ import print_function
from project.descriptors import DetectAndDescribe
from project.ir import BagOfVisualWords
from project.ir import Searcher
from project.ir import chi2_distance
from project import ResultsMontage
from scipy.spatial import distance
from redis import Redis
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
import argparse
import pickle
import imutils
import json
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory of indexed images")
ap.add_argument("-f", "--features-db", required=True, help="Path to the features database")
ap.add_argument("-b", "--bovw-db", required=True, help="Path to the bag-of-visual-words database")
ap.add_argument("-c", "--codebook", required=True, help="Path to the codebook")
ap.add_argument("-i", "--idf", type=str, help="Path to inverted document frequencies array")
ap.add_argument("-r", "--relevant", required=True, help = "Path to relevant dictionary")
ap.add_argument("-q", "--query", required=True, help="Path to the query image")
args = vars(ap.parse_args())

# initialize the keypoint detector, local invariant descriptor, descriptor pipeline,
# distance metric, and inverted document frequency array
detector = FeatureDetector_create("SURF")
descriptor = DescriptorExtractor_create("RootSIFT")
dad = DetectAndDescribe(detector, descriptor)
distanceMetric = chi2_distance
idf = None

# if the path to the inverted document frequency array was supplied, then load the
# idf array and update the distance metric

if args["idf"] is not None:
    idf = pickle.loads(open(args["idf"], "rb").read())
    distanceMetric = distance.cosine

# load the codebook vocabulary and initialize the bag-of-visual-words transformer
vocab = pickle.loads(open(args["codebook"], "rb").read())
bovw = BagOfVisualWords(vocab)

# load the relevant queries dictionary and lookup the relevant results for the query image
relevant = json.loads(open(args["relevant"]).read())
queryFilename = args["query"][args["query"].rfind("/") + 1:]
queryRelevant = relevant[queryFilename]

# load the query image and process it
queryImage = cv2.imread(args["query"])
queryImage = imutils.resize(queryImage, width=320)
cv2.imshow("Query", queryImage)
queryImage = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)

# extract features from the query image and construct a bag-of-visual-words from it
(_, descs) = dad.describe(queryImage)
hist = bovw.describe(descs).tocoo()

# connect to Redis and perform the search
redisDB = Redis(host='localhost', port=6379, db=0)
searcher = Searcher(redisDB, args["bovw_db"], args["features_db"], idf=idf, distanceMetric=distanceMetric)
sr = searcher.search(hist, numResults=20)
print(f"[INFO] search took {sr.search_time:.2f}s")

# initialize the results montage
montage = ResultsMontage((240,320), 5, 20)

# loop over the individual results
for i, (score, resultID, resultIdx) in enumerate(sr.results):
    # load the result image and display it
    print(f"[RESULT] {i+1}. {resultID} - {score:.2f}")
    result = cv2.imread(f"{args['dataset']}/{resultID}")
    montage.addResult(result, text=f"#{i+1}", highlight=resultID in queryRelevant)

# show the output image of results
cv2.imshow("Results", imutils.resize(montage.montage, height=700))
cv2.waitKey(0)
searcher.finish()



