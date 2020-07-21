from __future__ import print_function
import argparse
import dlib

ap = argparse.ArgumentParser()
ap.add_argument("-x", "--xml", required=True, help="Path to input XML file")
ap.add_argument("-d", "--detector", required=True, help="Path to output detector")
args = vars(ap.parse_args())

# grab the default training options for the HOG + Linear SVM detector, then
# train the detector -- in practice, the `C` parameter should be cross-validated
print("[INFO] training detector...")

options = dlib.simple_object_detector_training_options()
options.C = 1.0
options.num_threads = 8
options.be_verbose = True
dlib.train_simple_object_detector(args["xml"], args["detector"], options)

print(f"[INFO] training accuracy: {dlib.test_simple_object_detector(args['xml'], args['detector'])}")

# load the detector and visualize the HOG filter
detector = dlib.simple_object_detector(args["detector"])
win = dlib.image_window()
win.set_image(detector)
dlib.hit_enter_to_continue()