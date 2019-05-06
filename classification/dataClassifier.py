# This file contains feature extraction methods and harness
# code for data classification

import naiveBayes
import mira
import perceptron
import samples
import sys
import util
import math
import time
import random

TEST_SET_SIZE = 200
DIGIT_DATUM_WIDTH = 28
DIGIT_DATUM_HEIGHT = 28
FACE_DATUM_WIDTH = 60
FACE_DATUM_HEIGHT = 70


def basicFeatureExtractorDigit(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is white (0) or gray/black (1)
    """
    a = datum.getPixels()
    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x, y)] = 1
            else:
                features[(x, y)] = 0
    return features


def basicFeatureExtractorFace(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is an edge (1) or no edge (0)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(FACE_DATUM_WIDTH):
        for y in range(FACE_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x, y)] = 1
            else:
                features[(x, y)] = 0
    return features


def enhancedFeatureExtractorDigit(datum):
    """
    Your feature extraction playground.
    You should return a util.counter() of features
    for this datum (datum is of type samples.Datum).
    ## DESCRIBE YOUR ENHANCED FEATURES HERE...
    ##
    """

    features = basicFeatureExtractorDigit(datum)
    return features


def enhancedFeatureExtractorFace(datum):
    """
    Your feature extraction playground for faces.
    It is your choice to modify this.
    """

    features = basicFeatureExtractorFace(datum)
    return features

def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.
    Use the printImage(<list of pixels>) function to visualize features.
    An example of use has been given to you.
    - classifier is the trained classifier
    - guesses is the list of labels predicted by your classifier on the test set
    - testLabels is the list of true labels
    - testData is the list of training datapoints (as util.Counter of features)
    - rawTestData is the list of training datapoints (as samples.Datum)
    - printImage is a method to visualize the features
    (see its use in the odds ratio part in runClassifier method)
    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    for i in range(len(guesses)):
        prediction = guesses[i]
        truth = testLabels[i]
        if prediction != truth:
            print("===================================")
            print("Mistake on example %d" % i)
            print("Predicted %d; truth is %d" % (prediction, truth))
            print("Image: ")
            print(rawTestData[i])
            break


## =====================
## You don't have to modify any code below.
## =====================

class ImagePrinter:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def printImage(self, pixels):
        """
        Prints a Datum object that contains all pixels in the
        provided list of pixels.  This will serve as a helper function
        to the analysis function you write.
        Pixels should take the form
        [(2,2), (2, 3), ...]
        where each tuple represents a pixel.
        """
        image = samples.Datum(None, self.width, self.height)
        for pix in pixels:
            try:
                # This is so that new features that you could define which
                # which are not of the form of (x,y) will not break
                # this image printer...
                x, y = pix
                image.pixels[x][y] = 2
            except:
                print("new features:", pix)
                continue
        print(image)


def default(str):
    return str + ' [Default: %default]'


def readCommand(argv):
    """Processes the command used to run from the command line."""
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)


    parser.add_option('-r', '--run',  help=default('automatically runs training and test cycle for 5 times'),
                      default= False, action='store_true')

    parser.add_option('-c', '--classifier', help=default('The type of classifier'),
                      choices=['perceptron', 'naiveBayes', 'mira'],
                      default='naiveBayes')
    parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces'], default='digits')
    parser.add_option('-t', '--training', help=default('The ratio of the training set to use'), default=1.0,
                      type="float")
    parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False,
                      action="store_true")
    parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False,
                      action="store_true")
    parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
    parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
    parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"),
                      type="float", default=2.0)
    parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False,
                      action="store_true")
    parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print("Doing classification")
    print("--------------------")
    print("data:\t\t" + options.data)
    print("classifier:\t\t" + options.classifier)
    print("using enhanced features?:\t" + str(options.features))

    if options.data == "digits":
        printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
        if options.features:
            featureFunction = enhancedFeatureExtractorDigit
        else:
            featureFunction = basicFeatureExtractorDigit
    elif options.data == "faces":
        printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
        if options.features:
            featureFunction = enhancedFeatureExtractorFace
        else:
            featureFunction = basicFeatureExtractorFace
    else:
        print("Unknown dataset", options.data)
        print(USAGE_STRING)
        sys.exit(2)

    if options.data == "digits":
        legalLabels = range(10)
    else:
        legalLabels = range(2)

    if options.training <= 0:
        print("Training set size should be a positive integer (you provided: %d)" % options.training)
        print(USAGE_STRING)
        sys.exit(2)

    if options.smoothing <= 0:
        print("Please provide a positive number for smoothing (you provided: %f)" % options.smoothing)
        print(USAGE_STRING)
        sys.exit(2)

    if options.odds:
        if options.label1 not in legalLabels or options.label2 not in legalLabels:
            print("Didn't provide a legal labels for the odds ratio: (%d,%d)" % (options.label1, options.label2))
            print(USAGE_STRING)
            sys.exit(2)

    if options.classifier == "mira":
        classifier = mira.MiraClassifier(legalLabels, options.iterations)
    elif options.classifier == "naiveBayes":
        classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
        classifier.setSmoothing(options.smoothing)
        if options.autotune:
            print
            "using automatic tuning for naivebayes"
            classifier.automaticTuning = True
        else:
            print("using smoothing parameter k=%f for naivebayes" % options.smoothing)
    elif options.classifier == "perceptron":
        classifier = perceptron.PerceptronClassifier(legalLabels, options.iterations)

    elif options.classifier == "knn":
        classifier = knn.KNN(legalLabels)

    else:
        print("Unknown classifier:", options.classifier)
        print(USAGE_STRING)

        sys.exit(2)

    args['classifier'] = classifier
    args['featureFunction'] = featureFunction
    args['printImage'] = printImage

    return args, options


USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """


# Main harness code

def runClassifier(args, options):
    featureFunction = args['featureFunction']
    classifier = args['classifier']
    printImage = args['printImage']

    # Load data
    trainingFactor = options.training

    print "training factor {}".format(trainingFactor)

    if options.data == "faces":
        TEST_SET_SIZE = 150

        numTraining = int(451 * trainingFactor)

        print "using {} datapoints out of {} ({}%) for faces".format(numTraining, 451, 100 * (numTraining/float(451)))

        rawTrainingData = samples.loadDataFile("facedata/facedatatrain", numTraining, FACE_DATUM_WIDTH,
                                               FACE_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTraining)
        rawValidationData = samples.loadDataFile("facedata/facedatatrain", TEST_SET_SIZE, FACE_DATUM_WIDTH,
                                                 FACE_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", TEST_SET_SIZE)
        rawTestData = samples.loadDataFile("facedata/facedatatest", TEST_SET_SIZE, FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", TEST_SET_SIZE)
    else:
        TEST_SET_SIZE = 1000

        numTraining = int(5000 * trainingFactor)

        print "using {} datapoints out of {} ({}%) for digits".format(numTraining, TEST_SET_SIZE, 100 * (numTraining/float(5000)))

        rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining, DIGIT_DATUM_WIDTH,
                                               DIGIT_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
        rawValidationData = samples.loadDataFile("digitdata/validationimages", TEST_SET_SIZE, DIGIT_DATUM_WIDTH,
                                                 DIGIT_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("digitdata/validationlabels", TEST_SET_SIZE)
        rawTestData = samples.loadDataFile("digitdata/testimages", TEST_SET_SIZE, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("digitdata/testlabels", TEST_SET_SIZE)

    # Extract features
    print("Extracting features...")
    trainingData = map(featureFunction, rawTrainingData)
    validationData = map(featureFunction, rawValidationData)
    testData = map(featureFunction, rawTestData)

    # Conduct training and testing
    automatic = options.run
    if automatic:
        outcomes = {}
        for iterator in range(5):
            print("Training...")

            if options.data == "faces":
                rawTrainingData = samples.loadDataFile("facedata/facedatatrain", 451, FACE_DATUM_WIDTH,
                                                       FACE_DATUM_HEIGHT)
                trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", 451)

                indices = []

                for x in range(numTraining):
                    indices.append(random.randint(0, 450))

                randomTrainingData = []
                randomTrainingLabels = []
                for index in indices:
                    randomTrainingData.append(rawTrainingData[index])
                    randomTrainingLabels.append(trainingLabels[index])

            else:
                rawTrainingData = samples.loadDataFile("digitdata/trainingimages", 5000, DIGIT_DATUM_WIDTH,
                                                       DIGIT_DATUM_HEIGHT)
                trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", 5000)

                indices = []

                for x in range(numTraining):
                    indices.append(random.randint(0, 4999))

                randomTrainingData = []
                randomTrainingLabels = []

                for index in indices:
                    randomTrainingData.append(rawTrainingData[index])
                    randomTrainingLabels.append(trainingLabels[index])


            trainingData = map(featureFunction, randomTrainingData)

            start = time.time()
            classifier.train(trainingData, randomTrainingLabels, validationData, validationLabels)
            print("Validating...")
            guesses = classifier.classify(validationData)
            correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
            print(str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (
                    100.0 * correct / len(validationLabels)))
            print("Testing...")
            guesses = classifier.classify(testData)
            correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
            print(str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels)))
            analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)
            interval = time.time() - start
            print "Training and testing time: " + str(interval)
            outcomes[str(iterator)] = ["Training and testing time: {}".format(interval), "accuracy of training: {}%".format((100.0 * correct / len(testLabels)))]

        print "outcomes: {}".format(outcomes)

    else:
        print("Training...")
        start = time.time()
        classifier.train(trainingData, trainingLabels, validationData, validationLabels)
        interval = time.time() - start
        print "Training time: " + str(interval)
        print("Validating...")
        guesses = classifier.classify(validationData)
        correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
        print(str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (
                100.0 * correct / len(validationLabels)))
        print("Testing...")
        guesses = classifier.classify(testData)
        correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
        print(
        str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels)))
        analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)


    # do odds ratio computation if specified at command line
    if options.odds & (options.classifier != "mostFrequent"):
        label1, label2 = options.label1, options.label2
        features_odds = classifier.findHighOddsFeatures(label1, label2)
        if options.classifier == "naiveBayes" or options.classifier == "nb":
            string3 = "=== Features with highest odd ratio of label %d over label %d ===" % (label1, label2)
        else:
            string3 = "=== Features for which weight(label %d)-weight(label %d) is biggest ===" % (label1, label2)

        print(string3)
        printImage(features_odds)


if __name__ == '__main__':
    # Read input
    args, options = readCommand(sys.argv[1:])
    # Run classifier
    runClassifier(args, options)

