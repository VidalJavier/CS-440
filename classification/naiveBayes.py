import util
import classificationMethod
import math
import collections

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.
    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1  # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False  # Look at this flag to decide whether to choose k automatically ** use this in your train method **
        self.odds = None
        self.cats = None
        self.prior = None

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def occurr(self, result):
        prob = dict(collections.Counter(result))
        for key in prob.keys():
            prob[key] = prob[key] / float(len(result))
        return prob

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        self.features = trainingData[0].keys()  # this could be useful for your code later...

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):

        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.
        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.
        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        
        prior = dict(collections.Counter(trainingLabels))
        for key in prior.keys():
            prior[key] = prior[key] / float(len(trainingLabels))

        self.prior = prior

        odds = dict()
        for c, prob in prior.items():
            odds[c] = collections.defaultdict(list)

        for c, prob in prior.items():
            row = list()
            for index, val in enumerate(trainingLabels):
                if c == val:
                    row.append(index)

            subset = list()
            for index in row:
                subset.append(trainingData[index])

            for r in range(len(subset)):
                for key, val in subset[r].items():
                    odds[c][key].append(val)

        cats = [k for k in prior]

        _odds = odds
        for c in cats:
            for key, val in odds[c].items():
                odds[c][key] = self.occurr(odds[c][key])

        self.odds = odds
        self.cats = cats
    
    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.
        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = []  # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
        """
        logJoint = util.Counter()
        
        "*** YOUR CODE HERE ***"
        for c in self.cats:
            catProb = self.prior[c]
            for key, val in datum.items():
                relFeatVal = self.odds[c][key]
                catProb += math.log(relFeatVal.get(datum[key], 0.01))

            logJoint[c] = catProb

        return logJoint

    def findHighOddsFeatures(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
                P(feature=1 | label1)/P(feature=1 | label2)
        """
        featuresOdds = []

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

        return featuresOdds
