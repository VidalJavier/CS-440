import util
import classificationMethod
from collections import Counter
from collections import defaultdict
import math


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
        self.classes = None
        self.prior_prob = None
        self.likelihoods = None

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

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

    def _occurrences(self, outcome):
        no_of_examples = len(outcome)
        prob = dict(Counter(outcome))
        for key in prob.keys():
            prob[key] = prob[key] / float(no_of_examples)
        return prob

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
        no_of_examples = len(trainingLabels)
        prior_prob = dict(Counter(trainingLabels))
        for key in prior_prob.keys():
            prior_prob[key] = prior_prob[key] / float(no_of_examples)

        self.prior_prob = prior_prob

        likelihoods = dict()
        for cls, prob in prior_prob.items():
            # initializing the dictionary
            likelihoods[cls] = defaultdict(list)

        for cls, prob in prior_prob.items():
            # taking samples of only 1 class at a time
            row_indices = list()
            for index, value in enumerate(trainingLabels):
                if value == cls:
                    row_indices.append(index)

            subset = list()
            for index in row_indices:
                subset.append(trainingData[index])

            for r in range(len(subset)):
                for key, value in subset[r].items():
                    likelihoods[cls][key].append(value)

        classes = [key for key in prior_prob]
        self.classes = classes
        _like = likelihoods
        for cls in classes:
            for key, value in likelihoods[cls].items():
                likelihoods[cls][key] = self._occurrences(likelihoods[cls][key])

        self.likelihoods = likelihoods

        # results = {}
        # correct = 0
        # for itr in range(len(validationData)):
        #     for cls in classes:
        #         class_probability = prior_prob[cls]
        #         for key, value in validationData[itr].items():
        #             relative_feature_values = likelihoods[cls][key]
        #             class_probability *= relative_feature_values.get(validationData[itr][key], 0.01)
        #
        #         results[cls] = class_probability
        #
        #     norm_factor = 0.0
        #
        #     for key, value in results.items():
        #         norm_factor += value
        #
        #     for key in results:
        #         try:
        #             results[key] = results[key]/norm_factor
        #         except ZeroDivisionError:
        #             pass
        #
        #     if (list(results.keys())[list(results.values()).index(max([value for key, value in results.items()]))]) == validationLabels[itr]:
        #         correct += 1
        #
        # print "validation accuracy: {}%".format((correct/float(len(validationLabels))) * 100)


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
        for cls in self.classes:
            class_probability = self.prior_prob[cls]
            for key, value in datum.items():
                relative_feature_values = self.likelihoods[cls][key]
                class_probability += math.log(relative_feature_values.get(datum[key], 0.01))

            logJoint[cls] = class_probability

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
