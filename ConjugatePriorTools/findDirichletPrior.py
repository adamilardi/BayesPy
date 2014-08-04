#!/usr/bin/python
#
# Finding the optimal dirichlet prior from counts
# By: Max Sklar
# @maxsklar
# https://github.com/maxsklar

# Copyright 2012 Max Sklar

# A sample of a file to pipe into this python script is given by test.csv

# ex
# cat test.csv | ./finaDirichletPrior.py --sampleRate 1

# Paper describing the basic formula:
# http://research.microsoft.com/en-us/um/people/minka/papers/dirichlet/minka-dirichlet.pdf

# Each columns is a different category, and it is assumed that the counts are pulled out of
# a different distribution for each row.
# The distribution for each row is pulled from a Dirichlet distribution; this script finds that
# dirichlet which maximizes the probability of the output.

# Parameter: the first param is the sample rate.  This is to avoid using the full data set when we
# have huge amounts of data.

import sys
import csv
import random
import time
import dirichletMultinomialEstimation as DME
import samplingTools as Sample
from optparse import OptionParser
import logging


def main(K, iterations, H, input_stream, sampleRate, M):
	startTime = time.time()
	logging.debug("K = " + str(K))
	logging.debug("iterations = " + str(iterations))
	logging.debug("H = " + str(H))
	logging.debug("sampleRate = " + str(sampleRate))
	logging.debug("M = " + str(M))

	# TODO(max): write up a paper describing the hyperprior and link it.
	W = 0
	Beta = [0]*K
	Hstr = H.split(",")
	hasHyperprior = False
	if (len(Hstr) == K + 1):
		for i in range(0, K): Beta[i] = float(Hstr[i])
		W = float(Hstr[K])
		hasHyperprior = True
	else:
		Beta = None
		W = None

	logging.debug("Beta = " + str(Beta))
	logging.debug("W = " + str(W))
	
	#####
	# Load Data
	#####
	csv.field_size_limit(1000000000)
	reader = csv.reader(input_stream, delimiter='\t')
	logging.debug("Loading data")
	priors = [0.]*K

	dataObj = DME.CompressedRowData(K)

	idx = 0
	for row in reader:
		idx += 1

		if (random.random() < float(sampleRate)):
			data = map(int, row)
			if (len(data) != K):
				logging.error("There are %s categories, but line has %s counts." % (K, len(data)))
				logging.error("line %s: %s" % (i, data))
			
			
			while sum(data) > M: data[Sample.drawCategory(data)] -= 1
			
			sumData = sum(data)
			weightForMean = 1.0 / (1.0 + sumData)
			for i in range(0, K): priors[i] += data[i] * weightForMean
			dataObj.appendRow(data, 1)

		if (idx % 1000000) == 0: logging.debug("Loading Data: %s rows done" % idx)

	dataLoadTime = time.time()
	logging.debug("loaded %s records into memory" % idx)
	logging.debug("time to load memory: %s " % (dataLoadTime - startTime))

	for row in dataObj.U:
		if len(row) == 0 and not hasHyperprior:
			# TODO(max): write up a paper describing the hyperprior and link it.
			raise Exception("You can't have any columns with all 0s, unless you provide a hyperprior (-H)")

	priorSum = sum(priors) + 0.01 # Nudge to prevent zero
	for i in range(0, K):
	  priors[i] /= priorSum
	  priors[i] += 0.01 # Nudge to prevent zero

	priors = DME.findDirichletPriors(dataObj, priors, iterations, Beta, W)	

	# print "Final priors: ", priors
	logging.debug("Final average loss: %s" % DME.getTotalLoss(priors, dataObj, Beta, W))

	totalTime = time.time() - dataLoadTime
	logging.debug("Time to calculate: %s" % totalTime)
	return priors


if __name__ == '__main__':
	parser = OptionParser()
	parser.add_option('-s', '--sampleRate', dest='sampleRate', default='1', help='Randomly sample this fraction of rows')
	parser.add_option('-K', '--numCategories', dest='K', default='2', help='The number of (tab separated) categories that are being counted')
	parser.add_option('-M', '--maxCountPerRow', dest='M', type=int, default=sys.maxint, help='The maximum number of the count per row.  Setting this lower increases the running time')
	parser.add_option("-L", '--loglevel', action="store", dest="loglevel", default='DEBUG', help="don't print status messages to stdout")
	parser.add_option('-H', '--hyperPrior', dest='H', default="", help='The hyperprior of the Dirichlet (paper coming soon!) comma separated K+1 values (Beta then W)')
	parser.add_option('-i', '--iterations', dest='iterations', default='50', help='How many iterations to do')

	(options, args) = parser.parse_args()

	#Set the log level
	log_level = options.loglevel
	numeric_level = getattr(logging, log_level, None)
	if not isinstance(numeric_level, int):
	    raise ValueError('Invalid log level: %s' % log_level)
	logging.basicConfig(level=numeric_level)

	main(int(options.K), int(options.iterations), options.H, sys.stdin, options.sampleRate, options.M)

