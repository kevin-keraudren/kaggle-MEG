#!/usr/bin/python
"""
Adapted from:

http://martinos.org/mne/stable/auto_examples/decoding/plot_decoding_sensors.html#example-decoding-plot-decoding-sensors-py

"""

import matplotlib
matplotlib.use('cairo')
import matplotlib.pyplot as plt

import pylab as pl

import numpy as np
from scipy.io import loadmat

from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score, ShuffleSplit

import joblib
from joblib import Parallel, delayed

def do_t(X, y, t):
    clf = SVC(C=1, kernel='linear')
    
    # Define a monte-carlo cross-validation generator (reduce variance):
    cv = ShuffleSplit(len(X), 10, test_size=0.2)
    
    Xt = X[:, :, t]
    # Standardize features
    Xt -= Xt.mean(axis=0)
    Xt /= Xt.std(axis=0)
    # Run cross-validation
    # Note : for sklearn the Xt matrix should be 2d (n_samples x
    # n_features)
    scores_t = cross_val_score(clf, Xt, y, cv=cv, n_jobs=1)

    return scores_t.mean(),scores_t.std()

all_scores = []
all_std = []
for subject in xrange(1,17):
    filename = 'data/train_subject%02d.mat' % subject
    print "Loading", filename
    data = loadmat(filename, squeeze_me=True)
    X = data['X']
    y = data['y']

    # X = X[:10]
    # y = y[:10]

    n_times = X.shape[2]

    mean_std = Parallel(n_jobs=-1)(delayed(do_t)(X,y,t) for t in range(n_times))

    scores = np.array( [s[0] for s in mean_std] )
    std_scores = np.array( [s[1] for s in mean_std] )
    
    all_scores.append(scores)
    all_std.append( std_scores)

scores = np.array(all_scores).mean(axis=0)
#std_scores = np.array(all_std).mean(axis=0)
std_scores = np.array(all_scores).std(axis=0)

np.save("numpy_data/mean_time_selection.npy", scores)
np.save("numpy_data/std_time_selection.npy", std_scores)

times = np.linspace( -0.5, 1.0, num=375 )    
scores *= 100  # make it percentage
std_scores *= 100
print len(times),len(scores)
plt.plot(times, scores, label="Classif. score")
plt.axhline(50, color='k', linestyle='--',
            label="Chance level")
plt.axvline(0, color='r', label='stim onset')
plt.legend()
hyp_limits = (scores - std_scores, scores +
              std_scores)
plt.fill_between(times, hyp_limits[0],
                 y2=hyp_limits[1], color='b', alpha=0.5)
plt.xlabel('Times (ms)')
plt.ylabel('CV classification score (% correct)')
#plt.ylim([30, 100])
plt.title('Sensor space decoding')
plt.savefig('img/time_selection.png' )

