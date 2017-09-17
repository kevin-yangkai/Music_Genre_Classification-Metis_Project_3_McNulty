## Music Genre Classification with Supervised Learning
###Background:
I DJ on the side for fun and one of the tasks that I have to do regularly is go through the tons of tracks that I accumulate and annote them with descriptor tags.  I do this for two main reasons.  First, I have a ton of tracks in my digital library.  I often come across a track that I cannot remember of the top of my head and it is very convenient to have an idea of the genre, the general mood of the track, and if there are any vocals on the track.  Second, if I want to play a series of tracks that follow a specific mood, it is great to be able to sort my library based on mood descriptor tags as well.

So when I begin to sort tracks the first things I do is categorize the tracks by genre and sub-genre. Thus the crux of this project.  In the end, it would be great to have a folder on my hard drive full of tracks of various genres of electronic music.  I would want to carry out two or three rounds of classification on the tracks.  The first pass would be to seperate them into different folders based on predicted genre and the second pass would then look at those folders and further seperate them into sub-genres. If I can achieve high accuracy, then I can then import the tracks into my library and fix the errors on the fly as I encounter them!!  Later, it would be great to include further descriptors such as the presence of vocals.

I have a background in signal processing and found a great tutorial with somewhat dated code by a reseach assistant, at the time I suppose, Alexander Schindler at the Vienna University of Technology in the Institute of Software Technology and Interactive Systems.  The code was all writen in python2 and the scripts for visualization needed some definite polishing.  I updated the code for python3, fixed issues due to moving to python3 in some of the dependencies, added various functionality, and vastly overhauled the visualization.  You will see that I tried to keep the original implementation of the basic math intact from the original version of the code.  In the scope of math implementation, I only alterated datatypes and how the program deals with nans, zeros, and infs as needed.  The tutoral by Alexander Schindler was an excellent source of guidance in learning how to scan and process acoustic data.  So thanks a ton Alexander Schindler!!

For this project I did not go crazy and scan thousands of tracks for training or exaustively classify tracks into all genres of electronic music.  I wanted a proof of concept and a minimal viable product to exand upon in the future.  For genre classification, chose drum and bass, breatbeats, house, and hip hop.  For sub-genre classification I chose drum and bass, attempting to classify the sub-genres of Jungle, Neuro/big energy, and minimal drum and bass.  However to get to that point, much coding and signal processing had to be done!!  Improvements and future features are listed below in the "Still to do" section of this README.

I used various supervised learning algorithms in this project, but it is not exhaustive and many of the boosted algorithms should be explored still.  Here you will see implementation of: K nearest neighbor (KNN), Logistic Regression with regularization, both Gaussian GNB) and Bernoulli Naive Bayes (BNB), both linear and non-linear Singular vector machines (SVM), Decision trees, and Random Forrests.

### Results:  
**genre: 4 group classification**  
**knn:** accuracy 0.67  

**LogReg model using L1 regularization**  
Training Data Accuracy: 0.85  
Test Data Accuracy:     0.66  
Precision:              1.00  
Recall:                 0.89  
F1 Score:                 0.94  

**LogReg model using L2 regularization**  
Training Data Accuracy: 0.92  
Test Data Accuracy:     0.63  
Precision:              0.94  
Recall:                 0.89  
F1 Score:                 0.91  

**GaussianNB**  
Training Data Accuracy: 0.67  
Test Data Accuracy:     0.58  
Precision:              0.75  
Recall:                 0.95  
F1 Score:                 0.84  

**BernoulliNB**  
Training Data Accuracy: 0.60  
Test Data Accuracy:     0.58  
Precision:              0.74  
Recall:                 0.94  
F1 Score:                 0.83  

**Linear SVC**  
Training Data Accuracy: 1.00  
Test Data Accuracy:     0.72  
Precision:              0.94  
Recall:                 0.84  
F1 Score:                 0.89  

**SVC (linear and non-linear)**  
Best parameters:
 {'C': 10.0, 'kernel': 'rbf'} SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,  
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',  
  max_iter=-1, probability=True, random_state=None, shrinking=True,  
  tol=0.001, verbose=False)  
accuracy: 0.71  

**Decision Tree Classifier**  
Training Data Accuracy: 1.00  
Test Data Accuracy:     0.54  
Precision:              0.88  
Recall:                 0.93  
F1 Score:                 0.90  

**Random Forrest Classifier**  
Training Data Accuracy: 0.97  
Test Data Accuracy:     0.59  
Precision:              0.89  
Recall:                 0.89  
F1 Score:                 0.89  

sub-genre: 3 group classification  
**knn:** accuracy 0.71  

**LogReg model using L1 regularization**  
Training Data Accuracy: 0.64  
Test Data Accuracy:     0.59  
Precision:              0.93  
Recall:                 0.67  
F1 Score:                 0.78  

**LogReg model using L2 regularization**  
Training Data Accuracy: 0.82  
Test Data Accuracy:     0.67  
Precision:              0.84  
Recall:                 0.64  
F1 Score:                 0.73  

**GaussianNB**  
Training Data Accuracy: 0.59  
Test Data Accuracy:     0.58  
Precision:              0.89  
Recall:                 0.81  
F1 Score:                 0.85  

**BernoulliNB**  
Training Data Accuracy: 0.62  
Test Data Accuracy:     0.59  
Precision:              0.76  
Recall:                 0.79  
F1 Score:                 0.78  

**SVC (linear and non-linear)**  
Best parameters:  
 {'C': 1.0, 'kernel': 'rbf'} SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,  
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',  
  max_iter=-1, probability=True, random_state=None, shrinking=True,  
  tol=0.001, verbose=False)  
accuracy: 0.65  

**Decision Tree Classifier**  
Training Data Accuracy: 1.00  
Test Data Accuracy:     0.62  
Precision:              0.81  
Recall:                 0.74  
F1 Score:                 0.77  

**Random Tree Classifier**  
Training Data Accuracy: 0.97  
Test Data Accuracy:     0.55  
Precision:              0.73  
Recall:                 0.48  
F1 Score:                 0.58  

### The python scripts:
1. **convert_mp3_to_wav.py -** MP3 to WAV file convertion in bulk.  I'm put this together because most of my tracks are in MP3 form and the open-source nature of this project made me not want to go the MP3 route.  WAVs have no proprietary leanings. 
2. **audio_processing_pipeline.py -** extraction of MFCC information, low level audio track information, and rhythmic features from tracks in bulk
3. **Classifiers.ipynb -** After running bulk processing, this script implements the various supervised learning models for classification.  You will find quantifications of performance metrics and various visualizations.
4.  **visualization_of_audio_features_group_preprocessed.ipynb -** This python notebook is for visually comparing the features across genres and sub-genres as well as visualization of the many processing steps required to compute classification features.  The visualization of processing uses a single track as an examplar.  This script does not do bulk processing of tracks on the fly and requires you to do this prior to running this script using audio_processing_pipeline.py.
5. **visualization_of_audio_features.ipynb -** This script is essentially the same as visualization_of_audio_features_group_preprocessed.ipynb but does do bulk processing of tracks on the fly.  You will diffinetly not want to run this script if you have hundreds of tracks in each training group however if you have a few tracks in each group and you just want a visual estimate on how different the various features are between groups, this is a quick way of doing so.
6. **sound_processing_functions.py -** this is a library where the majority of the functions I put together exist.
7. **mfcc_mp.py** - this library is specifically for computing mfccs.
8. **segment_axis.py -** this library contains code I took from a depreciated package and updated to run with this program using python3.  Sorry, I just cant remember where I grabed it from...
### key packages to install:
1. pydub (also requires installation of ffmpeg.  On linux box: apt-get install ffmpeg)
2. scipy
3. numpy
4. matplotlib (also requires installation of python3-tk. On linux box: apt-get install python3-tk) 

### Still to do
1. OPTIMIZATION!!!  Right now the model is a bit bloated and can be slimmed down and streamlined.  I need to go through my feature set and remove those features that are contrubuting little or no useful information.
2. I am currently analyzing tracks at a very high granularity, which I believe is unnecissary for what I am using these algorithms for.  This may change depending on what categories I am trying to classify tracks into.  I'm alluding to detection an classification of vocals.
3.  Test the ability of this algorithm to detect vocals and to seperate them into groups of tracks containing female vocalist, male vocalists, and both.  Going back to number 2 of my "still to do list", is an analysis of tracks at high granularity going to yield better model performance due to the high frequencies of human voice relative to the musical beats?
4.  Setup automatic movement of files to new folders based on classification.
5.  Setup sequential scanning for various features.  Right now you have to run the script twice.  Onces for genre classification and once for sub-genre classification.
6.  Get together a larger pool of tracks to train classifiers to increase accuracy.
7.  Currently, the script is setup for experimentation.  Specifically, the models are trained on a subset of the available tracks and tested on the remaining tracks.  It would be nice to have a solid model that can be used for everyday use.
8. Set the script up to convert the tracks back to mp3 and then automatically add the descriptor labels.  Better yet, given that the tracks were originally MP3s prior to conversion to WAVs for analysis, it would be great to just find the originalMP3 versions of the tracks and tag those files with descriptors.
9.  It would be nice to forego the conversion from MP3 to WAV.  Ill have to look into this.
10. MORE COMMENTS IN THE CODE!!!

### References:
1. [A great tutorial on processing music](http://www.ifs.tuwien.ac.at/~schindler/lectures/MIR_Feature_Extraction.html) - plenty of the signal processing code used in this project stems directly from this site with significant modification.  The code on the site is written for python2.  I have modified the code to run on python3.  I have vastly updated the visualization and graphing functions as well.
