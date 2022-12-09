# DetectSimilarFaces
The labelled faces dataset of sckit-learn contains gray scale images of 62 differnet famous personalites
from politics.we assume that there are no target labels, i.e. the names of
the persons are *unknown*. We want to find a method to cluster similar images. This can be done
using a _dimensionality reduction algorithm like PCA_ for feature generation and a subsequent
clustering e.g. using `DBSCAN`
we would have to take this into account. We extract the first 50 images of each person and put them into a flat array called X_people. The correspinding targets (y-values, names), are storeed in the y_people array.
Apply a principal component analysis `X_pca=pca.fit_transform(X_people)` and
extract the first 100 components of each image. Reconstruct the first 10 entries of the dataset
using the 100 components of the PCA transformed data by applying the
`pca.inverse_transform` method and reshaping the image to the original size using
np.reshape.
# Apply DBSCAN on these features
Import DBSCAN class from sklearn.cluster, generate an instance called dbscan and
apply it to the pca transformed data X_pca and extract the cluster labels using labels
= dbscan.fit_predict(X_pca). Useing first the standard parameters for the method and
check how many unique clusters the algorithm could find by analyzing the number of
unique entries in the predicted cluster labels.
Change the parameter eps of the dbscan using `dbscan(min_samples=3, eps=5)`. Change
the value of eps in the range from 5 to 10 in steps of 0.5 using a for loop and check for
each value of eps how many clusters could be determined.
Select the value of `eps` where the numbers of clusters found is maximum and plot the
members of the clusters found using the follwing python code.
# using other cluster algorithms learner on the pca transformed data
