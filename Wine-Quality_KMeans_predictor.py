# Yasmin Gil
# yasmingi@usc.edu
# Wine Quality KMeans predictor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier

def WineQualityAnalysis():
    # read data into frame, extract Quality and drop wine & quality columns
    wine_quality_raw = pd.read_csv('wineQualityReds.csv')
    quality_df = wine_quality_raw['quality']
    wine_quality = wine_quality_raw.drop(['Wine', 'quality'], axis=1)

    # normalize the columns
    normalize = Normalizer()
    normalize.fit(wine_quality)
    wine_quality = pd.DataFrame(normalize.transform(wine_quality), columns=wine_quality.columns)
    print(wine_quality.head())

    # range for clustering and plot inertia vs k
    k_array = np.arange(1, 11)
    inertias = []
    for k in k_array:
        wine_model = KMeans(n_clusters=k)
        wine_model.fit(wine_quality)
        inertias.append(wine_model.inertia_)

    fig, ax = plt.subplots(1, 1)
    ax.plot(k_array, inertias, '-o')
    ax.set(xlabel='Number of Clusters, k', ylabel='Inertia', title='Inertia vs Num Clusters')
    ax.set_xticks(k_array)

    plt.show()
    # question: What value of k would be optimal
    print("I would pick 6 number of clusters")

    # cluster wines into 6 clusters, random_state=2021, assign cluster num
    wine_model = KMeans(n_clusters=6, random_state=2021)
    wine_model.fit(wine_quality)
    wine_quality['clusters'] = wine_model.labels_

    # add quality back to DF
    wine_quality['quality'] = quality_df

    print(pd.crosstab(wine_quality['clusters'], wine_quality['quality']))
    # question: do clusters represent quality of wine?
    print("No, the clusters do not give good representation according to this cross tab.\n"
          "There is no noticeable separation between the qualities of wine\n"
          "And no column has a solid cluster we can identify as the same value is repeated.")
def main():
    WineQualityAnalysis()

if __name__ == '__main__':
    main()