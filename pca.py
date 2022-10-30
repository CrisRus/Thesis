import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.decomposition import PCA
import seaborn as sns


raw_data = pd.read_csv("resized_dataset2.csv", index_col=False)
y = raw_data['Target']
X_raw = raw_data.drop("Target", axis=1)
Categories = ["COVID19","NORMAL","PNEUMONIA", "TURBERCULOSIS"]



def show_clusters(data):
    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init="k-means++", n_clusters=4, n_init=4)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    plt.title(
        "K-means clustering (PCA-reduced data)"
    )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()



def find_number_of_components(pca):
	# find number of components
	per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
	ratio_sum = 0
	for order in range(0, len(pca.explained_variance_ratio_)):
		ratio_sum = ratio_sum + pca.explained_variance_ratio_[order]
		print("{0}: {1}".format(order, ratio_sum))

	labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
	plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
	plt.ylabel("Percentage of Explained Variance")
	plt.xlabel("Principal Component")
	plt.title("Explained variance")
	plt.show()


def find_the_number_of_clusters(principal_components, limit):
	# Find the number of clusters
	wcss = []
	for i in range(1, limit + 1):
		print("Fitting components {0}/{1}".format(i, limit))
		kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=42)
		kmeans_pca.fit(principal_components)
		wcss.append(sum(np.min(cdist(principal_components, kmeans_pca.cluster_centers_, 'euclidean'), axis=1)) /
                    principal_components.shape[0])

	# Plot the figure
	plt.figure(figsize=(16, 10))
	plt.plot(range(1, limit + 1), wcss, marker='o', linestyle='--')
	plt.xlabel("Number of clusters")
	plt.ylabel("WCSS")
	plt.title("K-Means PCA")

	order = np.linspace(1, limit, limit)
	# find the elbow
	# https://github.com/arvkevi/kneed/blob/master/notebooks/decreasing_function_walkthrough.ipynb
	kn = KneeLocator(order, wcss, curve='convex', direction='decreasing')
	plt.vlines(kn.elbow, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
	plt.show()
	print("Number of clusters: {0}".format(kn.elbow))
	return int(kn.elbow)


def cluster_with_kmeans(number_of_clusters, principal_components, principal_df):
	# do some clustering
    kmeans_pca = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
    prediction = kmeans_pca.fit_predict(principal_components)

    final_df = pd.concat([principal_df], axis=1)
    final_df['Cluster'] = kmeans_pca.labels_
    final_df.rename({0: 'PC1', 1: 'PC2', 2: 'PC3'}, axis=1, inplace=True)
    # plot the thing
    sns.set()
    plt.figure(figsize=(12, 7))
    plt.title("K-Means PCA")
    sns.scatterplot(
		x="PC1", y="PC2",
		hue="Class",
        style="Cluster",
        palette=sns.color_palette("tab10"),
		data=final_df,
		s=75,
		alpha=0.7
	)
    plt.show()


def main():
    pca = PCA()
    principal_components = pca.fit_transform(X_raw)
    # find_number_of_components(pca)
    # find_the_number_of_clusters(principal_components, 15)
    principal_df = pd.DataFrame(data=principal_components)
    labels = []
    for code in y:
        labels.append(Categories[code])
    principal_df['Class'] = labels
    cluster_with_kmeans(4, principal_components, principal_df)
    # show_clusters(X_raw)

if __name__ == '__main__':
    main()
