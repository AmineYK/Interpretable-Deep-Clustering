# Interpretable-Deep-Clustering

Clustering is a fundamental learning task widely used as a first step in data analysis.
For example, biologists often use cluster assignments to analyze genome sequences,
medical records, or images. Since downstream analysis is typically performed at
the cluster level, practitioners seek reliable and interpretable clustering models.
We propose a new deep-learning framework that predicts interpretable cluster
assignments at the instance and cluster levels. First, we present a self-supervised
procedure to identify a subset of informative features from each data point. Then,
we design a model that predicts cluster assignments and a gate matrix that leads
to cluster-level feature selection. We show that the proposed method can reliably
predict cluster assignments using synthetic and real data. Furthermore, we verify
that our model leads to interpretable results at a sample and cluster level.
