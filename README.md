# Abstract:
 Nowadays, graph theory has a wide variety of usage domains such
as world-wide-web, neuroscience, and social networks. Graph mining is based
on insights of graph data. Link prediction, graph classification and community
detection are made via graph mining in order to maximize influence on target
people. On the other hand, real-world data can be a challenging issue to represent
with the graph due to the complexity and noise of its nature [1]. Graph neural network
has given better performance as an important graph representation method
based on machine learning and has drawn significant interest in study. However
while calculating such features for graph classification, most of the approaches
like GNN deal with the entire graph. Recently the attention which is inspiration
of brain functions and based on the relevant parts of the data have been
implemented to address this issue, is the most thrilling developments in machine
learning, whose huge potential was well shown in invariable fields. In this study,
we propose Graph-Level Similarity-Based GAT (GrlS-GAT) and Transformed
Graph Classification using Structural Attention (TGCSA) to detect the illness of
the brain. GrlS-GAT is a novel architecture that strengthen the raw GAT. GrlSGAT
combines an ensemble learning technique Extra-Tree Classifier (use to learn
importance of the feature to outcome), a graph attention network and graph convolution
network. According to Veliˇckovi´c et al. GATs is a new implementation
of the neural networks operating on graph-structured data, utilizing concealed
self-attention layers to resolve the deficient prior methods based on approximating
graph convolutions [2]. By loading layers in which nodes (entities) are able
to participate over the features of their neighborhoods, they allow assigning different
sizes to various nodes in the neighborhood, without needing any kind of
expensive matrix process or understanding the layout of the graph beforehand.
Throughout this way, they tackle many main problems of spectral-based neural
graph networks collectively and render our concept conveniently accessible to
both inductive and transductive challenging issues. Moreover, the use of attention
mechanism allows us to collect information and give our attention on small
parts of the graph. With the help of this approach, we can avoid noise in the rest
of the graph.We present a combination of GVT (Graph View Transformer) and
RNN model, called the Graph Classification using Structural Attention (TGCSA),
that processes only a part of the graph by selecting a sequence of important nodes
with an attention mechanism. And since we use GVT on the data set, we combine
several graph views into one and create better results. The attention mechanism
can be used to solve the real-word data challenging issues on the graph representation.
With this study, we expect our combination model can detect brain illness
at the high accuracy.
