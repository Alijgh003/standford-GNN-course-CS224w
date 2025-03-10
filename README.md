# standford-GNN-course-CS224w
the course page: https://web.stanford.edu/class/cs224w/


## Colab 1 : node_embedding
In this Colab, we have wrote a full pipeline for learning node embeddings. We will go through the following 3 steps.

To start, we will load a classic graph in network science, the Karate Club Network. We will explore multiple graph statistics for that graph.

We will then work together to transform the graph structure into a PyTorch tensor, so that we can perform machine learning over the graph.

Finally, we will finish the first learning algorithm on graphs: a node embedding model. For simplicity, our model here is simpler than DeepWalk / node2vec algorithms taught in the lecture. But it's still rewarding and challenging, as we will write it from scratch via PyTorch.

## Colab 2 : introduction to pytorch geometric and ogd (open graph benchmark)
In Colab 2, we have worked to construct our own graph neural network using PyTorch Geometric (PyG) and then apply that model on two Open Graph Benchmark (OGB) datasets. These two datasets will be used to benchmark your model's performance on two different graph-based tasks: 1) node property prediction, predicting properties of single nodes and 2) graph property prediction, predicting properties of entire graphs or subgraphs.

First, we have learned how PyTorch Geometric stores graphs as PyTorch tensors.

Then, we have loaded and inspect one of the Open Graph Benchmark (OGB) datasets by using the ogb package. OGB is a collection of realistic, large-scale, and diverse benchmark datasets for machine learning on graphs. The ogb package not only provides data loaders for each dataset but also model evaluators.

Lastly, we have built our own graph neural network using PyTorch Geometric. We have then trained and evaluated our model on the OGB node property prediction and graph property prediction tasks.

## Colab 3 : GraphSAGE implemention
In Colab 2 we constructed GNN models by using PyTorch Geometric's built in GCN layer, GCNConv. In this Colab we have gone a step deeper and implemented the GraphSAGE (Hamilton et al. (2017)) layer directly. Then we have runned our models on the CORA dataset, which is a standard citation network benchmark dataset.

## Colab 4 : GAT (Graph Attnetion Network) 
In Colab 2 we constructed GNN models by using PyTorch Geometric's built in GCN layer, GCNConv. In Colab 3 we implemented the GraphSAGE (Hamilton et al. (2017)) layer. In this colab have used what we've learned and implemented a more powerful layer: GAT (Veličković et al. (2018)). Then we will run our models on the CORA dataset, which is a standard citation network benchmark dataset.

## Colab 5 : H-GNN (Heterogeneous Graphs Neural Network)
In this Colab, we have shifted our focus from homogenous graphs to heterogeneous graphs. Heterogeneous graphs extend the traditional homogenous graphs that we have been working with by incorperating different node and edge types. This additional information allows us to extend the graph neural nework models that we have worked with before. Namely, we can apply heterogenous message passing, where different message types now exist between different node and edge type relationships.

In this notebook, we have first learned how to transform NetworkX graphs into DeepSNAP representations. Then, we have dived deeper into how DeepSNAP stores and represents heterogeneous graphs as PyTorch Tensors.

Lastly, we have built our own heterogenous graph neural netowrk models using PyTorch Geometric and DeepSNAP. We have then applied our models for a node property prediction task; specifically, we will evaluate these models on the heterogeneous ACM node prediction dataset.
