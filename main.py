"""Running the GAM model."""
import torch
import numpy as np
import matplotlib.pylab as plt
import json
import glob
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from texttable import Texttable
import scipy.io
import matplotlib as plt
import numpy as np
from sklearn.model_selection import KFold
import os
liste=[]
listefeature=[]
traine=[]
test=[]
listefeature = [None] * 77
dosya = [None] * 77
dosya2 = [None] * 77
liste = [None] * 77

carpım = [None] * 77
logs = dict()
logs["edges"] = []
logs["target"]= []
mean = [None] * 77
yeni = [None] * 77
arr1=[None]*77

import scipy.io
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pickle
from torch.autograd import Variable
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from scipy.stats import norm
import sys
import pandas as pd

import graph_data_loader
from model import GTN
from model import PopulationWeightedFusion

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import mlab
arr2=[None]*77
import json
import glob
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from texttable import Texttable

#from param_parser import parameter_parser
import glob
import json
import torch
import random
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm, trange
sonuc=[None]*5
torch.manual_seed(20)
torch.cuda.manual_seed(20)
np.random.seed(20)
random.seed(20)
torch.backends.cudnn.deterministic=True
sayac11=1




class StepNetworkLayer(torch.nn.Module):
    """
    Step Network Layer Class for selecting next node to move.
    """
    def __init__(self, args, identifiers):
        """
        Initializing the layer.
        :param args: Arguments object.
        :param identifiers: Node type -- id hash map.
        """
        super(StepNetworkLayer, self).__init__()
        self.identifiers = identifiers
        self.args = args
        self.setup_attention()
        self.create_parameters()

    def setup_attention(self):
        """
        Initial attention generation with uniform attention scores.
        """
        self.attention = torch.ones((len(self.identifiers)))/len(self.identifiers)

    def create_parameters(self):
        """
        Creating trainable weights and initlaizing them.
        """
        self.theta_step_1 = torch.nn.Parameter(torch.Tensor(len(self.identifiers),
                                                            self.args.step_dimensions))

        self.theta_step_2 = torch.nn.Parameter(torch.Tensor(len(self.identifiers),
                                                            self.args.step_dimensions))

        self.theta_step_3 = torch.nn.Parameter(torch.Tensor(2*self.args.step_dimensions,
                                                            self.args.combined_dimensions))

        torch.nn.init.uniform_(self.theta_step_1, -1, 1)
        torch.nn.init.uniform_(self.theta_step_2, -1, 1)
        torch.nn.init.uniform_(self.theta_step_3, -1, 1)

    def sample_node_label(self, orig_neighbors, graph, features,seed=10):
        """
        Sampling a label from the neighbourhood.
        :param original_neighbors: Neighbours of the source node.
        :param graph: NetworkX graph.
        :param features: Node feature matrix.
        :return label: Label sampled from the neighbourhood with attention.
        """
        neighbor_vector = torch.tensor([1.0 if n in orig_neighbors else 0.0 for n in graph.nodes()])
        neighbor_features = torch.mm(neighbor_vector.view(1, -1), features)
        attention_spread = self.attention * neighbor_features
        normalized_attention_spread = attention_spread / attention_spread.sum()
        normalized_attention_spread = normalized_attention_spread.detach().numpy().reshape(-1)
        #print("1")
        np.random.seed(seed)
        try:
            label = np.random.choice(np.arange(len(self.identifiers)), p=normalized_attention_spread)
            return label
        except:
            pass
        
        return label

    def make_step(self, node, graph, features, labels, inverse_labels,seed=10):
        """
        :param node: Source node for step.
        :param graph: NetworkX graph.
        :param features: Feature matrix.
        :param labels: Node labels hash table.
        :param inverse_labels: Inverse node label hash table.
        """
        orig_neighbors = set(nx.neighbors(graph, node))
        label = self.sample_node_label(orig_neighbors, graph, features)
        labels = list(set(orig_neighbors).intersection(set(inverse_labels[str(label)])))
        #print("2")
        random.seed(seed)
        new_node = random.choice(labels)
        new_node_attributes = torch.zeros((len(self.identifiers), 1))
        new_node_attributes[label, 0] = 1.0
        attention_score = self.attention[label]
        return new_node_attributes, new_node, attention_score

    def forward(self, data, graph, features, node):
        """
        Making a forward propagation step.
        :param data: Data hash table.
        :param graph: NetworkX graph object.
        :param features: Feature matrix of the graph.
        :param node: Base node where the step is taken from.
        :return state: State vector.
        :return node: New node to move to.
        :return attention_score: Attention score of chosen node.
        """
        feature_row, node, attention_score = self.make_step(node, graph, features,
                                                            data["labels"], data["inverse_labels"])

        hidden_attention = torch.mm(self.attention.view(1, -1), self.theta_step_1)
        hidden_node = torch.mm(torch.t(feature_row), self.theta_step_2)
        combined_hidden_representation = torch.cat((hidden_attention, hidden_node), dim=1)
        state = torch.mm(combined_hidden_representation, self.theta_step_3)
        state = state.view(1, 1, self.args.combined_dimensions)
        return state, node, attention_score

class DownStreamNetworkLayer(torch.nn.Module):
    """
    Neural network layer for attention update and node label assignment.
    """
    def __init__(self, args, target_number, identifiers):
        """
        :param args:
        :param target_number:
        :param identifiers:
        """
        super(DownStreamNetworkLayer, self).__init__()
        self.args = args
        self.target_number = target_number
        self.identifiers = identifiers
        self.create_parameters()

    def create_parameters(self):
        """
        Defining and initializing the classification and attention update weights.
        """
        self.theta_classification = torch.nn.Parameter(torch.Tensor(self.args.combined_dimensions, self.target_number))
        self.theta_rank = torch.nn.Parameter(torch.Tensor(self.args.combined_dimensions, len(self.identifiers)))
        torch.nn.init.xavier_normal_(self.theta_classification)
        torch.nn.init.xavier_normal_(self.theta_rank)

    def forward(self, hidden_state):
        """
        Making a forward propagation pass with the input from the LSTM layer.
        :param hidden_state: LSTM state used for labeling and attention update.
        """
        predictions = torch.mm(hidden_state.view(1, -1), self.theta_classification)
        attention = torch.mm(hidden_state.view(1, -1), self.theta_rank)
        attention = torch.nn.functional.softmax(attention, dim=1)
        return predictions, attention

class GAM(torch.nn.Module):
   
    
    """
    Graph Attention Machine class.
    """
    def __init__(self, args):
        global sayac11
        """
        Initializing the machine.
        :param args: Arguments object.
        """
        super(GAM, self).__init__()
        self.args = args
        self.identifiers, self.class_number = read_node_labels(self.args)
        self.step_block = StepNetworkLayer(self.args, self.identifiers)
        self.recurrent_block = torch.nn.LSTM(self.args.combined_dimensions,
                                             self.args.combined_dimensions, 1)

        self.down_block = DownStreamNetworkLayer(self.args, self.class_number, self.identifiers)
        self.reset_attention()

    def reset_attention(self):
        """
        Resetting the attention and hidden states.
        """
        self.step_block.attention = torch.ones((len(self.identifiers)))/len(self.identifiers)
        torch.manual_seed(10)
        self.lstm_h_0 = torch.randn(1, 1, self.args.combined_dimensions)
        torch.manual_seed(10)
        self.lstm_c_0 = torch.randn(1, 1, self.args.combined_dimensions)

    def forward(self, data, graph, features, node):
        """
        Doing a forward pass on a graph from a given node.
        :param data: Data dictionary.
        :param graph: NetworkX graph.
        :param features: Feature tensor.
        :param node: Source node identifier.
        :return label_predictions: Label prediction.
        :return node: New node to move to.
        :return attention_score: Attention score on selected node.
        """
        self.state, node, attention_score = self.step_block(data, graph, features, node)
        lstm_output, (self.h0, self.c0) = self.recurrent_block(self.state,
                                                               (self.lstm_h_0, self.lstm_c_0))
        label_predictions, attention = self.down_block(lstm_output)
        self.step_block.attention = attention.view(-1)
        label_predictions = torch.nn.functional.log_softmax(label_predictions, dim=1)
        return label_predictions, node, attention_score

class GAMTrainer(object):
    
    
    
    """
    Object to train a GAM model.
    """
    def __init__(self, args):
        self.args = args
        self.model = GAM(args)
        self.setup_graphs()
        self.logs = create_logs(self.args)

    def setup_graphs(self):
        global sayac11
        print(sayac11)
        """
        Listing the training and testing graphs in the source folders.
        """
        self.training_graphs = glob.glob("./input/train {}/".format(sayac11) + "*.json")
        self.test_graphs = glob.glob("./input/test {}/".format(sayac11) + "*.json")

    def process_graph(self, graph_path, batch_loss,seed=10):
        """
        Reading a graph and doing a forward pass on a graph with a time budget.
        :param graph_path: Location of the graph to process.
        :param batch_loss: Loss on the graphs processed so far in the batch.
        :return batch_loss: Incremented loss on the current batch being processed.
        """
        data = json.load(open(graph_path))
        graph, features = create_features(data, self.model.identifiers)
        #print("3")
        random.seed(seed)
        node = random.choice(list(graph.nodes()))
        attention_loss = 0
        for t in range(self.args.time):
            predictions, node, attention_score = self.model(data, graph, features, node)
            target, prediction_loss = calculate_predictive_loss(data, predictions)
            batch_loss = batch_loss + prediction_loss
            if t < self.args.time-2:
                attention_loss += (self.args.gamma**(self.args.time-t))*torch.log(attention_score)
        reward = calculate_reward(target, predictions)
        batch_loss = batch_loss-reward*attention_loss
        self.model.reset_attention()
        return batch_loss

    def process_batch(self, batch):
        """
        Forward and backward propagation on a batch of graphs.
        :param batch: Batch if graphs.
        :return loss_value: Value of loss on batch.
        """
        self.optimizer.zero_grad()
        batch_loss = 0
        for graph_path in batch:
            batch_loss = self.process_graph(graph_path, batch_loss)
        batch_loss.backward(retain_graph=True)
        self.optimizer.step()
        loss_value = batch_loss.item()
        self.optimizer.zero_grad()
        return loss_value

    def update_log(self):
        """
        Adding the end of epoch loss to the log.
        """
        average_loss = self.epoch_loss/self.nodes_processed
        self.logs["losses"].append(average_loss)

    def fit(self,seed=10):
        """
        Fitting a model on the training dataset.
        """
        print("\nTraining started.\n")
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        self.optimizer.zero_grad()
        epoch_range = trange(self.args.epochs, desc="Epoch: ", leave=True)
        for _ in epoch_range:
            
            random.seed(seed)
            random.shuffle(self.training_graphs)
            batches = create_batches(self.training_graphs, self.args.batch_size)
            self.epoch_loss = 0
            self.nodes_processed = 0
            batch_range = trange(len(batches))
            for batch in batch_range:
                self.epoch_loss = self.epoch_loss + self.process_batch(batches[batch])
                self.nodes_processed = self.nodes_processed + len(batches[batch])
                loss_score = round(self.epoch_loss/self.nodes_processed, 4)
                batch_range.set_description("(Loss=%g)" % loss_score)
            self.update_log()

    def score_graph(self, data, prediction):
        """
        Scoring the prediction on the graph.
        :param data: Data hash table of graph.
        :param prediction: Label prediction.
        """
        target = data["target"]
        is_it_right = (target == prediction)
        self.predictions.append(is_it_right)

    def score(self,seed=10):
        global sayac11
        global sonuc
        """
        
        Scoring the test set graphs.
        """
        print("\n")
        print("\nScoring the test set.\n")
        self.model.eval()
        self.predictions = []
        for data in tqdm(self.test_graphs):
            data = json.load(open(data))
            graph, features = create_features(data, self.model.identifiers)
            node_predictions = []
            for _ in range(self.args.repetitions):
                #print("4")
                random.seed(seed)
                node = random.choice(list(graph.nodes()))
                for _ in range(self.args.time):
                    prediction, node, _ = self.model(data, graph, features, node)
                node_predictions.append(np.argmax(prediction.detach()))
                self.model.reset_attention()
            prediction = max(set(node_predictions), key=node_predictions.count)
            self.score_graph(data, prediction)
        self.accuracy = float(np.mean(self.predictions))
        print("\nThe test set accuracy is: "+str(round(self.accuracy, 4))+".\n")
        sonuc[sayac11-1]=round(self.accuracy, 4)

    def save_predictions_and_logs(self):
        """
        Saving the predictions as a csv file and logs as a JSON.
        """
        self.logs["test_accuracy"] = self.accuracy
        with open(self.args.log_path, "w") as f:
            json.dump(self.logs, f)
        cols = ["graph_id", "predicted_label"]
        predictions = [[self.test_graphs[i], self.predictions[i].item()] for i in range(len(self.test_graphs))]
        self.output_data = pd.DataFrame(predictions, columns=cols)
        self.output_data.to_csv(self.args.prediction_path, index=None)

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def read_node_labels(args):
    """
    Reading the graphs from disk.
    :param args: Arguments object.
    :return identifiers: Hash table of unique node labels in the dataset.
    :return class_number: Number of unique graph classes in the dataset.
    """
    global sayac11
    print("\nCollecting unique node labels.\n")
    labels = set()
    targets = set()
    graphs = glob.glob("./input/train {}/".format(sayac11) + "*.json")
    try:
        graphs = graphs + glob.glob("./input/test {}/".format(sayac11) + "*.json")
    except:
        pass
    for g in tqdm(graphs):
        data = json.load(open(g))
        #print(data.shape)
        
        labels = labels.union(set(list(data["labels"].values())))
       
        
        targets = targets.union(set([data["target"]]))
      
        
    identifiers = {label: i for i, label in enumerate(list(labels))}
    class_number = len(targets)
    print("\n\nThe number of graph classes is: "+str(class_number)+".\n")
    return identifiers, class_number

def create_logs(args):
    """
    Creates a dictionary for logging.
    :param args: Arguments object.
    :param log: Hash table for logs.
    """
    log = dict()
    log["losses"] = []
    log["params"] = vars(args)
    return log

def create_features(data, identifiers):
    """
     Creates a tensor of node features.
    :param data: Hash table with data.
    :param identifiers: Node labels mapping.
    :return graph: NetworkX object.
    :return features: Feature Tensor (PyTorch).
    """
    graph = nx.from_edgelist(data["edges"])
    features = []
    for node in graph.nodes():
        features.append([1.0 if data["labels"][str(node)] == i else 0.0 for i in range(len(identifiers))])
    features = np.array(features, dtype=np.float32)
    features = torch.tensor(features)
    return graph, features

def create_batches(graphs, batch_size):
    """
    Creating batches of graph locations.
    :param graphs: List of training graphs.
    :param batch_size: Size of batches.
    :return batches: List of lists with paths to graphs.
    """
    batches = [graphs[i:i + batch_size] for i in range(0, len(graphs), batch_size)]
    return batches

def calculate_reward(target, prediction):
    """
    Calculating a reward for a prediction.
    :param target: True graph label.
    :param prediction: Predicted graph label.
    """
    reward = (target == torch.argmax(prediction))
    reward = 2*(reward.float()-0.5)
    return reward

def calculate_predictive_loss(data, predictions):
    """
    Prediction loss calculation.
    :param data: Hash with label.
    :param prediction: Predicted label.
    :return target: Target tensor.
    :prediction loss: Loss on sample.
    """
    target = [data["target"]]
    target = torch.tensor(target)
    prediction_loss = torch.nn.functional.nll_loss(predictions, target)
    return target, prediction_loss


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def my_loss(H, graphs_torch):
    distance=0

    for subject in range(graphs_torch.shape[0]):
        for view in range(graphs_torch.shape[3]):
            d = graphs_torch[subject,:,:,view].float()
            distance+=torch.dist(H, d, 2)

    return distance
    
def minmax_sc(x):
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    return x

def train(args, graph_dataloader, graphs_torch, model_GTN, model_PopulationFusion):
    params = list(model_GTN.parameters()) + list(model_PopulationFusion.parameters())
    optimizer = torch.optim.Adam(params , lr=0.005, weight_decay=0.001)
    Epoch_losses=[]
    GTN_losses=[]
    Fusion_losses=[]
    for epoch in range(args.num_epochs):
        model_GTN.train()
        model_PopulationFusion.train()
        idx_subject=0
        loss_GTN=0
        for batch_idx, data in enumerate(graph_dataloader):
            adj = Variable(data['adj'].float(), requires_grad=False).to(device)
            model_GTN.zero_grad()
            H = model_GTN(adj)
            loss_GTN += my_loss(H,graphs_torch)
            if idx_subject==0:
                H_population=torch.unsqueeze(H, dim=0)
                idx_subject=1
            else:
                H_tmp = torch.unsqueeze(H, dim=0)
                H_population = torch.cat([H_population,H_tmp])
        loss_GTN/=graphs_torch.shape[0]
        H_fusion = model_PopulationFusion(H_population.permute(1,2,0))
        loss_Fusion = my_loss(H_fusion,graphs_torch)
        loss = args.lambda_1 * loss_GTN + loss_Fusion
        loss.backward()
        optimizer.step()
        
        print("Epoch ",epoch,"Loss ",loss.item(), " ",loss_GTN.item()," ", loss_Fusion.item())
        Epoch_losses.append(loss.item())    
        GTN_losses.append(loss_GTN.item())
        Fusion_losses.append(loss_Fusion.item())
        if (epoch+1)%20==0:
            print(H_fusion)
    return H_fusion.detach().numpy(), Epoch_losses, GTN_losses, Fusion_losses

    

def arg_parse():
    parser = argparse.ArgumentParser(description='Graph Classification')
    parser.add_argument('--dataset', type=str, default='RH',
                        help='Dataset')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Training Epochs')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layer')
    parser.add_argument('--lambda_1', type=float, default=0.3,
                        help='weight of GT loss')
    return parser.parse_args()

def main():
    for i in range(77):
        args = arg_parse()
        print("Main : ",args)
        with open('LH_edges','rb') as f:
            graphs_list = pickle.load(f)
            graphs_list=graphs_list[i]
            graphs_list=np.asarray(graphs_list)
            graphs_list=np.reshape(graphs_list,(1,35,35,4))
           
        for subject in range(len(graphs_list)):
            for view in range(graphs_list[0].shape[2]):
                graphs_list[subject][:,:,view] = minmax_sc(graphs_list[subject][:,:,view])
        
      
        print(graphs_list.shape)
        
        graphs_stacked = np.stack(graphs_list, axis=0)
        graphs_torch = torch.from_numpy(graphs_stacked)
        print(graphs_torch.shape)
        
        
        edges_number=graphs_list[0].shape[-1]
        
        graph_dataloader = graph_data_loader.GraphDataLoader(graphs_list)
        
        model_GTN = GTN(num_edge=edges_number,
                        num_channels=2,
                        num_layers=args.num_layers,
                        norm=True)
        model_PopulationFusion = PopulationWeightedFusion(num_subjects=len(graphs_list))
        H_population, Epoch_losses, GTN_losses, Fusion_losses = train(args, graph_dataloader, graphs_torch, model_GTN, model_PopulationFusion)
        arr2[i]=H_population
        mean[i] = arr2[i].mean()
        print(arr2[i])
        # save epoch losses and fusion output tensor
        with open('H_population{}'.format(i), 'wb') as f:
            pickle.dump(H_population, f)
        with open('epoch_losses{}'.format(i), 'wb') as f:
            pickle.dump(Epoch_losses, f)
        
        # plot loss evolution across epochs
        x_epochs = [i for i in range(args.num_epochs)]
        plt1, = plt.plot(x_epochs, Epoch_losses, label='Total loss'.format(i))
        plt2, = plt.plot(x_epochs, GTN_losses, label='Transformer loss'.format(i))
        plt3, = plt.plot(x_epochs, Fusion_losses, label='Fusion loss'.format(i))
        plt.legend(handles=[plt1,plt2,plt3])
        plt.xlabel('Epochs'.format(i))
        plt.ylabel('Loss'.format(i))
        plt.grid(True)
        plt.savefig("H{}".format(i))
        
        # plot fusion template tensor 
        mask_adj = np.zeros_like(H_population)
        mask_adj[np.triu_indices_from(mask_adj)] = True

        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(30, 30))
            ax = sns.heatmap(H_population, mask=mask_adj, square=True,annot=True)
        Epoch_losses=0
        GTN_losses=0
        Fusion_losses=0
   

if __name__ == "__main__":
    main()

logs["labels"]={"0":1,"1": 2, "2": 3, "3": 1, "4": 4, "5": 1, "6": 1, "7": 5, "8": 1, "9": 1, "10": 4, "11": 3, "12": 4, "13": 3, "14": 1, "15": 1, "16": 5, "17": 3, "18": 3, "19": 3, "20": 4, "21": 5, "22": 2, "23": 3, "24": 5,"25": 2,"26":3,"27":3,"28":5,"29":1,"30":5,"31":3,"32": 1,"33": 1,"34":1}
logs["inverse_labels"]={"1":[0,3,5,6,8,9,14,15,29,32,33,34],"2": [1,22,25], "3": [2,11,13,17,18,19,23,26,27,31], "4": [4,10,12,20], "5": [7,16,21,23,28,30]}



arr3=np.asarray(arr2,dtype=np.float32)
   
    
arr4=arr3.copy()



mean=np.asarray(mean,dtype=np.float32)
#print(mean.mean())
#print(mean.std())




arr4[arr3<(mean.mean())]=0
arr4[arr3>=(mean.mean())]=1

#print(arr4)



for i in range(77):


    graphses=nx.from_numpy_matrix(arr4[i]) #creating graph from numpy array
    
    yeni[i]=list(graphses.edges)
    #print(yeni[i])


for i in range(77):  


    logs["edges"]=yeni[i]

    if(i<=41):
        logs["target"]=1
    elif(i>41 and i<77):
        logs["target"]=0


    dosya[i]=logs
    
    dosya2[i]=dosya[i].copy()
    print(dosya2[i])

dosya2=np.array(dosya2)




kf = KFold(n_splits=5,shuffle=True,random_state=5)
kf.get_n_splits(dosya2)
print(kf)
say=5
for train_index, test_index in kf.split(dosya2):
    print("TRAIN:", train_index, "TEST:", test_index)
    
    traine=list(traine)
    
    
    os.chdir('/Users/Ahmet Zafer/Desktop/birleşimbitirme son kod/input')
   
    test=list(test)
    os.mkdir("train {}".format(say))
    os.chdir("train {}".format(say))
    
    
    for a in train_index:
        with open("{}.json".format(a), "w") as f:
            json.dump(dosya2[a], f)
            
    os.chdir('/Users/Ahmet Zafer/Desktop/birleşimbitirme son kod/input')
    
    os.mkdir("test {}".format(say))
    os.chdir("test {}".format(say))
    
    for b in  test_index:
        with open("{}.json".format(b), "w") as t: 
           
            json.dump(dosya2[b], t)
    os.chdir('/Users/Ahmet Zafer/Desktop/birleşimbitirme son kod/input')
    
    say=say-1   
    
os.chdir('/Users/Ahmet Zafer/Desktop/birleşimbitirme son kod')    
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)



        
args = Namespace(batch_size=32, combined_dimensions=64,epochs=10, gamma=0.99, learning_rate=0.001, log_path='./logs/erdos_gam_logs.json', prediction_path='./output/erdos_predictions.csv', repetitions=10, step_dimensions=32, test_graph_folder='./input/test {}/'.format(sayac11), time=10, train_graph_folder='./input/train {}/'.format(sayac11), weight_decay=1e-05)
"""
Parsing command line parameters, processing graphs, fitting a GAM.
"""
print(sayac11,"sayac")
#global sayac11
#global sonuc
#args = parameter_parser()
tab_printer(args)
for z in range(5):
    model = GAMTrainer(args)
    model.fit()
    model.score()
    model.save_predictions_and_logs()
    sayac11=sayac11+1
sonuc=np.asarray(sonuc)
print(sonuc.mean(),"son accuracy")