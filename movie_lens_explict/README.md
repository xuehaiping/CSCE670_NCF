## Implementation details

### Embedding layer
- input -> embedding is a fully connected layer (sparse to dense vectors). Produces embeddings=latent vectors
- we build 4 embeddings, user-MLP, user-GMF, movie-MLP, movie-GMF
- if MLP is 3 layers, then embedding length is 2x predictive factors (see [evaluation](### Evaluation))

### MLP
- Feed by embedding layer
- ReLu activation. [Here](http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/) is an example
- Each higher layer has 1/2 the number of units
- 3 hidden layers
- Last layer is length of predictive factors
    - each earlier layer is 2x the previous

### GMF
- Input latent vectors (embeddings)
- The mapping function of the first layer is the dot product multiplied by edge weights h
- h is learned from data with the log loss (binary cross-entropy loss)
- Activation function: Sigmoid function

### Total system
- Pretrain GMF and MLP
- ADAM during pretraining, vanilla SGD after

### Evaluation
- Leave one out evaluation
- Randomly sample 100 items not interacted by the user, and rank the test item among 100 items (k fold cross-validation)
- Hit, rate and NDCG
- Random sample 1 interaction for each user as validation data, and tune hyper-parameters on it
- Sample four negative instance per positive instance
- Last layer of NCF = predictive factors, evaluated at [8,16,32,64]



# Running Details

### Prepare data for training

1. Download MovieLens data
2. Change file_path variable name in the input_data.py script to the MovieLens path
3. Run input_data.py


### Pre-train the MLP and GMF models to get its weights
1. Run MLP.py
2. Run GMF.py

### Train the full-model
1. Run FULL_MODEL.py
