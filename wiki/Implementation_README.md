### Here's how we plan to implement it

##### Creating the ml model

Before beginning the creation of the ml model, the following things have to be taken care of:
- The raw data has to be parsed by using the [nltk](https://www.nltk.org/) library available in python in order to achieve tokenisation, lemmatization, etc.
- Mappings have to be generated using the object-relational mapper software to enable interaction with our relational database.
- After mappings have been obtained, [BERT](https://cloud.google.com/ai-platform/training/docs/algorithms/bert-start) – designed by google, can be used to generate contextual embeddings and cosine similarity search (again available in python libraries) can be used to find similarities between various mappings.
- Principal component analysis, available in [scitik-learn](https://scikit-learn.org/stable/), must be used for dimensionality reduction to ensure simplicity in visualising and designing the model architecture.

Our ml model will aim to draw a comparison between the accuracy and precision of 5 ml and dl-based algorithms, in outputting the relevant API. All ml-dl algorithms will be implemented by making use of [keras](https://keras.io/), a deep-learning API for implementing neural networks, scikit learn, an open-source data analysis library or [OpenAI Gym](https://www.gymlibrary.dev/), used for reinforcement learning. Visualisation of the results after each training epoch and plotting the results will be achieved by making use of [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/) libraries available in python.
- **GRUS and LSTM** can easily be implemented by using [keras.model.layers](https://keras.io/api/layers). Various parameters like number of neurons per layer, learning rate and momentum optimizer can be tampered with easily using keras.
- **SVR** can be implemented using scikit learn
- **Naïve Bayes** can be implemented as follows:
```py
#finding prior probability
def prior(Y_train,label):
    num = np.sum(Y_train==label)
    den = Y_train.shape[0]
    return num/den
```
```py
#finding likelihood
def conditional(X_train,Y_train,feature_col,feature_val,label):
    X_new = X_train[Y_train==label]
    num = np.sum(X_new[:,feature_col]==feature_val)
    den = X_new.shape[0]
    return float(num/den)
```
```py
def predict_class(Y_train,X_train):
    classes = np.unique(Y_train)
    no_features = X_train.shape[1]
    posterior = []
    for label in classes:
        likelihood = 1.0
        for feature in range(no_features):
        cond = conditional(X_train,Y_train,feature,X_train[feature],label)
        likelihood = likelihood*cond
        post = likelihood*prior(Y_train,label)
        posterior.append(post)
        prediction = np.argmax(posterior)
        return prediction
```
- We realised that another algorithm that can be used to perform the given task is the **Markov Decision Process** (or MDP), which is a framework that can solve most RL problems. This can help solve the given problem statement due to 2 of its key properties, the next state depends only on the current state and not on any other previous states (hence mappings can be generated with ease), and the ability to assign weighted rewards based on the agent’s action results in the most optimal mapping. MDP can be implemented using OpenAI Gym. Deep Q-learning and Deep Sarsa can be implemented using the same library.
  ```py
  class ReplayMemory:
    
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def insert(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        assert self.can_sample(batch_size)

        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)
        return [torch.cat(items) for items in batch]

    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size * 10

    def __len__(self):
        return len(self.memory)
  ```

  ##### Creating the Web App

  As of now we are going to use the [Django REST Framework](https://www.django-rest-framework.org/) to deploy the ML Algorithms. We plan to use [React](https://reactjs.org/) for the frontend.
  > Will update this portion once sufficient development is done.