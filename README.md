# NOTDOT

### Collaborative Filtering using Neural Networks
Based on the paper: [Neural Collaborative Filtering][1]

#### Team members:

* Roger Solis
* Qiancheng Liu
* Haiping Xue
* Henry Qiu

#### Presentation video: [TODO][2]

#### Data bases:
* [Movie lens 1 million data set][3]
* [Yelp Dataset Challenge][4]

#### Code structure
The latest changes are mantained on the **master** branch
* "**data**" Contains the data sets
* "**movie_lens_explict**"	Contains the improved version of the NCF paper. Here we use explicit ratings
* "**movie_lens_implict**"	Contains the built as the NCF paper states. Here we only consider user interaction with movies
* "**yelp_lens_explict**" Contains the model that works with user ratings and user revies using doc2vec


#### General instructions

##### Setting up environment
You will need to install the following packages on you python environment
* [keras][5]
* [tensorflow][6]
* [numpy][7]
* [gensim][8]

##### Running the models
Once you have dedided the approach of you preference (implicit, explicit, yelp datase), you should switch to the proper folder and then follow the next steps to see the results we got.
1. Make sure the data files are present on the **data** directory. If they are not, download them and put them into its respective folder right before running the model.
2. We have included the pretraining for the MLP and GMF models in a single file  along with the complete model. Therefore, in order to run our code you should run the scrip  **full_model.py**. 
 
 
**full_model.py** will train the GMF and MLP models, save its weights, feed them into the full model, train the full model, store the model for future references and finally, will load the testing data and feed it into the trained model. Right after, evaluation is performed over the predicted data.

[1]: http://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf "NCF"
[2]: https://www.youtube.com/watch?v=R9N8Xw3kvG8 "Video"
[3]: https://grouplens.org/datasets/movielens/1m "Movie Lens"
[4]: https://www.yelp.com/dataset_challenge "Yelp"
[5]: https://keras.io/ "Keras"
[6]: https://www.tensorflow.org/ "Tensorflow"
[7]: http://www.numpy.org/ "numpy"
[8]: https://radimrehurek.com/gensim/ "gensim"
