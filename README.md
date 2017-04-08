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






## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/xuehaiping/CSCE670_NCF/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/xuehaiping/CSCE670_NCF/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
