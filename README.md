## Implementation details

### embedding layer
- input -> embedding is a fully connected layer
- we build 4 embeddings, user-MLP, user-GMF, movie-MLP, movie-GMF
- if MLP is 3 layers, then embedding length is 2x predictive factors (see [evaluation](### evaluation))
### MLP
- ReLu activation
- each higher layer has 1/2 the number of units
- 3 hidden layers
- last layer is lenth of predictive factors
    - each earlier layer is 2x the previous

### GMF

### total system
- pretrain GMF and MLP
- ADAM during pretraining, vanilla SGD after

### evaluation
- leave one out evaluation
- randomly sample 100 items not interacted by the user, and rank the test item among 100 items
- Hit rate and NDCG
- random sample 1 interaction for each user as validation data, and tune hyper-parameters on it
- sample four negative instance per positive instance
- last layer of NCF = predictive factors, evaluated at [8,16,32,64]



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
