# Adversarial Attacks on Graph Neural Networks via Meta Learning

<p align="center">
<img src="https://www.in.tum.de/fileadmin/w00bws/daml/gnn-meta-attack/figure3.png" width="400">
</p>

Implementation of the paper:   
**[Adversarial Attacks on Graph Neural Networks via Meta Learning](https://openreview.net/forum?id=Bylnx209YX&noteId=r1xNHe2tAQ)**

by Daniel Zügner and Stephan Günnemann.   
Published at ICLR'19, May 2019, New Orleans, USA

Copyright (C) 2019   
Daniel Zügner   
Technical University of Munich    

## Requirements
* Python 3.6 or newer
* `numpy`
* `scipy`
* `scikit-learn`
* `tensorflow`
* `matplotlib` (for the demo notebook)
* `seaborn` (for the demo notebook)

## Installation
`python setup.py install`

## Run the code
 
 To try our code, you can use the IPython notebook `demo.ipynb`.
 
## Contact
Please contact zuegnerd@in.tum.de in case you have any questions.


## References
### Datasets
In the `data` folder we provide the following datasets originally published by   
#### Cora
McCallum, Andrew Kachites, Nigam, Kamal, Rennie, Jason, and Seymore, Kristie.  
*Automating the construction of internet portals with machine learning.*   
Information Retrieval, 3(2):127–163, 2000.

and the graph was extracted by

Bojchevski, Aleksandar, and Stephan Günnemann. *"Deep gaussian embedding of   
attributed graphs: Unsupervised inductive learning via ranking."* ICLR 2018.

#### Citeseer
Sen, Prithviraj, Namata, Galileo, Bilgic, Mustafa, Getoor, Lise, Galligher, Brian, and Eliassi-Rad, Tina.   
*Collective classification in network data.*   
AI magazine, 29(3):93, 2008.
#### PolBlogs
Lada A Adamic and Natalie Glance. 2005. *The political blogosphere and the 2004   
US election: divided they blog.*   
In Proceedings of the 3rd international workshop on Link discovery. 36–43.

### Graph Convolutional Networks
Our implementation of the GCN algorithm is based on the authors' implementation,
available on GitHub [here](https://github.com/tkipf/gcn).

The paper was published as  

Thomas N Kipf and Max Welling. 2017.  
*Semi-supervised classification with graph
convolutional networks.* ICLR (2017).

## Cite
Please cite our paper if you use the model or this code in your own work:

```
@inproceedings{zugner_adversarial_2019,
	title = {Adversarial Attacks on Graph Neural Networks via Meta Learning},
	author={Z{\"u}gner, Daniel and G{\"u}nnemann, Stephan},
	booktitle={International Conference on Learning Representations (ICLR)},
	year = {2019}
}
```
