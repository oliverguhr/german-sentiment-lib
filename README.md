[![Downloads](https://pepy.tech/badge/germansentiment)](https://pepy.tech/project/germansentiment)

# German Sentiment Classification with Bert

This package provides a very simple interface to [detect the sentiment](https://de.wikipedia.org/wiki/Sentiment_Detection) of German texts. It uses the Googles Bert architecture trained on 1.834 million samples. The training data contains texts from various domains like Twitter, Facebook and movie, app and hotel reviews. You can find more information about the dataset and the training process in the [paper](https://www.aclweb.org/anthology/2020.lrec-1.202.pdf).

## Install 

To get started install the package from [pypi](https://pypi.org/project/germansentiment/):

```bash
pip install germansentiment
```

## Usage

```python
from germansentiment import SentimentModel

model = SentimentModel()

texts = [
    "Mit keinem guten Ergebniss","Das ist gar nicht mal so gut",
    "Total awesome!","nicht so schlecht wie erwartet",
    "Der Test verlief positiv.","Sie fährt ein grünes Auto."]
       
result = model.predict_sentiment(texts)
print(result)
```

The code above will output following list:

```python
["negative","negative","positive","positive","neutral", "neutral"]
```

### Output class probabilities

```python
from germansentiment import SentimentModel

model = SentimentModel()

classes, probabilities = model.predict_sentiment(["das ist super"], output_probabilities = True) 
print(classes, probabilities)
```
```python
['positive'] [[['positive', 0.9761366844177246], ['negative', 0.023540444672107697], ['neutral', 0.00032294404809363186]]]
```

## Results

If you are interested in code and data that was used to train this model please have a look at [this repository](https://github.com/oliverguhr/german-sentiment) and our [paper](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.201.pdf). Here is a table of the F1 scores that his model achieves on following datasets. Since we trained this model on a newer version of the transformer library, the results are slightly better than reported in the paper.

| Dataset                                                      | F1 micro Score |
| :----------------------------------------------------------- | -------------: |
| [holidaycheck](https://github.com/oliverguhr/german-sentiment) |         0.9568 |
| [scare](https://www.romanklinger.de/scare/)                  |         0.9418 |
| [filmstarts](https://github.com/oliverguhr/german-sentiment) |         0.9021 |
| [germeval](https://sites.google.com/view/germeval2017-absa/home) |         0.7536 |
| [PotTS](https://www.aclweb.org/anthology/L16-1181/)          |         0.6780 |
| [emotions](https://github.com/oliverguhr/german-sentiment)  |         0.9649 |
| [sb10k](https://www.spinningbytes.com/resources/germansentiment/) |         0.7376 |
| [Leipzig Wikipedia Corpus 2016](https://wortschatz.uni-leipzig.de/de/download/german) |         0.9967 |
| all                                                          |         0.9639 |

## Cite

For feedback and questions contact me via e-mail or Twitter [@oliverguhr](https://twitter.com/oliverguhr). Please cite us if you found this useful:

```
@InProceedings{guhr-EtAl:2020:LREC,
  author    = {Guhr, Oliver  and  Schumann, Anne-Kathrin  and  Bahrmann, Frank  and  Böhme, Hans Joachim},
  title     = {Training a Broad-Coverage German Sentiment Classification Model for Dialog Systems},
  booktitle      = {Proceedings of The 12th Language Resources and Evaluation Conference},
  month          = {May},
  year           = {2020},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association},
  pages     = {1620--1625},
  url       = {https://www.aclweb.org/anthology/2020.lrec-1.202.pdf}
}
```


