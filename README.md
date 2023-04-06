This curriculum is copied from [Melanie Walsh's *Introduction to Cultural Analytics & Python*](https://melaniewalsh.github.io/Intro-Cultural-Analytics/05-Text-Analysis/09-Topic-Modeling-Without-Mallet.html). However, the dataset is different. This script uses the U.S. Inaugural Addresses corpus, which is available for download [here](https://melaniewalsh.github.io/Intro-Cultural-Analytics/_downloads/64e2547e2d86c20cc2a74f660143cfeb/US_Inaugural_Addresses.zip). 

# Topic Modeling — With Tomotopy

In this workshop, we're learning about a text analysis method called *topic modeling*. This method will help us identify the main topics or discourses within a collection of texts or single text that has been separated into smaller text chunks.

In this particular lesson, we're going to use [Tomotopy](https://github.com/bab2min/tomotopy) to topic model The U.S. Inaugural Addresses [zip file (.zip) of text files (.txt)](https://melaniewalsh.github.io/Intro-Cultural-Analytics/_downloads/64e2547e2d86c20cc2a74f660143cfeb/US_Inaugural_Addresses.zip) contains U.S. Inaugural Addresses ranging from President George Washington (1789) to President Donald Trump (2017). Each text file is titled with a number, the corresponding last name of the U.S. President, and the corresponding year of the Inaugural Address..

Tomotopy is a topic modeling tool that is written purely in Python.

___

## Install Packages


```python
# !pip install tomotopy
```


```python
# !pip install little_mallet_wrapper
```

Note: A “wrapper” is a Python package that makes complicated code easier to use and/or makes code from a different programming language accessible in Python.

Since Little MALLET Wrapper also uses the data visualization library seaborn, we’re also going to pip install seaborn:


```python
# !pip install seaborn
```

## Import Packages

Now let's import `tomotopy`, `little_mallet_wrapper` and the data viz library `seaborn`.

We're also going to import [`glob`](https://docs.python.org/3/library/glob.html) and [`pathlib`](https://docs.python.org/3/library/pathlib.html#basic-use) for working with files and the file system.

Finally, we will also import `pandas` to organize our data


```python
import tomotopy as tp
import little_mallet_wrapper
import seaborn
import glob
from pathlib import Path
import pandas as pd
```

## Get Training Data From Text Files

Before we topic model the US Innaugural Address files, we need to process the text files and prepare them for analysis. The steps below demonstrate how to process texts if your corpus is a collection of separate text files. 

<div class="admonition note" name="html-admonition" style="background: lightblue; padding: 10px">
    
<p class="title">Note</p>
    
We're calling these text files our *training data*, because we're *training* our topic model with these texts. The topic model will be learning and extracting topics based on these texts.
    
</div>

To get the necessary text files, we're going to make a variable and assign it the file path for the directory that contains the text files.

</br>

*Note: make sure to save your files in the same folder as your Jupyter Notebook!*


```python
directory = "US_Inaugural_Addresses"
```

Then we're going to use the `glob.gob()` function to make a list of all (`*`) the `.txt` files in that directory.


```python
files = glob.glob(f"{directory}/*.txt")
```

Next we process our texts with the function `little_mallet_wrapper.process_string()`.

This function will take every individual text file, transform all the text to lowercase as well as remove stopwords, punctuation, and numbers, and then add the processed text to our master list `training_data`.


```python
training_data = []
original_texts = []
titles = []

for file in files:
    text = open(file, encoding='utf-8').read()
    processed_text = little_mallet_wrapper.process_string(text, numbers='remove')
    training_data.append(processed_text)
    original_texts.append(text)
    titles.append(Path(file).stem)
```


```python
len(training_data), len(original_texts), len(titles)
```

## Train Topic Model


```python
# Number of topics to return
num_topics = 15
# Numer of topic words to print out
num_topic_words = 10

# Intialize the model
model = tp.LDAModel(k=num_topics)

# Add each document to the model, after splitting it up into words
for text in training_data:
    model.add_doc(text.strip().split())
    
print("Topic Model Training...\n\n")
# Iterate over the data 10 times
iterations = 10
for i in range(0, 100, iterations):
    model.train(iterations)
    print(f'Iteration: {i}\tLog-likelihood: {model.ll_per_word}')

print("\nTopic Model Results:\n\n")
# Print out top 10 words for each topic
topics = []
topic_individual_words = []
for topic_number in range(0, num_topics):
    topic_words = ' '.join(word for word, prob in model.get_topic_words(topic_id=topic_number, top_n=num_topic_words))
    topics.append(topic_words)
    topic_individual_words.append(topic_words.split())
    print(f"✨Topic {topic_number}✨\n\n{topic_words}\n")
```

## Examine Top Documents and Titles

Load topic distributions


```python
topic_distributions = [list(doc.get_topic_dist()) for doc in model.docs]
```

Make functions for displaying top documents. The `get_top_docs()` function is taken from Maria Antoniak's [Little Mallet Wrapper](https://github.com/maria-antoniak/little-mallet-wrapper/blob/c89bfbeddb11ddc2a6874476985275a7b2a6c1fd/little_mallet_wrapper/little_mallet_wrapper.py#L164)


```python
from IPython.display import Markdown, display
import re

def make_md(string):
    display(Markdown(str(string)))

def get_top_docs(docs, topic_distributions, topic_index, n=5):
    
    sorted_data = sorted([(_distribution[topic_index], _document) 
                          for _distribution, _document 
                          in zip(topic_distributions, docs)], reverse=True)
    
    topic_words = topics[topic_index]
    
    make_md(f"### ✨Topic {topic_index}✨\n\n{topic_words}\n\n---")
    
    for probability, doc in sorted_data[:n]:
        # Make topic words bolded
        for word in topic_words.split():
            if word in doc.lower():
                doc = re.sub(f"\\b{word}\\b", f"**{word}**", doc, re.IGNORECASE)
        
        make_md(f'✨  \n**Topic Probability**: {probability}  \n**Document**: {doc}\n\n')
    
    return
```

Display top titles


```python
get_top_docs(titles, topic_distributions, topic_index=0, n=6)
```


```python
get_top_docs(titles, topic_distributions, topic_index=1, n=5)
```

Display top documents with topic words bolded


```python
get_top_docs(original_texts, topic_distributions, topic_index=1, n=5)
```

## Heatmap

Make a heatmap. This function is taken from Maria Antoniak's [Little Mallet Wrapper](https://github.com/maria-antoniak/little-mallet-wrapper/blob/c89bfbeddb11ddc2a6874476985275a7b2a6c1fd/little_mallet_wrapper/little_mallet_wrapper.py#L171)


```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks', font_scale=1.2)
def plot_categories_by_topics_heatmap(labels, 
                                      topic_distributions, 
                                      topic_keys, 
                                      output_path=None,
                                      target_labels=None,
                                      color_map = sns.cm.rocket_r,
                                      dim=None):
    
    # Combine the labels and distributions into a list of dictionaries.
    dicts_to_plot = []
    for _label, _distribution in zip(labels, topic_distributions):
        if not target_labels or _label in target_labels:
            for _topic_index, _probability in enumerate(_distribution):
                dicts_to_plot.append({'Probability': float(_probability),
                                      'Category': _label,
                                      'Topic': 'Topic ' + str(_topic_index).zfill(2) + ': ' + ' '.join(topic_keys[_topic_index][:5])})

    # Create a dataframe, format it for the heatmap function, and normalize the columns.
    df_to_plot = pd.DataFrame(dicts_to_plot)
    df_wide = df_to_plot.pivot_table(index='Category', 
                                     columns='Topic', 
                                     values='Probability')
    df_norm_col=(df_wide-df_wide.mean())/df_wide.std()
        
    # Show the final plot.
    if dim:
        plt.figure(figsize=dim)
    sns.set(style='ticks', font_scale=1.2)
    ax = sns.heatmap(df_norm_col, cmap=color_map)    
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.xticks(rotation=30, ha='left')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()
```


```python
target_labels = titles
```


```python
plot_categories_by_topics_heatmap(titles,
                                  topic_distributions,
                                  topic_individual_words,
                                  target_labels=target_labels,
                                  color_map = 'Blues',
                                 dim=(12,9))
# For all possible color maps, see https://matplotlib.org/stable/tutorials/colors/colormaps.html#miscellaneous
```

## Output a CSV File


```python
topic_results = []
for title, topic_distribution in zip(titles, topic_distributions):
    topic_results.append({'document': title, 'topic_distribution': topic_distribution})

df = pd.DataFrame(topic_results)
column_names = [f"Topic {number} {' '.join(topic[:4])}" for number, topic in enumerate(topic_individual_words)]
df[column_names] = pd.DataFrame(df['topic_distribution'].tolist(), index = df.index)
df = df.drop('topic_distribution', axis='columns')
```


```python
df
```


```python
df.sort_values(by='Topic 3 never well liberty would', ascending=False)[:6]
```


```python
df.to_csv('Topic-Distributions.csv', encoding='utf-8', index=False)
```


```python

```
