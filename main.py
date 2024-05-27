import os
import nltk
import time
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist
from nltk import ngrams
import numpy as np
from itertools import chain

nltk.download('punkt')


#converting corpus to lower case
def convert_to_lower(corpus):
    return corpus.lower()

#merging text sentences
def clean_and_merge(raw_corpus):
    print('Text cleaning started ..', end = " ")
    clean_content = ""
    for sentence in raw_corpus:
        sentence = sentence.strip().replace('\n', ' ')
        clean_content += sentence
    print('Text cleaning ended')
    return clean_content

# Converts the input text string into tokens
def word_tokenizer(corpus):
    try:
        print('Tokenization started ..', end = " ")
        if isinstance(corpus, (bytes, bytearray)):
            corpus = corpus.decode('utf-8') 
        corpus_tokens = word_tokenize(corpus)
        print('Tokenization ended')
        print('Total count of tokens: ', len(corpus_tokens))
        return corpus_tokens
    except Exception as error:
        print('Error encountered while tokenizing', str(error))


# Fetches the stop words from 'English' lang in NLTK library
def get_stopwords():
    stop_words = set(stopwords.words('english'))
    print('Count of stop words: ', len(stop_words))
    print('writing to file started')
    write_to_file(stop_words, 'Stopwords.txt')
    # returning fetched stop words
    return stop_words

# Fetches the multilingual stop words from NLTK library
def get_all_stopwords():
    # Get all the languages present in the NLTK library.
    print('Feching global stop words started...', end=' ')
    languages = stopwords.fileids()
    
    global_stop_words = list(chain.from_iterable(stopwords.words(lang) for lang in languages))
    write_to_file(global_stop_words, 'GlobalStopWords.txt')
    print('Feching global stop words completed')
    print('Count of stop words: ', len(global_stop_words))
    return global_stop_words


# code to write out stop words to file
def write_to_file(stopwords, name):
    with open(name, 'w') as file:
        for word in stopwords:
            file.write(word + '\n')

#processing only text data into tokens
def process_text_to_tokens(corpus_tokens, stop_words, remove_stopwords):
    processed_tokens = []
    try:
        # performed stemming and without stop words removal
        if (not(remove_stopwords)):
            print("No stopword removal.")
            print('Stemming in progress ...', end = ' ')
            stemmer = PorterStemmer()
            stemmed_text_tokens = [stemmer.stem(token) for token in corpus_tokens if token.isalpha()]
            processed_tokens = stemmed_text_tokens
            print('Stemming completed.')
        # performed stemming and stop words removal
        elif remove_stopwords:
            print('Stemming and stop word removal in progress ...', end = " ")
            stemmer = PorterStemmer()
            stemmed_text_tokens = [stemmer.stem(token) for token in corpus_tokens if token.isalpha() and token not in stop_words]
            print('Stemming completed.')
            processed_tokens = stemmed_text_tokens
        print('Count of processed tokens: ', len(processed_tokens))
        return processed_tokens
    except Exception as error:
        print('Exception occurred while processing text tokens', str(error))


# Preprocessing the data
def preprocessing(raw_corpus, remove_stopwords, global_stopwords):
    # Step 1: Clean and merge the raw corpus text data.
    merged_corpus = clean_and_merge(raw_corpus)
    # Step 2: Convert the merged corpus to lowercase to ensure uniformity.
    corpus_lowercase = convert_to_lower(merged_corpus)
    # corpus_punctuation = remove_punctuation(corpus_lowercase)
    # Step 3: Tokenize the lowercase corpus into individual words.
    corpus_tokens = word_tokenizer(corpus_lowercase)
    # Step 4: Initialize stop words English/Global.
    stop_words = global_stopwords if len(global_stopwords) > 0 else get_stopwords()
    # Step 5: Process the tokens based on specified options (stemming and stopwords removal).
    processed_tokens = process_text_to_tokens(corpus_tokens, stop_words, remove_stopwords)
    # Step 6: Return the processed tokens for further analysis.
    return processed_tokens


def staging(folder_name, remove_stopwords, global_stopwords):
    print('******** {} stemming , stop-word removal - {} ********'.format(folder_name, remove_stopwords))
    # Recording the start time for measuring the execution time
    start_time = datetime.fromtimestamp(time.time())
    dir_path = os.path.join('/Users/ishumoganti/Downloads', folder_name)
    raw_corpus = fetch_raw_corpus(dir_path)
    processed_tokens = preprocessing(raw_corpus, remove_stopwords, global_stopwords)
    
    # Return a list containing the start time and the processed tokens
    return [start_time, processed_tokens]



#code to crawl through all files in the directory
def fetch_raw_corpus(dir_path):
    corpus_list = []
    print('path is:', dir_path)
    try:
        for root, folders, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                #print('file path is', file_path)
                content = ''
                with open(file_path, 'rb') as f:
                    for sentence in f.readlines():
                        content += sentence.decode('utf8', 'replace')
                    corpus_list.append(content)
        print('returned from here')
        return corpus_list
    except Exception as error:
        print("error is:", error)

#To find out most frequent words
def frequency_Distribution(processed_tokens, top_word_count):
    top_words = FreqDist(processed_tokens).most_common(top_word_count)
    print(top_words)

#code for frequency distribution
def top_words(processed_tokens, top_word_count, start_time):
    print('Top {} words: \n'.format(top_word_count))
    frequency_Distribution(processed_tokens, top_word_count)
    end_time = datetime.fromtimestamp(time.time())    
    print('Time taken in sec: ', (end_time - start_time).seconds)
    print()


# Invoking for corpus 1
[start_time, c1_stemmed_with_stop_words] = staging('Corpus1', False, [])
print(start_time)

#With Stemming and NO stop word removal.
top_words(c1_stemmed_with_stop_words, 30, start_time)
top_words(c1_stemmed_with_stop_words, 50, start_time)
top_words(c1_stemmed_with_stop_words, 70, start_time)

#With Stemming and with stop word removal.
[start_time, c1_stemmed_no_stop_words] = staging('Corpus1', True, [])

top_words(c1_stemmed_no_stop_words, 30, start_time)
top_words(c1_stemmed_no_stop_words, 50, start_time)
top_words(c1_stemmed_no_stop_words, 70, start_time)

# Invoking for corpus 2
[start_time, c2_stemmed_with_stop_words] = staging('Corpus2', False, [])
print(start_time)

#With Stemming and NO stop word removal.
top_words(c2_stemmed_with_stop_words, 30, start_time)
top_words(c2_stemmed_with_stop_words, 50, start_time)
top_words(c2_stemmed_with_stop_words, 70, start_time)

# With Stemming and with stop word removal.
[start_time, c2_stemmed_no_stop_words] = staging('Corpus2', True, [])

top_words(c2_stemmed_no_stop_words, 30, start_time)
top_words(c2_stemmed_no_stop_words, 50, start_time)
top_words(c2_stemmed_no_stop_words, 70, start_time)


#This function generates n-grams from the given corpus depending on the value of 'n'.
def n_grams(processed_tokens, n, words_req):
    grams = ngrams(processed_tokens, n)
    print('Top {} {}-grams: \n'.format(words_req, n))
    frequency_Distribution(list(grams), words_req)
    print()

#CORPUS1
# Performed stemming and did not remove stop words
n_grams(c1_stemmed_with_stop_words, 2, 30)
n_grams(c1_stemmed_with_stop_words, 2, 50)
n_grams(c1_stemmed_with_stop_words, 2, 70)

#Performed stemming and removed stop words
n_grams(c1_stemmed_no_stop_words, 2, 30)
n_grams(c1_stemmed_no_stop_words, 2, 50)
n_grams(c1_stemmed_no_stop_words, 2, 70)

#CORPUS 2
#Performed stemming and did not remove stop words
n_grams(c2_stemmed_with_stop_words, 2, 30)
n_grams(c2_stemmed_with_stop_words, 2, 50)
n_grams(c2_stemmed_with_stop_words, 2, 70)

#Performed stemming and removed stop words
n_grams(c2_stemmed_no_stop_words, 2, 30)
n_grams(c2_stemmed_no_stop_words, 2, 50)
n_grams(c2_stemmed_no_stop_words, 2, 70)

'''
Up to this point, our focus has been on eliminating stop words specifically from the 'English' language. 
However, it is conceivable that the text may contain words from other languages, and to address this possibility, 
we have implemented extra measures to filter out stop words from those foreign languages as well.

Please note that handling the provided excerpts might require more time due to the presence of 
approximately 1.6 crore tokens and approximately 10,000 stop words that need to be analyzed.

'''

# Fetch global stop words 
global_stopwords = get_all_stopwords()

# CORPUS 1
print('with global stop words removal started\n\n')
[start_time, c1_stemmed_with_stop_words] = staging('Corpus1', True, global_stopwords)

# Invoking the functions
top_words(c1_stemmed_with_stop_words, 30, start_time)
n_grams(c1_stemmed_with_stop_words, 2, 30)

# CORPUS 2
[start_time, c2_stemmed_with_stop_words] = staging('Corpus2', True, global_stopwords)

# Invoking the functions
top_words(c2_stemmed_with_stop_words, 30, start_time)
n_grams(c2_stemmed_with_stop_words, 2, 30)