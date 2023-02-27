#!/usr/bin/env python
# coding: utf-8

'''
Mini Project 1: Group-3
********************************************************************************************
1. Sai Phani Ram Popuri - 2205577
2. Prathima Mettu - 2160335

Functionality: The code analyzes frequency distribution for unigrams and bigrams of corpora.
Created Date: 20 Feb, 2023.
********************************************************************************************
'''


# Importing necessary modules

import os
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
import time
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist
from nltk import ngrams


# Crawls through the directory and reads all the files present inside it.
def fetch_raw_corpus(dir_path):
    corpus_list = []
    for root, folders, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            content = ''
            with open(file_path, 'rb') as f:
                # Reads the file line by line for improved efficiency.
                for sentence in f.readlines():
                    content += sentence.decode('utf8', 'replace')
                corpus_list.append(content)
    return corpus_list


# Function to convert the input string into lowercase.
def convert_to_lower(corpus):
    return corpus.lower()


# Removes extra spaces and replaces escape characters from the text string

def clean_n_merge(raw_corpus):
    print('Cleaning Text ..', end = " ")
    clean_content = ""
    for sentence in raw_corpus:
        sentence = sentence.strip().replace('\n', ' ')
        clean_content += sentence
    print('Done.')
    return clean_content

# Converts the input text string into tokens
def word_tokenizer(corpus):
    print('Tokenization starts ...', end = " ")
    corpus_tokens = word_tokenize(corpus)
    print('Done')
    print('Total count of tokens: ', len(corpus_tokens))
    return corpus_tokens

# Writes the stop words to the file 'Stopwords.txt'
def write_to_file(stopwords):
    with open('Stopwords.txt', 'w') as file:
        for word in stopwords:
            file.write(word + '\n')


# Fetches the stop words from 'English' lang in NLTK library
def get_stopwords_english():
    stop_words = set(stopwords.words('english'))
    print('Count of stop words: ', len(stop_words))
    return stop_words


# Fetches the multilingual stop words from NLTK library
def get_stopwords():
    # fileids() returns all the languages present in the NLTK library.
    languages = stopwords.fileids()
    global_stop_words = []
    try:
        temp = []
        for lang in languages:
            lang_stop_words = set(stopwords.words(lang))
            temp.append(lang_stop_words)
    except:
        print('Encontered an error with : ', lang)
    
    for sublist in temp:
        for word in sublist:
            global_stop_words.append(word)
    
    write_to_file(global_stop_words)
    print('Count of stop words: ', len(global_stop_words))
    return global_stop_words


# Performs stemming operation on tokens and keeps only alpha tokens (excludes numbers, special chars ..)
'''
    The four combinations:
    1. With Stemming and with stop-word removal.
    2. With Stemming and without stop-word removal.
    3. Without Stemming and with stop-word removal.
    4. Without Stemming and without stop-word removal.
'''

def process_text_only_tokens(corpus_tokens, stop_words, perform_Stemming, remove_stopwords):
    processed_tokens = []
    # with stemming and stop words removal
    if(perform_Stemming and remove_stopwords):
        print('Stemming and stop word removal in progress ...', end = " ")
        stemmer = PorterStemmer()
        stemmed_text_tokens = [stemmer.stem(token) for token in corpus_tokens if token.isalpha() and token not in stop_words]
        print('Done.')
        processed_tokens = stemmed_text_tokens
    # with stemming and without stop words removal
    elif (perform_Stemming and not(remove_stopwords)):
        print("No stopword removal.")
        print('Stemming in progress ...', end = ' ')
        stemmer = PorterStemmer()
        stemmed_text_tokens = [stemmer.stem(token) for token in corpus_tokens if token.isalpha()]
        processed_tokens = stemmed_text_tokens
        print('Done.')
    # without stemming and stop words removal
    elif (not(perform_Stemming) and remove_stopwords):
        print('No stemming performed.')
        print('stop word removal in progress ...', end = " ")
        unstemmed_text_tokens = [token for token in corpus_tokens if token.isalpha() and token not in stop_words]
        processed_tokens = unstemmed_text_tokens
        print('Done.')
    # without stemming and without stop words removal
    elif (not(perform_Stemming) and not(remove_stopwords)):
        print('No stemming and stop word removal.')
        unstemmed_text_tokens = [token for token in corpus_tokens if token.isalpha()]
        processed_tokens = unstemmed_text_tokens

    print('Count of processed tokens: ', len(processed_tokens))
    
    return processed_tokens


# Wrapper method that performs the preprocessing operation
def preprocessing_text(raw_corpus, perform_stemming, remove_stopwords, custom_stopwords):
    merged_corpus = clean_n_merge(raw_corpus)
    corpus_lowercase = convert_to_lower(merged_corpus)
    corpus_tokens = word_tokenizer(corpus_lowercase)
    stop_words = custom_stopwords if len(custom_stopwords) > 0 else get_stopwords_english()
    processed_tokens = process_text_only_tokens(corpus_tokens, stop_words, perform_stemming, remove_stopwords)
    return processed_tokens


# Returns the list of most frequent words as per the requirement
def freq_Distribution(processed_tokens, top_word_count):
    top_words = FreqDist(processed_tokens).most_common(top_word_count)
    print(top_words)

# Staging and preprocess the text.
def staging_n_preprocessing(folder_name, perform_stemming, remove_stopwords, custom_stopwords):
    print('*********************** {} stemming - {}, stop-word removal - {} ***********************'.format(folder_name, perform_stemming, remove_stopwords))
    start_time = datetime.fromtimestamp(time.time())
    dir_path = os.path.join('/Users/beingrampopuri/Desktop/Personal/Text_Mining_1', folder_name)
    raw_corpus = fetch_raw_corpus(dir_path)
    processed_tokens = preprocessing_text(raw_corpus, perform_stemming, remove_stopwords, custom_stopwords)
    return [start_time, processed_tokens]


# Driver code for freq distribution

def top_words(processed_tokens, top_word_count, start_time):
    print('Top {} words: \n'.format(top_word_count))
    freq_Distribution(processed_tokens, top_word_count)
    end_time = datetime.fromtimestamp(time.time())    
    print('Time taken in sec: ', (end_time - start_time).seconds)
    print()


# Corpus 1: Without Stemming and without stop word removal.
[start_time, cor1_unstem_without_stop_words] = staging_n_preprocessing('Corpus1', False, False, [])

# Invoking the function with different values of 'K'
# top_words(tokens, words_req, start_time)
top_words(cor1_unstem_without_stop_words, 30, start_time)
top_words(cor1_unstem_without_stop_words, 50, start_time)
top_words(cor1_unstem_without_stop_words, 70, start_time)


# Corpus 1: Without Stemming and with stop word removal.
[start_time, cor1_unstem_with_stop_words] = staging_n_preprocessing('Corpus1', False, True, [])

# Invoking the function
top_words(cor1_unstem_with_stop_words, 30, start_time)
top_words(cor1_unstem_with_stop_words, 50, start_time)
top_words(cor1_unstem_with_stop_words, 70, start_time)


# Corpus 1: With Stemming and without stop word removal.
[start_time, cor1_stem_without_stop_words] = staging_n_preprocessing('Corpus1', True, False, [])

# Invoking the function
top_words(cor1_stem_without_stop_words, 30, start_time)
top_words(cor1_stem_without_stop_words, 50, start_time)
top_words(cor1_stem_without_stop_words, 70, start_time)


# Corpus 1: With Stemming and with stop word removal.
[start_time, cor1_stem_with_stop_words] = staging_n_preprocessing('Corpus1', True, True, [])

# Invoking the function
top_words(cor1_stem_with_stop_words, 30, start_time)
top_words(cor1_stem_with_stop_words, 50, start_time)
top_words(cor1_stem_with_stop_words, 70, start_time)


# CORPUS 2

# Corpus 2: without Stemming and without stop word removal.
[start_time, cor2_unstem_without_stop_words] = staging_n_preprocessing('Corpus2', False, False, [])

# Invoking the function
top_words(cor2_unstem_without_stop_words, 30, start_time)
top_words(cor2_unstem_without_stop_words, 50, start_time)
top_words(cor2_unstem_without_stop_words, 70, start_time)


# Corpus 2: without Stemming and with stop word removal.
[start_time, cor2_unstem_with_stop_words] = staging_n_preprocessing('Corpus2', False, True, [])

# Invoking the function
top_words(cor2_unstem_with_stop_words, 30, start_time)
top_words(cor2_unstem_with_stop_words, 50, start_time)
top_words(cor2_unstem_with_stop_words, 70, start_time)

# Corpus 2: with Stemming and without stop word removal.
[start_time, cor2_stem_without_stop_words] = staging_n_preprocessing('Corpus2', True, False, [])

# Invoking the function
top_words(cor2_stem_without_stop_words, 30, start_time)
top_words(cor2_stem_without_stop_words, 50, start_time)
top_words(cor2_stem_without_stop_words, 70, start_time)

# Corpus 2: with Stemming and with stop word removal.
[start_time, cor2_stem_with_stop_words] = staging_n_preprocessing('Corpus2', True, True, [])

# Invoking the function
top_words(cor2_stem_with_stop_words, 30, start_time)
top_words(cor2_stem_with_stop_words, 50, start_time)
top_words(cor2_stem_with_stop_words, 70, start_time)

# Function that returns the n-grams of the corpus based on the input 'n'
def n_grams(processed_tokens, n, words_req):
    grams = ngrams(processed_tokens, n)
    print('Top {} {}-grams: \n'.format(words_req, n))
    freq_Distribution(list(grams), words_req)
    print()

# Corpus 1: Without stemming and Without stop word removal
n_grams(cor1_unstem_without_stop_words, 2, 30)
n_grams(cor1_unstem_without_stop_words, 2, 50)
n_grams(cor1_unstem_without_stop_words, 2, 70)

# Corpus 1: Without stemming and With stop word removal
n_grams(cor1_unstem_with_stop_words, 2, 30)
n_grams(cor1_unstem_with_stop_words, 2, 50)
n_grams(cor1_unstem_with_stop_words, 2, 70)

# Corpus 1: With stemming and Without stop word removal
n_grams(cor1_stem_without_stop_words, 2, 30)
n_grams(cor1_stem_without_stop_words, 2, 50)
n_grams(cor1_stem_without_stop_words, 2, 70)


# Corpus 1: With stemming and With stop word removal
n_grams(cor1_stem_with_stop_words, 2, 30)
n_grams(cor1_stem_with_stop_words, 2, 50)
n_grams(cor1_stem_with_stop_words, 2, 70)


# CORPUS-2
# Corpus 2: Without stemming and Without stop word removal

n_grams(cor2_unstem_without_stop_words, 2, 30)
n_grams(cor2_unstem_without_stop_words, 2, 50)
n_grams(cor2_unstem_without_stop_words, 2, 70)


# Corpus 2: Without stemming and With stop word removal
n_grams(cor2_unstem_with_stop_words, 2, 30)
n_grams(cor2_unstem_with_stop_words, 2, 50)
n_grams(cor2_unstem_with_stop_words, 2, 70)


# Corpus 2: With stemming and Without stop word removal
n_grams(cor2_stem_without_stop_words, 2, 30)
n_grams(cor2_stem_without_stop_words, 2, 50)
n_grams(cor2_stem_without_stop_words, 2, 70)


# Corpus 2: With stemming and With stop word removal
n_grams(cor2_stem_with_stop_words, 2, 30)
n_grams(cor2_stem_with_stop_words, 2, 50)
n_grams(cor2_stem_with_stop_words, 2, 70)


'''
So far the stop words of only 'English' language are removed.
There might be a possibility that the text contains words from different languages.
Therefore, we have walked an extra mile to remove the stop words from those languages as well.

Note: Below snippets take longer time to process, as there are approx ~ 1.5 crore tokens and ~10K stop words to check.
'''

# The above operation is performed only on unstemmed versions. 

# Fetch global stop words.
global_stopwords = get_stopwords()

# Corpus 1:  Unstemmed and with stop word removal.
[start_time, cor1_unstem_with_global_stop_words] = staging_n_preprocessing('Corpus1', False, True, global_stopwords)

# Invoking the functions
top_words(cor1_unstem_with_global_stop_words, 30, start_time)
n_grams(cor1_unstem_with_global_stop_words, 2, 30)

# Corpus 2:  Unstemmed and with stop word removal.
[start_time, cor2_unstem_with_global_stop_words] = staging_n_preprocessing('Corpus2', False, True, global_stopwords)

# Invoking the functions
top_words(cor2_unstem_with_global_stop_words, 30, start_time)
n_grams(cor2_unstem_with_global_stop_words, 2, 30)
