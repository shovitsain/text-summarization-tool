import nltk
nltk.download("stopwords")  # Download the stopwords corpus
nltk.download("punkt")  # Download the punkt tokenizer (for sentence tokenization)
from nltk.corpus import stopwords  # Import stopwords from nltk corpus
from nltk.cluster.util import cosine_distance  # Import cosine distance function
import numpy as np
import networkx as nx  # Import NetworkX for graph-based algorithms like PageRank

# Function to read the article from a file and return the sentences as a list of words
def read_article(file_name):
    # Open the file and read all lines
    file = open(file_name, "r")
    filedata = file.readlines()
    
    # Split the first line by ". " to break it into sentences
    article = filedata[0].split(". ")
    
    sentences = []
    for sentence in article:
        # Replace non-alphabetic characters with a space and split by spaces to get words
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    
    # Remove the last empty sentence if any (due to split operation)
    sentences.pop()
    
    return sentences

# Function to compute the similarity between two sentences
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []  # Use an empty list if no stopwords are provided
    
    # Convert each word to lowercase
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    
    # Create a list of unique words from both sentences
    all_words = list(set(sent1 + sent2))

    # Initialize vectors for the two sentences (each will have length equal to the number of unique words)
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    # Fill the vectors with counts of words in the sentences (excluding stopwords)
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    
    # Return the cosine similarity (1 - cosine distance)
    return 1 - cosine_distance(vector1, vector2)

# Function to generate a similarity matrix between all pairs of sentences
def gen_sim_matrix(sentences, stop_words):
    # Initialize a similarity matrix of size len(sentences) x len(sentences)
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    # Calculate the similarity for each pair of sentences (excluding comparisons of a sentence with itself)
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue  # Skip comparing a sentence with itself
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    
    return similarity_matrix

# Function to generate a summary from the article
def generate_summary(file_name, top_n=5):
    stop_words = stopwords.words("english")  # Get the list of stopwords in English
    summarize_text = []  # List to store the top sentences for the summary
    
    # Read the article and get the sentences
    sentences = read_article(file_name)
    
    # Generate the similarity matrix for the sentences
    sentence_similarity_matrix = gen_sim_matrix(sentences, stop_words)
    
    # Create a graph where each sentence is a node and edges represent sentence similarity
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    
    # Apply the PageRank algorithm to rank the sentences based on importance
    scores = nx.pagerank(sentence_similarity_graph)
    
    # Sort the sentences based on their scores (importance), in descending order
    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    # Select the top_n sentences based on their rank
    for i in range(top_n):
        # Join the words in the selected sentence to form a coherent sentence
        summarize_text.append(" ".join(ranked_sentence[i][1]))
    
    # Print the summary (with sentences joined together and separated by ". ")
    print("Summary \n", ". ".join(summarize_text))

generate_summary("msft.txt",2)
