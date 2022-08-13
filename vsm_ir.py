import sys
import os
import xml.etree.ElementTree as ET
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math
import json


ps = PorterStemmer()
word_tokenize = RegexpTokenizer(r'\w+')
documents = dict()
word_dic = dict()
# inverted_index[0] = true inverted index like in class
# inverted_index[1] = document vector lengths
# inverted_index[2] = mapping from document id to dict containing for each word its count
# inverted_index[3] = dict mapping document to |D|
# inverted_index[4] holds the average document length
# inverted_index[5] holds stop words
inverted_index = [dict(),dict(),dict(),dict(),0,dict()]
total_docs = 0
query_hm = dict()

BM25_K1 = 1.4
BM25_B = 0.75

def prologue():
    stop = set()
    try:
        stop = set(stopwords.words('english'))
    except:
        nltk.download('stopwords')
        stop = set(stopwords.words('english'))
    dic = dict()
    for val in stop:
        dic[val] = 1
    inverted_index[5] = dic

def calc_documents_length():
    for token in inverted_index[0]:
        idf = inverted_index[0][token][0]
        for k in range(2, len(inverted_index[0][token])):
            doc_id = inverted_index[0][token][k][0]
            count = inverted_index[0][token][k][1]
            if doc_id not in inverted_index[1]:
                inverted_index[1][doc_id] = 0
            inverted_index[1][doc_id] += pow((idf*count), 2)
    for doc in inverted_index[1]:
        inverted_index[1][doc] = math.sqrt(inverted_index[1][doc])


def create_index():
    global documents
    global inverted_index
    global total_docs

    directory = sys.argv[2]

    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        if os.path.isfile(file) and file.endswith(".xml"):
            tree = ET.parse(file)
            root = tree.getroot()
            for child in root.findall("./RECORD"):  # extracts all the text from file.
                doc_text = []
                doc_id = 0
                for entry in child:
                    if entry.tag == "RECORDNUM":
                        doc_id = int(entry.text)
                        total_docs += 1
                    elif entry.tag == "EXTRACT":
                        doc_text += list(str(entry.text).lower() + " ")
                    elif entry.tag == "ABSTRACT":
                        doc_text += list(str(entry.text).lower() + " ")
                    elif entry.tag == "TITLE":
                        doc_text += list(str(entry.text).lower() + " ")
                doc_text = word_tokenize.tokenize("".join(doc_text))
                stemmed_text = []
                for token in doc_text:
                    if token not in inverted_index[5]:
                        stemmed_text.append(ps.stem(token))
                inverted_index[3][doc_id] = len(stemmed_text) # save |D| for each doc in order to calculate BM25 faster in real time!
                inverted_index[4] += len(stemmed_text)
                document = dict()
                for token in stemmed_text:
                    if token not in document:
                        document[token] = 0
                    document[token] += 1

                documents[doc_id] = document

                for token in document:
                    if token not in inverted_index[0]:
                        inverted_index[0][token] = [0,0]
                    inverted_index[0][token].append((doc_id,document[token]))
    inverted_index[4] = inverted_index[4] / total_docs
    inverted_index[2] = documents

def calculate_idf():
    global inverted_index
    for token in inverted_index[0]:
        docs_with_the_word = (len(inverted_index[0][token]) - 2)
        inverted_index[0][token][0] = math.log((total_docs / docs_with_the_word),2)

def calculate_bm25_idf():
    global inverted_index
    for token in inverted_index[0]:
        docs_with_the_word = (len(inverted_index[0][token]) - 2)
        nom = total_docs - docs_with_the_word + 0.5
        denom = docs_with_the_word + 0.5
        inverted_index[0][token][1] = math.log(nom/denom + 1,2)


def dic_to_json_file(dic,file_name):
    j = json.dumps(dic)
    f = open(file_name, "w")
    f.write(j)
    f.close()

def handle_creating_index_request():
    prologue()
    create_index()
    calculate_idf()
    calculate_bm25_idf()
    calc_documents_length()
    dic_to_json_file(inverted_index,"vsm_inverted_index.json")


def create_hash_map_vector_for_query(query):
    global query_hm
    query = word_tokenize.tokenize("".join(query))
    for word in query:
        if word not in inverted_index[5]:
            word = word.lower()
            word = ps.stem(word)
            if word not in query_hm:
                query_hm[word] = 0
            query_hm[word] += 1

def calc_similarity_tf_idf():
    similarity = dict()
    query_length = 0

    for word in query_hm:
        if word in inverted_index[0]:
            idf = inverted_index[0][word][0]
            K = query_hm[word]
            weight = idf * K
            query_length += pow(weight, 2)
            lst = inverted_index[0][word]
            for i in range(2,len(lst)):
                doc = lst[i][0]
                C = lst[i][1]
                if doc not in similarity:
                    similarity[doc] = 0
                similarity[doc] += weight*idf*C

    query_length = math.sqrt(query_length)

    for doc in similarity:
        similarity[doc] = similarity[doc] / (query_length * inverted_index[1][str(doc)])
    return similarity

def bm25():
    similarity = dict()
    for doc_id in inverted_index[2]:
        sm = 0
        for qi in query_hm:
            if qi in inverted_index[2][doc_id]:
                idf = float(inverted_index[0][qi][1])
                fqid = int(inverted_index[2][doc_id][qi])
                nom = fqid * (BM25_K1 + 1)
                denom = fqid + BM25_K1 * (1 - BM25_B + BM25_B * int(inverted_index[3][doc_id])/float(inverted_index[4]))
                sm += query_hm[qi] * idf * nom / denom
        if sm > 0:
            similarity[doc_id] = sm
    return similarity

def retrive_best_documents(sim_func, index_path, query):
    global inverted_index
    global query_hm
    with open(index_path) as json_file:
        inverted_index = json.load(json_file)

    query_hm = dict()
    create_hash_map_vector_for_query(query)
    sim = []
    if sim_func == "tfidf":
        sim = calc_similarity_tf_idf()
    elif sim_func == "bm25":
        sim = bm25()

    result = sorted(sim.items(), key=lambda item: item[1], reverse=True)

    end = len(result)

    if sim_func == "tfidf":
        for i in range(len(result)-2):
            if result[0][1] > 3*result[i][1] and result[i//8][1] > 2*result[i][1]:
                end = i
                break
        result = [int(result[i][0]) for i in range(end) if result[i][1] > 0.075]
    elif sim_func == "bm25":
        for i in range(len(result)-2):
            if result[0][1] > 2.1*result[i][1]:
                end = i
                break
        result = [int(result[i][0]) for i in range(int(float(end)/1.5)) if int(result[i][1])>5]

    return result


if __name__ == '__main__':
    operation = sys.argv[1]
    if operation == "create_index":
        handle_creating_index_request()
    elif operation == "query":
        docs = retrive_best_documents(sys.argv[2], sys.argv[3], sys.argv[4].lower())
        with open("ranked_query_docs.txt", 'w') as fp:
            for doc in docs:
                # write each item on a new line
                fp.write("%s\n" % doc)
