# coding: utf-8
"""
Example of a Streamlit app for an interactive spaCy model visualizer. You can
either download the script, or point streamlit run to the raw URL of this
file. For more details, see https://streamlit.io.
Installation:
pip install streamlit
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download de_core_news_sm
Usage:
streamlit run streamlit_spacy.py
"""
from __future__ import unicode_literals

import streamlit as st
import spacy
from spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS

import pandas as pd
from textblob import TextBlob
import pke

import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

DEFAULT_TEXT = "Mark Zuckerberg is the CEO of Facebook. He is a great guy! No, he is a BAD guy!!  No, he is an ok guy..."
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

SPACY_MODEL_NAMES = []
lang = "en"
if lang == "en": 
    SPACY_MODEL_NAMES += ["en_core_web_sm"]

stopwords = STOP_WORDS

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#helper functions
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results
#main
def get_keywords_tfidf(text, n_keywords=5, stopwords=stopwords):
        #paramters of Vectorizers should be configurable vs hardcoded (by us, not user)

        #generate tf-idf for the given document
        tf_idf_vector=tfidf_transformer.transform(cv.transform([text]))

        #sort the tf-idf vectors by descending order of scores
        sorted_items=sort_coo(tf_idf_vector.tocoo())

        keywords=extract_topn_from_vector(feature_names,sorted_items,n_keywords)

    #filter by score; Threshold = paramter we need to configure
        TFIDF_THRES = 0.45
        final_keywords=[k for k,v in keywords.items() if v >TFIDF_THRES]
        return final_keywords

def get_keywords_topicrank(text, lang=lang, n_keywords=5, stopwords=stopwords, normalization=None, return_scores = False): 

    # initialize keyphrase extraction model, here TopicRank
    extractor = pke.unsupervised.TopicRank()
    # load the content of the document and perform French stemming

    extractor.load_document(input=text,
                        language=lang,
                        normalization=normalization)

    # keyphrase candidate selection, here sequences of nouns and adjectives
    # defined by the Universal PoS tagset
    extractor.candidate_selection(pos={"NOUN", "PROPN" "ADJ"}, stoplist=list(stopwords))

    # candidate weighting, here using a random walk algorithm
    extractor.candidate_weighting()

    # N-best selection, keyphrases contains the 10 highest scored candidates as
    # (keyphrase, score) tuples
    keyphrases = extractor.get_n_best(n=n_keywords)

    if not return_scores:
        kws = [k[0] for k in keyphrases]
        return kws
    else:
        return keyphrases


@st.cache(allow_output_mutation=True)
def load_model(name):
    return spacy.load(name)


@st.cache(allow_output_mutation=True)
def process_text(model_name, text):
    nlp = load_model(model_name)
    return nlp(text)


st.sidebar.title("brAInsuite Demo/Test Lab - NLP ")
st.sidebar.markdown(
    """
a) sentence-based
b) corpus-based
"""
)

spacy_model = st.sidebar.selectbox("Model name", SPACY_MODEL_NAMES)
model_load_state = st.info(f"Loading model '{spacy_model}'...")
nlp = load_model(spacy_model)
model_load_state.empty()
from spacy_langdetect import LanguageDetector
if "language_detector" not in nlp.pipe_names:
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

text = st.text_area("Text to analyze", DEFAULT_TEXT)
doc = process_text(spacy_model, text)

st.header("detected language")
st.write(doc._.language)
st.sidebar.header("translate to")

trans_lang = "de"
trans_lang = st.sidebar.selectbox(
     'languages',
     ('de', 'es', 'it'))

st.header("translation")
orig_text = TextBlob(text)
trans_text = orig_text.translate(to=trans_lang)
st.write(trans_lang, trans_text)

st.header("Sentence-based analyses")

split_sents = st.sidebar.checkbox("Split sentences", value=True)
docs = [span.as_doc() for span in doc.sents] if split_sents else [doc]

sents = []
for sent in docs: 
    sents.append(sent.text)
    
GET_KEYWORDS = True
if GET_KEYWORDS: 
    st.header("Keywords")
    st.subheader("tfidf")
    cv=CountVectorizer(min_df=1, max_df=0.85,stop_words=stopwords)
    word_count_vector=cv.fit_transform(sents)
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)
 
    feature_names=cv.get_feature_names()

    for sent in docs:
        keywords = get_keywords_tfidf(sent.text)
        st.write(" ".join(keywords))


st.header("Sentiment")
for sent in docs:
    sentiment = TextBlob(sent.text).polarity
    st.write(sent.text, sentiment)


#sent_df = pd.DataFrame({"text": text})
#sent_df["cleaned"] = list(clean_text_soft(sent_df.text))
#sent_df["lemma_tokens"] = list(lemmatize_text_soft(sent_df["cleaned"],lang=lang))
#sent_df["lemma_tokens"] = sent_df["lemma_tokens"].apply(lambda w: " ".join(w))

#textblob
#sent_df["sentiment_tb"] = sent_df["lemma_tokens"].apply(lambda text: TextBlob(text).polarity)


if "parser" in nlp.pipe_names:
    st.header("Dependency Parse & Part-of-speech tags")
    st.sidebar.header("Dependency Parse")
    collapse_punct = st.sidebar.checkbox("Collapse punctuation", value=True)
    collapse_phrases = st.sidebar.checkbox("Collapse phrases",value=True)
    compact = st.sidebar.checkbox("Compact mode",value=True)
    options = {
        "collapse_punct": collapse_punct,
        "collapse_phrases": collapse_phrases,
        "compact": compact,
    }
    docs = [span.as_doc() for span in doc.sents] if split_sents else [doc]
    for sent in docs:
        html = displacy.render(sent, options=options)
        # Double newlines seem to mess with the rendering
        html = html.replace("\n\n", "\n")
        if split_sents and len(docs) > 1:
            st.markdown(f"> {sent.text}")
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

if "ner" in nlp.pipe_names:
    st.header("Named Entities")
    st.sidebar.header("Named Entities")
    label_set = nlp.get_pipe("ner").labels
    labels = st.sidebar.multiselect("Entity labels", label_set, label_set)
    html = displacy.render(doc, style="ent", options={"ents": labels})
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
    attrs = ["text", "label_", "start", "end", "start_char", "end_char"]
    if "entity_linker" in nlp.pipe_names:
        attrs.append("kb_id_")
    data = [
        [str(getattr(ent, attr)) for attr in attrs]
        for ent in doc.ents
        if ent.label_ in labels
    ]
    df = pd.DataFrame(data, columns=attrs)
    st.dataframe(df)


if "textcat" in nlp.pipe_names:
    st.header("Text Classification")
    st.markdown(f"> {text}")
    df = pd.DataFrame(doc.cats.items(), columns=("Label", "Score"))
    st.dataframe(df)


vector_size = nlp.meta.get("vectors", {}).get("width", 0)
if vector_size:
    st.header("Vectors & Similarity")
    st.code(nlp.meta["vectors"])
    text1 = st.text_input("Text or word 1", "apple")
    text2 = st.text_input("Text or word 2", "orange")
    doc1 = process_text(spacy_model, text1)
    doc2 = process_text(spacy_model, text2)
    similarity = doc1.similarity(doc2)
    if similarity > 0.5:
        st.success(similarity)
    else:
        st.error(similarity)

st.header("Token attributes")

if st.button("Show token attributes"):
    attrs = [
        "idx",
        "text",
        "lemma_",
        "pos_",
        "tag_",
        "dep_",
        "head",
        "ent_type_",
        "ent_iob_",
        "shape_",
        "is_alpha",
        "is_ascii",
        "is_digit",
        "is_punct",
        "like_num",
    ]
    data = [[str(getattr(token, attr)) for attr in attrs] for token in doc]
    df = pd.DataFrame(data, columns=attrs)
    st.dataframe(df)


st.header("Corpus-based analyses")

WORD_CLOUD = False
if WORD_CLOUD: 
    st.header("Word Cloud")


    # Create some sample text
    text = 'Fun, fun, awesome, awesome, tubular, astounding, superb, great, amazing, amazing, amazing, amazing'

    # Create and generate a word cloud image:
    wordcloud = WordCloud().generate(text)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    #st.pyplot()


import os
import ftfy
st.write(os.getcwd())
#os.chdir(r"C:\Users\us1145\Desktop\DataFrames/")
#home = os.environ
#st.write(home)
#st.write(os.getcwd())
path_to_file = (r".")
file_list = [f for f in os.listdir(path_to_file) if "xlsx" in f]
file_select = st.sidebar.selectbox("Select a file", file_list)
st.sidebar.title("Load Dataset")
data_loaded=False 
if st.sidebar.button("Click to load data"):
    with st.spinner("Loading "+ file_select + " ..."):
        data_df = pd.read_excel(file_select)
    st.success("File loaded!")
    pd.set_option('display.max_colwidth', -1)
    st.write("Processing data...")
    data_df["cleaned"] = data_df["Review"].apply(lambda text: ftfy.fix_text if isinstance(text, str) else "UNK")
    
    data_df["nlp"] = data_df["cleaned"].apply(nlp)
    data_df["lemma"] = data_df["nlp"].apply(lambda doc: " ".join([tok.text for tok in doc if not tok.is_stop and not tok.is_punct]))
    st.dataframe(data_df)
    data_loaded=True
    
if data_loaded:
    
    from corextopic import corextopic as ct
    st.header("Topics")

    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(
        max_df=.5,
        min_df=5,
        max_features=None,
        ngram_range=(1, 2),
        norm=None,
        binary=True,
        use_idf=True,
        sublinear_tf=False
    )

    vectorizer = vectorizer.fit(data_df['lemma'])
    tfidf = vectorizer.transform(data_df['lemma'])
    vocab = vectorizer.get_feature_names()
    N_TOPICS = 10

    anchors = []
    model = ct.Corex(n_hidden=N_TOPICS, seed=42)
    model = model.fit(
        tfidf,
        words=vocab
    )

    for i, topic_ngrams in enumerate(model.get_topics(n_words=20)):
        topic_ngrams = [ngram[0] for ngram in topic_ngrams if ngram[1] > 0]
        st.write("Topic #{}: {}".format(i+1, ", ".join(topic_ngrams)))
