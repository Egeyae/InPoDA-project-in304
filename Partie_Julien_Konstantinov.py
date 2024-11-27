import json
import unicodedata
import re
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from bertopic import BERTopic


def file_open(path):
    """
    Fonction qui permet d'ouvrir un fichier 
    """
    if type(path)!=str:
        print("Erreur sur le type du chemin")

    with open(path,"r") as file:
        jason=json.load(file)
    return jason



def text_cleaning(texte):
    """
    Fonction qui permet de filtrer les caractères spéciaux d'un texte.
    """
    texte_normalise = unicodedata.normalize('NFD', texte)
    texte_sans_accents = ''.join(c for c in texte_normalise if unicodedata.category(c) != 'Mn')
    texte_sans_retours = texte_sans_accents.replace('\n', '').replace('\r', '').replace("'", '') .replace("’", '').replace("\u200d",'')
    pattern_emojis = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
                                "\U0001F680-\U0001F6FF\U0001F700-\U0001F77F"
                                "\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF"
                                "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F"
                                "\U0001FA70-\U0001FAFF\U00002702-\U000027B0"
                                "\U000024C2-\U0001F251]+", flags=re.UNICODE)
    texte_sans_emojis = pattern_emojis.sub(r'', texte_sans_retours)
    return texte_sans_emojis



def special_caracters(jason):
    """
    Fonction qui permet de stocker les tweets sans caractères spéciaux dans un fichier JSON.
    """
    length=len(jason)
    dico={}
    for i in range(0,length):
        if jason[i]["text"].isalnum()==True:
            dico[i]=jason[i]["text"]
        else:
            dico[i]=text_cleaning(jason[i]["text"])
    
    with open("zone_d'atterissage.json",'w') as file2:
        json.dump(dico,file2,indent=2)
    return dico



def authors_list(jason):
    """
    Fonction permettant d'obtenir l'identification de l'auteur d'un tweet.
    """
    author_list=[]
    length=len(jason)
    for i in range(0,length):
        author_list.append(jason[i]["author_id"])
    return author_list



def hashtags_list(jason):
    """
    Fonction permettant d'obtenir la liste des hashtags présents dans un tweet. 
    """
    hashtag_list=[]
    length=len(jason)
    for i in range(0,length):
        keys1=list(jason[i].keys())
        if "entities" in keys1:
            keys2=list(jason[i]["entities"].keys())
            if "hashtags" in keys2:
                liste=jason[i]["entities"]["hashtags"]
                sous_liste=[]
                for j in range(0,len(liste)):
                    sous_liste.append("#"+str(liste[j]["tag"]))
                hashtag_list.append(str(sous_liste))
            else :
                hashtag_list.append([])
        else:
            hashtag_list.append([])
    return hashtag_list
    


def users_list(jason):
    """
    Fonction permettant d'obtenir la liste des utilisateurs mentionnés dans un tweet. 
    """
    user_list=[]
    length=len(jason)
    for i in range(0,length):
        keys1=list(jason[i].keys())
        if "entities" in keys1:
            keys2=list(jason[i]["entities"].keys())
            if "mentions" in keys2:
                liste=jason[i]["entities"]["mentions"]
                sous_liste=[]
                for j in range(0,len(liste)):
                    sous_liste.append("@"+liste[j]["username"])
                user_list.append(str(sous_liste))
            else :
                user_list.append([])
        else:
            user_list.append([])
    return user_list


#def top_K_hashtags(k,df):




# def topics(jason):

#     with open('topics_dict.json') as file:
#         topics_dico=json.load(file)

#     model = BERTopic()
#     print("Domaines détectés pour chaque phrase :")
#     length=len(jason)
#     for i in range(0,length):
#         topic = model.fit_transform(jason[i]["text"])
#         domain = topics_dico.get(topic, "Inconnu")  
#         print(f"Phrase : {jason[i]["text"]}")
#         print(f" - Sujet détecté : {topic}")
#         print(f" - Domaine général : {domain}")
#         print()

#     print("Détails sur les sujets et mots-clés associés :")
#     for topic_id in set(topics):
#         print(f"Sujet {topic_id} : {model.get_topic(topic_id)}")
#     return


def train_topic_model(phrases_dict):
    """
    Entraîne un modèle BERTopic basé sur les phrases du dictionnaire.
    Retourne le modèle et les résultats de l'entraînement.
    """
    phrases = list(phrases_dict.values())
    
    # Initialisation d'un modèle BERTopic avec un vectoriseur optimisé
    vectorizer_model = CountVectorizer(stop_words="english", max_df=0.9)
    topic_model = BERTopic(vectorizer_model=vectorizer_model)
    
    # Entraîne le modèle
    topics, probs = topic_model.fit_transform(phrases)
    
    return topic_model, topics, probs

def find_topic_for_new_phrase(topic_model, new_phrase, topics_dict):
    """
    Utilise un modèle BERTopic pour trouver le sujet le plus proche d'une nouvelle phrase.
    Retourne l'identifiant et le libellé du sujet.
    """
    topic_id = topic_model.transform([new_phrase])[0][0]  # Récupère l'ID du sujet
    topic_name = topics_dict.get(str(topic_id), "Sujet inconnu")
    return topic_id, topic_name


    

def main(path):
    data={}
    jason=file_open(path)
    clean_text=special_caracters(jason)
    data["Auteur"]=authors_list(jason)
    data["Hashtags"]=hashtags_list(jason)
    data["Mentions"]=users_list(jason)
    #print(topics(jason))
    df=pd.DataFrame(data)
    compte_authors=df["Auteur"].value_counts()
    compte_hashtags=df["Hashtags"].value_counts()
    compte_mentions=df["Mentions"].value_counts()

    topics_file = "topics_dict.json"  # Fichier contenant les sujets (clé: ID, valeur: nom du sujet)
    phrases_file = "sentences_dict.json"  # Fichier contenant les phrases associées aux sujets
    topics_dict = file_open(topics_file)
    phrases_dict = file_open(phrases_file)
    print("Entraînement du modèle BERTopic...")
    topic_model, topics, probs = train_topic_model(phrases_dict)
    print("Modèle entraîné avec succès.")
    new_phrase = input("Entrez une nouvelle phrase pour identifier son sujet : ")
    new_phrase = "".join([x for x in new_phrase.split() if not x.startswith("#")])
    topic_id, topic_name = find_topic_for_new_phrase(topic_model, new_phrase, topics_dict)
    
    print(f"Sujet identifié : {topic_name} (ID: {topic_id})")
   
    return


print(main('versailles_tweets_100.json'))

