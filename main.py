import ast
import json
import re
import unicodedata #module pour la manipulation de caractères unicode
from collections import Counter
import torch
import pandas as pd
from transformers import pipeline, XLMRobertaTokenizer
from textblob import TextBlob


def file_open(path):
    """
    Fonction qui permet d'ouvrir un fichier 
    """
    if type(path) != str: #On vérifie que le chemin renseigné est bien de type str
        print("Erreur sur le type du chemin")

    with open(path, "r", encoding="utf-8", errors="replace") as file: #On ouvre le fichier json en remplaçant les erreurs d'encodage si il y en a
        jason = json.load(file)
    
    return jason


def text_cleaning(texte):
    """
    Fonction qui permet de filtrer les caractères spéciaux d'un texte.
    """
    texte_normalise = unicodedata.normalize('NFD', texte) #On décompose le texte pour accéder aux codes des caractères spéciaux
    texte_sans_accents = ''.join(c for c in texte_normalise if unicodedata.category(c) != 'Mn') #On enlève les accents, qui sont des caractères de catégorie 'Mn'
    #On compile les émojis en renseignant leur pattern
    pattern_emojis = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
                                "\U0001F680-\U0001F6FF\U0001F700-\U0001F77F"
                                "\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF"
                                "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F"
                                "\U0001FA70-\U0001FAFF\U00002702-\U000027B0"
                                "\U000024C2-\U0001F251]+", flags=re.UNICODE)
    texte_sans_emojis = pattern_emojis.sub(r'', texte_sans_accents) #On enlève les émojis en fonction de l'expression régulière de leur pattern
    texte_sans_lien = re.sub(r"https?://\S+", "", texte_sans_emojis) #On enlève tous les liens du texte
    texte_sans_arobases = re.sub(r"@\w+", "", texte_sans_lien) #On enlève tous les arobases et ce qui suit
    texte_sans_hashtags = re.sub(r"#\w+", "", texte_sans_arobases) #On enlève tous les hashtags et ce qui suit
    texte_sans_parasites = texte_sans_hashtags.replace('\n', '').replace('\r', '').replace("'", '').replace("’",'').replace("\u200d", '').replace("@", '').replace("#", '') #On enlève les retours à la ligne, les apostrophes, etc...
    texte_final = re.sub(r"\s+", " ", texte_sans_parasites).strip() #On détermine notre texte final sans espaces en trop
    
    return texte_final


def special_caracters(jason):
    """
    Fonction qui permet de stocker les tweets sans caractères spéciaux dans un fichier JSON.
    """
    length = len(jason)
    dico = {}
    for i in range(0, length): 
        #On parcourt le fichier jason, si le texte est constitué de lettres et de chiffres uniquement, on l'ajoute au dictionnaire, sinon on nettoie le texte d'abord
        if jason[i]["text"].isalnum():
            dico[i] = jason[i]["text"]
        else:
            dico[i] = text_cleaning(jason[i]["text"])

    with open("zone_d'atterissage.json", 'w') as file2:
        json.dump(dico, file2, indent=2)

    return dico


def authors_list(jason):
    """
    Fonction qui permet d'obtenir la liste des auteurs des tweets.
    """
    author_list = []
    length = len(jason)
    for i in range(0, length): 
        author_list.append(jason[i]["author_id"])

    return author_list


def hashtags_list(jason):
    """
    Fonction qui permet d'obtenir la liste des hashtags présents dans un tweet. 
    """
    hashtag_list = []
    length = len(jason)
    for i in range(0, length): 
        #Si le tweet contient une section "entities" puis "hashtags", alors on ajoute le hashtag, sinon on ajoute une liste vide
        keys1 = list(jason[i].keys())
        if "entities" in keys1:
            keys2 = list(jason[i]["entities"].keys())
            if "hashtags" in keys2:
                liste = jason[i]["entities"]["hashtags"]
                sous_liste = []
                for j in range(0, len(liste)):
                    sous_liste.append("#" + str(liste[j]["tag"]))
                hashtag_list.append(str(sous_liste))
            else:
                hashtag_list.append([])
        else:
            hashtag_list.append([])
            
    return hashtag_list


def users_list(jason):
    """
    Fonction qui permet d'obtenir la liste des utilisateurs mentionnés dans un tweet. 
    """
    df = pd.DataFrame(columns=["Users"])
    user_list = []
    length = len(jason)
    for i in range(0, length):
        #Si le tweet contient une section "entities" puis "mentions", alors on ajoute le hashtag, sinon on ajoute une liste vide
        keys1 = list(jason[i].keys())
        if "entities" in keys1:
            keys2 = list(jason[i]["entities"].keys())
            if "mentions" in keys2:
                liste = jason[i]["entities"]["mentions"]
                sous_liste = []
                for j in range(0, len(liste)):
                    sous_liste.append("@" + liste[j]["username"])
                user_list.append(str(sous_liste))
            else:
                user_list.append([])
        else:
            user_list.append([])

    return user_list


def contenu_list(jason):
    """
    Fonction qui permet de retourner la liste des textes des tweets.
    """
    contenus = []
    length = len(jason)
    for i in range(0, length):
        contenus.append(jason[i]["text"])

    return contenus


def sentiment_list(jason):
    sentiments = []
    length = len(jason)

    def polarity_to_sentiment(polarity):
        if polarity == 0:
            return "neutral"
        elif polarity > 0:
            return "positive"
        else:
            return "negative"

    for i in range(0, length):
        text = text_cleaning(jason[i]["text"])
        if text:
            sentiments.append(polarity_to_sentiment(TextBlob(text).sentiment.polarity))
        else:
            sentiments.append("N/A")

    return sentiments


def start_model():
    """
    Fonction qui permet de démarrer le modèle Roberta de transformers.
    """
    models = {
        "small": "xlm-roberta-base", # Warning ! This is not very precise
        "big": "joeddav/xlm-roberta-large-xnli"  # Warning ! This is very slow
    }
    model_name = models["small"] #Choix du modèle
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    #Utilisation de la pipeline de classification zéro-shot de Hugging Face Transformers
    classifier = pipeline("zero-shot-classification", model=model_name, tokenizer=tokenizer, device=device) 

    return classifier


def topics(jason, classifier, k=5):
    """
    Fonction qui permet de trouver parmis une liste de sujets celui qui correspond le plus pour une phrase donnée.
    """
    topic_list = []
    score_list = []

    labels = ["Physique", "Chimie", "Biologie", "Astronomie", "Géologie",
              "Climatologie", "Océanographie", "Écologie", "Zoologie", "Botanique",
              "Génétique", "Biotechnologie", "Histoire", "Géographie", "Sociologie",
              "Anthropologie", "Archéologie", "Psychologie", "Sciences politiques",
              "Économie", "Démographie", "Philosophie", "Éthique", "Linguistique",
              "Études culturelles", "Ingénierie civile", "Ingénierie mécanique",
              "Ingénierie électrique", "Ingénierie informatique", "Ingénierie aérospatiale",
              "Ingénierie chimique", "Robotique", "Intelligence artificielle", "Blockchain",
              "Cybersécurité", "Internet des objets", "Nanotechnologie", "Médecine générale",
              "Médecine dentaire", "Pharmacie", "Infirmerie", "Nutrition", "Kinésithérapie",
              "Chirurgie", "Psychiatrie", "Pédiatrie", "Médecine vétérinaire", "Santé publique",
              "Neurosciences", "Mathématiques", "Statistiques", "Informatique", "Logique",
              "Théorie des jeux", "Recherche opérationnelle", "Cryptographie", "Littérature",
              "Arts visuels", "Musique", "Théâtre", "Cinéma", "Design", "Architecture",
              "Études religieuses", "Histoire de l'art", "Poésie", "Finance", "Comptabilité",
              "Marketing", "Gestion des ressources humaines", "Entrepreneuriat", "Commerce international",
              "Logistique", "Stratégie d’entreprise", "Économie comportementale",
              "Économie verte", "Droit civil", "Droit pénal", "Droit commercial",
              "Droit du travail", "Droit international", "Droit de la propriété intellectuelle",
              "Droit des technologies", "Criminologie", "Médiation juridique", "Pédagogie",
              "Didactique", "Éducation spécialisée", "Formation continue", "Psychologie de l’éducation",
              "Politique éducative", "Journalisme", "Relations publiques", "Publicité",
              "Cinématographie", "Rédaction technique", "Études médiatiques", "Communication numérique",
              "Réseaux sociaux", "Agriculture", "Agronomie", "Sylviculture", "Horticulture",
              "Agroécologie", "Gestion des ressources naturelles", "Énergies renouvelables",
              "Développement durable", "Pollution et gestion des déchets", "Entraînement sportif",
              "Kinésiologie", "Médecine du sport", "Tourisme", "Hôtellerie", "Gastronomie",
              "Jeux vidéo", "Esports", "Sécurité informatique", "Sécurité nationale",
              "Politiques de défense", "Renseignement", "Gestion des catastrophes",
              "Sécurité privée", "Droit international humanitaire", "Études interdisciplinaires",
              "Science des données", "Sciences des matériaux", "Études sur le genre",
              "Études sur les minorités", "Recherche scientifique", "Innovation sociale"]

    length = len(jason)
    texts = [text_cleaning(jason[i]["text"]) for i in range(length)] #On nettoie le texte pour avoir une bonne base pour l'analyse

    for i, txt in enumerate(texts):
        if txt:
            result = classifier(txt, candidate_labels=labels)
            result = zip(result["labels"], result["scores"]) #Création de tuples de la forme (label, score)
            result = sorted(result, key=lambda x: x[1], reverse=True) #Tri en fonction du score
            #Ajout des premiers topics et des premiers scores obtenus
            topic_list.append(result[:k][0][0]) 
            score_list.append(result[:k][0][1])
        else: #Si il n'y a pas de texte
            topic_list.append("None")
            score_list.append("None")
        print(f"Tweet n°{i} analysé")

    return topic_list, score_list


# def topics():

#     nlp = spacy.load('fr_core_news_sm')
#     sentence = "Le réchauffement climatique fait peur."
#     doc = nlp(sentence)
#     for token in doc :
#         if 'subj' in token.dep_ :
#             print("Sujet trouvé :", token.text)

#     return


def column_to_list(liste):
    """
    Fonction qui permet d'éliminer les sous-listes d'une liste pour avoir tous les éléments dans une liste.
    """
    liste_plate = []
    for element in liste:
        #Si l'élément est une sous-liste, on ajoute ses éléments a la n ouvelle liste, sinon si c'est un élément seul, on l'ajoute aussi
        if isinstance(element, str) and element.startswith("["):
            sous_liste = ast.literal_eval(element)
            liste_plate.extend(sous_liste)
        elif isinstance(element, list):
            liste_plate.extend(element)

    return liste_plate


def top_K_hashtags(df, K):
    """
    Fonction qui permet de renvoyer le top des hashtags les plus récurrents.
    """
    dataframe_list = df["Hashtags"].to_list()
    hashtags_list = column_to_list(dataframe_list)
    compteur = Counter(hashtags_list) #On utilise la classe Counter pour compter l'occurence des éléments
    max_hashtags = compteur.most_common(K)

    return max_hashtags


def top_K_authors(df, K):
    """
    Fonction qui permet de renvoyer le top des auteurs les plus actifs.
    """
    compte_authors = df["Auteur"].value_counts()
    compteur = Counter(compte_authors.to_dict()) #On convertit en dictionnaire pour pouvoir utiliser Counter
    max_authors = compteur.most_common(K)
    return max_authors


def top_K_mentions(df, K):
    """
    Fonction qui permet de renvoyer le top des auteurs les plus mentionnés.
    """
    dataframe_list = df["Mentions"].to_list()
    mentions_list = column_to_list(dataframe_list)
    compteur = Counter(mentions_list)
    max_mentions = compteur.most_common(K)

    return max_mentions


def top_K_topics(df, K):
    """
    Fonction qui permet de renvoyer le top des sujets les plus récurrents.
    """
    compte_topics = df["Topics"].value_counts()
    compteur = Counter(compte_topics.to_dict())
    max_topics = compteur.most_common(K)

    return max_topics


def nombre_publications_authors(df):
    """
    Fonction qui permet de compter le nombre de publications par auteur.
    """
    compte_authors = df["Auteur"].value_counts()
    authors_df = compte_authors.reset_index() #On réinitialise les indexs
    authors_df.columns = ["Auteur", "Nombre de publications"]

    return authors_df


def nombre_publications_hashtags(df):
    """
    Fonction qui permet de compter le nombre de publications par hashtag.
    """
    dico = {}
    dataframe_list = df["Hashtags"].to_list()
    hashtags_list = column_to_list(dataframe_list)
    compteur = Counter(hashtags_list)
    dico["Hashtag"] = list(compteur.keys())
    dico["Nombre de publications"] = list(compteur.values())

    return pd.DataFrame(dico)


def nombre_publications_topics(df):
    """
    Fonction qui permet de compter le nombre de publications par sujet.
    """
    compte_topics = df["Topics"].value_counts()
    topics_df = compte_topics.reset_index()
    topics_df.columns = ["Topic", "Nombre de publications"]

    return topics_df


def tweets_mentionning_specific_user(user, df):
    """
    Fonction qui permet de retourner tous les tweets qui mentionnent un utilisateur en particulier.
    """
    #On parcourt la colonne "Mentions" et on voit les lignes contenant l'utilisateur choisi, on affiche ensuite l'utilisateur et le contenu du tweet associé
    result = df.loc[df["Mentions"].apply(lambda users: user in users), ["Contenu", "Mentions"]]
    return result


def users_mentionning_specific_hashtag(hashtag, df):
    """
    Fonction qui permet de retourner tous les auteurs qui mentionnent un hashtag en particulier.
    """
    result = df.loc[df["Hashtags"].apply(lambda hashtags: hashtag in hashtags), ["Auteur", "Hashtags"]]
    return result


def tweets_to_df(jason):
    """
    Fonction qui retourne un dataframe à partir d'un dictionnaire.
    """
    data = {
        "Auteur": authors_list(jason),
        "Hashtags": hashtags_list(jason),
        "Mentions": users_list(jason),
        "Contenu": contenu_list(jason),
        "Sentiment": sentiment_list(jason),
    }
    classifier = start_model()
    data["Topics"] = topics(jason, classifier, k=5)[0]
    return pd.DataFrame(data)


def main(path):
    data = {}
    jason = file_open(path)
    clean_text = special_caracters(jason)
    K = int(input("Entrez un entier"))
    df = tweets_to_df(jason)
    top_hashtags = top_K_hashtags(df, K)
    top_authors = top_K_authors(df, K)
    top_mentions = top_K_mentions(df, K)
    top_topics = top_K_topics(df, K)
    publications_authors = nombre_publications_authors(df)
    publications_hashtags = nombre_publications_hashtags(df)
    publications_topics = nombre_publications_topics(df)

    return tweets_mentionning_specific_user("leonna_julie", df)


if __name__ == "__main__":
    print(main('versailles_tweets_100.json'))
