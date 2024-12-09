import ast
import json
import re
import unicodedata
from collections import Counter

import pandas as pd
from transformers import pipeline, XLMRobertaTokenizer


def file_open(path):
    """
    Fonction qui permet d'ouvrir un fichier 
    """
    if type(path) != str:
        print("Erreur sur le type du chemin")

    with open(path, "r", encoding="utf-8", errors="replace") as file:
        jason = json.load(file)
    return jason


def text_cleaning(texte):
    """
    Fonction qui permet de filtrer les caractères spéciaux d'un texte.
    """
    texte_normalise = unicodedata.normalize('NFD', texte)
    texte_sans_accents = ''.join(c for c in texte_normalise if unicodedata.category(c) != 'Mn')
    pattern_emojis = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
                                "\U0001F680-\U0001F6FF\U0001F700-\U0001F77F"
                                "\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF"
                                "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F"
                                "\U0001FA70-\U0001FAFF\U00002702-\U000027B0"
                                "\U000024C2-\U0001F251]+", flags=re.UNICODE)
    texte_sans_emojis = pattern_emojis.sub(r'', texte_sans_accents)
    texte_sans_lien = re.sub(r"https?://\S+", "", texte_sans_emojis)
    texte_sans_arobases = re.sub(r"@\w+", "", texte_sans_lien)
    texte_sans_hashtags = re.sub(r"#\w+", "", texte_sans_arobases)
    texte_sans_parasites = texte_sans_hashtags.replace('\n', '').replace('\r', '').replace("'", '').replace("’",
                                                                                                            '').replace(
        "\u200d", '').replace("@", '').replace("#", '')
    texte_final = re.sub(r"\s+", " ", texte_sans_parasites).strip()
    return texte_final


def special_caracters(jason):
    """
    Fonction qui permet de stocker les tweets sans caractères spéciaux dans un fichier JSON.
    """
    length = len(jason)
    dico = {}
    for i in range(0, length):
        if jason[i]["text"].isalnum() == True:
            dico[i] = jason[i]["text"]
        else:
            dico[i] = text_cleaning(jason[i]["text"])

    with open("zone_d'atterissage.json", 'w') as file2:
        json.dump(dico, file2, indent=2)
    return dico


def authors_list(jason):
    """
    Fonction qui permet d'obtenir l'identification de l'auteur d'un tweet.
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
    contenus = []
    length = len(jason)
    for i in range(0, length):
        contenus.append(jason[i]["text"])
    return contenus


def start_model():
    models = {
        "small": "xlm-roberta-base",
        "big": "joeddav/xlm-roberta-large-xnli"  # Warning ! This is very slow
    }
    model_name = models["small"]
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    classifier = pipeline("zero-shot-classification", model=model_name, tokenizer=tokenizer)

    return classifier


def topics(jason, classifier, k=5):
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

    texts = [text_cleaning(jason[i]["text"]) for i in range(length)]

    for i, txt in enumerate(texts):
        if txt:
            result = classifier(txt, candidate_labels=labels)
            result = zip(result["labels"], result["scores"])
            result = sorted(result, key=lambda x: x[1], reverse=True)
            topic_list.append(result[:k][0][0])
            score_list.append(result[:k][0][1])
        else:
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
    liste_plate = []
    for element in liste:
        if isinstance(element, str) and element.startswith("["):
            sous_liste = ast.literal_eval(element)
            liste_plate.extend(sous_liste)
        elif isinstance(element, list):
            liste_plate.extend(element)
    return liste_plate


def top_K_hashtags(df, K):
    dataframe_list = df["Hashtags"].to_list()
    hashtags_list = column_to_list(dataframe_list)
    compteur = Counter(hashtags_list)
    max_hashtags = compteur.most_common(K)

    return max_hashtags


def top_K_authors(df, K):
    compte_authors = df["Auteur"].value_counts()
    #max_authors = compte_authors.nlargest(K)
    compteur = Counter(compte_authors)
    max_authors = compteur.most_common(K)
    return max_authors


def top_K_mentions(df, K):
    compte_mentions = df["Mentions"].value_counts()
    dataframe_list_2 = df["Mentions"].to_list()
    mentions_list = column_to_list(dataframe_list_2)
    compteur2 = Counter(mentions_list)
    max_mentions = compteur2.most_common(K)

    return max_mentions


def top_K_topics(df, K):
    compte_topics = df["Topics"].value_counts()
    # max_topics = compte_topics.nlargest(K)
    topics_list = column_to_list(compte_topics)
    compteur = Counter(topics_list)
    max_topics = compteur.most_common(K)

    return max_topics


def nombre_publications_authors(df):
    compte_authors = df["Auteur"].value_counts()
    return compte_authors


def nombre_publications_hashtags(df):
    dataframe_list = df["Hashtags"].to_list()
    hashtags_list = column_to_list(dataframe_list)
    compteur = Counter(hashtags_list)
    for element, count in compteur.items():
        print(element, count)
    return


def nombre_publications_topics(df):
    compte_topics = df["Topics"].value_counts()
    return compte_topics


def tweets_to_df(jason):
    data = {
        "Auteur": authors_list(jason),
        "Hashtags": hashtags_list(jason),
        "Mentions": users_list(jason),
        "Contenu": contenu_list(jason)
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

    return df, top_hashtags, top_authors, top_mentions, top_topics, publications_authors, publications_hashtags, publications_topics


if __name__ == "__main__":
    print(main('versailles_tweets_100.json'))
