import json
import unicodedata
import re
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



def author_ID(jason):
    """
    Fonction permettant d'obtenir l'identification de l'auteur d'un tweet.
    """
    length=len(jason)
    for i in range(0,length):
        print("L'auteur du tweet n°"+str(i)+" est "+str(jason[i]["author_id"]))
    return



def hashtags_list(jason):
    """
    Fonction permettant d'obtenir la liste des hashtags présents dans un tweet. 
    """
    length=len(jason)
    for i in range(0,length):
        keys1=list(jason[i].keys())
        if "entities" in keys1:
            keys2=list(jason[i]["entities"].keys())
            if "hashtags" in keys2:
                liste=jason[i]["entities"]["hashtags"]
                liste2=[]
                for j in range(0,len(liste)):
                    liste2.append("#"+liste[j]["tag"])
                print("Tweet n°"+str(i)+" : "+str(liste2))
            else :
                print("Tweet n°"+str(i)+" : Il n'y a pas de hashtags sur ce tweet")
        else:
            print("Tweet n°"+str(i)+" : Il n'y a pas d'entité sur ce tweet")
    return
    


def users_list(jason):
    """
    Fonction permettant d'obtenir la liste des utilisateurs mentionnés dans un tweet. 
    """
    length=len(jason)
    for i in range(0,length):
        keys1=list(jason[i].keys())
        if "entities" in keys1:
            keys2=list(jason[i]["entities"].keys())
            if "mentions" in keys2:
                liste=jason[i]["entities"]["mentions"]
                liste2=[]
                for j in range(0,len(liste)):
                    liste2.append("@"+liste[j]["username"])
                print("Tweet n°"+str(i)+" : "+str(liste2))
            else :
                print("Tweet n°"+str(i)+" : Il n'y a pas d'utilisateurs mentionnés sur ce tweet")
        else:
            print("Tweet n°"+str(i)+" : Il n'y a pas d'entité sur ce tweet")
    return


def topics(jason):
    topic_list=[]
    topic_model=BERTopic.load("MaartenGr/BERTopic_Wikipedia")
    length=len(jason)
    for i in range(0,length):
        topic=topic_model.transform(jason[i]["text"])
        topic_list.append(topic)
    for j in range(len(topic_list)):
        print("Le sujet du tweet n°"+str(j)+" est : "+str(topic_list[j]))
    return topic_list



def main(path):
    jason=file_open(path)
    clean_text=special_caracters(jason)
    print(clean_text)
    print(author_ID(jason))
    print(hashtags_list(jason))
    print(users_list(jason))
    print(topics(jason))
    return


print(main('versailles_tweets_100.json'))

