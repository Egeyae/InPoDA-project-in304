import json


def special_caracters(path):
    """
    Fonction qui permet de filtrer les caractères spéciaux de tweets.
    """
    if type(path)!=str:
        print("Erreur sur le type du chemin")

    with open(path,"r") as file:
        jason=json.load(file)

    length=len(jason)
    dico={}
    for i in range(0,length):
        if jason[i]["text"].isalnum()==True:
            dico[i]=jason[i]["text"]
        else:
            dico[i]=''.join(filter(str.isalnum,jason[i]["text"]))
    
    with open("zone_d'atterissage.json",'w') as file2:
        json.dump(dico,file2)
    
    return dico



def author_ID(path):
    """
    Fonction permettant d'obtenir l'identification de l'auteur d'un tweet.
    """
    if type(path)!=str:
        print("Erreur sur le type du chemin")

    with open(path,"r") as file:
        jason=json.load(file)

    length=len(jason)
    for i in range(0,length):
        print("L'auteur du tweet n°"+str(i)+" est "+str(jason[i]["author_id"]))

    return



def hashtags_list(path):
    """
    Fonction permettant d'obtenir la liste des hashtags présents dans un tweet. 
    """
    if type(path)!=str:
        print("Erreur sur le type du chemin")

    with open(path,"r") as file:
        jason=json.load(file)

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
    


def users_list(path):
    """
    Fonction permettant d'obtenir la liste des utilisateurs mentionnés dans un tweet. 
    """
    if type(path)!=str:
        print("Erreur sur le type du chemin")

    with open(path,"r") as file:
        jason=json.load(file)

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

