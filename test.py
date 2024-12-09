# from transformers import pipeline, XLMRobertaTokenizer
#
# # Load the correct tokenizer for the model
# model_name = "joeddav/xlm-roberta-large-xnli"
# tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
# classifier = pipeline("zero-shot-classification", model=model_name, tokenizer=tokenizer)
#
#
# text = "Un mouton cest  bien plus intelligent que toi! "
# labels = list({"Politique", "Économie", "Santé", "Environnement", "Éducation", "Justice", "Sécurité", "Crise sociale",
#                "Intelligence artificielle", "Réseaux sociaux", "Cryptomonnaies", "Startups", "Gadgets", "Blockchain",
#                "Cyber-sécurité", "Innovation", "Musique", "Cinéma", "Séries", "Livres", "Art", "Jeux vidéo", "Humour",
#                "Mèmes", "Mode", "Football", "Basketball", "Tennis", "Cyclisme", "Natation", "Athlétisme", "Fitness",
#                "Jeux Olympiques", "Voyages", "Destinations", "Tourisme", "Aventure", "Nature", "Randonnée", "Cuisine",
#                "Recettes", "Restaurants", "Vins", "Nutrition", "Plats végétariens", "Vegan", "Fast-food", "Astronomie",
#                "Biologie", "Climat", "Écologie", "Santé", "Découvertes scientifiques", "Bien-être",
#                "Développement personnel", "Relations", "Psychologie", "Motivation", "Concerts", "Festivals",
#                "Expositions", "Spectacles", "Carnavals", "Investissements", "Cryptomonnaies", "Banques", "Finances",
#                "Entrepreneuriat", "Journalisme", "Actualités", "Réseaux sociaux", "Vidéos virales", "Podcast",
#                "Égalité", "LGBTQ+", "Changement climatique", "Droits de l'homme", "Justice sociale", "Albums",
#                "Chanteurs", "Groupes", "Artistes", "Expositions", "Peinture", "Sculpture", "Smartphones",
#                "Appareils électroniques", "Technologie mobile", "Applications", "Techwear", "Animal",
#                "Animaux de compagnie", "Vétérinaire", "Protection animale", "Complot"})
#
#
# def get_most_likely_topic(tweet, k=5):
#     result = classifier(text, candidate_labels=labels)
#     result = zip(result["labels"], result["scores"])
#     result = sorted(result, key=lambda x: x[1], reverse=True)
#
#     return result[:k]
#
# print(get_most_likely_topic(text))

import matplotlib.pyplot as plt

# Example data
hashtags = ['#python', '#datascience', '#ai', '#machinelearning', '#python',
            '#ai', '#python', '#datascience', '#ai', '#bigdata']

# Count occurrences of hashtags
from collections import Counter

hashtag_counts = Counter(hashtags)

# Get the top K hashtags
K = 3
top_k = hashtag_counts.most_common(K)

# Split data into labels and values
labels, values = zip(*top_k)

# Plot
plt.figure(figsize=(10, 8))  # Set figure size
plt.bar(labels, values, color='skyblue')  # Create bar chart
plt.xlabel('Hashtags', fontsize=12)
plt.ylabel('Counts', fontsize=12)
plt.title(f'Top {K} Most Used Hashtags', fontsize=14)
plt.tight_layout()  # Adjust layout for better fit
plt.show()