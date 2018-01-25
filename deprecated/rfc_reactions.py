from deprecated import fetch_reactions

token_matrix, reaction_matrix = fetch_reactions.reactions("data/user")

# w2v = glove6B()
#
# rfc_w2v = Pipeline([
#     ("MeanEmbeddingVectorizer", MeanEmbeddingVectorizer(w2v)),
#     ("RandomForestClassifier", RandomForestClassifier(n_estimators=200, n_jobs=7))])

print("Reaction skew for +1", sum(reaction_matrix[:, 4])/len(reaction_matrix[:, 4]))

# print(cross_val_score(rfc_w2v, token_matrix, reaction_matrix[:, 0], cv=5, scoring='recall_macro'))

# rfc_w2v_tfidf = Pipeline([
#     ("TfidfEmbeddingVectorizer", TfidfEmbeddingVectorizer(w2v)),
#     ("RandomForestClassifier", RandomForestClassifier(n_estimators=200))])