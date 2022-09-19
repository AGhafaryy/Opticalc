import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
ds = pd.read_csv('course data.csv') #replace with your own csv file
ds = ds.iloc[:,:3]


tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words= "english")
tfidf_matrix = tf.fit_transform(ds["description"])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix) 
results = {}

for idx, row in ds.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], ds['ID'][i]) for i in similar_indices]
    results[row['ID']] = similar_items[1:]

    

def index(course):
  return ds.loc[ds['course ']==course].iat[0,0] #fast access using https://stackoverflow.com/questions/16729574/how-to-get-a-value-from-a-cell-of-a-dataframe
def recommend(course, n):
    id = index(course)
    print("Here are "+str(n)+" courses similar to "+course+":")   
    print("   ")
    recomms = results[id][:n]   
    for recomm in recomms: 
       print(ds['course '][recomm[1]-1]+" (description:" +ds['description'][recomm[1]-1] + ")")
       print("")

recommend('EECE 310',3)

