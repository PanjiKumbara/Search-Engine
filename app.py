import os
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Muat dokumen dari folder
def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # Hanya proses file .txt
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents

# Muat dataset dari folder
documents_folder = 'documents'  # Sesuaikan path ini jika perlu
df = pd.DataFrame(load_documents(documents_folder), columns=['text'])

# Vektorisasi TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['text'])

@app.route('/', methods=['GET', 'POST'])
def index():
    query = None
    results = pd.DataFrame(columns=['Document', 'Similarity'])  # DataFrame kosong sebagai default

    if request.method == 'POST':
        query = request.form['query']
        query_vec = vectorizer.transform([query])

        # Hitung cosine similarity antara query dan dokumen
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

        # Buat DataFrame hasil pencarian
        results = pd.DataFrame({
            'Document': df['text'],
            'Similarity': similarities
        }).sort_values(by='Similarity', ascending=False)

    # Render template dengan query dan hasil
    return render_template('index.html', query=query, results=results)

if __name__ == '__main__':
    app.run(debug=True)
