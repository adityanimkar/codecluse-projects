from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

def perform_kmeans(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(data)
    data['Cluster'] = clusters
    return data, kmeans

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            data = pd.read_csv(file_path)
            columns = data.columns.tolist()
            return render_template('index.html', data=data.head().to_html(), columns=columns)

    return render_template('index.html')

@app.route('/cluster', methods=['POST'])
def cluster():
    file_path = os.path.join('uploads', request.form['file_name'])
    data = pd.read_csv(file_path)
    columns = request.form.getlist('columns')
    num_clusters = int(request.form['num_clusters'])
    selected_data = data[columns]
    clustered_data, kmeans_model = perform_kmeans(selected_data, num_clusters)
    cluster_centers = pd.DataFrame(kmeans_model.cluster_centers_, columns=columns)

    if len(columns) == 2:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=columns[0], y=columns[1], hue='Cluster', data=clustered_data, palette='viridis')
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        plt.title('Customer Clusters')
        plot_path = os.path.join('static', 'cluster_plot.png')
        plt.savefig(plot_path)
        plt.close()

    return render_template('index.html', clustered_data=clustered_data.to_html(), cluster_centers=cluster_centers.to_html(), plot_path=plot_path if len(columns) == 2 else None)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
