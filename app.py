from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from flask_bootstrap import Bootstrap

app = Flask(__name__)
Bootstrap(app)

# AMBIL DATASET HARGA RUMAH
path_dataset = "data/harga_rumah.csv"
data = pd.read_csv(path_dataset)
features = data.columns[:-1]

# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    rounded_prediction = None  # Initialize outside the if block

    if request.method == 'POST':
        # SIMPAN INPUTAN USER
        user_input = [float(request.form[feature]) for feature in ["luas", "kasur", "km"]]

        # LOAD TRAINED MODEL
        model = KNeighborsRegressor(n_neighbors=3)
        model.fit(data[["luas", "kasur", "km"]], data["harga"])

        # MEMBUAT PREDIKSI HARGA
        prediction = model.predict([user_input])[0]
        rounded_prediction = int(round(prediction))

    return render_template('index.html', features=features, prediction=rounded_prediction)

if __name__ == '__main__':
    app.run(debug=True)
