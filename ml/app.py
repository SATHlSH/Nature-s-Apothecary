from flask import Flask, render_template, request, jsonify, send_file
import pickle
import os

app = Flask(__name__)

# Load the model and TF-IDF vectorizer
with open('model/herb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/predict', methods=['POST']) 
def predict():
    symptoms = request.form['symptoms']
    
    # Preprocess the input
    symptoms_vectorized = tfidf_vectorizer.transform([symptoms])

    # Make prediction
    prediction = model.predict(symptoms_vectorized)

    return jsonify({'herb': prediction[0]})

@app.route('/output')
def output():
    """Generate output file and send it for download."""
    # Create or generate the output data
    output_data = "This is the output from the Python script."
    
    # Save the output to a file
    output_file_path = "output.txt"
    with open(output_file_path, "w") as file:
        file.write(output_data)

    return send_file(output_file_path, as_attachment=True)

if __name__ == '__main__':
    # Ensure output.txt is removed on each run for fresh output generation
    if os.path.exists("output.txt"):
        os.remove("output.txt")
    app.run(debug=True)