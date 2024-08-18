from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import os
import pandas as pd
from stream_topic.utils import TMDataset  # Importing your package
from stream_topic.models import KmeansTM, LDA, NMFTM  # Replace with actual models
from stream_topic.visuals import (
    visualize_topic_model,
    visualize_topics,
    visualize_topics_as_wordclouds,
)

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Replace with a secure key
app.config["UPLOAD_FOLDER"] = "web_app/uploads/"
app.config["DATASET_FOLDER"] = "web_app/datasets/"  # Directory to save datasets

# Global variables to store the dataset and selected model type
dataset = None
selected_model_type = None  # New global variable to store selected model type


@app.route("/")
def upload_file():
    return render_template("upload_files.html")


@app.route("/upload", methods=["POST"])
def upload_file_post():
    global dataset
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)

    file = request.files["file"]

    if file.filename == "":
        flash("No selected file")
        return redirect(request.url)

    if file:
        # Ensure the upload and dataset folders exist
        if not os.path.exists(app.config["UPLOAD_FOLDER"]):
            os.makedirs(app.config["UPLOAD_FOLDER"])
        if not os.path.exists(app.config["DATASET_FOLDER"]):
            os.makedirs(app.config["DATASET_FOLDER"])

        # Save the uploaded file
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # Load the CSV file into a pandas DataFrame
        data = pd.read_csv(filepath)[0:500]
        data["Labels"] = None

        if "text" not in data.columns:
            flash('CSV file must contain a "text" column.')
            return redirect(request.url)

        # Create TMDataset
        dataset = TMDataset()
        dataset.create_load_save_dataset(
            data=data,
            dataset_name="data",
            save_dir=f"{app.config['DATASET_FOLDER']}/",
            doc_column="text",
            label_column="Labels",
        )

        # Redirect to the model selection page
        return redirect(url_for("preprocess"))


@app.route("/preprocess", methods=["GET", "POST"])
def preprocess():
    global dataset, selected_model_type

    if request.method == "POST":
        selected_model_type = request.form["model"]  # Store the selected model globally

        dataset.fetch_dataset(
            name="data",
            dataset_path=app.config["DATASET_FOLDER"],
            source="local",
        )

        # Preprocess the dataset with the selected model
        dataset.preprocess(model_type=selected_model_type)

        # Show preprocessing completed message
        return render_template(
            "preprocess.html",
            preprocessing_message="Preprocessing completed. You can now download the dataset or continue without downloading.",
        )

    return render_template("preprocess.html")


@app.route("/download_and_continue")
def download_and_continue():
    global dataset
    processed_file_path = os.path.join(
        app.config["DATASET_FOLDER"], "processed_dataset.csv"
    )
    dataset.dataframe.to_csv(processed_file_path, index=False)

    # Send the file and redirect to model fitting after download
    return send_file(
        processed_file_path, as_attachment=True, download_name="processed_dataset.csv"
    )


@app.route("/fit_model", methods=["GET", "POST"])
def fit_model():
    global model, dataset, selected_model_type

    if request.method == "POST":
        num_topics = int(request.form["num_topics"])

        # Ensure the dataset exists and has been preprocessed
        if dataset is None or not hasattr(dataset, "dataframe"):
            flash("Dataset not found or not preprocessed.")
            return redirect(url_for("upload_file"))

        # Add message to be displayed before fitting the model
        message = "Starting model fitting now..."

        # Fit the model based on the globally selected model type
        if selected_model_type == "KmeansTM":
            model = KmeansTM()
        elif selected_model_type == "LDA":
            model = LDA()
        elif selected_model_type == "NMF":
            model = NMFTM()
        else:
            flash("Invalid model selection.")
            return redirect(url_for("upload_file"))

        # Fit the model
        model.fit(dataset, n_topics=num_topics)

        topics = model.get_topics(n_words=15)  # Assuming this method exists
        return render_template("results.html", results=topics, message=message)

    return render_template("fit_model.html")


@app.route("/visualizations", methods=["GET", "POST"])
def visualizations():
    global model, dataset  # Declare the model as global to access the fitted model

    if request.method == "POST":
        visualization_choice = request.form["visualization"]

        # Handle the visualization based on the user's choice
        if visualization_choice == "topic_overview":
            visualize_topic_model(
                model,
                dataset=dataset,
                reduce_first=True,
                port=8052,
            )
        elif visualization_choice == "word_cloud":
            visualize_topics_as_wordclouds(
                model,
                max_words=20,
            )
        else:
            flash("Invalid visualization selection.")
            return redirect(url_for("visualizations"))

        return redirect(url_for("upload_file"))  # Redirect after visualization

    return render_template("visualizations.html")


if __name__ == "__main__":
    app.run(debug=True)
