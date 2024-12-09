from flask import Flask, render_template, request, send_from_directory
import os
import torch
import open_clip
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

app = Flask(__name__)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"


class ImageRetriever:
    def __init__(
        self,
        embeddings_file="image_embeddings.pickle",
        image_dir="coco_images_resized",
    ):
        if not os.path.exists(embeddings_file):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        self.image_dir = image_dir
        self.data_frame = pd.read_pickle(embeddings_file)
        self.image_names = self.data_frame["file_name"].values
        self.feature_vectors = np.stack(self.data_frame["embedding"].values)

        # Load the CLIP model
        self.model_name = "ViT-B-32"
        self.pretrained = "openai"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self.model = self.model.to(device)
        self.model.eval()

        self.tokenizer = open_clip.get_tokenizer(self.model_name)

        # Precompute PCA embeddings
        self.pca = PCA(n_components=50)
        self.embeddings_pca = self.pca.fit_transform(self.feature_vectors)

    def encode_text(self, text_query):
        with torch.no_grad():
            tokens = self.tokenizer([text_query]).to(device)
            text_emb = self.model.encode_text(tokens)
            return text_emb.cpu().numpy().squeeze()

    def encode_image(self, image_file):
        img = Image.open(image_file).convert("RGB")
        img_tensor = self.preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            img_emb = self.model.encode_image(img_tensor)
        return img_emb.cpu().numpy().squeeze()

    def combine_embeddings(self, text_emb, image_emb, lam=0.5):
        combined = lam * text_emb + (1 - lam) * image_emb
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined /= norm
        return combined

    def search_similar(self, query_emb, top_k=5, use_pca=False):
        if use_pca:
            query_emb = self.pca.transform(query_emb.reshape(1, -1))[0]
            query_emb /= np.linalg.norm(query_emb)
            embeddings = self.embeddings_pca
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        else:
            query_emb /= np.linalg.norm(query_emb)
            embeddings = self.feature_vectors
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        sim_scores = embeddings @ query_emb
        top_indices = np.argsort(sim_scores)[::-1][:top_k]
        return [(self.image_names[i], sim_scores[i]) for i in top_indices]


retriever = ImageRetriever(
    embeddings_file="image_embeddings.pickle", image_dir="coco_images_resized"
)


@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    error_message = None

    if request.method == "POST":
        query_type = request.form.get("query_type", "text")
        text_query = request.form.get("text_query", "").strip()
        use_pca = request.form.get("use_pca") == "on"
        image_query_file = request.files.get("image_query")
        hybrid_weight = float(request.form.get("hybrid_weight", 0.5))

        try:
            if query_type == "text" and text_query:
                query_emb = retriever.encode_text(text_query)
            elif query_type == "image" and image_query_file:
                query_emb = retriever.encode_image(image_query_file)
            elif query_type == "hybrid" and text_query and image_query_file:
                text_emb = retriever.encode_text(text_query)
                img_emb = retriever.encode_image(image_query_file)
                query_emb = retriever.combine_embeddings(
                    text_emb, img_emb, lam=hybrid_weight
                )
            else:
                raise ValueError("Invalid input for the selected query type.")

            results = retriever.search_similar(query_emb, top_k=5, use_pca=use_pca)
        except Exception as e:
            error_message = str(e)

    return render_template("index.html", results=results, error_message=error_message)


@app.route("/coco_images_resized/<path:filename>")
def serve_image(filename):
    image_folder = "coco_images_resized"
    if not os.path.exists(os.path.join(image_folder, filename)):
        return f"Image {filename} not found in {image_folder}", 404
    return send_from_directory(image_folder, filename)


if __name__ == "__main__":
    app.run(debug=True)
