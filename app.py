from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

MODEL_PATH = "model_artifacts.pkl"
with open(MODEL_PATH, "rb") as f:
    artifacts = pickle.load(f)

best_algo = artifacts["best_algo"]
ratings_df = artifacts["ratings_df"]
tfidf = artifacts["tfidf"]
item_texts = artifacts["item_texts"]
item_index_map = artifacts["item_index_map"]
item_similarity = artifacts["item_similarity"]
ALPHA = artifacts["alpha"]

app = Flask(__name__, template_folder="View")

def hybrid_recommendations(user_id, top_n=10):
    user_items = ratings_df[ratings_df['UserId'] == user_id]['ProductId'].unique()
    user_unrated = [iid for iid in item_texts['ProductId'].unique() if iid not in user_items]

    cf_predictions = []
    for iid in user_unrated:
        pred = best_algo.predict(user_id, iid)
        cf_predictions.append((iid, pred.est))
    cf_predictions.sort(key=lambda x: x[1], reverse=True)

    user_highly_rated = ratings_df[(ratings_df['UserId'] == user_id) & (ratings_df['Score'] >= 4)]['ProductId'].unique()
    user_highly_rated_indices = [item_index_map[i] for i in user_highly_rated if i in item_index_map]

    hybrid_scores = []
    for (iid, cf_score) in cf_predictions:
        if iid not in item_index_map:
            hybrid_scores.append((iid, cf_score))
            continue
        idx = item_index_map[iid]
        if len(user_highly_rated_indices) > 0:
            sim_scores = item_similarity[idx, user_highly_rated_indices]
            content_score = np.mean(sim_scores)
        else:
            content_score = 0

        hybrid_score = ALPHA * cf_score + (1 - ALPHA) * content_score
        hybrid_scores.append((iid, hybrid_score))

    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    return hybrid_scores[:top_n]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    user_id = request.form.get("user_id")
    try:
        recommendations = hybrid_recommendations(user_id, top_n=5)
        output = [
            {"rank": idx + 1, "ProductID": rec[0], "Score": round(rec[1], 4)}
            for idx, rec in enumerate(recommendations)
        ]
        return jsonify(output)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/random_recommend", methods=["GET"])
def random_recommend():
    try:
        sample_user = ratings_df['UserId'].sample(1).iloc[0]
        recommendations = hybrid_recommendations(sample_user, top_n=5)
        output = {
            "user_id": sample_user,
            "recommendations": [
                {"rank": idx + 1, "ProductID": rec[0], "Score": round(rec[1], 4)}
                for idx, rec in enumerate(recommendations)
            ]
        }
        return jsonify(output)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)