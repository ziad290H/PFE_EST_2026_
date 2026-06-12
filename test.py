import pickle
with open("all_trained_models.pkl", "rb") as f:
    model = pickle.load(f)

print(model)
# picking the best model 'random forest'
best_model = model["RandomForest"]
print("saving the best model in best_model.pkl")
with open("return_predictor/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("return_predictor/best_model.pkl", "rb") as f:
    the_best_model = pickle.load(f)

print(the_best_model)