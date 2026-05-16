import gdv_vis

X = [[0.5, 1.0], [1.0, 2.0], [2.0, 4.0]]
y = [0, 0, 1]

model = gdv_vis.LogisticRegression()
model.fit(X, y, learning_rate=0.1, iterations=100)

proba = model.predict_proba([2.0, 4.0])
print("Predicts:", proba)

pred = model.predict([2.0, 4.0])
print("Class:", pred)