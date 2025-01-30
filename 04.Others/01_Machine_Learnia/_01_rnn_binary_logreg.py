import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

def init_data():
    x, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
    y = y.reshape((y.shape[0], 1))
    print(f"dimensions de x: {x.shape}")
    print(f"dimensions de y: {y.shape}")
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap='summer')
    plt.show()
    return x, y

def init_params(x):
    w = np.random.randn(x.shape[1], 1)
    b = np.random.rand(1)
    return w, b

def model(x, w, b):
    z = x.dot(w) + b
    a = 1/(1+np.exp(-z))
    return a

def log_loss(a,y):
    return 1/len(y) * np.sum(-y * np.log(a) + (1 - y) * np.log(1 - a))

def gradient(a, x, y):
    dw = 1/len(y) * np.dot(x.T, a-y)
    db = 1/len(y) * np.sum(a-y)
    return dw, db

def update(dw, db, w, b, learning_rate):
    w = w - learning_rate * dw
    b = b - learning_rate * db
    return w, b

def predict(x,w,b):
    a = model(x,w,b)
    return a >= 0.5

def artificial_neuron(x, y, learning_rate=0.1, epoch=100):
    #initialisation
    w, b = init_params(x)

    loss_history = []
    for i in range(epoch):
        a = model(x, w, b)
        loss_val =  log_loss(a,y)
        loss_history.append(loss_val)
        dw, db = gradient(a, x, y)
        w, b = update(dw, db, w, b, learning_rate)
        print(f"Model epoch {i}, loss val: {loss_val}")

    y_pred = predict(x,w,b)
    model_score = accuracy_score(y,y_pred)
    print(f"Loss history: {loss_history}")
    print(f"Accuracy : {model_score}")

    plt.plot(loss_history)
    plt.show()

    plot_frontière_decision(x, y, w, b)
    visualisation_3d(x, y)
    visualisation2_3d(x, y, w, b)


def plot_frontière_decision(x, y, w, b):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x[:,0], x[:,1], c=y, cmap='summer')

    x1 = np.linspace(-1, 4, 100)
    x2 = (-w[0]*x1-b)/w[1]

    ax.plot(x1, x2, lw=3)
    fig.show()

def visualisation_3d(x_data,y_data):
    fig = go.Figure(data=[go.Scatter3d(
        x=x_data[:, 0].flatten(),
        y=x_data[:, 1].flatten(),
        z=y_data.flatten(),
        mode='markers',
        marker=dict(
            size=5,
            color=y_data.flatten(),
            colorscale='YlGn',
            opacity=0.8,
            reversescale=True
        )
    )])

    fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, b=0, t=0))
    fig.layout.scene.camera.projection.type = "orthographic"
    fig.show()

def visualisation2_3d(x_data,y_data, w, b):
    x0 = np.linspace(x_data[:, 0].min(), x_data[:, 0].max(), 100)
    x1 = np.linspace(x_data[:, 1].min(), x_data[:, 1].max(), 100)
    xx0, xx1 = np.meshgrid(x0, x1)
    z = w[0] * xx0 + w[1] * xx1 + b
    a = 1 / (1 + np.exp(-z))

    fig = (go.Figure(data=[go.Surface(z=a, x=xx0, y=xx1, colorscale='YlGn', opacity=0.7, reversescale=True)]))

    fig.add_scatter3d(x=x_data[:, 0].flatten(), y=x_data[:, 1].flatten(), z=y_data.flatten(), mode='markers',
                      marker=dict(size=5, color=y_data.flatten(), colorscale='YlGn', opacity=0.9, reversescale=True))

    fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, b=0, t=0))
    fig.layout.scene.camera.projection.type = "orthographic"
    fig.show()

def main():
    x, y = init_data()
    artificial_neuron(x, y, learning_rate=0.1, epoch=100)

if __name__ == "__main__":
    main()