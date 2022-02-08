def plot_loss(loss, type, epochs):
    n = len(loss)
    for _ in range(epochs-n):
        loss.append(loss[-1])
    print(len(loss))
    fig, ax = plt.subplots(figsize=(12,8))
    ax.margins(0.05) 

    ax.plot(np.arange(epochs), loss, ms=5, label='0')

    ax.legend()
    ax.legend(loc=2)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(type)
    ax.set_title('Tricky 3 Class Classification')
    plt.show()

def plot_decision_boundary(NN):
    X_test = np.random.uniform(low=-3.0, high=3.0, size=(100000,2))
    y_pred = NN.test(X_test, [])
    plot_data(X_test, y_pred)