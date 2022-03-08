import matplotlib.pyplot as plt
acc = [[],[],[]]
for i in range(3):
    with open(f"client{i+1}.txt") as f:
        c = 0
        for line in f.readlines():
            line = line.strip() 
            if line and line[0] == "e":
                c += 1
                acc[i].append(float(line[-4:]))
    plt.plot(range(1,21), acc[i], label= f"Client {i+1}")
    plt.xticks(range(1,21))
    # ax.xaxis.get_major_locator().set_params(integer=True)
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.savefig("client_accuracy")
losses = [[],[],[]]
for i in range(3):
    with open(f"client{i+1}.txt") as f:
        c = 0
        for line in f.readlines():
            line = line.strip() 
            if line and line[0] == "e":
                c += 1
                losses[i].append(float(line[17:25]))
    plt.plot(range(1,21), losses[i], label= f"Client {i+1}")
    plt.xticks(range(1,21))
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig("client_loss")
