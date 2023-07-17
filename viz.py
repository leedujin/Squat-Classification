import matplotlib.pyplot as plt
from config import get_parser

# .log 파일에서 데이터 추출
train_accuracy = []
val_accuracy = []
test_accuracy = []

args = get_parser().parse_known_args()[0]

with open(args.log_file, "r") as file:
    lines = file.readlines()
    for line in lines:
        if "train accuracy" in line:
            train_acc = float(line.split(":")[1].strip())
            train_accuracy.append(train_acc)
        elif "val accuracy" in line:
            val_acc = float(line.split(":")[1].strip())
            val_accuracy.append(val_acc)
        elif "test accuracy" in line:
            test_acc = float(line.split(":")[1].strip())
            test_accuracy.append(test_acc)

# 정확도 시각화
epochs = range(len(train_accuracy))

plt.plot(epochs, train_accuracy, label='Train Accuracy')
plt.plot(epochs, val_accuracy, label='Validation Accuracy')
plt.plot(epochs, test_accuracy, label='Test Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.show()
