import csv
import matplotlib.pyplot as plt
path_to_csv = "C:/STUDIA/INZYNIERKA/chameleon/image_colorization/data/run-Oct25_19-54-49_DESKTOP-K2JRB94-tag-._Loss_train.csv"


def main():
    with open(path_to_csv, "r") as plot_results:
        csv_reader = csv.reader(plot_results)
        headers = next(csv_reader)
        time_stamp = []
        step = []
        loss_value = []
        for row in csv_reader:
            time_stamp.append(float(row[0]))
            step.append(int(row[1]))
            loss_value.append(float(row[2]))

        plt.plot(time_stamp, loss_value)
        plt.show()
        print("End")


if __name__ == "__main__":
    main()
