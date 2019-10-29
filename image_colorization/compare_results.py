import matplotlib.pyplot as plt
import glob
import matplotlib.image as mpimg

path_to_results_folder = "results/"

chosen_versions = ["original", "V84", "V130", "V140", "V150"]


def main():

    paths_list = []
    for i, results_dir in enumerate(chosen_versions):
        paths_list.append([])
        for file in glob.glob(path_to_results_folder + results_dir + "/*.png"):
            paths_list[i].append(file)

    number_of_subplots = len(paths_list)
    for list_of_imgs in zip(*paths_list):
        fig = plt.figure(figsize=(16, 8))
        for i, img_path in enumerate(list_of_imgs):
            img = mpimg.imread(img_path)
            ax1 = fig.add_subplot(1, number_of_subplots, i+1)
            ax1.imshow(img)
            ax1.title.set_text(img_path.split("/")[-1])
        plt.show()


if __name__ == "__main__":
    main()
