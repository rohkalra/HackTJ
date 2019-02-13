import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

df = pd.read_csv("/Users/Rohkal/Desktop/final_dataset.csv")
textVector = TfidfVectorizer(stop_words="english")
df["Description"] = df["Description"].fillna(" ")
textVectorM = textVector.fit_transform(df["Description"])

similarity_score = linear_kernel(textVectorM, textVectorM)

array_of_indices = pd.Series(df.index, index=df['Name']).drop_duplicates()


def recommended_item(name, similarity_score=similarity_score):

    name.lower()
    index = array_of_indices[name]
    list_of_similar_items = list(enumerate(similarity_score[index]))
    list_of_similar_items = sorted(list_of_similar_items, key=lambda x: x[1], reverse=True)

    list_of_similar_items = list_of_similar_items[1:2]

    grocery_indices = [i[0] for i in list_of_similar_items]
    first_string = str(df['Name'].iloc[grocery_indices])
    # print(first_string[7:])
    second_string = str(df['Location'].iloc[grocery_indices])
    # print(second_string[7:])
    return first_string[5:] + "\n" + second_string[5:]


def main():

    fileIn = open("input.txt", "r")
    fileOut = open("output.txt", "w")

    for line in fileIn:
        if "\n" in line:
            line = line.replace("\n", "")
        fileOut.write(str(line) + ", ")
        fileOut.write("" + str(recommended_item(line)))
        fileOut.write("\n")
    fileOut.close()

    fileA = open("output.txt", "r")
    lines = fileA.readlines()
    fileA.close()
    fileA = open("output.txt", "w")
    for line in lines:
        if line != "Name: Location, dtype: object" + "\n":
            fileA.write(line)

    fileA = open("output.txt", "r")
    lines = fileA.readlines()
    fileA.close()
    fileA = open("output.txt", "w")
    for line in lines:
        if line != "Name: Name, dtype: object" + "\n":
            fileA.write(line)


if __name__ == "__main__":
    main()
