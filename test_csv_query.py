
import pandas as pd
from ingestiondata.csv_query import answer_csv_question

df = pd.read_csv("ingestiondata/Social_Network_Ads.csv")

print("Columns:", df.columns.tolist())

while True:
    question = input("\nAsk a question (or type 'exit'): ").strip()

    if question.casefold() == "exit":
        break

    answer = answer_csv_question(df, question)

    if answer is None:
        print("This question needs another query method.")
    else:
        print(answer)
