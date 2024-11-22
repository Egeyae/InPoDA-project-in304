import io

import pandas as pd

def load_split_raw_data():
    # Load the raw file and split it by 100k batch, for easier use
    # Also remove unnecessary columns
    with open("./data/raw_sentiment140.csv", encoding="utf-8", errors="ignore") as f:
        raw_data = f.read().splitlines()

    def remove_mentions(sentence: str) -> str:
        return " ".join([w for w in sentence.split() if not w.startswith("@")])

    def process_save_chunk(chunk: list[str], idx) -> None:
        with open(f"./data/chunk{idx}.csv", "w", encoding="utf-8") as f:
            df = pd.read_csv(io.StringIO("\n".join(chunk)))
            df = df.drop(columns=["id", "date", "query", "user"])
            df["text"] = df["text"].apply(remove_mentions)
            df.to_csv(f, index=False)
    i = len(raw_data)
    while i > 0:
        size = min(100_000, i)
        process_save_chunk(["sentiment,id,date,query,user,text"] + raw_data[i-size:i], i)
        i -= size


if __name__ == "__main__":
    load_split_raw_data()