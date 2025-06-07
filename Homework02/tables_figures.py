import matplotlib.pyplot as plt
import pandas as pd
import os

TRAIN_FILE = os.path.join("DATA", "df_train.csv")
TEST_FILE = os.path.join("DATA", "df_test.csv")


def count_types(df):
    artist_count = (df['type'] == 'Artist').sum()
    woa_count = (df['type'] == 'WoA').sum()
    none_count = len(df) - artist_count - woa_count

    return {'Artist': artist_count, 'WoA': woa_count, 'None': none_count}


def table1(df_train, df_test):
    train_counts = count_types(df_train)
    test_counts = count_types(df_test)

    df_summary = pd.DataFrame({
        'Train': train_counts,
        'Test': test_counts
    }).reset_index().rename(columns={'index': 'Type'})

    print(df_summary.to_markdown(index=False))


def top_entities(df):
    top_artists = (df[df["type"] == "Artist"]['named_entity'].value_counts().
                   head(10).reset_index().rename(columns={
                       'index': 'Artist name',
                       'named_entity': 'Artist count'
                   }))

    top_woas = (df[df["type"] == "WoA"]['named_entity'].value_counts().head(
        10).reset_index().rename(columns={
            'index': 'WoA name',
            'named_entity': 'WoA count'
        }))

    return pd.concat([top_artists, top_woas], axis=1)


def table2(df_train, df_test):
    train_top = top_entities(df_train)
    test_top = top_entities(df_test)

    combined = pd.concat([train_top, test_top], axis=1)
    combined.columns = [
        "Train Artist", "Count", "Train WoA", "Count", "Test Artist", "Count",
        "Test WoA", "Count"
    ]

    print("### Top 10 Artists and WoAs – Train and Test Combined")
    print(combined.to_markdown(index=False))


def count_entities_per_query(df):
    entity_counts = df.groupby("query")["named_entity"].count()

    count_0 = (entity_counts == 0).sum()
    count_1 = (entity_counts == 1).sum()
    count_2 = (entity_counts == 2).sum()
    count_3_plus = (entity_counts >= 3).sum()

    return {"0": count_0, "1": count_1, "2": count_2, "3+": count_3_plus}


def table3(df_train, df_test):
    train_dist = count_entities_per_query(df_train)
    test_dist = count_entities_per_query(df_test)

    df_summary = pd.DataFrame({
        "Train queries": train_dist,
        "Test queries": test_dist
    }).reset_index().rename(columns={"index": "Named Entities"})

    print(df_summary.to_markdown(index=False))

def top_first_words_counts(df, prefix):
        df = df.copy()
        df["first_word"] = df["query"].str.strip().str.lower().str.split(
        ).str[0]
        counts = (df.drop_duplicates("query")["first_word"].value_counts().
                  head(10).reset_index().rename(columns={
                      "index": f"{prefix} First Word",
                      "first_word": f"{prefix} #"
                  }))
        return counts

def table4(df_train, df_test):
    train_counts = top_first_words_counts(df_train, "Train")
    test_counts = top_first_words_counts(df_test, "Test")

    combined = pd.concat([train_counts, test_counts], axis=1)

    print(combined.to_markdown(index=False))

def table5(df_train, df_test):
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    df_all = df_all.dropna(subset=["named_entity", "type"])

    unique_counts = (
        df_all.groupby("type")["named_entity"]
        .nunique()
        .reset_index()
        .rename(columns={"named_entity": "# Unique Named Entities", "type": "Type"})
    )

    print(unique_counts.to_markdown(index=False))

def figure1(df_train, df_test):
    def get_lengths(df):
        return df["query"].str.strip().str.split().str.len()

    train_lengths = get_lengths(df_train)
    test_lengths = get_lengths(df_test)

    plt.figure(figsize=(8, 5))
    plt.hist(train_lengths, bins=20, alpha=0.6, label="Train", edgecolor='black')
    plt.hist(test_lengths, bins=20, alpha=0.6, label="Test", edgecolor='black')
    plt.xlabel("Query Length (words)")
    plt.ylabel("Number of Queries")
    plt.title("Figure 1 – Query Length Distribution (Train vs Test)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def figure2(df_train, df_test, top_n=20):
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    entity_counts = df_all['named_entity'].value_counts().head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(entity_counts.index[::-1], entity_counts.values[::-1], color="tab:purple", edgecolor="black")
    plt.xlabel("Frequency")
    plt.title(f"Figure 3 – Top {top_n} Named Entities (Train + Test Combined)")
    plt.tight_layout()
    plt.show()


def main():
    df_train = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)

    table1(df_train, df_test)
    table2(df_train, df_test)
    table3(df_train, df_test)
    table4(df_train, df_test)
    table5(df_train, df_test)

    figure1(df_train, df_test)
    figure2(df_train, df_test)


if __name__ == '__main__':
    main()
