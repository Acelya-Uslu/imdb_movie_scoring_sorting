# IMDB Movie Scoring & Sorting

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)  # Bütün sütunları gösterir
pd.set_option("display.max_rows", None)  # Bütün satırları gösterir
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.5f" % x)  # 0 dan sonra 5 basamak gösterir

df = pd.read_csv("datasets/movies_metadata.csv",
                 low_memory=False)  # Dtype Warning kapamak için

df.head()

df = df[["title", "vote_average", "vote_count"]]

df.head()
df.shape

# Vote Average ( Oy Ortalamasına )Göre Sıralama

df.sort_values("vote_average", ascending=False).head(10)

df["vote_count"].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T

# Az oy alan filmleri görmek istemediğim için bir filtre koyuyorum.
df[df["vote_count"] > 400].sort_values("vote_average", ascending=False).head(10)

# vote_count u vote average ile aynı ölçeğe getiriyorum.


df["vote_count_score"] = MinMaxScaler(feature_range=(1, 10)). \
    fit(df[["vote_count"]]). \
    transform(df[["vote_count"]])

# Vote average * Vote Count ile Sıralama


df["average_count_score"] = df["vote_average"] * df["vote_count_score"]
df.sort_values("average_count_score", ascending=False).head(10)

# IMDB Weighted Rating
# weighted_rating = (v/(v+M) * r) + (M/(v+M) * C)

# wr = weighted rating
# r = vote average --> mevcut film puanı
# v = vote count --> ilgili filmin oy sayısı
# M = minimum votes required to be listed in the Top 250 --> min gereken oy sayısı
# C = the mean vote across the whole report (currently 7.0) --> tüm filmlerin genel ortalaması

M = 2500
C = df["vote_average"].mean()


def weighted_rating(r, v, M, C):
    return (v / (v + M) * r) + (M / (v + M) * C)


df.sort_values("average_count_score", ascending=False).head(10)

weighted_rating(8.50000, 8000.0000, M, C)

df["weighted_rating"] = weighted_rating(df["vote_average"],
                                        df["vote_count"], M, C)

df.sort_values("weighted_rating", ascending=False).head(10)


# Bayesian Average Rating Score ( Bayes Derecelendirme Puanı)
# Bar yöntemi Ratinglerin olasılıksal ağırlıklı ortalamasına göre sıralama yapar.


def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score


bayesian_average_rating([34733, 4355, 4704, 6561, 13515, 26183, 87368, 273082, 600260, 1295351])

df = pd.read_csv("datasets/imdb_ratings.csv")
df = df.iloc[0:, 1:]  # problemli satırlardan kurtulmak için yapılan bir işlem

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four", "five",
                                                                "six", "seven", "eight", "nine", "ten"]]), axis=1)

df.sort_values("bar_score", ascending=False).head(10)
