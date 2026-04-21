# ===== Imports =====

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix


# ===== Load dataset =====

matches = pd.read_csv("matches.csv", index_col=0)

# Convert date column to datetime format
matches["date"] = pd.to_datetime(matches["date"])


# ===== Feature Engineering =====

# Convert categorical features into numeric codes
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes

# Extract kickoff hour
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype(int)

# Extract day of week
matches["day_code"] = matches["date"].dt.dayofweek

# Create prediction target (Win = 1, otherwise = 0)
matches["target"] = (matches["result"] == "W").astype(int)


# ===== Train/Test Split =====

train = matches[matches["date"] < "2022-01-01"]
test = matches[matches["date"] > "2022-01-01"]

predictors = [
    "venue_code",
    "opp_code",
    "hour",
    "day_code"
]


# ===== Train Initial Model =====

rf = RandomForestClassifier(
    n_estimators=50,
    min_samples_split=10,
    random_state=1
)

rf.fit(train[predictors], train["target"])

preds = rf.predict(test[predictors])


# ===== Evaluate Initial Model =====

accuracy = accuracy_score(test["target"], preds)
precision = precision_score(test["target"], preds)

print("Initial Model Accuracy:", accuracy)
print("Initial Model Precision:", precision)


# ===== Rolling Average Feature Generator =====

def rolling_averages(group, cols, new_cols):

    group = group.sort_values("date")

    rolling_stats = group[cols].rolling(
        3,
        closed="left",
        min_periods=3
    ).mean()

    group[new_cols] = rolling_stats

    group = group.dropna(subset=new_cols)

    return group


# Performance columns for rolling stats

cols = [
    "gf",
    "ga",
    "xg",
    "xga",
    "poss",
    "sh",
    "sot",
    "dist",
    "fk",
    "pk",
    "pkatt"
]

new_cols = [f"{c}_rolling" for c in cols]


# Apply rolling averages team-by-team

matches_rolling = (
    matches
    .groupby("team", group_keys=False)
    .apply(lambda x: rolling_averages(x, cols, new_cols))
    .reset_index(drop=True)
)


# ===== Retrain Model with Rolling Features =====

def make_predictions(data, predictors):

    train = data[data["date"] < "2022-01-01"]
    test = data[data["date"] > "2022-01-01"]

    model = RandomForestClassifier(
        n_estimators=50,
        min_samples_split=10,
        random_state=1
    )

    model.fit(train[predictors], train["target"])

    preds = model.predict(test[predictors])

    combined = pd.DataFrame(
        dict(actual=test["target"], predicted=preds),
        index=test.index
    )

    precision = precision_score(test["target"], preds)

    return combined, precision


combined, precision = make_predictions(
    matches_rolling,
    predictors + new_cols
)

print("Improved Model Precision:", precision)


# ===== Merge Metadata Back =====

combined = combined.merge(
    matches_rolling[
        ["date", "team", "opponent", "result"]
    ],
    left_index=True,
    right_index=True
)


# ===== Confusion Matrix Visualization =====

cm = confusion_matrix(
    combined["actual"],
    combined["predicted"]
)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title("PL Match Win Prediction Evaluation")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# ===== Accuracy Over Time Visualization =====

combined["correct"] = (
    combined["actual"] == combined["predicted"]
)

accuracy_by_date = combined.groupby("date")[
    "correct"
].mean()

accuracy_by_date.plot(figsize=(10,5))

plt.title("Prediction Accuracy Over Time")
plt.ylabel("Accuracy")
plt.xlabel("Match Date")

plt.show()


# ===== Rolling Goals Distribution =====

matches_rolling["gf_rolling"].hist(bins=30)

plt.title("Rolling Goals Scored Distribution")
plt.xlabel("Rolling Goals Average")
plt.ylabel("Frequency")

plt.show()
