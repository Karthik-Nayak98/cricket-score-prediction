import pandas as pd
import os
import json
from tqdm import tqdm

filenames = []
for file in os.listdir("./data/ipl_json"):
    filenames.append(
        os.path.join("./data/ipl_json", file)
    )


final_df = pd.DataFrame()
counter = 1
for file in tqdm(filenames):
    with open(file, "r") as f:
        df = pd.json_normalize(json.load(f))
        df["match_id"] = counter
        final_df = pd.concat([final_df, df], ignore_index=True)
        counter += 1

backup = final_df.copy()

final_df = backup.copy()


feature_cols = ["innings", "info.teams", "info.city", "match_id"]

final_df = final_df[feature_cols]


retain_city = [
    "Mohali",
    "Kolkata",
    "Mumbai",
    "Delhi",
    "Pune",
    "Chennai",
    "Ahmedabad",
    "Lucknow",
    "Jaipur",
    "Raipur",
    "Hyderabad",
    "Bengaluru",
    "Chandigarh",
    "Visakhapatnam",
    "Dharamsala",
    "Ranchi",
]

retain_cities = set(retain_city)

final_df = final_df[final_df["info.city"].isin(retain_cities)]

team_mapping = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
}

final_df["info.teams"] = final_df["info.teams"].apply(
    lambda teams: [team_mapping.get(team, team) for team in teams]
)

teams_to_remove = [
    "Rising Pune Supergiant",
    "Rising Pune Supergiants",
    "Kochi Tuskers Kerala",
    "Deccan Chargers",
    "Gujarat Lions",
    "Pune Warriors",
]

# Convert list to a set for faster lookup
remove_set = set(teams_to_remove)

# Keep rows where neither team in the 'info.teams' list intersects with our remove_set
final_df = final_df[
    ~final_df["info.teams"].apply(lambda teams: any(t in remove_set for t in teams))
]


def update_innings(innings):
    for inning in innings:
        inning["team"] = team_mapping.get(inning["team"], inning["team"])
    return innings


final_df["innings"] = final_df["innings"].apply(update_innings)


first_innings = final_df.iloc[1]["innings"][0]

for items in first_innings:
    batting_team = first_innings["team"]
    overs = first_innings["overs"]

delivery_df = pd.DataFrame()

for index, row in final_df.iterrows():
    first_innings = row["innings"][0]
    city = row["info.city"]
    mtch_id = row["match_id"]
    teams = row["info.teams"]
    for items in first_innings:
        team = first_innings["team"]
        overs = first_innings["overs"]

        match_id = []
        cities = []
        batting_team = []
        bowling_team = []
        balls_bowled = []
        runs = []
        current_score = []
        current_run_rate = []
        balls_left = []
        total_runs = 0
        player_of_dismissed = []
        for single_over in overs:
            over = single_over["over"]
            for idx, delivery in enumerate(single_over["deliveries"]):
                match_id.append(mtch_id)
                cities.append(city)
                batting_team.append(team)
                runs.append(int(delivery["runs"]["total"]))
                balls_bowled.append((over * 6) + idx + 1)
                bowling_team.append(
                    [t for t in teams if t != team][0]
                )  # Assuming the other team is the bowling team
                balls_left.append(120 - ((over * 6) + idx + 1))

                if delivery["runs"]["total"] > 0:
                    total_runs += int(delivery["runs"]["total"])
                current_score.append(total_runs)
                current_run_rate.append(
                    round((total_runs / ((over * 6 + idx + 1) / 6)), 2)
                )

                if "wickets" in delivery:
                    player_of_dismissed.append(1)
                else:
                    player_of_dismissed.append(0)

    loop_df = pd.DataFrame(
        {
            "match_id": match_id,
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "balls_bowled": balls_bowled,
            "balls_left": balls_left,
            "current_score": current_score,
            "current_run_rate": current_run_rate,
            "runs": runs,
            "player_dismissed": player_of_dismissed,
            "city": cities,
        }
    )

    delivery_df = pd.concat([delivery_df, loop_df], ignore_index=True)

total_runs = (
    delivery_df.groupby("match_id")["runs"].sum().reset_index(name="total_runs")
)

delivery_df = delivery_df.merge(total_runs, on="match_id", how="left")


delivery_df["wickets_fallen"] = delivery_df.groupby("match_id")[
    "player_dismissed"
].cumsum()

delivery_df["wickets_left"] = 10 - delivery_df["wickets_fallen"]

delivery_df.drop(columns=["player_dismissed", "wickets_fallen"], inplace=True)
delivery_df["last_5_overs_runs"] = delivery_df.groupby("match_id")["runs"].transform(
    lambda x: x.rolling(30).sum()
)

delivery_df.dropna(inplace=True)


delivery_df.to_csv(
    "/Users/karnayak/Personal/cricket_mlops/data/match_summary.csv", index=False
)
