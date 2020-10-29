import pandas as pd
import pickle

train_df = pd.read_csv(f'data/new_sqai/preprocessed_data_train.csv', sep="\t")
users_train = train_df["user_id"].unique()
items_train = train_df["item_id"].unique()
skills_train = train_df["skill_id"].unique()
schools_train = train_df["school_id"].unique()
info_dict = {
    'user_id': (min(users_train), max(users_train), len(users_train)),
    'item_id': (min(items_train), max(items_train), len(items_train)),
    'skill_id': (min(skills_train), max(skills_train), len(skills_train)),
    'school_id': (min(schools_train), max(schools_train), len(schools_train))
}
print(info_dict)
with open('train_id_stats.pickle', 'wb') as handle:
    pickle.dump(info_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)