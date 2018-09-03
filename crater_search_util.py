import numpy as np


def add_votes_to_db(database, key, data):
    # adds votes to the tally in the database
    entry = database.get(key, None)
    if entry is None:
        votes = add_votes(np.zeros(78287, dtype=np.uint8), data)
        database[key] = votes
    elif data is not None:
        database[key] = add_votes(database[key], data)
    return database


def add_ids_to_db(database, key, data):
    # add possible crater candidates to database if there are any.
    # this is done via intersection of new crater candidates with
    # already existing ones (if there is new ones and already
    # existing ones)
    entry = database.get(key, None)
    if entry is None or len(entry) == 0:
        database[key] = data
    elif data is not None:
        database[key] = np.intersect1d(entry, data)
    return database


def add_votes(target, new):
    target[new] += 1
    return target


def sort_indices_by_votes(database):
    indices = []
    for v in database.values():
        sort_by_most_votes = np.argsort(v)[::-1]
        indices.append(sort_by_most_votes)
    return indices
