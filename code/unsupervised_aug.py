import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity


num = 1000

def aug_random(train, test, num_aug):
    print("random")
    new = []
    for i in range(num):
        index = random.sample(list(np.arange(len(train))), 2)
        new_sample = 0.5*train[index[0], :, :, :] + 0.5*train[index[1], :, :, :]
        new.append(new_sample)

    new = np.array(new)
    new = aug_sim(new, test, num_aug)
    return new


def aug_appliance(train, test, num_aug):
    print("appliance")
    new = np.zeros((num, 6, 112, 24))
    for i in range(num):
        home_agg = np.zeros((112, 24))
        for appliance in range(1, 6):
            index = np.random.choice(list(range(len(train))))
            new[i, appliance, :, :] = train.copy()[index, appliance, : :]
            home_agg += train.copy()[index, appliance, :, :]
        new[i, 0, :, :] = home_agg
    new = aug_sim(new, test, num_aug)
    return new


def aug_noise(train, test, num_aug):
    print("noise")
    new = []
    for i in range(num):
        index = np.random.choice(list(range(len(train))))
        noise = np.random.normal(0, 1, 112*24*5).reshape(5, 112, 24)
        new_sample = train.copy()[index]
        new_sample[1:] = new_sample[1:] + noise
        new_sample[0] = 0 
        for j in range(1, 6):
            new_sample[0] += new_sample[j]
        new.append(new_sample)

    new = np.array(new)
    new = aug_sim(new, test, num_aug)

    return new


def aug_sim(train, test, num_aug):

    test_aggregate = test[:, 0, :, :]
    train_aggregate = train[:, 0, :, :]
    
    test_aggregate = test_aggregate.reshape(test_aggregate.shape[0], -1)
    train_aggregate = train_aggregate.reshape(train_aggregate.shape[0], -1)
    
    train_test_agg = np.vstack([train_aggregate, test_aggregate])
    
    similarity = cosine_similarity(train_test_agg)
    similarity = similarity[:len(train_aggregate), len(train_aggregate):]
    
    train_max = similarity.max(axis=1)
    
    # k = int(0.5*len(train_aggregate))
    index = np.argpartition(train_max, -num_aug)[-num_aug:]
    
    return train[index]


def augmented_data(train, num_aug, case, test):
    
    if num_aug == 0:
        # print("no augmentation data")
        return train

    new = []

    if case == 1:
        new = aug_random(train, test, num_aug)
    if case == 2:
        new = aug_appliance(train, test, num_aug)
    if case == 3:
        new = aug_noise(train, test, num_aug)
    # if case == 4:
    #     new = aug_sim(train, test, num_aug)

    return np.vstack([train, new])
