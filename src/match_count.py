# File: ./src/match_count.py

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import datetime
import json
from .utils import dump_dataset_to_csv  # Relative import

class MatchCount:
    """
    Class for generating Counting Task instances.
    The task is to count the number of occurrences of a specific element in a sequence.
    CoT includes filler tokens to simulate reasoning steps.
    """
    
    def __init__(self, length, target, mod=10, transform=None, inverse_transform=None):
        if not isinstance(target, int):
            raise ValueError("`target` must be a single integer.")
        self.length = length
        self.target = target
        self.mod = mod
        self.transform = transform
        self.inverse_transform = inverse_transform
        self.random = np.random.default_rng()
    
    def get_true_instance(self):
        """
        Generates a true instance where the count is correctly embedded in the CoT.
        """
        sequence = self.random.integers(0, self.mod, size=self.length)
        # print(f"Generated sequence shape: {sequence.shape}")  # Removed Debugging
        count = np.sum(sequence == self.target)
        
        if self.transform:
            transform_sequence, inverse_transform_sequence = self.transform(sequence)
            # print(f"Transformed sequence shape: {transform_sequence.shape}")  # Removed Debugging
            sequence_transformed = inverse_transform_sequence
        else:
            transform_sequence = sequence.copy()
            sequence_transformed = sequence.copy()
        
        solution = count
        return sequence, sequence_transformed, solution
    
    def get_corrupted_instance(self, corruption_rate=4/3):
        """
        Generates a corrupted instance by altering some elements in the sequence.
        """
        sequence, transform_sequence, solution = self.get_true_instance()
        corruptions = self.random.geometric(1/corruption_rate)
        corruptions = min(corruptions, self.length)
        indices = self.random.choice(self.length, size=corruptions, replace=False)
        # print(f"Corrupting indices: {indices}")  # Removed Debugging
        for idx in indices:
            transform_sequence[idx] = self.random.integers(0, self.mod)
            sequence[idx] = self.inverse_transform(transform_sequence)[idx]
        
        # Recalculate solution
        new_solution = np.sum(sequence == self.target)
        # print(f"New solution after corruption: {new_solution}")  # Removed Debugging
        
        return sequence, transform_sequence, new_solution

# Transformation functions
def identity_transform(sequence, mod=10):
    return sequence.copy(), sequence.copy()

def random_transform(sequence, mod=10):
    transform_seqs = np.random.default_rng().integers(0, mod, size=len(sequence))
    transformed = (sequence + transform_seqs) % mod
    inverse = (-transform_seqs) % mod
    return transformed, inverse

# String formatting functions
def count_basic_string(inputs, transform_inputs, solution, transform_params, rng=None, mod=10):
    """
    Basic string format with CoT.
    """
    st = ' ' + ' '.join([str(x) for x in inputs])
    st += ' T ' + ' '.join([str(x) for x in transform_params])
    st += ' CoT ' + f"The count of {transform_params[0]} is {solution}"
    return st

def count_fillers_string(inputs, transform_inputs, solution, transform_params, filler_token='.', rng=None, mod=10):
    """
    String format with fillers in CoT.
    """
    st = ' ' + ' '.join([str(x) for x in inputs])
    st += ' T ' + ' '.join([str(x) for x in transform_params])
    st += ' ' + ' '.join([filler_token]*5)  # Insert fillers
    st += ' CoT ' + f"The count of {transform_params[0]} is {solution}"
    return st

STRING_FUNCTION_MAPPING = {
    'basic': count_basic_string,
    'fillers': count_fillers_string
}

def generate_sample(length, target, mod, transform, type='True', corruption_rate=4/3, rng=None):
    """
    Generates a single data sample.
    """
    if transform == 'random':
        transform_params = (rng.integers(0, mod),)
        transform_fn, inverse_transform_fn = random_transform, lambda x: random_transform(x, mod)[1]
    elif transform == 'identity':
        transform_params = ()
        transform_fn, inverse_transform_fn = identity_transform, identity_transform
    else:
        raise ValueError('Unsupported transform type')
    
    m_count = MatchCount(length=length, target=target, mod=mod, transform=transform_fn, inverse_transform=inverse_transform_fn)
    
    if type == 'True':
        inputs, transform_inputs, solution = m_count.get_true_instance()
    elif type == 'Corrupted':
        inputs, transform_inputs, solution = m_count.get_corrupted_instance(corruption_rate=corruption_rate)
    else:
        raise ValueError('Type must be True or Corrupted')
    
    return inputs, transform_inputs, solution, transform_params

def GenerateMatchCountDataset(name, train_samples, test_samples,
                              length, target, mod=10, 
                              true_instance_rate=0.5, 
                              cot_rate=0.5, 
                              corruption_rate=4/3,
                              transform='random', 
                              filler_to_string=None, cot_to_string=None,
                              data_path='./data/'):
    """
    Generate a dataset for the Counting Task.
    
    Args:
    - name (str): Name of the dataset
    - train_samples (int): Number of training samples to generate
    - test_samples (int): Number of test samples to generate
    - length (int): Length of the sequence
    - target (int): The element to count
    - mod (int): Modulus for transformations
    - true_instance_rate (float): Rate at which true instances are used
    - cot_rate (float): Rate at which CoT is included
    - corruption_rate (float): Rate of corruption
    - transform (str): Type of transformation ('random' or 'identity')
    - filler_to_string (callable): Function to format filler strings
    - cot_to_string (callable): Function to format CoT strings
    - data_path (str): Path to save data
    
    Returns:
    - None
    """
    if transform not in ['random', 'identity']:
        raise ValueError('Transform must be random or identity')
    
    randomizer = np.random.default_rng()
    corruption_vec = randomizer.binomial(1, true_instance_rate, size=train_samples + test_samples)
    corruption_vec = np.where(corruption_vec == 1, 'True', 'Corrupted')
    
    assert cot_rate <= 1, "cot_rate must be <= 1"
    filler_rate = 1 - cot_rate
    filler_vec = randomizer.choice([0, 1], p=[cot_rate, filler_rate], size=train_samples + test_samples)  # 0 is CoT, 1 is filler
    
    # Generate Training Data
    train_dataset = []
    for i in range(train_samples):
        sample = generate_sample(length, target, mod, transform, type=corruption_vec[i], corruption_rate=corruption_rate, rng=randomizer)
        if filler_vec[i] == 0:
            formatted = cot_to_string(*sample, rng=randomizer, mod=mod)
        else:
            formatted = filler_to_string(*sample)
        train_dataset.append(formatted)
    
    # Generate Test Data
    test_dataset = []
    for i in range(train_samples, train_samples + test_samples):
        sample = generate_sample(length, target, mod, transform, type=corruption_vec[i], corruption_rate=corruption_rate, rng=randomizer)
        if filler_vec[i] == 0:
            formatted = cot_to_string(*sample, rng=randomizer, mod=mod)
        else:
            formatted = filler_to_string(*sample)
        test_dataset.append(formatted)
    
    # Save Hyperparameters
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    hyperparameters_filename = f"args_{name}_{today}.json"
    args = {
        "name": name,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "length": length,
        "target": target,
        "mod": mod,
        "true_instance_rate": true_instance_rate,
        "transform": transform,
        "filler": filler_to_string.__name__ if filler_to_string else None,
        "cot": cot_to_string.__name__ if cot_to_string else None
    }
    
    with open(os.path.join(data_path, hyperparameters_filename), 'w') as f:
        json.dump(args, f)
    
    # Save Datasets
    train_desc = f"trainset_{today}.csv"
    dump_dataset_to_csv(train_dataset, os.path.join(data_path, f"{name}_{train_desc}"))
    
    test_desc = f"testset_{today}.csv"
    dump_dataset_to_csv(test_dataset, os.path.join(data_path, f"{name}_{test_desc}"))
    
    return
def GenerateMatchCountDataset(name, train_samples, test_samples,
                              length, target, mod=10, 
                              true_instance_rate=0.5, 
                              cot_rate=0.5, 
                              corruption_rate=4/3,
                              transform='random', 
                              filler_to_string=None, cot_to_string=None,
                              data_path='./data/'):
    """
    Generate a dataset for the Counting Task.
    
    Args:
    - name (str): Name of the dataset
    - train_samples (int): Number of training samples to generate
    - test_samples (int): Number of test samples to generate
    - length (int): Length of the sequence
    - target (int): The element to count
    - mod (int): Modulus for transformations
    - true_instance_rate (float): Rate at which true instances are used
    - cot_rate (float): Rate at which CoT is included
    - corruption_rate (float): Rate of corruption
    - transform (str): Type of transformation ('random' or 'identity')
    - filler_to_string (callable): Function to format filler strings
    - cot_to_string (callable): Function to format CoT strings
    - data_path (str): Path to save data
    
    Returns:
    - None
    """
    # Ensure the data_path directory exists
    os.makedirs(data_path, exist_ok=True)
    
    if transform not in ['random', 'identity']:
        raise ValueError('Transform must be random or identity')
    
    randomizer = np.random.default_rng()
    corruption_vec = randomizer.binomial(1, true_instance_rate, size=train_samples + test_samples)
    corruption_vec = np.where(corruption_vec == 1, 'True', 'Corrupted')
    
    assert cot_rate <= 1, "cot_rate must be <= 1"
    filler_rate = 1 - cot_rate
    filler_vec = randomizer.choice([0, 1], p=[cot_rate, filler_rate], size=train_samples + test_samples)  # 0 is CoT, 1 is filler
    
    # Generate Training Data
    train_dataset = []
    for i in range(train_samples):
        sample = generate_sample(length, target, mod, transform, type=corruption_vec[i], corruption_rate=corruption_rate, rng=randomizer)
        if filler_vec[i] == 0:
            formatted = cot_to_string(*sample, rng=randomizer, mod=mod)
        else:
            formatted = filler_to_string(*sample)
        train_dataset.append(formatted)
    
    # Generate Test Data
    test_dataset = []
    for i in range(train_samples, train_samples + test_samples):
        sample = generate_sample(length, target, mod, transform, type=corruption_vec[i], corruption_rate=corruption_rate, rng=randomizer)
        if filler_vec[i] == 0:
            formatted = cot_to_string(*sample, rng=randomizer, mod=mod)
        else:
            formatted = filler_to_string(*sample)
        test_dataset.append(formatted)
    
    # Save Hyperparameters
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    hyperparameters_filename = f"args_{name}_{today}.json"
    args = {
        "name": name,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "length": length,
        "target": target,
        "mod": mod,
        "true_instance_rate": true_instance_rate,
        "transform": transform,
        "filler": filler_to_string.__name__ if filler_to_string else None,
        "cot": cot_to_string.__name__ if cot_to_string else None
    }
    
    with open(os.path.join(data_path, hyperparameters_filename), 'w') as f:
        json.dump(args, f)
    
    # Save Datasets
    train_desc = f"trainset_{today}.csv"
    dump_dataset_to_csv(train_dataset, os.path.join(data_path, f"{name}_{train_desc}"))
    
    test_desc = f"testset_{today}.csv"
    dump_dataset_to_csv(test_dataset, os.path.join(data_path, f"{name}_{test_desc}"))
    
    return

