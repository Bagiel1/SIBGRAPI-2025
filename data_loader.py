import numpy as np

def load_data(dataset_name, feat_extractor):

    if dataset_name == 'flowers':
        if feat_extractor == 'resnet':
            data = np.load('data/Features-Labels-Lists/cnn-last_linear-resnet152.npz')
            features = data['features']
        elif feat_extractor == 'senet':
            data = np.load('data/Features-Labels-Lists/cnn-last_linear-senet154.npz')
            features = data['features']
        elif feat_extractor == 'vit':
            features = np.load('data/Features-Labels-Lists/features_vit-b16_flowers.npy')
        else:
            raise ValueError(f"Extrator de features '{feat_extractor}' não suportado para o dataset 'flowers'.")
        
        with open('data/Features-Labels-Lists/listFlowers.txt', 'r') as file:
            dataset_elements = [line.strip() for line in file.readlines()]
            class_size = 80
            labels = [i // class_size for i in range(len(dataset_elements))]

        return np.array(features), np.array(labels)

    else:
        raise ValueError(f"Dataset '{dataset_name}' não suportado.")
