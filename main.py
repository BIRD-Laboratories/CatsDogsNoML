import os
import numpy as np
import argparse
from PIL import Image
from transformers import AutoModel, AutoFeatureExtractor
import torch
import gradio as gr

def load_model_and_feature_extractor():
    """Load the pre-trained model and feature extractor."""
    model_name = "OpenGVLab/InternViT-300M-448px"
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    return model, feature_extractor

def extract_features_from_dataset(dataset_dir, categories, model, feature_extractor):
    """Extract features from the dataset and create a dictionary."""
    feature_dict = {category: [] for category in categories}

    for category in categories:
        category_dir = os.path.join(dataset_dir, category)
        for filename in os.listdir(category_dir):
            if filename.endswith('.jpg'):
                image_path = os.path.join(category_dir, filename)
                image = Image.open(image_path).convert('RGB')
                inputs = feature_extractor(images=image, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                    features = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
                feature_dict[category].append(features)

    return feature_dict

def average_features(feature_dict, categories):
    """Average the features for each category."""
    average_features = {}
    for category in categories:
        average_features[category] = np.mean(feature_dict[category], axis=0)
    return average_features

def save_features_as_tiles(features, output_dir, category):
    """Save the average features as 32x32 tiles and create 64 of these tiles."""
    features_normalized = ((features - features.min()) / (features.max() - features.min()) * 255).astype(np.uint8)
    features_reshaped = features_normalized.reshape(32, 32, -1)
    num_tiles = min(64, features_reshaped.shape[2])

    for i in range(num_tiles):
        tile = features_reshaped[:, :, i]
        tile_rgb = np.repeat(tile[:, :, np.newaxis], 3, axis=2)
        tile_image = Image.fromarray(tile_rgb)
        tile_image.save(os.path.join(output_dir, f'{category}_tile_{i+1}.png'))

def classify_image(image_path, average_features, model, feature_extractor):
    """Classify a new image using the average features."""
    image = Image.open(image_path).convert('RGB')
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

    similarities = {}
    for category in average_features.keys():
        similarity = np.dot(features, average_features[category]) / (np.linalg.norm(features) * np.linalg.norm(average_features[category]))
        similarities[category] = similarity

    return max(similarities, key=similarities.get)

def load_average_features(cat_feature_dir, dog_feature_dir):
    """Load the average features from the saved tiles."""
    cat_features = []
    dog_features = []

    for i in range(1, 65):
        cat_tile_path = os.path.join(cat_feature_dir, f'cat_tile_{i}.png')
        dog_tile_path = os.path.join(dog_feature_dir, f'dog_tile_{i}.png')
        cat_tile = Image.open(cat_tile_path).convert('RGB')
        dog_tile = Image.open(dog_tile_path).convert('RGB')
        cat_features.append(np.array(cat_tile).mean(axis=(0, 1)).flatten())
        dog_features.append(np.array(dog_tile).mean(axis=(0, 1)).flatten())

    average_features = {
        'cat': np.mean(cat_features, axis=0),
        'dog': np.mean(dog_features, axis=0)
    }

    return average_features

def test_classifier(test_dir, average_features, model, feature_extractor):
    """Test the classifier on a test dataset and report accuracy."""
    categories = ['cat', 'dog']
    correct = 0
    total = 0

    for category in categories:
        category_dir = os.path.join(test_dir, category)
        for filename in os.listdir(category_dir):
            if filename.endswith('.jpg'):
                image_path = os.path.join(category_dir, filename)
                prediction = classify_image(image_path, average_features, model, feature_extractor)
                if prediction == category:
                    correct += 1
                total += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2%}")

def gradio_interface(image_path, cat_feature_dir, dog_feature_dir, model, feature_extractor):
    """Gradio interface for classifying a new image."""
    average_features = load_average_features(cat_feature_dir, dog_feature_dir)
    result = classify_image(image_path, average_features, model, feature_extractor)
    return f"The image is classified as a {result}."

def main():
    parser = argparse.ArgumentParser(description="Cat vs Dog Image Classification using Transformers")
    parser.add_argument('--find_features', action='store_true', help="Find features from the dataset and save them as images")
    parser.add_argument('--classify', type=str, help="Classify a new image using preloaded features")
    parser.add_argument('--test', type=str, help="Test the classifier on a test dataset")
    parser.add_argument('--dataset_dir', type=str, default='catsvsdogs', help="Path to the dataset directory")
    parser.add_argument('--test_dir', type=str, default='test', help="Path to the test dataset directory")
    parser.add_argument('--cat_feature_dir', type=str, default='cat_features', help="Path to the cat feature directory")
    parser.add_argument('--dog_feature_dir', type=str, default='dog_features', help="Path to the dog feature directory")
    parser.add_argument('--gradio', action='store_true', help="Run Gradio interface for classification")

    args = parser.parse_args()

    model, feature_extractor = load_model_and_feature_extractor()
    categories = ['cat', 'dog']

    if args.find_features:
        feature_dict = extract_features_from_dataset(args.dataset_dir, categories, model, feature_extractor)
        average_features = average_features(feature_dict, categories)
        os.makedirs(args.cat_feature_dir, exist_ok=True)
        os.makedirs(args.dog_feature_dir, exist_ok=True)
        save_features_as_tiles(average_features['cat'], args.cat_feature_dir, 'cat')
        save_features_as_tiles(average_features['dog'], args.dog_feature_dir, 'dog')
        print(f"Features saved to {args.cat_feature_dir} and {args.dog_feature_dir}")
    elif args.classify:
        average_features = load_average_features(args.cat_feature_dir, args.dog_feature_dir)
        result = classify_image(args.classify, average_features, model, feature_extractor)
        print(f"The image is classified as a {result}.")
    elif args.test:
        average_features = load_average_features(args.cat_feature_dir, args.dog_feature_dir)
        test_classifier(args.test, average_features, model, feature_extractor)
    elif args.gradio:
        gr.Interface(
            fn=lambda image_path: gradio_interface(image_path, args.cat_feature_dir, args.dog_feature_dir, model, feature_extractor),
            inputs=gr.Image(type="filepath"),
            outputs=gr.Textbox(),
            title="Cat vs Dog Image Classification",
            description="Upload an image to classify it as a cat or a dog."
        ).launch()
    else:
        print("Please specify either --find_features, --classify, --test, or --gradio.")

if __name__ == "__main__":
    main()
