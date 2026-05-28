"""
Dataset management and preprocessing for face recognition
"""

import os
from pathlib import Path
import cv2
import numpy as np


class FaceDataset:
    """Manage and organize face recognition dataset"""

    def __init__(self, dataset_path):
        """Initialize dataset with path"""
        self.dataset_path = Path(dataset_path)
        self.faces = []
        self.labels = []

    def load_dataset(self):
        """Load all face images from dataset directory"""
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path {self.dataset_path} does not exist")

        label = 0
        for person_dir in sorted(self.dataset_path.iterdir()):
            if not person_dir.is_dir():
                continue

            person_name = person_dir.name
            print(f"Loading faces for {person_name}...")

            for image_file in person_dir.glob("*.*"):
                if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    img = cv2.imread(str(image_file))
                    if img is not None:
                        self.faces.append(img)
                        self.labels.append(label)

            label += 1

        print(f"Loaded {len(self.faces)} face images")
        return len(self.faces) > 0

    def get_statistics(self):
        """Get dataset statistics"""
        if not self.faces:
            return None

        return {
            "total_images": len(self.faces),
            "total_people": len(set(self.labels)),
            "unique_labels": sorted(set(self.labels))
        }

    def split_dataset(self, train_ratio=0.8):
        """Split dataset into train and test sets"""
        total = len(self.faces)
        train_size = int(total * train_ratio)

        indices = np.random.permutation(total)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        train_faces = [self.faces[i] for i in train_indices]
        train_labels = [self.labels[i] for i in train_indices]

        test_faces = [self.faces[i] for i in test_indices]
        test_labels = [self.labels[i] for i in test_indices]

        return (train_faces, train_labels), (test_faces, test_labels)

    def get_face_count_per_person(self):
        """Get count of faces per person"""
        counts = {}
        for label in set(self.labels):
            count = sum(1 for l in self.labels if l == label)
            counts[f"Person_{label}"] = count
        return counts


def create_dataset_directory(base_path, person_names):
    """Create directory structure for dataset"""
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    for person_name in person_names:
        person_dir = base_path / person_name
        person_dir.mkdir(exist_ok=True)
        print(f"Created directory: {person_dir}")


if __name__ == "__main__":
    print("Face dataset management module loaded")

