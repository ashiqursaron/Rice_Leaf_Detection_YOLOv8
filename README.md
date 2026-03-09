# Rice Leaf Disease Detection using YOLOv8

## Project Description
This project focuses on the detection and classification of various rice leaf diseases using the state-of-the-art YOLOv8 object detection model. The goal is to provide an automated system for identifying common rice plant afflictions, aiding in early disease management and improving crop yield.

## Features
-   **Multi-Class Detection**: Identifies 8 distinct rice leaf diseases (Bacterial_Leaf_Blight, Brown_Spot, HealthyLeaf, Leaf_Blast, Leaf_Scald, Narrow_Brown_Leaf_Spot, Neck_Blast, Rice_Hispa).
-   **YOLOv8 Implementation**: Utilizes the Ultralytics YOLOv8 architecture for efficient and accurate object detection.
-   **Custom Dataset Training**: Trained on a custom annotated dataset of rice leaf images.
-   **Performance Evaluation**: Includes evaluation metrics such as mAP@0.5, Precision, and Recall to assess model performance.

## Dataset
The project uses a custom dataset named `RiceLeafAnnotatedDataset`, which includes `train`, `valid`, and `test` splits with corresponding image and label files in YOLO format.

## Installation
To set up the environment and run the project, follow these steps:

1.  **Clone the repository** (if applicable):
    ```bash
    git clone <your-repo-link>
    cd <your-repo-name>
    ```

2.  **Install required libraries**:
    ```bash
    !pip install ultralytics
    !pip install opencv-python
    ```

3.  **Mount Google Drive** (if your dataset is stored there, as in the notebook):
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

## Usage

### Model Prediction
To use the trained model for prediction on new images:

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('/content/drive/MyDrive/RiceLeafAnnotatedDataset/yolo_train_2/weights/best.pt') # Or last.pt

# Run prediction on an image
results = model.predict(source='/path/to/your/image.jpg', save=True, conf=0.25)

# Display results (if running in an environment like Colab)
# from IPython.display import Image, display
# display(Image(filename=results[0].save_dir + '/image.jpg')) # Adjust path as needed
Model Training
If you wish to retrain or fine-tune the model, use the following code (ensure data.yaml points to your dataset configuration):

from ultralytics import YOLO
import torch

model = YOLO('/content/drive/MyDrive/RiceLeafAnnotatedDataset/yolo_train_2/weights/last.pt') # Optionally start from a pretrained model

model.train(
    data='/content/drive/MyDrive/RiceLeafAnnotatedDataset/data.yaml',
    epochs=200,  # total target epochs
    batch=16,
    save=True,
    save_period=1,
    project='/content/drive/MyDrive/RiceLeafAnnotatedDataset',
    name='yolo_train_2',
    resume=False, # Set to True to continue interrupted training
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
Model Validation
To validate the model on the validation set:

!yolo val model='/content/drive/MyDrive/RiceLeafAnnotatedDataset/yolo_train/weights/best.pt' data='/content/drive/MyDrive/RiceLeafAnnotatedDataset/data.yaml' imgsz=640
Results and Performance
The training process logs various metrics. Key performance indicators are typically:

mAP@0.5: Mean Average Precision at an IoU threshold of 0.5.
Precision: The accuracy of positive predictions.
Recall: The ability of the model to find all positive instances.
Example validation results (from the notebook):

                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all        400        656      0.602      0.344      0.386      0.187
 Bacterial_Leaf_Blight         50         58      0.623      0.517      0.543      0.304
            Brown_Spot         49        128      0.321      0.438      0.347      0.122
           HealthyLeaf         50         76      0.661      0.276      0.343      0.209
            Leaf_Blast         49         63      0.915     0.0159      0.159     0.0539
            Leaf_Scald         49         60      0.531     0.0333       0.23      0.109
Narrow_Brown_Leaf_Spot         48         66      0.534      0.278      0.254      0.113
            Neck_Blast         50        100      0.584       0.79      0.726      0.335
            Rice_Hispa         50        105      0.649      0.405      0.483      0.253
Plots such as confusion_matrix.png and results.png (showing mAP, Precision, Recall over epochs) are generated during training to visualize performance.

Acknowledgments
Ultralytics YOLOv8 team for the powerful object detection framework.
The creators of the RiceLeafAnnotatedDataset for providing the training data.
Feel free to reach out for any questions or collaborations! ```
