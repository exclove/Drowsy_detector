# Drowsy_detector
ResNet, Mediapipe based Drowsy detector Using webcam



A ResNet and Mediapipe-based system to detect drowsiness using a webcam. This project tracks eye closure and issues warnings if eyes remain closed for over 5 seconds, enhancing safety and productivity.






**Features**

	1.	Dataset:
	    •	Source: MRL Eye Dataset (Creative Commons Zero (CC0) 1.0 License)
	    •	Content: 4000 images of closed eyes, 4000 images of open eyes.
	2.	Preprocessing:
	    •	Images were colorized for enhanced model input.
	3.	Model Training:
	    •	A ResNet model was trained to classify open and closed eyes.
	    •	Utilized the Tadam optimizer, an improved version of Adam, for efficient training. For more details, refer to the Tadam GitHub repository.
	4.	Real-Time Detection:
	    •	Integrated with a webcam to monitor eye closure in real-time.
	    •	Triggers a warning if eyes remain closed for more than 5 seconds.








**File Descriptions and Workflow**


*colorizing.py*
	
 Purpose: Preprocesses the dataset by converting grayscale images into colorized images.
 
 How it works:
 
	    •	Loads raw grayscale images from the dataset.
     
	    •	Applies a predefined colorization algorithm to enhance image quality.
     
	    •	Saves the processed images for use in model training.

 *global_name_space.py*
 
Purpose: Centralizes configuration and global variables for the project.

How it works:

	•	Defines paths for the dataset, processed images, and model checkpoints.
 
	•	Includes helper functions for logging and parameter management.

*train.py*

Purpose: Trains the ResNet-based model on the preprocessed dataset.

How it works:

	•	Loads colorized images and their corresponding labels.
 
	•	Configures and initializes a ResNet model.
 
	•	Uses Tadam, an enhanced optimizer based on Adam, for improved performance during training.
 
	•	Trains the model using a standard training loop with validation.
 
	•	Saves the trained model for real-time use in detection.

*main.py*

Purpose: Implements the real-time drowsiness detection system.

How it works:

	•	Loads the pre-trained ResNet model.
 
	•	Uses Mediapipe to track eye landmarks via a webcam feed.
 
	•	Classifies eye states (open/closed) using the ResNet model.
 
	•	Monitors the duration of closed eyes and triggers a warning if they remain closed for more than 5 seconds.



**Acknowledgements**

	•	The project utilizes the MRL Eye Dataset available on Kaggle.
 
	•	Special thanks to the creators of ResNet, Mediapipe, and the developers of Tadam.
