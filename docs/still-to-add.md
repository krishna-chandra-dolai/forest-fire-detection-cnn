# Still To Add

The project is now structured and uploaded, but these items would make it more complete:

1. Original CNN image dataset

   Add the actual `fire` and `no_fire` image folders under:

   ```text
   data/
   |-- train/
   |   |-- fire/
   |   `-- no_fire/
   `-- test/
       |-- fire/
       `-- no_fire/
   ```

2. Trained model file

   Add the final trained model under `models/`, for example:

   ```text
   models/forest_fire_resnet50.keras
   ```

3. Executed CNN notebook

   After adding the dataset, run:

   ```text
   notebooks/forest_fire_detection_cnn_resnet50.ipynb
   ```

   Then save the notebook with its real output cells.

4. Demo application

   A simple Streamlit or Flask app can be added for uploading an image and showing the prediction result.

5. License

   Add a license file if the GitHub repository should be reusable by others.
