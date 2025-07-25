Autoencoders for waveform feature extraction
----------------------------------------------------

This repository contains the code for training and testing autoencoder models as well as
generating traditional features from ECG waveform data. 

<br/>

> Repository Credits: <br/>
> * Roy van Mierlo (TU/e, [r.r.m.v.mierlo@tue.nl](mailto:r.r.m.v.mierlo@tue.nl))
> * Freek Relouw (TU/e, [f.j.a.relouw@tue.nl](mailto:f.j.a.relouw@tue.nl))

> Authors:
> * Roy van Mierlo (Biomedical Engineering, TU/e, [r.r.m.v.mierlo@tue.nl](mailto:r.r.m.v.mierlo@tue.nl))
> * Freek Relouw (Biomedical Engineering, TU/e, [f.j.a.relouw@tue.nl](mailto:f.j.a.relouw@tue.nl))
> * Natal van Riel (Biomedical Engineering, TU/e, [n.a.w.v.riel@tue.nl](mailto:n.a.w.v.riel@tue.nl))

Python version 3.11 was used to run the code in this repository. The code is written on a Windows system, 
but should also work on Linux systems.


---

---

## Datasets information

#### ECG waveform data from VitalDB
Raw data used for training is available upon request at [VitalDB.net](https://vitaldb.net/dataset/).
ECG (and other) waveforms, and demograpic information are available for download using
the [vitaldb python package](https://vitaldb.net/dataset/?query=lib).
ECG waveforms are found under the 'SNUADC/ECG' track. A .csv file of the demographic information can be downloaded by selecting all caseID's
in the [online data viewer](https://vitaldb.net/dataset/?query=viewer#) and then clicking
```Actions > Dowload Clinical Info```.

## Preparations (do before running code!)
Create a ```vars.env``` file in project folder containing your data folder path, the
file name of the .csv file containing the demographic information, your
Neptune project path, and your Neptune API token in the structure shown below. (No ```'```, 
```"```, or ```{}``` is needed around the values!) You can find the API token when logging
into neptune account, click on your username in the left bottom corner and then on
'Get your API token'.
    ```
    DATA_FOLDER_PATH={path to your data folder}
    DEMOGRAPHICS_FILE_NAME={e.g. demographics.csv}
    NPT_PROJECT={path to your Neptune project, e.g. royvanmierlo/AE_waveforms}
    NPT_API_TOKEN={your Neptune API token}, }
    ```

---

---

## Project Structure

###

### Python scripts

* ```config.py```: Contains the configuration for the project, such as the data folder path, demographics file name,
   Neptune project path, and Neptune API token. It reads the variables from the ```vars.env``` file.

* ```data_extract_vitaldb.py```: ECG data extraction and preprocessing. Finds case ID's in the vitalDB database 
   that have the 'SNUADC/ECG' track available. Then for each given case ID, it loads the record, preprocesses the 
   signals, extracts single beat samples, and removes unsuitable samples based on physiological thresholds. It also
   calculates the median waveforms and traditional features for the ECG signal. Finally, it saves processed samples per
   case ID in a .parquet file.

* ```data_create_arrays.py```: Combines the extracted data from the VitalDB database into arrays that can be used as 
   input for training autoencoder neural network (NN) models. Datasets are saved as .npy files and are named and shaped
   as follows: <br/>

   | Variable Name    | Data Type   | Shape of data | Description                                                                       |
   |------------------|-------------|---------------|-----------------------------------------------------------------------------------|
   | `np_waves`       | Numpy array | (batch, 320)  | ECG wave data, single 0.64 second 500Hz segment                                   |
   | `np_features`    | Numpy array | (batch, 35)   | Extracted fiducial points and traditionally used ECG waveform features            |
   | `np_info`        | Numpy array | (batch, 2)    | Case ID's and time indices of each data point (used to split dataset on patients) |

* ```main_ae.py```: Main script to train and test autoencoder models. It contains the configurations for the training 
   and testing processes, which can be changed in the argument parser. <br/>

* ```analysis_latent_size.py```: Script to analyze influence of latent size on autoencoder model performance. It 
   contains the trained model numbers and their latent sizes and visualizes the performances over the different latent
   sizes in a plot.

* ```analysis_performance_traditional_vs_ae.py```: Script to reconstruct waveforms from fiducial points using the 
   traditional method with Gaussian-based interpolation. Finds the optimal set of sigma parameters for the Gaussian 
   kernel to minimize the error between the reconstructed and original waveforms. Then it reconstructs the waveforms 
   using the optimal sigma's and calculates the error between the reconstructed and original waveforms. Finally, it
   compares the errors of the traditional method with the best performing autoencoder models and plots the results.

### Function packages:

* ```./data_load_preprocess```: Functions to load and preprocess the data.
  * ```load_record.py```: Returns the record of the queried tracks for the given the case_id if all tracks contain data.
  * ```filters.py```: Contains functions ```butterworth``` and ```zscore``` for filtering the records.
  * ```sample_from_record.py```: Sample 20-second segments from the full waveform record by slicing without overlap.
  * ```remove_samples.py```: Applies beat-detection (R-peak) and removes samples with an HR <30 bpm or >180 bpm, with
    frequent (>50%) premature ventricular contractions, or with a large SD compared to nr of found beats ratio > 0.0335.
  * ```median_sample.py```: Contains functions to calculate the median waveform and the median fiducial points from the 
    samples.
  * ```./feature_extraction```: Contains functions to extract features from waveform data.
    * ```features_ecg.py```: Contains functions to convert calculated fiducial points to points relative to the
      R-peak, and to calculate the traditional ECG morphology features.
<br/><br/>

* ```./functions_ae```: Functions to support the autoencoder model training and testing.
  * ```run_train.py```: Contains the functions for model starting model training. 
    * ```main_train```: The main model training process. Calls load_datasets and split_datasets, creates the data 
      generators for the train and validation set, then creates the network structure and calls train_with_lr_scheduler.
      <br/><br/>
    * ```train_with_lr_scheduler```: Trains the model using a learning rate scheduler. Selects the loss, optimizer, and
      learning rate scheduler based on the arguments passed to the function. Then loops over the epochs and internally
      loops over the batches. Calls the validate function after valid_interval batches. Saves the model with the best
      validation loss. 
    * ```cosine_annealing_warmup_lr```: Creates a schedule with a learning rate that decreases following the values of the
      cosine function between the initial lr set in the optimizer to eta_min, with several hard restarts, after a warmup 
      period during which it increases linearly between 0 and the initial lr set in the optimizer.
    * ```validate```: Function for the validation process. It evaluates the model on the validation set.
    * ```class EarlyStopping```: Class for early stopping during training. It monitors the validation loss and stops the 
      training if the validation loss does not improve for a certain number of epochs (patience). Also saves the best 
      model.
  * ```run_test.py```: Main function for testing the model. Loads the data, runs through the test set and saves the 
    predictions to .npy files.
  * ```load_data.py```: Contains functions to load the datasets from the .npy files and to split the dataset into train, 
    validation and test sets based on the unique ids.
  * ```load_model.py```: Load a trained model from the given directory.
  * ```model_classes.py```: Contains several autoencoder model classes. 
  * ```ranger_optimizer.py```: Contains the Ranger optimizer class, which is a combination of RAdam and LookAhead 
    optimizers.
  * ```loss_functions.py```: Contains custom loss functions for autoencoder evaluation.
  * ```custom_plots.py```: Contains a function to plot RMSE distributions between the original and reconstructed
    waveforms. 
<br/><br/>

* ```./functions_traditional```: Functions to support the traditional method signal reconstruction. 
  * ```signal_reconstruction.py```: Contains functions to reconstruct the ECG waveform from the fiducial points using 
    Gaussian-based interpolation as well as functions for alternative interpolation strategies.
<br/><br/>

### Other (generated) folders:
* ```./results```: Contains the results of the autoencoder model training and testing. It contains the trained models, 
  training and validation results, and the test results.
