Thanks for taking the time to look at the codebase for my Honours Project! If you plan on actually running the code, there are a few things to take note of.

- The transfer learning script has been removed, as the majority of it was previously written by my supervisor with some edits from me to feed in the neural network features.
- The iNat dataset is not included as it would greatly increase the size of the submitted folder, and because I do not have permission to redistribute it.
- When you want to run an experiment, please create a folder following this structure:

E.g. To generate a synthetic dataset using the 'mini' format with 1000 classes, create the folder 'synth_data_1000_mini' under the 'synth_data' folder. For a 'full' dataset, replace mini with full. The same can be done with iNat. This is to prevent the code from throwing an error when a target folder doesn't exist.

Once the folder exists, you can run the code with the INIT_SYNTH or INIT_INAT settings, and the generated data will be sent to the relevant folder.

You will need to separately download the inat data from https://github.com/visipedia/inat_comp/tree/master/2021 and place in the inat_2021_data folder. The specific data you will need is the train annotations, train_mini annotations and validation annotations.

Once you have generated a dataset, you can run the code with LOAD_SYNTH, LOAD_SYNTH_MINI, LOAD_INAT or LOAD_INAT_MINI in order to train a model and output results.

Lastly, you will need to run the code in an evironment with the relevant packages:
numpy
sklearn
pandas
pytorch
matplotlib

Most versions of these should work, but the code is designed to run on Python 3.8 ideally.

There are basic comments throughout explaining the structure, but I will be happy to answer any code-related questions!