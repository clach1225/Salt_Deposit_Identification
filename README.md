This is Christopher Lach's Repo for the Kaggle TGS Salt Identification Challenge. The repo was created to fulfilment of my Practicum 2 Project at Regis Univerisity in Denver, CO. This project is my final step in completing my studies at Regis.

The Project consisted of properly classifing individual pixels of seismic images as 0 (sediment/black) or 1 (salt/white). The data was downloaded from Kaggle. The dataset consist of 4,000 traning images as well as their associated mask. The test set is can be unzipped in a new directory which contains 18,000 images to classify for submission.

The following Notebooks are useful for understanding how we generate masks: IMAGE_AUG_FOLDERS.ipynb and Run_Python_Scripts.ipynb.

My orginal notebook created was IMAGE_AUG_FOLDERS.ipynb and this consist of a more hard coded, ridgid approach to the project. The user may use this file to run their own experiments but may have to adjust the parameters and file paths to make the code execute.

The other provided notebook, Run_Python_Scripts.ipynb, is my finalized notebook for this project and is more user-friendly. I would recommend going to this file to test out generating masks. 

Both of the previously mentioned notebooks will Read the path to the training/test data, convert the images/masks into numpy arrays as well as conver the images/masks to 128 x 128 pixels, augment the images/masks, use a U-Net CNN to classify pixels, resize the predict arrays back to the orginal 101 x 101 pixels. The final step is a function used to generate a csv for Kaggle submission.

This project was a very interesting one for me and I hope to anyone who stumbles upon this github will understand the process that I went through in its design. I hope that you will take away some aspect of it in a similar future project. 

All the best,
Christopher Lach
