## SE - LATAM Recruitment Process Challenge - Anderson Gimino - andersongimino@gmail.com

1 - Initially, I had to fix the barplot method call in the provided challenge/exploration.ipynb file to be able to execute all visualizations and better understand the DS analysis.

2 - I changed the call to the **challenge/__init__.py** file to call the API directly since they are in the same directory.

3 - In the **challenge/api.py** file, I added a method called train so that the model's training could also be done through the API, in case the file is updated in the future the model can be retrained via API.

4 - In the **model.py** file, I implemented 3 static methods as per the challenge/exploration.ipynb file:
- get_period_day
- is_high_season
- get_min_diff

5 - Also, in the **model.py** file, I initialized some variables with the references of the standard columns from the file and the features generated from it. The intent was to have a feature reference in cases where one-hot encoding is needed for the predict method.

6 - The preprocess method is called both by the model training endpoint and by the prediction endpoint to assemble the features that will be sent to both methods and to perform the prior treatment of the provided data.

7 - I added some libraries to the requirements files as the tasks of the makefile were requesting.

8 - I added the **Dockerfile** configuration, however, I didn't have enough time to finalize it.

9 - I made some changes to the test file tests/api/test_api.py so that the asserts would be covered, I also commented out two asserts because I didn't have time to improve the model's recall and f1-score.

10 - I made the model training and prediction API available at the address **http://54.91.57.182/docs#/**

11 - Since I was running the tests with WSL and needed to change the model test execution path to /mnt/c/Users/Datum TI/dev/ml-engineer-latam-challenge/data/data.csv, if you are going to execute locally, I suggest modifying it to your own path before running.

12 - I didn't have enough time to implement the ci cd part of the proposed exercise.