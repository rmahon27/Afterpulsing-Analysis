# Afterpulsing-Analysis
This is a method of performing analysis on data from a Tpx3Cam Camera to ger results pertaining mainly to afterpusling. This analysis will provide many plots neeeded for afterpulsing analysis including histograms of various data values like the x and y positions of hits on the camera, ToA (time of Arrival), ToT (Time over Threshold) and more. This analysis will also provide 2d histograms of dx and dy (difference in x and y values between successive hits), all of the final plots produced from this analsyis will return different versions for both corrected data values and non corrected values. The respective plots are sent to their own separate, labeled folders which the code also produces for you. The most significant results this analsyis produces are histograms of the time difference and distance values of successive hits. These histograms provide the afterpulsing fraction given in percent form and produces a shaded region for the region you choose as the access region (region of the data where afterpulsing occurs). As of now, this analysis only works with a single data file. The runtime for this analysis varies. For a smaller file like used in the example jupyter notebook, this analysis cna take as little as 5 minutes. Longer files however (around say 1.5GB) can take around a half hour for the computer used. 

# Code Process
The code uses a class based method where the data file you are using is inputted into a python class. Inside the class, the data file is parsed separated into various values like x,y etc. A correction file is also needed for this analysis and is similarly parsed inside the python class. Inside the class, many different functions are created which can be called inside a jupyter notebook file. An example jupyter notebook file is provided where a file called "DCR_hiQEred_100s_W0057_H07-220814-105649-1_cent.csv" is used with a correction file called "50um_100s_1_hiQEred_TOTcorr.csv". The plotting functions are fairly straightforward so I won't into to much detail on them here, the Analysis_single.py file can be looked at if needed. From this point on, the major steps in the analysis are as follows. 

## Step 1
Constructing the corrected time array using the provided correction file 

## Step 2 
Selecting an area of the sensor to select data from and appending values within this area to new lists. 

## Step 3 
Sorting all of the new lists using the sorting values of the time array. This is done using numpy.argsort. 

## Step 4
Finding the difference values by subtracting successive elements of each list (i.e t[i+1] - t[i]) and appending to new lists. 

## Step 5
Constructing final histograms of the analyzed data and acquiring such results as the afterpusling fraction. This is done by fitting a line to either the baseline of the dt histogram or a high order polynomial to the distance histogram. Either the fit line or fitted polynomial is then integrated over the access region of afterpulsing using simpsons rule. This integral value is then subtracted from the amount of data values falling within the access region. The last step is to divide this difference by the total number of data values. 
