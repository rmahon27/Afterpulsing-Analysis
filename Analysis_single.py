import numpy as np
from matplotlib import pyplot as plt 
from tqdm import tqdm 
import matplotlib as mpl
from matplotlib.ticker import NullFormatter
import matplotlib.mlab as mlab
import time
import pandas as pd 
from tqdm import tqdm
from scipy.optimize import curve_fit
import sys
import os 


## NOTE: This file is meant for producing specific results with a single file of outputted data taken from a Tpx3Cam Camera. This file will not work 
## to produce results for multiple files. This file uses a class system to perform the analysis which is a very flexible method of data analysis. In 
## principle, you should be able to add as many definitions as you would like to this file or change definitions as you wish to produce different 
## results or get new results altogether. This file was simply made with the goal to speed up computation time when performing analysis. 

## Extra NOTE: It's likely anyone viewing this file might not be aware of the library tqdm and as a result may be confused about its usage. It
#  functionally acts the same as a regular loop, however it constructs a loading bar for you which tells you the percent of progress through the loop
#  and gives an estimate for how much time is left to complete the loop. 


current_dir = sys.path[0]                   # Locates the current system pathway 

# Definition to create a folder called 'plots' to store most results
def make_folder():
    directory = 'Plots'
    if_exists = os.path.exists(directory)
    if if_exists != True:
        path = os.path.join(current_dir,directory)
        os.mkdir(path)
    else: 
        return None 

# Definition to define a variable line to fit a straight line to data
def line(x,m,B):
    return m*x + B


# Defining a general high order polynomial equation for fitting the data of the distance histogram. This polynomial very likely may need to be modified 
# if it does not properly fit your data. 
def poly(x,c1,c2,c3,c4,c5,c6,c7,c8):
    return c1*x**7 + c2*x**6 + c3*x**5 + c4*x**4 + c5+x**3 +c6*x**2 + c7*x + c8 

# Definition to integrate a function over a range starting with "a" and ending with "b" using n steps. Can change n if you feel more steps are required. 
# This function uses Simpsons method to perform integration. 
def Simpsons(f,a,b,n = 1e6):
    Sum = 0 
    dx = (b-a)/n
    i = 1 
    xi = a
    while i < n:
        fi = f(xi)
        fm = f(xi+0.5*dx)
        fe = f(xi+dx)
        Sum += (1/6)*(fi + 4*fm + fe)*dx
        xi += dx 
        i += 1 
    return Sum

#Returns a specific column from an array 
def get_column(arr,col_num):
    return arr[:,col_num]

#returns an array of differences between each element in the input array
def delement(arr):
    output_arr = []
    for i in range(len(arr)-1):
        delem = arr[i+1] - arr[i]
        output_arr.append(delem)
    return output_arr

# Class for the initial batch of results obtained from the data
class init_results:
    # Initializing function to call in and read the data from the data file and create separate variables for each column of data. Pandas is used here 
    # to read in the csv file as I've found pandas is significantly faster at reading in csv files compared to numpy or other methods. The correction 
    # file is also called in here and a separate variable is created for the corrected times.  
    def __init__(self, file, file_corr):
        array = pd.read_csv(file,  dtype= float, delimiter=",", usecols = (0,1,2,3,4,5))

        data = np.array(array)
        self.x = get_column(data,0)
        self.y = get_column(data,1)
        self.t = get_column(data,2)
        self.a = get_column(data,3)
        self.A = get_column(data,4)
        self.n = get_column(data,5)

        correction = np.loadtxt(file_corr, unpack=True, delimiter=",")
        TOT_i  = [int(tot/25) for tot in self.a]
        self.t_corr = self.t/4096*25. + correction[1][TOT_i]*1000.

    # Function to retrieve a histogram of the time values throughout the data set.  
    def get_TOA(self):
        fig, ax0 = plt.subplots(ncols=1, figsize=(10, 4))
        plt.hist(self.t/4096*25, bins = 100, color = 'r', ec = 'k')
        plt.title("TOA", fontsize = 12)
        plt.xlabel('TOA, ns',fontsize = 12)
        plt.ylabel('counts',fontsize = 12)
        plt.savefig('Plots/t_hist')
        plt.show()

    # Function to retrieve a 2D histogram of the x and y values of the data set retrieved from the camera. 
    def get_2D_xy(self):
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(9.5, 4))

        h = ax0.hist2d(self.x, self.y, bins = 256, range = [(0, 256), (0, 256)])
        fig.colorbar(h[3], ax = ax0)
        h = ax1.hist2d(self.x, self.y, bins = 256, range = [(0, 256), (0, 256)],
                      norm=mpl.colors.LogNorm())
        fig.colorbar(h[3], ax = ax1)
        fig.tight_layout()
        plt.savefig('Plots/2D_hist_xy')
        plt.show()

    # Function to retrieve a histogram of the  "time over threshold" values from the Tpx3cam which tells you the amount of energy absorbed in each pixel
    # of each hit.
    def get_a_hist(self):
        fig, ax = plt.subplots(ncols = 1, figsize = (10,4))
        ax.hist(self.a, bins = 100, range = (0, 2500),
                color = 'r', ec = 'k', histtype = 'stepfilled')
        plt.xlabel('TOT(ns)',fontsize = 12)
        plt.ylabel('counts',fontsize = 12)
        plt.savefig('Plots/a_hist')
        plt.show()

    # Function to retrieve a histogram of the "centroid" values retrieved from the camera. 
    def get_n_hist(self):
        fig, ax = plt.subplots(ncols = 1, figsize = (10,4))
        ax.hist(self.n, bins = 50, range = (0, 50), 
                color = 'r', ec = 'k', histtype = 'stepfilled')
        plt.xlabel('Npixels',fontsize = 12)
        plt.ylabel('counts',fontsize = 12)
        plt.yscale('log')
        plt.savefig('Plots/n_hist')
        plt.show()

    # Function to retrieve subplots of histograms of both the x and y values from the data. 
    def get_x_y_subplot(self):
        fig, (ax0, ax1) = plt.subplots(ncols = 2, figsize = (10,4))
        ax0.hist(self.x, bins = 256, range = (0,256), color = 'g',
                 alpha = 0.5, histtype = 'stepfilled')
        ax0.set_title('x', fontsize = 12)
        ax0.set_xlabel('x, pixel', fontsize = 12)
        ax0.set_ylabel('counts', fontsize = 12)

        ax1.hist(self.y, bins = 256, range = (0,256), color = 'g',
                 alpha = 0.5, histtype = 'stepfilled')
        ax1.set_title('y', fontsize = 12)
        ax1.set_xlabel('y, pixel', fontsize = 12)
        ax1.set_ylabel('counts', fontsize = 12)
        plt.savefig('Plots/x_y_subplot')
        plt.show()

    # Function to select a specific area on the camera and to create new variables housing the data within the conditions of this new selected area. 
    def select_area(self):
        x1 = []; y1 = []; t1 = []; a1 = []; n1 = []; tc1 = []
        px1min = 2;  px1max = 254; py1min = 2; py1max = 254;
        nmin = 1; nmax = 50; amin = 100; amax = 3000

        for i in tqdm(range(len(self.x)-1)):
            if ( px1min <= self.x[i] <= px1max and py1min <= self.y[i] <= py1max 
                and nmin<=self.n[i]<=nmax and amin<=self.a[i]<=amax):

                x1.append(self.x[i])
                y1.append(self.y[i])
                t1.append(self.t[i]/4096.*25.)
                tc1.append(self.t_corr[i])
                a1.append(self.a[i])
                n1.append(self.n[i])
        self.x_new = x1 
        self.y_new = y1
        self.t_new = t1 
        self.t_corr_new = tc1
        self.a_new = a1
        self.n_new = n1

    # Function to retrieve subplots of histograms of the x and y data from the selected regions as explained in the previous function. 
    def get_new_x_y_subplots(self):
        fig, (ax0, ax1) = plt.subplots(ncols = 2, figsize = (10,4))
        ax0.hist(self.x_new, bins = 256, range = (0,256), color = 'g',
                 alpha = 0.5, histtype = 'stepfilled')
        ax0.set_title('x', fontsize = 12)
        ax0.set_xlabel('x, pixel', fontsize = 12)
        ax0.set_ylabel('counts', fontsize = 12)

        ax1.hist(self.y_new, bins = 256, range = (0,256), color = 'g',
                 alpha = 0.5, histtype = 'stepfilled')
        ax1.set_title('y', fontsize = 12)
        ax1.set_xlabel('y, pixel', fontsize = 12)
        ax1.set_ylabel('counts', fontsize = 12)
        plt.savefig('Plots/x1_y1_subplot')
        plt.show()
    
    # Function to retrieve histogram of ToT values of data from selected regions. 
    def get_new_a_hist(self, amax):
        #Choose amax based off the max range of the data shown from the original histogram from earlier in this class

        fig, ax = plt.subplots(ncols = 1, figsize = (10,4))
        ax.hist(self.a_new, bins = int(amax/25), range = (0,amax), 
                color = 'r', ec = 'k', histtype = 'stepfilled')
        ax.set_xlabel('TOT,ns', fontsize = 12)
        ax.set_ylabel('counts', fontsize = 12)
        ax.set_title('TOT', fontsize = 12)
        plt.savefig('Plots/a1_hist')
        plt.show()
    
    # Function to retrieve histogram of "centroid" values from selected regions. 
    def get_new_n_hist(self):
        fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
        alim = 20;
        nbins = alim; 
        h1 = ax0.hist(self.n_new, bins = nbins, range = (0, alim), 
                      color = 'r', histtype = 'step', linewidth = 2)
        ax0.set_xlim(0,alim)
        ax0.set_xlabel('# of pixels in cluster',fontsize = 18)
        ax0.set_ylabel('counts',fontsize = 18)
        plt.savefig('Plots/n1_hist')
        plt.show()
 
# Class to obtain the "final" results of the data analysis which is very subjective since final is being defined here based on what the final 
# results being obtained during the time of writing this.  
class Final_results:
    # Initializing definition to call in the "new" data from the previous class. 
    def __init__(self, x, y, t):
        self.x = x 
        self.y = y 
        self.t = t 
        self.sort_key = np.argsort(t)

    # This function checks to see if the current data involves corrected time data or regular time data. If the input is a "y", it treats it as
    # corrected data and vice versa. 
    def check_corr(self, cond):
        if cond == "y":
            plot_loc = "_corr"
        elif cond == "n":
            plot_loc = "_reg"
        self.plot_loc = plot_loc

        directory = 'Plots' + plot_loc
        
        ifexists = os.path.exists(directory)

        if ifexists != True:
            path = os.path.join(current_dir,directory)
            os.mkdir(path)
        else: 
            return None 
    
    # This function sorts all of the data based on the sorting of the time array. 
    def get_new_vars(self):
        t_new = []; x_new = []; y_new = []
        for i in tqdm(range(len(self.sort_key))):
            sort_iter = self.sort_key[i]
            t_new.append(self.t[sort_iter])
            x_new.append(self.x[sort_iter])
            y_new.append(self.y[sort_iter])
        self.x_new = x_new 
        self.y_new = y_new
        self.t_new = t_new
    
    # This function retrieves the time difference and location difference values and puts them into arrays. dt represents the difference in time between
    # successive hits, dx and dy represent the respective dimensional differences in distance between successive hits and dist represents the full
    # distance on the sensor between successive hits. 
    def get_difference_vals(self):
        dt = []; dx = []; dy = []; dist = []
        dt = delement(self.t_new)
        dx = delement(self.x_new)
        dy = delement(self.y_new)
        for i in tqdm(range(len(dx))):
            dist_iter = dx[i]**2 + dy[i]**2 
            dist.append(np.sqrt(dist_iter))
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.dist = dist
        
        self.dx = np.array(self.dx)
        self.dt = np.array(self.dt)
        self.dy = np.array(self.dy)
    
    # This function plots a histogram of the time differences to allow you to see if the histogram looks good and to choose which region you want to
    # use to construct a fit of the baseline. 
    def get_dt_plot_init(self, Tlim):
        self.Tlim = Tlim
        self.nbins = int(self.Tlim/(1.5625)); 
        hist_full = np.histogram(self.dt, bins = self.nbins, range = (0,self.Tlim))
        bins = hist_full[0]
        edges = hist_full[1]
        bin_centers = (edges[1:] + edges[:-1])/2
        
        plt.rcParams["figure.figsize"] = (10,5)
        plt.plot(bin_centers,bins, '.b')
        plt.xlabel("$\Delta T$, ns", fontsize = 12)
        plt.ylabel('counts', fontsize = 12)
        plt.savefig('Plots' + self.plot_loc + '/dt_plot_init')
        plt.show()
        
        self.dt_bins = bins
        self.dt_edges = edges
        self.dt_bin_centers = bin_centers
    
    # This function constructs a fit line of the baseline, shades the access region and calculates the afterpulsing fraction. 
    def get_dt_plot_final(self):
        fit_indices = []   

        xfit_low_lim = int(input("Enter the lower limit for the fit of the baseline: \n"))
        xfit_hi_lim = int(input("Enter the upper limit for the fit of the baseline: \n"))
        access_lim = int(input("Enter your estimate for the access limit (x value where afterpulsing begins): \n"))

        # Checking to see if data point is within the chosen range for the fit and appending to the new list if true. 
        for i in tqdm(range(len(self.dt_bin_centers))):
            if xfit_low_lim <= self.dt_bin_centers[i] <= xfit_hi_lim:
                fit_indices.append(i)
        self.counts_fit = self.dt_bins[fit_indices]
        self.x_fit = self.dt_bin_centers[fit_indices]

        #Constructing the fit line by using scipy's curve_fit function.
        popt, pcov = curve_fit(line, self.x_fit, self.counts_fit)

        x_range = np.linspace(0,200, 10000) # Creates a range of x values for the plotting of the fit line, might need to change 200 to a different value 
                                            # if you want to plot a fit line over a different range. 
        fit_line = line(x_range, *popt)

        m = popt[0]
        b = popt[1]
        
        self.fit_line = fit_line
        self.fit_slope = m
        self.fit_intercept = b
        
        fit_line_func = lambda x: line(x, self.fit_slope, self.fit_intercept)
        
        self.fit_func = fit_line_func
        
        access_sum = Simpsons(self.fit_func, 0,access_lim)         # Integrating the fit function over the range of the access 
        
        # Calculating the afterpulsing perecent by subtracting the integral value from the number of data points within the acess and diving by 
        # the total number of data points. 
        self.dt_AP_percent = ((len(self.dt[np.where(self.dt < access_lim)]) - access_sum)/len(self.dt))*100
        
        #Creating the shaded region on the plot to represents the region of access which shows afterpulsing. 
        x_shade_range = np.linspace(0,access_lim, len(self.dt_bins))
        fit_line_shade_region = fit_line_func(x_shade_range)
        
        plt.rcParams["figure.figsize"] = (10,5)
        plt.plot(self.dt_bin_centers,self.dt_bins, '.b')
        plt.plot(x_range, fit_line, '--b')
        plt.xlabel("$\Delta T$, ns", fontsize = 12)
        plt.ticklabel_format(style = 'sci', axis = 'y', scilimits=(0,0))
        plt.ylabel('counts', fontsize = 12)
        plt.axvline(x=access_lim, color = 'r')
        plt.xlim(0,self.Tlim/2)
        plt.text(0.6*self.Tlim/2,0.8*max(self.dt_bins),' Afterpulsing fraction: %.1f%% ' % (self.dt_AP_percent), fontsize = 16)
        plt.fill_between(self.dt_bin_centers, y1 = fit_line_shade_region, y2 = self.dt_bins, 
                         step = "pre", where = (self.dt_bin_centers>=0) & (self.dt_bin_centers <= access_lim), 
                         color = 'yellow')
        plt.ylim(0,max(self.dt_bins))
        plt.savefig('Plots' + self.plot_loc + '/dt_plot_final')
        plt.show()

    # Function which creates 2D histograms of dx and dy  
    def get_2D_dxdy_subplots(self):
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(9.5, 4))

        rr = 20

        h = ax0.hist2d(self.dx, self.dy, bins = rr*2+1, 
                       range = [(-rr, rr), (-rr, rr)])
        fig.colorbar(h[3], ax = ax0)
        h = ax1.hist2d(self.dx, self.dy, bins = rr*2+1, 
                       range = [(-rr, rr), (-rr, rr)], 
                       norm=mpl.colors.LogNorm())
        fig.colorbar(h[3], ax = ax1)
        fig.tight_layout()
        plt.savefig('Plots' + self.plot_loc + '/2D_subplot_dxdy')
        plt.show()
    
    # Function which creates a singular non logged 2D histogram of dx and dy
    def get_2D_dxdy(self):
        fig,ax0 = plt.subplots(ncols=1, figsize=(10, 4))

        rr = 20

        h = ax0.hist2d(self.dx, self.dy, bins = rr*2+1, 
                       range = [(-rr, rr), (-rr, rr)])
        fig.colorbar(h[3], ax = ax0)
        fig.tight_layout()
        plt.savefig('Plots' + self.plot_loc + '/2D_hist_dxdy')
        plt.show()

    # Function which displays a histogram of the distances between successive hits. dlim here can be changed if you want a larger/smaller range on the x axis 
    def get_dist_plot_init(self):
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12,4))

        self.dlim = 250
        self.dist_nbins = self.dlim
        
        h1 = ax.hist(self.dist, bins = self.dist_nbins, range = (0,self.dlim), color = 'b', histtype = 'step', linewidth = 2)
        ax.set_xlim(0,self.dlim)
        ax.set_xlabel('distance, pix', fontsize = 18)
        ax.set_ylabel('counts', fontsize = 18)
        plt.show()

    # Function which creates the final plot of the distances and calculates the afterpulsing fraction. This is similar to what was done before with the time differences. 
    def get_dist_plot_final(self):
        fit_indices = []   

        access_lim = int(input("Enter your estimate for the access limit (x value where afterpulsing begins): \n"))

        hist_full = np.histogram(self.dist, bins = self.dlim, range = (0, self.dlim))
        self.dist_bins = hist_full[0]
        self.dist_edges = hist_full[1]

        self.bin_centers = (self.dist_edges[1:] + self.dist_edges[:-1])/2

        for i in range(len(self.bin_centers)):
            if self.bin_centers[i] >= access_lim:
                fit_indices.append(i)

        counts_fit = self.dist_bins[fit_indices]
        x_fit = self.bin_centers[fit_indices]

        popt,pcov = curve_fit(poly, x_fit, counts_fit)
        x_range = np.linspace(0,250, 10000)
        
        fit_curve = poly(x_range, *popt)

        self.dist_fit_vals = popt

        fit_curve_func = lambda x: poly(x,*popt)

        self.access_sum = Simpsons(fit_curve_func, 0,access_lim)

        self.dist = np.array(self.dist)
        
        self.dist_AP_percent = ((len(self.dist[np.where(self.dist <= access_lim)]) - self.access_sum)/len(self.dist))*100

        plt.rcParams["figure.figsize"] = (10,5)
        plt.hist(self.dist, bins = self.dlim, range = (0, self.dlim), histtype = 'step', color = 'blue' )
        plt.plot(x_range, fit_curve, '--b')
        plt.xlabel("distance, pix", fontsize = 12)
        plt.ylabel('counts', fontsize = 12)
        plt.axvline(x=access_lim, color = 'red')
        plt.text(1.2*max(self.bin_centers)/2,0.95*max(self.dist_bins),' Afterpulsing fraction: %.1f%% ' % (self.dist_AP_percent), fontsize = 16)
        plt.xlim(0,self.dlim)
        plt.savefig("Plots" + self.plot_loc + "/dist_final_hist")
        plt.show()