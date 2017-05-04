import glob
import numpy
import numpy.ma as ma

"""Code to average raster files to represent for example, 10 year average value for a particular region- for Matlab purposes
Files need to be moved manually, for example folder from the file path below contains data for rainfall only for 
years 1910-1930 and then the final file represents the average over these 20 years; raster information such as number columns
of columns needs to be added manually"""

path1 = input('\nSpecify the path to files and end with / (for example /Users/krzysztofnalborski/Desktop/Climate_Ranges/meantemp_X/) : ')
path1 = str(path1)

VARIABLE = input('\nSpecify the variable: ')
VARIABLE = str(VARIABLE)

files1 = glob.glob(path1 + '*.txt') 

w1 = []
for file1 in files1:
    print (file1)
    array1 = numpy.loadtxt(file1, skiprows=6, comments=None)
    w1.append(array1)

w1 = numpy.array(w1)

average1 = numpy.mean(w1, axis=0)

mx = ma.masked_where(average1 == -9999.0, average1)
mx = ma.masked_where(average1 < -50, average1)

maxim = mx.max()
minim = mx.min()

print ('Max value is:', maxim)
print ('Min value is:', minim)

c1 = numpy.savetxt(VARIABLE + '.txt', average1)

