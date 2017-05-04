import numpy 
import glob

#Code to obtain min and max values for a particular variable to define range for viualization purposes- video in Matlab
path = input('\nSpecify the path to files and end with /  (for example /Users/krzysztofnalborski/Desktop/BD_Group/Rainfall/) : ')
files = glob.glob(path + '*.txt')

list_max =[]
list_min = []
for file in files:
    print(file)
    array = numpy.loadtxt(file, skiprows=6, comments=None)
    a_ = numpy.ma.masked_array(array, array == -9999.0)
    max_value = numpy.max(a_)
    min_value = numpy.min(a_)
    list_max.append("%.2f" % round(max_value,2))
    list_min.append("%.2f" % round(min_value,2))


float_max_list = []
for i in list_max:
    x = float(i)
    float_max_list.append(x)

float_min_list = []
for i in list_min:
    x = float(i)
    float_min_list.append(x)


print ('Max is:', max(float_max_list))
print ('Min is:', min(float_min_list))

