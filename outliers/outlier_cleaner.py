#!/usr/bin/python
from audioop import reverse


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    print "Hellow"
    errors=(predictions-net_worths)**2
    cleaned_data=zip(ages,net_worths,errors)
    print cleaned_data
    cleaned_data=sorted(cleaned_data,key=lambda x:x[2][0],reverse=True)
    limit=int(len(cleaned_data)*0.1)
    
    
    
    
    return cleaned_data[limit:]

