# -*- coding: utf-8 -*-
"""
Created on Wed May 16 12:56:59 2018

@author: rjagtani
"""

# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Use slicing to create downstairs
downstairs=areas[0:6]

# Use slicing to create upstairs
upstairs=areas[-4:]

# Print out downstairs and upstairs
print(upstairs)
print(downstairs)

x=[1,2,3]

y=list(x)
y[2]=5
x
y

# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Use append twice to add poolhouse and garage size
areas.append(24.5); areas.append(15.45)


# Print out areas
print(areas)

# Reverse the orders of the elements in areas
areas.reverse()

# Print out areas
print(areas)


####### For matehmatical constants and other functions and symbols

import math

######## Working with numpy arrays

# height and weight are available as a regular lists

# Import numpy
import numpy as np

# Calculate the BMI: bmi
np_height_m = np.array(height) * 0.0254
np_weight_kg = np.array(weight) * 0.453592
bmi = np_weight_kg / np_height_m ** 2

# Create the light array
light=bmi<21

# Print out light
print(light)

# Print out BMIs of all baseball players whose BMI is below 21
print(bmi[light])


######## Basic Stats Numpy

# np_baseball is available

# Import numpy
import numpy as np

# Print mean height (first column)
avg = np.mean(np_baseball[:,0])
print("Average: " + str(avg))

# Print median height. Replace 'None'
med = np.median(np_baseball[:,0])
print("Median: " + str(med))

# Print out the standard deviation on height. Replace 'None'
stddev = np.std(np_baseball[:,0])
print("Standard Deviation: " + str(stddev))

# Print out correlation between first and second column. Replace 'None'
corr = np.corrcoef(np_baseball[:,0],np_baseball[:,1])
print("Correlation: " + str(corr))
corr.shape

#################################################

# Build histogram with 5 bins
import matplotlib.pyplot as plt

plt.plot(x,y) ## line plot
plt.scatter(x,y) ## scatter plot 
plt.hist(life_exp, bins = 5)    ## histogram plot

# Show and clean up plot
plt.show()
plt.clf()

# Build histogram with 20 bins
plt.hist(life_exp,bins=20)

########## Customisation

# Basic scatter plot, log scale
plt.scatter(gdp_cap, life_exp)
plt.xscale('log') 

# Strings
xlab = 'GDP per Capita [in USD]'
ylab = 'Life Expectancy [in years]'
title = 'World Development in 2007'

# Add axis labels
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')


# Add title
plt.title('World Development in 2007')

# After customizing, display the plot
plt.show()

### Scatter with size argument
plt.scatter(x = gdp_cap, y = life_exp, s = np.array(pop) * 2,c=col,alpha=0.8)


##### Dictionary Syntax

# Definition of countries and capital
countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']

# From string in countries and capitals, create dictionary europe

europe={"spain":"madrid","france":"paris","germany":"berlin","norway":"oslo"}
#europe1={countries:capitals}
# Print europe
print(europe)

# Definition of dictionary

europe = {'spain':'madrid', 'france':'paris', 'germany':'bonn',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw',
          'australia':'vienna'}

# Update capital of germany

europe['germany']='berlin'

# Remove australia

del(europe['australia'])

# Print europe

print(europe)

# Dictionary of dictionaries
europe = { 'spain': { 'capital':'madrid', 'population':46.77 },
           'france': { 'capital':'paris', 'population':66.03 },
           'germany': { 'capital':'berlin', 'population':80.62 },
           'norway': { 'capital':'oslo', 'population':5.084 } }


# Print out the capital of France

print(europe['france']['capital'])

# Create sub-dictionary data

data={'capital':'rome','population':59.83}

# Add data to europe under key 'italy'

europe['italy']=data

# Print europe

print(europe)

#### If-else

area = 10.0
if(area < 9) :
    print("small")
elif(area < 12) :
    print("medium")
else :
    print("large")
    
    
###### Filtering pandas dataframe

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Convert code to a one-liner
dr = cars['drives_right']
sel = cars[cars['drives_right']==True]

# Print sel
print(sel)


######## while loop


x = 1
while x < 4 :
    print(x)
    x = x + 1


############ for loop
    
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Code the for loop
for tt in areas : 
    print(tt)
    
########### for loop with access to index
    
fam = [1.73, 1.68, 1.71, 1.89]
for index, height in enumerate(fam) :
    print("index " + str(index) + ": " + str(height))

############# looping over dictionary
    
    
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'bonn', 
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'australia':'vienna' }
          
# Iterate over europe

for kk,vv in europe.items() :  ##europe.keys() for keys and europe.values() for values
    print("the capital of "+kk+" is "+vv)


############### Iterating over rows of pandas dataframe

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Iterate over rows of cars

for l,r in cars.iterrows() : 
    print(l) ; print(r)
    
    
############################# printing values of one column
    
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Adapt for loop 
for lab, row in cars.iterrows() :
    print(lab+": "+str(row['cars_per_cap']))
    
###### Using apply for vectorized operations instead of looping over each record

for lab, row in brics.iterrows() :
    brics.loc[lab, "name_length"] = len(row["country"])

brics["name_length"] = brics["country"].apply(len)

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Use .apply(str.upper)

cars["COUNTRY"]=cars["country"].apply(str.upper)


############## FUNCTIONS

def square():
    new_value = 4 ** 2
    return new_value

########
    
# Define shout with the parameter, word
def shout(word):
    """Return a string with three exclamation marks"""
    # Concatenate the strings: shout_word
    
    shout_word=word+'!!!'

    # Replace print with return
    return shout_word


###### Tuples - immutable objects
    
# Unpack nums into num1, num2, and num3

num1,num2,num3=nums
# Construct even_nums

even_nums=(2,num2,num3)
print(even_nums)

############# Function with multiple arguments and return values


# Define shout_all with parameters word1 and word2
def shout_all(word1,word2):
    
    # Concatenate word1 with '!!!': shout1
    
    shout1=word1+'!!!'
    
    # Concatenate word2 with '!!!': shout2
    
    shout2=word2+'!!!'
    
    # Construct a tuple with shout1 and shout2: shout_words
    
    shout_words=(shout1,shout2)

    # Return shout_words
    return shout_words

# Pass 'congratulations' and 'you' to shout_all(): yell1, yell2

yell1,yell2=shout_all('congratulations','you')

# Print yell1 and yell2
print(yell1)
print(yell2)

############################ Another function

# Import pandas

import pandas as pd

# Import Twitter data as DataFrame: df
df = pd.read_csv('tweets.csv')

# Initialize an empty dictionary: langs_count

langs_count = {}

# Extract column from DataFrame: col
col = df['lang']

# Iterate over lang column in DataFrame
for entry in col:

    # If the language is in langs_count, add 1
    if entry in langs_count.keys():
        langs_count[entry]=langs_count[entry]+1
    # Else add the language to langs_count, set the value to 1
    else:
        langs_count[entry]=1

# Print the populated dictionary
print(langs_count)


############# Returning function with functions

def echo(n):
    """Return the inner_echo function."""

    # Define inner_echo
    def inner_echo(word1):
        """Concatenate n copies of word1."""
        echo_word = word1 * n
        return echo_word

    # Return inner_echo
    return(inner_echo) 

# Call echo: twice
twice = echo(2)

# Call echo: thrice

thrice=echo(3)

# Call twice() and thrice() then print
print(twice('hello'), thrice('hello'))


############## Function which takes arbitrary number of string arguments

# Define gibberish
def gibberish(*args):
    """Concatenate strings in *args together."""

    # Initialize an empty string: hodgepodge
    
    hodgepodge=str()

    # Concatenate the strings in args
    for word in args:
        hodgepodge += word

    # Return hodgepodge
    return hodgepodge

# Call gibberish() with one string: one_word
one_word = gibberish('luke')

# Call gibberish() with five strings: many_words
many_words = gibberish("luke", "leia", "han", "obi", "darth")

# Print one_word and many_words
print(one_word)
print(many_words)


############# Variable number of keyword arguments ( as kwargs is a dictionary)

# Define report_status
def report_status(**kwargs):
    """Print out the status of a movie character."""

    print("\nBEGIN: REPORT\n")

    # Iterate over the key-value pairs of kwargs
    for kk,vv in kwargs.items():
        # Print out the keys and values, separated by a colon ':'
        print(kk + ": " + vv)


################## Lambda function
        
# Define echo_word as a lambda function: echo_word
echo_word = (lambda word1,echo : word1*echo)

# Call echo_word: result
result = echo_word('hey',5)

# Print result
print(result)


######### Using map to apply lambda function to each element of list

# Create a list of strings: spells
spells = ["protego", "accio", "expecto patronum", "legilimens"]

# Use map() to apply a lambda function over spells: shout_spells
shout_spells = map(lambda item : item+'!!!', spells)

# Convert shout_spells to a list: shout_spells_list
shout_spells_list=list(shout_spells)

# Convert shout_spells into a list and print it
print(shout_spells_list)

################# Using filter to select records that satisfy condition evaluated on lambda function

# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Use filter() to apply a lambda function over fellowship: result
result = filter(lambda member : len(member)>6, fellowship)

# Convert result to a list: result_list
result_list=list(result)

# Convert result into a list and print it
print(result_list)


########## Using try except blocks

# Define shout_echo
def shout_echo(word1, echo=1):
    """Concatenate echo copies of word1 and three
    exclamation marks at the end of the string."""

    # Initialize empty strings: echo_word, shout_words
    
    echo_word=str()
    shout_words=str()

    # Add exception handling with try-except
    try:
        # Concatenate echo copies of word1 using *: echo_word
        echo_word = word1*echo

        # Concatenate '!!!' to echo_word: shout_words
        shout_words = echo_word+'!!!'
    except:
        # Print error message
        print("word1 must be a string and echo must be an integer.")

    # Return shout_words
    return shout_words

# Call shout_echo
shout_echo("particle", echo="accelerator")


########################## Raising value error

# Define shout_echo
def shout_echo(word1, echo=1):
    """Concatenate echo copies of word1 and three
    exclamation marks at the end of the string."""

    # Raise an error with raise
    if echo<0:
        raise ValueError('echo must be greater than 0')

    # Concatenate echo copies of word1 using *: echo_word
    echo_word = word1 * echo

    # Concatenate '!!!' to echo_word: shout_word
    shout_word = echo_word + '!!!'

    # Return shout_word
    return shout_word

# Call shout_echo
shout_echo("particle", echo=-2)

############## Using iterators 

# Create a list of strings: mutants
mutants = ['charles xavier', 
            'bobby drake', 
            'kurt wagner', 
            'max eisenhardt', 
            'kitty pride']

# Create a list of tuples: mutant_list
mutant_list = list(enumerate(mutants))

# Print the list of tuples
print(mutant_list)

# Unpack and print the tuple pairs
for index1,value1 in enumerate(mutants):
    print(index1, value1)

# Change the start index
for index2,value2 in enumerate(mutants,start=1):
    print(index2, value2)


############### Using Zip 
    
# Create a list of tuples: mutant_data
mutant_data = list(zip(mutants,aliases,powers))

# Print the list of tuples
print(mutant_data)

# Create a zip object using the three lists: mutant_zip
mutant_zip = zip(mutants,aliases,powers)

# Print the zip object
print(mutant_zip)

# Unpack the zip object and print the tuple values
for value1,value2,value3 in mutant_zip:
    print(value1, value2, value3)
    

############# Unpacking zip using *
    
# Create a zip object from mutants and powers: z1
z1 = zip(mutants,powers)

# Print the tuples in z1 by unpacking with *
print(*z1)

# Re-create a zip object from mutants and powers: z1
z1 = zip(mutants,powers)

# 'Unzip' the tuples in z1 by unpacking with * and zip(): result1, result2
result1, result2 = zip(*z1)
    
######### Using iterators to read large files 

# Initialize an empty dictionary: counts_dict

counts_dict={}

# Iterate over the file chunk by chunk
for chunk in pd.read_csv('tweets.csv',chunksize=10):

    # Iterate over the column in DataFrame
    for entry in chunk['lang']:
        if entry in counts_dict.keys():
            counts_dict[entry] += 1
        else:
            counts_dict[entry] = 1

# Print the populated dictionary
print(counts_dict)


########## List Comprehensions

squares = [i**2 for i in range(10)]

# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

####### List with filtered elements 

# Create list comprehension: new_fellowship
new_fellowship = [member for member in fellowship if len(member)>6]

# Print the new list
print(new_fellowship)

####### list with same number of elements as original list

# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Create list comprehension: new_fellowship
new_fellowship = [member if len(member)>6  else '' for member in fellowship]

# Print the new list
print(new_fellowship)

################## Dict comprehension

# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Create dict comprehension: new_fellowship
new_fellowship = {member:len(member) for member in fellowship}

# Print the new list
print(new_fellowship)


########### Reading large files 

# Define read_large_file()
def read_large_file(file_object):
    """A generator function to read a large file lazily."""

    # Loop indefinitely until the end of the file
    while True:

        # Read a line from the file: data
        data = file_object.readline()

        # Break if this is the end of the file
        if not data:
            break

        # Yield the line of data
        yield data
        
# Open a connection to the file
with open('world_dev_ind.csv') as file:

    # Create a generator object for the file: gen_file
    gen_file = read_large_file(file)

    # Print the first three lines of the file
    print(next(gen_file))
    print(next(gen_file))
    print(next(gen_file))

#####  Reading file with large size
    
import pandas as pd

# Initialize reader object: df_reader
df_reader = pd.read_csv('ind_pop.csv',chunksize=10)

# Print two chunks
print(next(df_reader))
print(next(df_reader))


######################

# Define plot_pop()
def plot_pop(filename,country_code):

    # Initialize reader object: urb_pop_reader
    urb_pop_reader = pd.read_csv(filename, chunksize=1000)

    # Initialize empty DataFrame: data
    data = pd.DataFrame()
    
    # Iterate over each DataFrame chunk
    for df_urb_pop in urb_pop_reader:
        # Check out specific country: df_pop_ceb
        df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == country_code]

        # Zip DataFrame columns of interest: pops
        pops = zip(df_pop_ceb['Total Population'],
                    df_pop_ceb['Urban population (% of total)'])

        # Turn zip object into list: pops_list
        pops_list = list(pops)

        # Use list comprehension to create new DataFrame column 'Total Urban Population'
        df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1]) for tup in pops_list]
    
        # Append DataFrame chunk to data: data
        data = data.append(df_pop_ceb)

    # Plot urban population data
    data.plot(kind='scatter', x='Year', y='Total Urban Population')
    plt.show()

# Set the filename: fn
fn = 'ind_pop_data.csv'

# Call plot_pop for country code 'CEB'

plot_pop(fn,'CEB')

# Call plot_pop for country code 'ARB'

plot_pop(fn,'ARB')
