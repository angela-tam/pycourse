import pandas
import os
import numpy as np
from copy import deepcopy
import scipy.stats as stat


## Part A -- Harmonize case  (Straightforward)
# Many columns recieve string input. However, some users have chosen to use capital letters
# while others have not. 
# See for example df["What color is the shirt/dress/upper-body-clothing you're wearing right now, if any?"].unique()
# Create a function that will make all string responses in a given column lowercase

def harmonize_case(col):
    ''' Given a pandas series or list, will change all rows with string values to lowercase
    Outputs a sequence in the same format as input'''
    col = deepcopy(col)
    for i,val in enumerate(col):
        if type(val) == str:
            col[i] = val.lower()
        else:
            print('value %s at row %s is not a str, but a %s. Skipping...'%(val,i,type(val)))
            continue
    return col

## Part B -- Character count   (Straightforward)
# For the column 'Fill this text box with gibberish by mashing random keyboard keys', users 
# have entered a random number of string characters. We can count the number of characters
# the users typed and use it as a measure of "aggression" or "inner-stress". Write a function
# that will count the number of characters for each value in a columns

def character_count(col):
    '''given a pandas series or list, will output a new series or list with the character
    count of string values'''
    col = deepcopy(col)
    for i,val in enumerate(col):
        if type(val) == str:
            col[i] = len(val)
        else:
            print('value %s at row %s is not a str, but a %s. Skipping...'%(val,i,type(val)))
    
    return col

    ## Part C -- Handle NaNs    (Just a bit bumpy)
# Many columns have NaNs, and depending on what program you use to analyze data, you may need
# do address them. Write a function that handles the NaNs in a column. There should be an 
# argument where the user can define what they wish to do with the NaNs. They can either
# remove all rows with a NaN from the column, or they can code the NaNs as something else.
# You will obviously need a third argument where the user inputs what they want to code the
# NaN as (example input could be 999 or np.nan or 'missing')

def handle_NaNs(col,mode='remove',code=None):
    """takes a pandas Series or list as input. If mode is set to remove, will physically remove
    all values (rows) with NaNs in them. If mode is set to "encode", will change NaNs to code.
    Returns a pandas Series
    
    Note: remove will delete rows, mean dimensions of may be different from col
     """
    
    if mode != 'remove' and mode != 'encode':
        raise ValueError('argument mode must be set to remove or encode')
    
    if mode == 'encode' and code == None:
        raise ReferenceError('if mode is set encode, argument code must be passed')
    
    col = pandas.Series(deepcopy(col))
    if mode == 'remove':
        o_len = len(col)
        col.dropna(inplace=True)
        n_len = len(col)
        print('%s rows removed'%(o_len - n_len))
    
    elif mode == 'encode':
        count = 0
        for i,val in enumerate(col):
            if not pandas.notnull(val):
                col[i] = code
                count = count+1
        print('%s rows changed'%(count))
    
    return col

    ## Part D -- Encode    (Just a bit bumpy)
# Write a function that will take a dictionary as input. The dictionary should have column
# values as keys and new (encoded) values as the corresponding dictionary values. For example,
# input could be {Yes: 1, No: 0}. The function should apply the code to the column.
# If the dictionary argument is set to None, the function should automatically encode unique
# values with sequential integers.

def encode(col, code_dict=None):
    '''col is a pandas Series or list as input. Code_dict is a dict such that keys are entries
    that exist in col, and values are new entries to replace the key, thus "encoding" the column.
    If code_dict = None, will automatically encode unique values with sequential integers.
    Returns pandas Series.'''
    
    col = pandas.Series(deepcopy(col))
    net = []
    if not code_dict:
        code_dict = {}
        for i,val in enumerate(col.unique()):
            if pandas.notnull(val):
                code_dict.update({val: i+1})
            else: 
                code_dict.update({val: 0})
    else:
        if type(code_dict) != dict:
            raise ValueError('code_dict must be a dictionary object')
        
    for i,val in enumerate(col):
        if val in code_dict.keys():
            col[i] = code_dict[val]
        else:
            net.append(val)
    if len(net) > 0:
        print('the following values were not assigned labels:')
        for miss in set(net):
            print(miss)
    
    return col
            

## Part E -- Binary encode      (Tricky)
# Many columns have users choose from several different choices. See for example:
# df['If you had to choose one, what would be your favorite type of beer?'].unique()
# Create a function that will set one of those values to 1, and the rest to 0. The
# value to be set as 1 should be specified by the user. I could use this for example
# if I wanted to have a new variable called "IPA Drinkers".

# The user can also input a list or Series, in which case all values within the list 
# are changed to 1, and everything else to 0. The function should call a specific error 
# if the user inputs a value that is not an existing value in the column. 

# BONUS: Add extra feature to allow the user to choose whether or not to leave NaNs as NaNs

def binarize(col,binval,ignore_nan=False):
    ''' Given input of a list or pandas Series, will change all entries of binval to 1
    and all other entries to 0. If binval is a list, will change all entries matching 
    items in binval to 1, and all other entries as 0. If ignore_nan is True, NaNs will
    be ignored and therefore set to 0. Otherwise, NaNs will remain as NaNs in the final
    output. Outputs a pandas series'''
    
    col = deepcopy(pandas.Series(col))
    u = col.unique()
    
    if not ignore_nan:
        nanz= []
        for i,val in enumerate(col):
            if not pandas.notnull(val):
                nanz.append(i)
                
    if type(binval) != str: # since str is subscriptable
        try:
            binval[0] # will fail if binval is not a subscriptable object
            # Deal with typos in binval
            for x in binval:
                if x not in u:
                    raise ValueError('%s was entered into binval, but %s is not a valid entry in this column'%(x,x))
            # encode using list of values
            for i,val in enumerate(col):
                if val in binval:
                    col[i] = 1
                else:
                    col[i] = 0
        except:
            # encode using single, non-str value
            if binval not in u:
                raise ValueError('%s was entered into binval, but %s is not a valid entry in this column'%(x,x))
            col[col!=binval] = 0
            col[col==binval] = 1
    else:
        # encode using str value
        if binval not in u:
            raise ValueError('%s was entered into binval, but %s is not a valid entry in this column'%(x,x))
        col[col!=binval] = 0
        col[col==binval] = 1
    
    # set NaNs values back to NaNs
    if not ignore_nan:
        for ind in nanz:
            col[ind] = np.nan
    
    return col
                
    
    ## Part F -- Purify int/floats        (Tricky)
# Many columns have examples where users have input string when they were suppose to input 
# an integer or float. See for example: 
# df["How many romantic relationships have you been in that have lasted at least 6 months"].unique()
# Write a function that handles string inputs when they are supposed to be floats or ints. The function
# should have two modes, evaluate and apply. If evaluate, the function will return all values (as well
# as their index) that are not floats. This will let you evaluate whether you can fix these values
# (either manually or with your encode function). If apply, the function will remove all non int/float
# values and replace them with NaNs

# If mode is set to apply, the function should return the modified column. If mode is set to evaluate,
# the function should return the indices of the non-float/int values

def purify_numbers(col,mode='evaluate'):
    '''takes pandas Series or list. If mode is set to 'evaluate', will report the index of values
    that are not number classes, and will return those indices. If set to 'apply', will return
    col with non-floats/ints converted to NaNs'''
    
    if mode != 'evaluate' and mode != 'apply':
        raise Warning('mode must be set to evaluate or apply,',
                     'you set mode as %s. Running script in evaluate mode...'%(mode))
        mode = 'evaluate'
    
    if mode == 'evaluate':
        fail_idx = []
    
    col = deepcopy(col)
    for i,val in enumerate(col):
        try:
            float(val)
        except ValueError:
            if mode == 'evaluate':
                print('value %s at index %s is not a number class'%(val,i))
                fail_idx.append(i)
            else:
                col[i] = np.nan
    
    if mode == 'evaluate':
        if len(fail_idx) == 0:
            print('all indices are numbers')
        else:
            return fail_idx
    else:
        return col



## Part G -- Handle lists           (Challenging)
# This one is going to be a bit different from the others. Instead of using entire columns as
# inputs/outputs, you will be using values, specifically, sequence or list values

# Several columns have sequences as values. See for example:
# df['Please select all of the following for which you have some experience']
# or
# df['Type 5 random (English) words'] 

# Write a function that handles these lists. First, it should take a separator, where
# the user inputs what string should separate items in these lists (for example ';').
# If the separator is None, it should automatically try several types of separators and
# evaluate each separator by whether the length of the separated item matches a user
# input value. 

# For example, the user input value for df['Type 5 random (English) words']
# would be 5. The function would test several separators for each list (' ' or ', ' or ',')
# until it finds a solution where the value is separated into a list of 5 items.

# If it cannot find a solution that is equal to the target, it should find the separator
# that has the closest number of separations to the target

# If no input value is set, it should use the separator that results in the
# greatest number of splits.

# Also, in all cases, the function should return which separator was used. 

# Additionally, you should add an optional argument where the user can choose to return only
# the length of the list in addition. (Sometimes we may only wish to get the length, consider
# df['Do you strongly dislike the taste or texture of any of the following things?'] where
# length of the sequence could be an measurement of whether someone is a picky eater!)

# Finally, the user can input whether they want the value to be returned a separated
# list, or as a string where the values are separated by a user defined string


# IM GOING TO HANDLE THIS WITH SUBFUNCTIONS, TO MAKE THE OVERALL READABILITY OF THE CODE 
# BETTER

def handle_list(val,sep=None,target=None,out_type='list',out_sep=', ',len_only=False,verbose=True):
    '''input is a string that needs to be split. If sep is specified, will try toseparate 
    string by specified sep. If sep = None, function will use multiple separators and 
    choose the one that separates the string to the length specified in target. If no
    string succeeds, the separator that results to the closest length to target is chosen.
    If no target is specified, function will choose the separator that returns the greatest
    number of separations. 
    Output changes depending on arguments. If out_type set to list, function will output
    a list. If set to str, function will output a str separated by out_sep. Finally, if
    len_only is True, function will return only the length of the separated list.
    If verbose set to False, script will not print output'''
    
    if out_type != 'list' and out_type != 'str':
        raise ValueError("out_type must be set to 'list' or 'str'")
    
    # Handle sep if sep is not a str
    if sep:
        if type(sep) != str:
            raise TypeError('sep must must a str object, you entered a %s object'%(type(sep)))
            
    # make sure target is specified and spawn sep_list
    else:
        if target: 
            if type(target) != int:
                raise ValueError('if sep is not specified, target must be set to an int')
        sep_list = [', ',',','; ',';',' ']
        # There's a more elegant way of doing this with regexp, but its a bit advanced for
        # where we are in the course, so I'll just do it this way.
    
    val = deepcopy(val)
    
    # Perform separation in situations where sep is specified
    if sep:
        try:
            val_list = val.split(sep)
            sep_used = sep
        except:
            # if separator doesn't work
            if verbose:
                print('could not separate %s, of the %s class, using sep %s'%(val,type(val),sep))
            sep_used = np.nan
            nval = val
        else:
            if len_only:
                nval = len(val_list)
            else:
                nval = construct_output(val_list,out_type,out_sep)
    
    # Perform iterative search through seps to find best separator      
    else:
        if target:
            nval,sep_used = find_best_sep_target(val,sep_list,target,out_type,out_sep,len_only,verbose)
        else:
            nval,sep_used = find_best_sep_notarget(val,sep_list,out_type,out_sep,len_only,verbose)
            
        if verbose:
            print('using %s as sep'%(sep_used))   
        
    return nval,sep_used


    
def find_best_sep_target(val,sep_list,target,out_type,out_sep,len_only,verbose):
    result = False
    x = 0
    # find separator that matches target
    while not result and x < len(sep_list):
        sep = sep_list[x]
        try:
            val_list = val.split(sep)
            if len(val_list) == target:
                result = True
            else:
                x = x+1
        except:
            x = x+1
                
    if not result:
        # if no separator found to match target
        if verbose:
            print('WARNING: could not find a valid parser for val %s'%(val))
            print('Instead, searching for sep with the closest match to target...')
        nval,sep_used = find_best_sep_notarget(val,sep_list,out_type,out_sep,len_only,target)
    else:
        if len_only:
            nval = len(val_list)
            sep_used = np.nan
        else:
            sep_used = sep_list[x]
            nval = construct_output(val_list,out_type,out_sep)
            
    return nval,sep_used

def find_best_sep_notarget(val,sep_list,out_type,out_sep,len_only,verbose,target=None):
    
    results = []
    
    for sep in sep_list:
        try:
            val_list = val.split(sep)
            results.append(len(val_list))
        except:
            results.append(0)
    
    if target:
        diff = []
        for r in results:
            adiff = abs(r - target)
            diff.append(adiff)
        best_ind = np.argmin(diff)
     
    else:
        best_ind = np.argmax(results)
    
    sep_used = sep_list[best_ind]
    
    try:
        val_list = val.split(sep_used)
    except:
        if verbose:
            print('no possible separator found for %s, of the %s class'%(val,type(val)))
        nval,sep_used = val,np.nan
    else:
        if len_only:
            nval = len(val_list)
        else:
            nval = construct_output(val_list,out_type,out_sep)
    
    return nval,sep_used
    
    
def construct_output(in_list,out_type,out_sep):
    if out_type == 'list':
        nval = in_list 
    else:
        nval=''
        for entry in in_list:
            nval = nval+entry+out_sep
    
    return nval


## Part H -- Simplify        (Challenging)
# A lot of people put smartass answers into some of the open response questions, or they gave a
# bit more detail than is useful. This is of course my fault for not being more specific with the
# questions and controlling the output. Write a function that will detect whether a given string
# value has another existing string value (from the same column) within it. For example, if one
# value say "green and white", and "white" is another existing value in the column, it will replace
# "green and white" with "white". There should be a "tie" argument where the user can choose between
# 'alert' or 'remove' modes. If 'alert', if more than one value fits, (for example if both "green" and 
# "white" already exist in the column), it should print the index, both "matching" values, and
# an instruction for the user to change it manually. If 'remove', it should replace the value with
# a np.nan. Finally, as with other functions, this function should have an evaluate vs apply argument.

def simplify(col, tie='alert', mode='evaluate',in_words=None):
    '''
    input should be a pandas Series or list. Function will search through entries in series and''
    will make suggestions as to what changes can be made to better harmonize list entries. Function
    works by checking if existing entries can be found within other entries. In case more than one
    suggestion can be made, if tie is set to alert, the user will merely be alerted of the tie and
    no changes will take place. If set to remove, the tied value will be set to NaN. 
    Output depends on mode. If mode set to evaluate, function will print suggestions and the will 
    return the index of the suggested items. If mode set to apply, function will make the
    suggestions and return a pandas Series with suggestions made.
    In_words can be a list of words that the function uses as a reference for simplification 
    '''
    
    if tie != 'alert' and tie != 'remove':
        raise Warning('tie must be set to alert or remove. Users specified %s,'%(tie),
                     'moving forward with tie set to alert')
    
    if mode != 'evaluate' and mode != 'apply':
        raise Warning('mode must be set to evaluate or apply,',
                     'you set mode as %s. Running script in evaluate mode...'%(mode))
        mode = 'evaluate'
    
    #col = deepcopy(pandas.Series(col))
    col = deepcopy(col)
    
    # initialize words
    if in_words:
        words=in_words
    else:
        words = list(col.unique())
        for x,word in enumerate(words):
            if type(word) != str:
                words.remove(words[x])
    
    if mode == 'evaluate':
        fail_inds = []
    
    for i,val in enumerate(col):
        
        if type(val) != str:
            continue           # skip non-str entries
        
        # find partial matches
        suggestions = []
        for word in words:
            if not word == val:
                if word in val:
                    suggestions.append(word)
        
        
        if len(suggestions) == 0: # if none found, keep moving
            continue
        elif len(suggestions) == 1: # if one found, suggest it
            if mode == 'evaluate':
                print('suggestion found: perhaps replace %s at index %s with %s' %(val,i,suggestions[0]))
                fail_inds.append(i)
            else:
                print('changing %s to %s'%(val,suggestions[0]))
                col[i] = suggestions[0]
                
        else: # If there's a tie
            if tie == 'alert':
                print('the following suggestions were made or %s at index %s'%(val,i))
                for sug in suggestions:
                    print(sug)
                print('please select the best option and change manually.',
                     'or rerun with tie = remove to set ties to NaN')
                fail_inds.append(i)
            else:
                print('tie found for %s.. setting to np.nan'%(val))
                col[i] = np.nan
    
    if mode == 'evaluate':
        return fail_inds
    else:
        return col
                

## Part I -- Time Code       (Just a bit bumpy)
# Two columns have date/time information in them, df.ix['Timestamp'] and df['What time is it right now?']
# Write a function that will take these times and encode whether the survey was taken in morning,
# afternoon, evening or night.
# You may need to read about how to deal with datetime data: 
#http://pandas.pydata.org/pandas-docs/stable/timeseries.html
# or --> https://dateutil.readthedocs.io/en/stable/

from dateutil import parser

def time_of_day(col):
    '''function will take a pandas Series or list of datetimes. Function will return a new list or
    Series indicating if the datetime corresponds to morning, afternoon, evening or night'''

    col = deepcopy(col)
    for i,val in enumerate(col): 
        try:
            time = parser.parse(val)
        except:
            raise TypeError('could not parse %s at index %s, please make sure your inputs are datetimes'%(val,i))
        if time.hour in range(0,6):
            col[i] = 'night'
        elif time.hour in range(6,12):
            col[i] = 'morning'
        elif time.hour in range(12,18):
            col[i] = 'afternoon'
        else:
            col[i] = 'evening'
    
    return col


## Part J -- Uniqueness   (Challenging)
# There are two columns that involve users to input a list of five words:
# df['Name the first five animals you can think of'] and df['Type 5 random (English) words']
# Think of a function that calculates how unique each word is that the user gave compared
# to all words given by all users, and assign a single score of uniqueness for each row based on
# the words given. How you do the scoring is up to you

def uniqueness(col,seq=False,seq_sep=None,seq_target=None,score_bins=None):
    
    col = deepcopy(pandas.Series(col))
    
    # adjust default scoring parameters...
    if score_bins:
        if seq:
            score_bins = 10
        else:
            score_bins = 5
    
    # First iteration: get word list
    if not seq:
        words = {}
        for word in col.unique():
            if type(word) == str and len(word) > 1:
                if word in words.keys():
                    words.update({word: words[word]+1})
                else:
                    words.update({word: 1})
    else:
        words = {}
        for val in col:
            val_list,jnk = handle_list(val,seq_sep,seq_target,verbose=False)
            if type(val_list) != list:
                continue
            for word in val_list:
                if len(word) < 2:
                    continue
                if word in words.keys():
                    words.update({word: words[word]+1})
                else:
                    words.update({word: 1})
    
    # for scoring
    try:
        jnk,values = np.histogram(list(words.values()),bins=score_bins)
    except:
        raise ValueError('score_bins is set too high. Reduce it!')
    print(values)
    
    # Second iteration, calculate score
    if not seq:
        for i,val in enumerate(col):
            if val in words.keys():
                score = calc_word_score(words[val],values)
                col[i] = score
            else:
                print('could not calculate score for index %s with value %s'%(i,val))
                col[i] = np.nan
    else:
        for i,val in enumerate(col):
            val_list,jnk = handle_list(val,seq_sep,seq_target,verbose=False)
            
            if type(val_list) != list:
                print('could not calculate score for index %s with value %s'%(i,val))
                col[i] = np.nan
                
            sub_score = []
            if seq_target:
                if len(val_list) > seq_target:
                    print('index %s had more than %s words,'%(i,seq_target), 
                            'removing final word %s'%(val_list[-1]))
                    val_list = val_list[:seq_target]
            for word in val_list:
                if word not in words.keys():
                    continue
                score = calc_word_score(words[word],values)
                sub_score.append(score)
            
            col[i] = sum(sub_score)
        
        return col,list(words.keys())
                
    
def calc_word_score(freq,values):
        
    binned = False
    count = 0
    while binned == False:
        if freq <= values[count]:
            binned = True
        else:
            count = count + 1
        
    score = len(values) - count

    return score

def data_miner(xcols,ycols,data,return_ps = False, verbose = True, print_errors=False):
	'''quick and extremely dirty method for finding significant 
	associations in data. Not recommended for real analyses!!! 
	For each column in xcols, searches for associations between 
	each column in ycols. Will automatically figure out whether 
	to use pearsons correlation, anova, ttest or chi-square. Data
	refers to the dataframe containing the columns.
	Function will print sigificant associations.
	If return_ps is True, will return a dataframe of pvalues for
	FDR correction purposes
	If print errors is True, will also print relationships for 
	which the script did not know what to do.
	If verbose set to False, will not print significant
	relationships. However, if verbose AND return_ps are both
	false, script won't output anything.

	** WARNING ** Do not take any of these associations at face
	value! This script is just meant to highlight basic 
	parametric associations in a large messy dataset!
	''' 

	sigcols = []
	done = []
	ps = {}
	for k,icol in enumerate(xcols):
	    print('working on %s'%(icol))
	    i = data[icol]
	    for l,jcol in enumerate(ycols):
	        if k == l:
	            continue
	        j = data[jcol]
	        if (jcol,icol) in done:
	            continue
	        else:
	            done.append((icol,jcol))
	        if i.dtype == 'float64' and j.dtype == 'float64':
	            if len(i.unique()) > 4 and len(j.unique()) > 4:
	                coef,p = stat.pearsonr(i.tolist(),j.tolist())
	                test = 'r'
	                ps.update({(jcol,icol): p})
	            elif len(i.unique()) > 4 or len(j.unique()) > 4:
	                coef,p = j_anova_ttest(icol,jcol,data)
	                test = 'tf'
	                ps.update({(jcol,icol): p})
	            else:
	                coef,p = j_chi(i,j)
	                test = 'chi'
	                ps.update({(jcol,icol): p})
	        elif i.dtype == 'O' and j.dtype == 'O':
	            coef,p = j_chi(i,j)
	            test = 'chi'
	            ps.update({(jcol,icol): p})
	        else:
	            coef,p = j_anova_ttest(icol,jcol,data)
	            test = 'tf'
	            ps.update({(jcol,icol): p})
	        if verbose:
		        if p < 0.001:
		            print(('*'*3),icol,'   vs  ',jcol,test,coef,p,('*'*3))
		            sigcols.append((icol,jcol))
		        elif p < 0.01:
		            print(('*'*2),icol,'   vs  ',jcol,test,coef,p,('*'*2))
		            sigcols.append((icol,jcol))
		        elif p < 0.05:
		            print('*',icol,'   vs  ',jcol,coef,test,p,'*')
		            sigcols.append((icol,jcol))
		        else:
		            if print_errors:
		            	print('______ %s    vs    %s was not assessed _____'%(icol,jcol))
	        
	if return_ps:
		pdf =  pandas.DataFrame(index=ps.keys())
		for k,v in ps.items():
			pdf.ix[k,'p'] = v
		return pdf 
            
                


def j_anova_ttest(x,y,data):
    '''quick and extremely dirty script to a) figure out whether
    to perform a t-test or anova, b) figure out which variables 
    (if any) are categorical, c) organize data for a ttest or 
    ANOVA and d) run the test
    ''' 

    # figure out which is the grouping variable
    
    try:
        data[x].astype(float)
    except:
        x,y = y,x
    else:
        try:
            data[y].astype(float)
            if len(data[x].unique()) < len(data[y].unique()):
                x,y = y,x
        except:
            pass

    gps = data[y].unique()
    res = []
    for g in gps:
        if pandas.notnull(g):
            g_dat = np.array(data[data[y] == g][x].dropna())
            if len(g_dat) > 3:
                res.append(g_dat)

    if len(res) == 2:
        tf,p = stat.ttest_ind(res[0],res[1])
    elif len(res) == 3:
        tf,p = stat.f_oneway(res[0],res[1],res[2])
    elif len(res) == 4:
        tf,p = stat.f_oneway(res[0],res[1],res[2],res[3])
    elif len(res) == 5:
        tf,p = stat.f_oneway(res[0],res[1],res[2],res[3],res[4])
    elif len(res) == 6:
        tf,p = stat.f_oneway(res[0],res[1],res[2],res[3],res[4],res[5])
    elif len(res) == 7:
        tf,p = stat.f_oneway(res[0],res[1],res[2],res[3],res[4],res[5],res[6])
    elif len(res) == 8:
        tf,p = stat.f_oneway(res[0],res[1],res[2],res[3],res[4],res[5],res[6],res[7])
    else:
        tf,p = 0,1
    
    return tf,p

def j_chi(x,y):
    ''' a quick and VERY dirty script to organize data for a chi
    square test and run the test
    '''
    jnk1 = encode(x.dropna())
    jnk2 = encode(y.dropna())
    try:
        x_in = jnk1.astype(float).tolist()
        x_out = jnk2.astype(float).tolist()
    except:
        chi,p = 0,1
    else:
        try:
            chi,p = stat.chisquare(jnk1.tolist(),jnk2.tolist())
        except:
            chi,p = 0,1
    
    return chi,p

#def multivar_miner(tstcols,data,only_binary_ivs=True,verbose=False):

#    rdf,dvs,ivs,col_dir = prep_frame(tstcols,data)
#    if not only_binary_ivs:
#        ivs = rdf.columns
#
#    if not verbose:
#        associations = []
#
#    done = []
#    for i,dv_col in enumerate(dvs):
#        print('working on DV %s of %s'%(i,len(dvs)))
#        dv = data[dv_col]
#        for iva_col in ivs:
#            if iva_col == dv_col:
#                continue
#            if (iva_col,dv_col) in done:
#                continue
#            else:
#                done.append((dv_col,iva_col))
#            iva = data[iva_col]
#            for ivb_col in ivs:
#                if ivb_col == iva_col or ivb_col == dv_col:
#                    continue
#                if (dv_col,ivb_col,iva_col) in done:
#                    continue
#                else:
#                    done.append((dv_col,iva_col,ivb_col))
#                ivb = data[ivb_col]
#                inter,changer = multivariate_test(dv,iva,ivb,data)
#                if inter:
#                    assos = '>>>>Interaction: %s      vs     %s    on    %s <<<<<<<<<'%(
#                            col_dir[iva_col], col_dir[ivb_col], col_dir[dv_col])
#                    if verbose:
#                        print('\n',assoc)
#                    else:
#                        associations.append(assos)
#                if changer:
#                    assos = '*****Change: %s     vs     %s     on     %s'%(
#                             col_dir[iva_col],col_dir[ivb_col],col_dir[dv_col])
#                    if verbose:
#                        print('\n',assoc)
#                    else:
#                        associations.append(assos)
#	if not verbose:
#		return(associations)
#
#
#def prep_frame(tstcols,data):
#    
#    # make new frame of only tstcols
#    rdf = deepcopy(data[tstcols])
#    
#    # fix headers
#    ncols = []
#    col_dir = {}
#    for col in tstcols:
#        ncols.append(col[:14])
#        col_dir.update({col[:14]:col})
#    rdf.columns = ncols
#    
#    # find feasable dependent variables
#    dvs = []
#    for col in rdf.columns:
#        if rdf[col].dtype == 'float64' and len(rdf[col].unique()) > 4:
#            dvs.append(col)
#            
#    # find feasible independent variable (in case user specifies)
#    ivs = []
#    for col in rdf.columns:
#        if rdf[col].dtype == 'O' and len(rdf[col].unique()) < 4:
#            ivs.append(col)
#
#    # encode categorical variables
#    for col in rdf.columns:
#        if rdf[col].dtype == 'O':
#            rdf[col] = us.encode(rdf[col]).astype(float)
#    
#    
#    return rdf,dvs,ivs,col_dir
#
#def multivariate_test(dv,iva,ivb,data):
#    
#    inter=False
#    changer=False
#    
#    tsta_p = smf.ols('dv ~ iva',data=data).fit().pvalues[1]
#    tstb_p = smf.ols('dv ~ ivb',data=data).fit().pvalues[1]
#    
#    tstmv_ps = smf.ols('dv ~ iva + ivb',data=data).fit().pvalues
#    
#    if tsta_p > 0.05 and tstmv_ps[1] < 0.05:
#        changer=True
#    if tstb_p > 0.05 and tstmv_ps[2] < 0.05:
#        changer = True
#    
#    tstint_ps = smf.ols('dv ~ iva * ivb',data=data).fit().pvalues[-1]
#    if tstint_ps < 0.05:
#        inter = True
#    
#    return inter,changer
#
