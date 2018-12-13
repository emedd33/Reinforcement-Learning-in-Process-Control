# The functions in this file are used to generate datasets for machine-learning problems.

import tensorflow as tf
import numpy as np
import copy
import math
import os  # For starting up tensorboard from inside python
import matplotlib.pyplot as PLT
import scipy.cluster.hierarchy as SCH  # Needed for dendrograms
import numpy.random as NPR

# ****** SESSION HANDLING *******

def gen_initialized_session(dir='probeview'):
    sess = tf.Session()
    sess.probe_stream = viewprep(sess,dir=dir)  # Create a probe stream and attach to the session
    sess.viewdir = dir  # add a second slot, viewdir, to the session
    sess.run(tf.global_variables_initializer())
    return sess

def copy_session(sess1):
    sess2 = tf.Session()
    sess2.probe_stream = sess1.probe_stream
    sess2.probe_stream.reopen()
    sess2.viewdir = sess1.viewdir
    return sess2

def close_session(sess, view=True):
    sess.probe_stream.close()
    sess.close()
    if view: fireup_tensorboard(sess.viewdir)

# Simple evaluator of a TF operator.
def tfeval(operators):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    result = sess.run(operators) # result = a list of output values, one from each operator.
    sess.close()
    return result

# ***** TENSORBOARD SUPPORT ****

# This creates the main data for tensorboard viewing: the graph and variable histories.

def viewprep(session, dir='probeview',flush=120,queue=10):
    clear_tensorflow_log(dir)  # Without this, the directory fills up with unusable files
    return tf.summary.FileWriter(dir,session.graph,flush_secs=flush,max_queue=queue)

# To view probes, the function graph, etc., do this at the command line:
#        tensorboard --logdir=probeview
# Then open a Chrome browser and go to site:  localhost:6006

def fireup_tensorboard(logdir, logwash=True):
    os.system('tensorboard --logdir='+logdir)
    if logwash: clear_tensorflow_log(logdir)

def clear_tensorflow_log(logdir):
    os.system('rm ' + logdir +'/events.out.*')

# ***** GENERATING Simple DATA SETS for MACHINE LEARNING *****

# Generate all bit vectors of a given length (num_bits).
def gen_all_bit_vectors(num_bits):
    def bits(n):
        s = bin(n)[2:]
        return [int(b) for b in '0'*(num_bits - len(s))+s]
    return [bits(i) for i in range(2**num_bits)]

# Convert an integer to a bit vector of length num_bits, with prefix 0's as padding when necessary.
def int_to_bits(i,num_bits):
    s = bin(i)[2:]
    return [int(b) for b in '0' * (num_bits - len(s)) + s]

def all_ints_to_bits(num_bits):
    return [int_to_bits(i) for i in range(2**num_bits)]

# Convert an integer k to a sparse vector in which all bits are "off" except the kth bit.  Note: this
# is zero-based, so the one-hot vector for 0 is 10000..., and for 1 is 010000..

def int_to_one_hot(int,size,off_val=0, on_val=1,floats=False):
    if floats:
        off_val = float(off_val); on_val = float(on_val)
    if int < size:
        v = [off_val] * size
        v[int] = on_val
        return v

def one_hot_to_int(vect,on_val=1): return vect.index(on_val)

# Generate all one-hot vectors of length len
def all_one_hots(len, floats=False):
    return [int_to_one_hot(i,len,floats=floats) for i in range(len)]

# bits = list of 1's and 0's
def bits_to_str(bits): return ''.join(map(str,bits))
def str_to_bits(s): return [int(c) for c in s]

# ****** VECTOR SHIFTING ******
# Shift a vector right (dir=1) or left (dir= -1) and any number of spaces (delta).

def shift_vector(v,dir=1,delta=1):
    dx = dir*delta; vl = len(v)
    v2 = v.copy()
    for i in range(vl):
        j = (i + dx) % vl
        v2[j] = v[i]
    return v2

# Given one shift command (dir + delta), provide MANY examples using randomly-generated initial vectors.
def gen_vector_shift_cases(vlen,count, dir=1,delta=1, density=0.5):
    cases = []
    for i in range(count):
        v = gen_dense_vector(vlen,density=density)
        v2 = shift_vector(v,dir=dir,delta=delta)
        cases.append((v,v2))
    return cases

# ****** RANDOM VECTORS of Chosen Density *****

# Given a density (fraction), this randomly places onvals to produce a vector with the desired density.
def gen_dense_vector(size, density=.5, onval=1, offval=0):
    a = [offval] * size
    indices = np.random.choice(size,round(density*size),replace=False)
    for i in indices: a[i] = onval
    return a

def gen_random_density_vectors(count,size,density_range=(0,1)):
    return [gen_dense_vector(size,density=np.random.uniform(*density_range)) for c in range(count)]

# ***** SYMMETRIC VECTORS *****
# Symmetric vectors are binary vectors that are symmetric about their midpoint.  For example:
# 10011001, 01110 and 0001110111000 are symmetric.  Note that they can have even or odd length.

def gen_symmetric_vector(len):
    halflen = math.floor(len/2)
    v1 = gen_dense_vector(halflen,density=NPR.uniform(0,1))
    v2 = v1.copy()
    v2.reverse()
    v3 = v1 + v2
    if (len % 2) == 1:
        v3.insert(halflen,NPR.randint(0,2))  # Note that numpy.random's randint(0,k) uses k-1 as max int, not k.
    return v3

def check_vector_symmetry(v):
    vlen = len(v)
    for i in range(math.floor(vlen/2)):
        if v[i] != v[vlen-1-i]:
            return False
    return True

# This produces a set of symmetric vectors and appends the class label onto the end (for ease of use in ML).

def gen_symvect_cases(vlen,count,label=1):
    return [gen_symmetric_vector(vlen) + [label] for i in range(count)]

def gen_anti_symvect_cases(vlen,count,label=0):
    cases = []
    while len(cases) < count:
        v = gen_dense_vector(vlen,density=NPR.uniform(0,1))
        if not(check_vector_symmetry(v)):
            cases.append(v+[label])
    return cases

# Generate a dataset with an equal (or nearly so if vlen is odd) number of symmetric and anti-symmetric bit vectors
def gen_symvect_dataset(vlen,count):
    s1 = math.floor(count/2); s2 = count - s1
    cases = gen_symvect_cases(vlen,s1) + gen_anti_symvect_cases(vlen,s2)
    NPR.shuffle(cases)
    return cases

# ****** LINES (horiz and vert) in arrays *********

# This produces either rows or columns of values (e.g. 1's), where the bias controls whether or not
# the entire row/column gets filled in or not just some cells. bias=1 => fill all.  Indices are those of the
# rows/columns to fill.  This is mainly used for creating data sets for classification tasks: horiz -vs- vertical
# lines.

def gen_line_array(dims,indices,line_item=1,background=0,columns=False,bias=1.0):
    a = np.array([background]*np.prod(dims)).reshape(dims)
    if columns: a = a.reshape(list(reversed(dims)))
    for row in indices:
        for j in range(a.shape[1]):
            if np.random.uniform(0, 1) <= bias: a[row,j] = line_item
    if columns: a = a.transpose()
    return a


# ****** ML CASE GENERATORS *****
# A ML "case" is a vector with two elements: the input vector and the output (target) vector.  These are the
# high-level functions that should get called from ML code.  They invoke the supporting functions above.

# The simplest autoencoders use the set of one-hot vectors as inputs and target outputs.

def gen_all_one_hot_cases(len, floats=False):
    return [[c,c] for c in all_one_hots(len,floats=floats)]

# This creates autoencoder cases for vector's with any density of 1's (specified by density_range).
def gen_dense_autoencoder_cases(count,size,dr=(0,1)):
    return [[v,v] for v in gen_random_density_vectors(count,size,density_range=dr)]

# Produce a list of pairs, with each pair consisting of a num_bits bit pattern and a singleton list containing
# the parity bit: 0 => an even number of 1's, 1 => odd number of 1's.  When double=True, a 2-bit vector is the
# target, with bit 0 indicating even parity and bit 1 indicating odd parity.

def gen_all_parity_cases(num_bits, double=True):
    def parity(v): return sum(v) % 2
    def target(v):
        if double:
            tg = [0,0].copy()
            tg[parity(v)] = 1
            return tg
        else: return [parity(v)]

    return [[c, target(c)] for c in gen_all_bit_vectors(num_bits)]

# This produces "count" cases, where features = random bit vectors and target = a one-hot vector indicating
# the number of 1's in the feature vector(default) or simply the count label.  Note that the target vector is one bit
# larger than the feature vector to account for the case of a zero-sum feature vector.

def gen_vector_count_cases(num,size,drange=(0,1),random=True,poptarg=True):
    if random: feature_vectors = gen_random_density_vectors(num,size,density_range=drange)
    else: feature_vectors = gen_all_bit_vectors(size)
    if poptarg:
        targets = [int_to_one_hot(sum(fv),size+1) for fv in feature_vectors]
    else: targets = [sum(fv) for fv in feature_vectors]
    return [[fv,targ] for fv,targ in zip(feature_vectors,targets)]

def gen_all_binary_count_cases(size,poptarg=True): return gen_vector_count_cases(None,size,random=False,poptarg=poptarg)

# Generate cases whose feature vectors, when converted into 2-d arrays, contain either one or more horizontal lines
# or one or more vertical lines.  The argument 'min_opens' is the minimum number of rows (for horizontal lines) or
# columns (for vertical lines) that must NOT be filled. The class is then simply horizontal (0) or vertical (1).
# A bias of 1.0 insures that no noise will be injected into the lines, and thus classification should be easier.

def gen_random_line_cases(num_cases,dims,min_lines=1,min_opens=1,bias=1.0, mode='classify',
                          line_item=1,background=0,flat=True,floats=False):
    def gen_features(r_or_c):
        r_or_c = int(r_or_c)
        size = dims[r_or_c]
        count = np.random.randint(min_lines,size-min_opens+1)
        return gen_line_array(dims,indices=np.random.choice(size,count,replace=False), line_item=line_item,
                              background=background,bias=bias,columns =(r_or_c == 1))
    def gen_case():
        label = np.random.choice([0,1]) # Randomly choose to use a row or a column
        features = gen_features(label)
        if flat: features = features.flatten().tolist()
        if mode == 'classify':  # It's a classification task, so use 2 neurons, one for each class (horiz, or vert)
            target = [0]*2
            target[label] = 1
        elif mode == 'auto':  target = copy.deepcopy(features)  # Autoencoding mode
        else: target = [float(label)]  # Otherwise, assume regression mode.
        return (features, target)

    if floats:
        line_item = float(line_item); background = float(background)
    return [gen_case() for i in range(num_cases)]

# ********** SEGMENT VECTORS **********
# These have groups/segments/blobs of "on" bits interspersed in a background of "off" bits.  The key point is that we can
# specify the number of segments, but the sizes are chosen randomly.

def gen_segmented_vector(vectorsize,numsegs,onval=1,offval=0):
    if vectorsize >= 2*numsegs - 1:  # Need space for numsegs-1 gaps between the segments
        vect = [offval] * vectorsize
        if numsegs <= 0: return vect
        else:
            min_gaps = numsegs - 1 ;
            max_chunk_size = vectorsize - min_gaps; min_chunk_size = numsegs
            chunk_size = NPR.randint(min_chunk_size,max_chunk_size+1)
            seg_sizes = gen_random_pieces(chunk_size,numsegs)
            seg_start_locs = gen_segment_locs(vectorsize,seg_sizes)
            for s0,size in zip(seg_start_locs,seg_sizes): vect[s0:s0+size] = [onval]*size
            return vect

#  Randomly divide chunk_size units into num_pieces units, returning the sizes of the units.
def gen_random_pieces(chunk_size,num_pieces):
    if num_pieces == 1: return [chunk_size]
    else:
        cut_points = list(NPR.choice(range(1,chunk_size),num_pieces-1,replace=False)) # the pts at which to cut up the chunk
        lastloc = 0; pieces = []; cut_points.sort() # sort in ascending order
        cut_points.append(chunk_size)
        for pt in cut_points:
            pieces.append(pt-lastloc)
            lastloc = pt
        return pieces

def gen_segment_locs(maxlen,seg_sizes):
    locs = []; remains = sum(seg_sizes); gaps = len(seg_sizes) - 1; start_min = 0
    for ss in seg_sizes:
        space = remains + gaps
        start = NPR.randint(start_min,maxlen - space + 1)
        locs.append(start)
        remains -= ss; start_min = start + ss + 1; gaps -= 1
    return locs

# This is the high-level routine for creating the segmented-vector cases.  As long as poptargs=True, a
# population-coded (i.e. one-hot) vector will be created as the target vector for each case.

def gen_segmented_vector_cases(vectorlen,count,minsegs,maxsegs,poptargs=True):
    cases = []
    for c in range(count):
        numsegs = NPR.randint(minsegs,maxsegs+1)
        v = gen_segmented_vector(vectorlen,numsegs)
        case = [v,int_to_one_hot(numsegs-minsegs,maxsegs-minsegs+1)] if poptargs else [v,numsegs]
        cases.append(case)
    return cases

def segment_count(vect,onval=1,offval=0):
    lastval = offval; count = 0
    for elem in vect:
        if elem == onval and lastval == offval: count += 1
        lastval = elem
    return count

# This produces a string consisting of the binary vector followed by the segment count surrounded by a few symbols
# and/or blanks.  These strings are useful to use as labels during dendrogram plots, for example.
def segmented_vector_string(v,pre='** ',post=' **'):
    def binit(vect): return map((lambda x: 1 if x > 0 else 0), vect)
    return ''.join(map(str, binit(v))) + pre + str(segment_count(v)) + post


# ***** PRIMITIVE DATA VIEWING ******

def show_results(grabbed_vals,grabbed_vars=None,dir='probeview'):
    showvars(grabbed_vals,names = [x.name for x in grabbed_vars], msg="The Grabbed Variables:")

def showvars(vals,names=None,msg=""):
    print("\n"+msg,end="\n")
    for i,v in enumerate(vals):
        if names: print("   " + names[i] + " = ",end="\n")
        print(v,end="\n\n")

# Very simple printing of a matrix using the 'style' format for each element.
def pp_matrix(m,style='{:.3f}'):
    rows, cols = m.shape
    for r in range(rows):
        print()  # skips to next line
        for c in range(cols): print(style.format(m[r][c]), end=' ')
    print()

# *******  DATA PLOTTING ROUTINES *********

def simple_plot(yvals,xvals=None,xtitle='X',ytitle='Y',title='Y = F(X)'):
    xvals = xvals if xvals is not None else list(range(len(yvals)))
    PLT.plot(xvals,yvals)
    PLT.xlabel(xtitle); PLT.ylabel(ytitle); PLT.title(title)
    PLT.draw()

# Each history is a list of pairs (timestamp, value).
def plot_training_history(error_hist,validation_hist=[],xtitle="Epoch",ytitle="Error",title="History",fig=True):
    PLT.ion()
    if fig: PLT.figure()
    if len(error_hist) > 0:
        simple_plot([p[1] for p in error_hist], [p[0] for p in error_hist],xtitle=xtitle,ytitle=ytitle,title=title)
        PLT.hold(True)
    if len(validation_hist) > 0:
        simple_plot([p[1] for p in validation_hist], [p[0] for p in validation_hist])
    PLT.ioff()

# alpha = transparency
def simple_scatter_plot(points,alpha=0.5,radius=3):
    colors = ['red','green','blue','magenta','brown','yellow','orange','brown','purple','black']
    a = np.array(points).transpose()
    PLT.scatter(a[0],a[1],c=colors,alpha=alpha,s=np.pi*radius**2)
    PLT.draw()

# This is Hinton's classic plot of a matrix (which may represent snapshots of weights or a time series of
# activation values).  Each value is represented by a red (positive) or blue (negative) square whose size reflects
# the absolute value.  This works best when maxsize is hardwired to 1.  The transpose (trans) arg defaults to
# true so that matrices are plotted with rows along a horizontal plane, with the 0th row on top.

# The 'colors' argument, a list, is ordered as follows: background, positive-value, negative-value, box-edge.
# If you do not want to draw box edges, just use 'None' as the 4th color.  A gray-scale combination that
# mirrors Hinton's original version is ['gray','white','black',None]

def hinton_plot(matrix, maxval=None, maxsize=1, fig=None,trans=True,scale=True, title='Hinton plot',
                colors=['gray','red','blue','white']):
    hfig = fig if fig else PLT.figure()
    hfig.suptitle(title,fontsize=18)
    if trans: matrix = matrix.transpose()
    if maxval == None: maxval = np.abs(matrix).max()
    if not maxsize: maxsize = 2**np.ceil(np.log(maxval)/np.log(2))

    axes = hfig.gca()
    axes.clear()
    axes.patch.set_facecolor(colors[0]);  # This is the background color.  Hinton uses gray
    axes.set_aspect('auto','box')  # Options: ('equal'), ('equal','box'), ('auto'), ('auto','box')..see matplotlib docs
    axes.xaxis.set_major_locator(PLT.NullLocator()); axes.yaxis.set_major_locator(PLT.NullLocator())

    ymax = (matrix.shape[1] - 1)* maxsize
    for (x, y), val in np.ndenumerate(matrix):
        color = colors[1] if val > 0 else colors[2]  # Hinton uses white = pos, black = neg
        if scale: size = max(0.01,np.sqrt(min(maxsize,maxsize*np.abs(val)/maxval)))
        else: size = np.sqrt(min(np.abs(val),maxsize))  # The original version did not include scaling
        bottom_left = [x - size / 2, (ymax - y) - size / 2] # (ymax - y) to invert: row 0 at TOP of diagram
        blob = PLT.Rectangle(bottom_left, size, size, facecolor=color, edgecolor=colors[3])
        axes.add_patch(blob)
    axes.autoscale_view()
    PLT.draw()
    PLT.pause(.001)

# This graphically displays a matrix with color codes for positive, negative, small positive and small negative,
# with the latter 2 defined by the 'cutoff' argument.  The transpose (trans) arg defaults to
# True so that matrices are plotted with rows along a horizontal plane, with the 0th row on top.
# Colors denote: [positive, small positive, small negative, negative]

def display_matrix(matrix,fig=None,trans=True,scale=True, title='Matrix',tform='{:.3f}',tsize=12,
                   cutoff=0.1,colors=['red','yellow','grey','blue']):
    hfig = fig if fig else PLT.figure()
    hfig.suptitle(title,fontsize=18)
    if trans: matrix = matrix.transpose()
    axes = hfig.gca()
    axes.clear()
    axes.patch.set_facecolor('white');  # This is the background color.  Hinton uses gray
    axes.set_aspect('auto','box')  # Options: ('equal'), ('equal','box'), ('auto'), ('auto','box')..see matplotlib docs
    axes.xaxis.set_major_locator(PLT.NullLocator()); axes.yaxis.set_major_locator(PLT.NullLocator())

    ymax = matrix.shape[1] - 1
    for (x, y), val in np.ndenumerate(matrix):
        if val > 0: color = colors[0] if val > cutoff else colors[1]
        else: color = colors[3] if val < -cutoff else colors[2]
        botleft = [x - 1/2, (ymax - y) - 1/2] # (ymax - y) to invert: row 0 at TOP of diagram
        # This is a hack, but I seem to need to add these blank blob rectangles first, and then I can add the text
        # boxes.  If I omit the blobs, I get just one plotted textbox...grrrrrr.
        blob = PLT.Rectangle(botleft, 1,1, facecolor='white',edgecolor='white')
        axes.add_patch(blob)
        axes.text(botleft[0]+0.5,botleft[1]+0.5,tform.format(val),
                  bbox=dict(facecolor=color,alpha=0.5,edgecolor='white'),ha='center',va='center',
                  color='black',size=tsize)
    axes.autoscale_view()
    PLT.draw()
    PLT.pause(1)

# ****** Principle Component Analysis (PCA) ********
# This performs the basic operations outlined in "Python Machine Learning" (pp.128-135).  It begins with
# an N x K array whose rows are cases and columns are features.  It then computes the covariance matrix (of features),
# which is then used to compute the eigenvalues and eigenvectors.  The eigenvectors corresponding to the largest
# (absolute value) eigenvalues are then combined to produce a transformation matrix, which is applied to the original
# N cases to produce N new cases, each with J (ideally J << K) features.  This is UNSUPERVISED dimension reduction.

def pca(features,target_size,bias=True,rowvar=False):
    farray = features if isinstance(features,np.ndarray) else np.array(features)
    cov_mat = np.cov(farray,rowvar=rowvar,bias=bias) # rowvar=False => each var's values are in a COLUMN.
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    return gen_dim_reduced_data(farray,target_size,eigen_vals, eigen_vecs)

# Use the highest magnitude eigenvalues (and their eigenvectors) as the basis for feature-vector transformations that
# reduce the dimensionality of the data.  feature_array is N x M, where N = # cases, M = # features

def gen_dim_reduced_data(feature_array,target_size,eigen_values,eigen_vectors):
    eigen_pairs = [(np.abs(eigen_values[i]),eigen_vectors[:,i]) for i in range(len(eigen_values))]
    eigen_pairs.sort(key=(lambda p: p[0]),reverse=True)  # Sorts tuples by their first element = abs(eigenvalue)
    best_vectors = [pair[1] for pair in eigen_pairs[ : target_size]]
    w_transform = np.array(best_vectors).transpose()
    return np.dot(feature_array,w_transform)

# *************** DENDROGRAM*************************
# Options:
# orientation = top, bottom, left, right (refers to location of the root of the tree)
# mode = single, average, complete, centroid, ward, median
# metric = euclidean, cityblock (manhattan), hamming, cosine, correlation ... (see matplotlib distance.pdist for all 23)
def dendrogram(features,labels,metric='euclidean',mode='average',ax=None,title='Dendrogram',orient='top',lrot=90.0):
    ax = ax if ax else PLT.gca()
    cluster_history = SCH.linkage(features,method=mode,metric=metric)
    SCH.dendrogram(cluster_history,labels=labels,orientation=orient,leaf_rotation=lrot)
    PLT.tight_layout()
    ax.set_title(title)
    ax.set_ylabel(metric + ' distance')
    PLT.show()