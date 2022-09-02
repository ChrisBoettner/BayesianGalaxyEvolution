#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 18:16:20 2022

@author: chris
"""
import math
import numpy as np
from matplotlib.lines import Line2D
from matplotlib import text as mtext
from matplotlib.colors import to_rgb

from model.data.load import load_data_points
from model.helper import make_array, make_list, pick_from_list
from model.analysis.calculations import calculate_best_fit_ndf

################ PLOT DATA ####################################################

def plot_group_data(axes, ModelResult):
    ''' Use list of group objects to plot group data. '''
    groups = ModelResult.groups    
    for g in groups:
        for z in g.redshift:
            axes[z].errorbar(g.data_at_z(z).quantity,
                             g.data_at_z(z).phi,
                             [g.data_at_z(z).lower_error,
                              g.data_at_z(z).upper_error],
                             capsize=3,
                             fmt=g.marker,
                             color=g.color,
                             label=g.label,
                             alpha=ModelResult.quantity_options['marker_alpha'])
    return


def plot_best_fit_ndf(axes, ModelResult):
    ''' Calculate and plot best fit number density functions. '''
    ndfs = calculate_best_fit_ndf(ModelResult, ModelResult.redshift)

    for z in ModelResult.redshift:
        color = pick_from_list(ModelResult.color, z)
        label = pick_from_list(ModelResult.label, z)
        axes[z].plot(ndfs[z][:, 0],
                     ndfs[z][:, 1],
                     linestyle=ModelResult.linestyle,
                     label=label,
                     color=color)
    return(ndfs)

def plot_data_with_confidence_intervals(ax, data_percentile_dict, 
                                        color, data_masks=None):
    '''
    Plot data with confidence intervals. 
    The input data must be a dictonary, 
    where the keys must be the sigma equvialent values (1-5) and each dict 
    entry must contain an arry of the form (x, y, lower_y_bound, upper_y_bound)
    as returned by calculate_qhmr and calculate_q1_q2_relation in 
    model.calculations. 
    The color must be speficified either as rgb or str.
    Optional: Add data_masks for data that is or is not constrained by data,
    must be an array containing 3 mask arrays [data_mask, mask_beginning, 
    mask_trail].
    '''
    
    # convert color to RGB
    if type(color) is str:
        color = to_rgb(color)
    
    ## get sigma equiv values and define sigma properties
    sigma = list(data_percentile_dict.keys())
    # conversion table between sigma equivalents and percentiles
    sigma_equiv_table = {1: '68', 2: '95', 3: '99.7', 4: '99.993',
                         5: '99.99994'}
    # alpha values for different sigma equivalents
    sigma_color_alphas = {1: 0.6, 2: 0.5, 3: 0.3, 4: 0.2, 5: 0.1}
    if not set(sigma).issubset(sigma_equiv_table.keys()):
        raise ValueError('sigmas must be between 1 and 5 (inclusive).')
    
    # set default data_masks if none are given
    if data_masks is None:
        data_mask      = np.arange(len(data_percentile_dict[sigma[0]]))
        mask_beginning = []
        mask_trail     = []
    else:
        if len(data_masks)!=3:
            raise ValueError('data_masks must be array containing three '
                             'mask arrays: data_mask, mask_beginning, '
                             'mask_trail.')
        data_mask, mask_beginning, mask_trail = data_masks

    # plot confidence intervals
    for s in sigma:  
         if s == sigma[-1]:
             # plot medians
             ax.plot(data_percentile_dict[s][:,0][data_mask],
                     data_percentile_dict[s][:,1][data_mask],
                     label='constrained by data',
                     color=color)
             ax.plot(data_percentile_dict[s][:,0][mask_beginning],
                     data_percentile_dict[s][:,1][mask_beginning],
                     ':',label='not constrained by data',
                     color=color)
             ax.plot(data_percentile_dict[s][:,0][mask_trail],
                     data_percentile_dict[s][:,1][mask_trail],
                     ':', color=color)
         ax.fill_between(data_percentile_dict[s][:, 0],
                         data_percentile_dict[s][:, 2], 
                         data_percentile_dict[s][:, 3],
                         color=blend_color(color, sigma_color_alphas[s]),
                         label= sigma_equiv_table[s] + r'\% percentile')
    return()

def plot_data_points(ax, ModelResult1, ModelResult2=None, legend=True):
    '''
    Plot additional dataset to compare to Model.
    '''
    
    # figure out naming scheme, in order to look up if data is available
    if ModelResult2 is None:
        try:
            name = ModelResult1.quantity_name
            data = load_data_points(name)
        except NameError:
            raise NameError('Can\'t find data.')
            
    else:
        try:
            name = ModelResult1.quantity_name + '_' + ModelResult2.quantity_name
            data = load_data_points(name)
        except:
            raise NameError('Can\'t find data.')
    
    # so far we only have data for mstar_mbh, can be extended, include labels
    if name == 'mstar_mbh':
        ax.scatter(data[:,0], data[:,1], 
                   s=35, alpha=0.5,  
                   color='lightgrey',
                   label='Baron2019 (Type 1 and Type 2)',
                   marker='o')
    return()
    

################ LEGEND #######################################################


def add_legend(axes, ind, sort=False, **kwargs):
    '''
    Add legend at axis given by ind. If sort is true, sort labels before
    displaying legend.
    '''
    axes = make_array(axes)

    labels = remove_double_labels(axes)
    if sort:
        labels = dict(sorted(labels.items()))

    axes[ind].legend(list(labels.values()),
                     list(labels.keys()),
                     frameon=False,
                      **kwargs)
    return


def add_separated_legend(axes, separation_point, ncol=1, loc=0):
    '''
    Add part of legend to first subplot and part to last subplot, devided by
    separation_point. Can also adjust number of columns of legend.
    '''
    labels = remove_double_labels(axes)
    axes[-2].legend(list(labels.values())[:separation_point],
                   list(labels.keys())[:separation_point],
                   frameon=False,
                   loc=loc)
    axes[-1].legend(list(labels.values())[separation_point:],
                    list(labels.keys())[separation_point:],
                    frameon=False,
                    prop={'size': 14},
                    loc=[0,0], ncol=ncol)
    return


def remove_double_labels(axes):
    '''  
    Remove duplicates in legend that have same label. Also sorts labels so that
    Line2D objects appear first.
    '''
    from itertools import compress
    axes = make_array(axes)

    handles, labels = [], []
    for a in axes.flatten():
        handles_, labels_ = a.get_legend_handles_labels()
        handles += handles_
        labels += labels_

    # sort so that Lines2D objects appear first
    lines_idx      = [isinstance(handle, Line2D) for handle in handles]
    handles_sorted = list(compress(handles, lines_idx)) \
                     + list(compress(handles, np.logical_not(lines_idx)))
    labels_sorted = list(compress(labels, lines_idx)) \
                     + list(compress(labels, np.logical_not(lines_idx)))                
    by_label = dict(zip(labels_sorted, handles_sorted))
    return(by_label)

################ FRAMES #######################################################


def turn_off_frame(axes):
    '''
    Turn off top and right frame of all axes

    '''
    
    axes = make_list(axes)
    
    for ax in axes:
            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
    return


################ AXES AND LIMITS ##############################################


def turn_off_axes(axes):
    ''' Turn of axes for subplots that are not used. '''
    for ax in axes.flatten():
        if (not ax.lines) and (not ax.patches):
            ax.axis('off')
    return

def get_distribution_limits(ModelResults):  
    ''' 
    Get minimum and maximum values for distributions across redshifts
    and different models (e.g. for pdf plot limits).
    '''
    
    ModelResults = make_list(ModelResults)
    max_values, min_values = {}, {}
    for Model in ModelResults:
        for z in Model.redshift:
            distribution = Model.distribution.at_z(z)
            for i in range(distribution.shape[1]):
                max_values.setdefault(i, -np.inf) # check if key already exists,
                                                  # if not, create it and put 
                                                  # value to -infinity
                min_values.setdefault(i, np.inf)  
                
                current_max = np.amax(distribution[:,i])
                current_min = np.amin(distribution[:,i])
                
                # update maximum and minimum values
                if current_max>max_values[i]:
                    max_values[i] = current_max
                if current_min< min_values[i]:
                    min_values[i] = current_min    
                    
    # return as list of limits
    limits = list(zip(min_values.values(),max_values.values()))
    return(limits)
    
    
################ COLORS #######################################################  

def blend_color(color, alpha, bg_color=np.array([1,1,1])):
    '''
    Blend color with background color using alpha value. Color and background
    color must be given as array of RGB values. Default background color is 
    white.
    '''
    color    = make_array(color)
    bg_color = make_array(bg_color)
    return((1-alpha)*bg_color + alpha*color)
  
################ TEXT #########################################################


def add_redshift_text(axes, redshifts):
    ''' Add current redshift as text to upper plot corner. '''
    for z in redshifts:
        axes[z].text(0.97, 0.94, 'z=' + str(z),
                     horizontalalignment='right',
                     verticalalignment='top',
                     transform=axes[z].transAxes)
    return


class CurvedText(mtext.Text):
    """
    A text object that follows an arbitrary curve, taken from 
    https://stackoverflow.com/questions/19353576/curved-text-rendering-in-matplotlib.
    """
    def __init__(self, x, y, text, axes, **kwargs):
        super(CurvedText, self).__init__(x[0],y[0],' ', **kwargs)

        axes.add_artist(self)

        ##saving the curve:
        self.__x = x
        self.__y = y
        self.__zorder = self.get_zorder()

        ##creating the text objects
        self.__Characters = []
        for c in text:
            if c == ' ':
                ##make this an invisible 'a':
                t = mtext.Text(0,0,'a')
                t.set_alpha(0.0)
            else:
                t = mtext.Text(0,0,c, **kwargs)

            #resetting unnecessary arguments
            t.set_ha('center')
            t.set_rotation(0)
            t.set_zorder(self.__zorder +1)

            self.__Characters.append((c,t))
            axes.add_artist(t)


    ##overloading some member functions, to assure correct functionality
    ##on update
    def set_zorder(self, zorder):
        super(CurvedText, self).set_zorder(zorder)
        self.__zorder = self.get_zorder()
        for c,t in self.__Characters:
            t.set_zorder(self.__zorder+1)

    def draw(self, renderer, *args, **kwargs):
        """
        Overload of the Text.draw() function. Do not do
        do any drawing, but update the positions and rotation
        angles of self.__Characters.
        """
        self.update_positions(renderer)

    def update_positions(self,renderer):
        """
        Update positions and rotations of the individual text elements.
        """

        #preparations

        ##determining the aspect ratio:
        ##from https://stackoverflow.com/a/42014041/2454357

        ##data limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        ## Axis size on figure
        figW, figH = self.axes.get_figure().get_size_inches()
        ## Ratio of display units
        _, _, w, h = self.axes.get_position().bounds
        ##final aspect ratio
        aspect = ((figW * w)/(figH * h))*(ylim[1]-ylim[0])/(xlim[1]-xlim[0])

        #points of the curve in figure coordinates:
        x_fig,y_fig = (
            np.array(l) for l in zip(*self.axes.transData.transform([
            (i,j) for i,j in zip(self.__x,self.__y)
            ]))
        )

        #point distances in figure coordinates
        x_fig_dist = (x_fig[1:]-x_fig[:-1])
        y_fig_dist = (y_fig[1:]-y_fig[:-1])
        r_fig_dist = np.sqrt(x_fig_dist**2+y_fig_dist**2)

        #arc length in figure coordinates
        l_fig = np.insert(np.cumsum(r_fig_dist),0,0)

        #angles in figure coordinates
        rads = np.arctan2((y_fig[1:] - y_fig[:-1]),(x_fig[1:] - x_fig[:-1]))
        degs = np.rad2deg(rads)


        rel_pos = 10
        for c,t in self.__Characters:
            #finding the width of c:
            t.set_rotation(0)
            t.set_va('center')
            bbox1  = t.get_window_extent(renderer=renderer)
            w = bbox1.width
            h = bbox1.height

            #ignore all letters that don't fit:
            if rel_pos+w/2 > l_fig[-1]:
                t.set_alpha(0.0)
                rel_pos += w
                continue

            elif c != ' ':
                t.set_alpha(1.0)

            #finding the two data points between which the horizontal
            #center point of the character will be situated
            #left and right indices:
            il = np.where(rel_pos+w/2 >= l_fig)[0][-1]
            ir = np.where(rel_pos+w/2 <= l_fig)[0][0]

            #if we exactly hit a data point:
            if ir == il:
                ir += 1

            #how much of the letter width was needed to find il:
            used = l_fig[il]-rel_pos
            rel_pos = l_fig[il]

            #relative distance between il and ir where the center
            #of the character will be
            fraction = (w/2-used)/r_fig_dist[il]

            ##setting the character position in data coordinates:
            ##interpolate between the two points:
            x = self.__x[il]+fraction*(self.__x[ir]-self.__x[il])
            y = self.__y[il]+fraction*(self.__y[ir]-self.__y[il])

            #getting the offset when setting correct vertical alignment
            #in data coordinates
            t.set_va(self.get_va())
            bbox2  = t.get_window_extent(renderer=renderer)

            bbox1d = self.axes.transData.inverted().transform(bbox1)
            bbox2d = self.axes.transData.inverted().transform(bbox2)
            dr = np.array(bbox2d[0]-bbox1d[0])

            #the rotation/stretch matrix
            rad = rads[il]
            rot_mat = np.array([
                [math.cos(rad), math.sin(rad)*aspect],
                [-math.sin(rad)/aspect, math.cos(rad)]
            ])

            ##computing the offset vector of the rotated character
            drp = np.dot(dr,rot_mat)

            #setting final position and rotation:
            t.set_position(np.array([x,y])+drp)
            t.set_rotation(degs[il])

            t.set_va('center')
            t.set_ha('center')

            #updating rel_pos to right edge of character
            rel_pos += w-used

    