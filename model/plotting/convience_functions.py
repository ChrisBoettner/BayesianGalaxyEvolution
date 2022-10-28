#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 18:16:20 2022

@author: chris
"""
import math
import numpy as np
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib import text as mtext
from matplotlib.colors import to_rgb

from model.data.load import load_data_points
from model.helper import make_array, make_list, pick_from_list, \
                         calculate_percentiles
from model.analysis.calculations import calculate_best_fit_ndf,\
                                        calculate_expected_black_hole_mass_from_ERDF

################ PLOT DATA ####################################################

def plot_group_data(axes, ModelResult, redshift=None):
    ''' Use list of group objects to plot group data at redshift z. '''
    axes = make_list(axes)
    groups = ModelResult.groups
    for g in groups:
        if redshift is None:
            rs = g.redshift
        elif np.isscalar(redshift):
            rs = make_list(redshift)
            
        for z in rs:
            if z not in g.redshift:
                continue
            if len(axes) == 1:
                ax = axes[0]
            else:
                ax = axes[z]
            ax.errorbar(g.data_at_z(z).quantity,
                        g.data_at_z(z).phi,
                        [g.data_at_z(z).lower_error,
                         g.data_at_z(z).upper_error],
                        capsize=3,
                        elinewidth = mpl.rcParams['lines.markersize']/2,
                        fmt=g.marker,
                        color=g.color,
                        label=g.label,
                        alpha=ModelResult.quantity_options['marker_alpha'])
    return


def plot_best_fit_ndf(axes, ModelResult, redshift=None, **kwargs):
    ''' Calculate and plot best fit number density functions at redshift z. '''
    axes = make_list(axes)
    
    ndfs = calculate_best_fit_ndf(ModelResult, ModelResult.redshift)
    
    if redshift is None:
        redshift = ModelResult.redshift
    elif np.isscalar(redshift):
        redshift = make_list(redshift)
    
    for z in redshift:
        color = pick_from_list(ModelResult.color, z)
        label = pick_from_list(ModelResult.label, z)
        if len(axes) == 1:
            ax = axes[0]
        else:
            ax = axes[z]
        ax.plot(ndfs[z][:, 0],
                ndfs[z][:, 1],
                linestyle=ModelResult.linestyle,
                #label=label,
                color=color,
                **kwargs)
    return(ndfs)

def plot_data_with_confidence_intervals(ax, data_percentile_dict, 
                                        color, data_masks=None, median=True,
                                        alpha=0.8, linewidth=5):
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
    mask_trail]. If median=False, only plot confidence intervals. alpha 
    controls alpha level of shaded area.
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
    sigma_color_alphas = get_sigma_color_alphas()
    if not set(sigma).issubset(sigma_equiv_table.keys()):
        raise ValueError('sigmas must be between 1 and 5 (inclusive).')
    
    # set default data_masks if none are given
    if data_masks is None:
        data_mask      = np.arange(len(data_percentile_dict[sigma[0]]))
        mask_beginning = np.array([])
        mask_trail     = np.array([])
    else:
        if len(data_masks)!=3:
            raise ValueError('data_masks must be array containing three '
                             'mask arrays: data_mask, mask_beginning, '
                             'mask_trail.')
        data_mask, mask_beginning, mask_trail = data_masks
    
    # differentiate which label should be used in case masks (data outside
    # observations) are plotted or not
    masks_shown = (mask_beginning.any() or mask_trail.any())
    if masks_shown:
        label = 'Constrained by data'
    else:
        label = 'Model Median'

    # plot confidence intervals
    for s in np.sort(make_array(sigma))[::-1]: # sort sigma in reverse order 
         if s == sigma[-1] and median:
             # plot medians
             ax.plot(data_percentile_dict[s][:,0][data_mask],
                     data_percentile_dict[s][:,1][data_mask],
                     #label=label,
                     color=color,
                     linewidth=linewidth)
             if masks_shown:
                 ax.plot(data_percentile_dict[s][:,0][mask_beginning],
                         data_percentile_dict[s][:,1][mask_beginning],
                         ':',
                         #label='Not constrained by data',
                         color=color,
                         linewidth=linewidth)
                 ax.plot(data_percentile_dict[s][:,0][mask_trail],
                         data_percentile_dict[s][:,1][mask_trail],
                         ':', color=color)
         ax.fill_between(data_percentile_dict[s][:, 0],
                         data_percentile_dict[s][:, 2], 
                         data_percentile_dict[s][:, 3],
                         color=blend_color(color, sigma_color_alphas[s]),
                         edgecolor=blend_color(color, 
                                               sigma_color_alphas[s]+0.05),
                         alpha=alpha,
                         #label= sigma_equiv_table[s] + r'\% Percentile'
                         )
    return()

def plot_data_points(ax, ModelResult1, ModelResult2=None, z=0, legend=True,
                     **kwargs):
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
        except NameError:
            raise NameError('Can\'t find data.')
    
    if name == 'mstar_mbh':
        ax.scatter(data[:,0], data[:,1], 
                   s=35, alpha=0.5,  
                   color='lightgrey',
                   label='Baron2019 (Type 1 and Type 2)',
                   marker='o',
                   **kwargs)
    
    elif name == 'mbh_Lbol':
        ax.scatter(data[:,0], data[:,1], 
                   s=35, alpha=0.5,  
                   color='lightgrey',
                   label='Baron2019 (Type 1)',
                   marker='o',
                   **kwargs)
    
    elif name == 'Muv_mstar':
        if z not in data.keys():
            raise KeyError('Redshift not in Mainsequence data dictonary.')
        d = data[z]
        
        zorder = 0 if z<7 else 100 # decide if datapoint should be in front or
                                   # back
        ax.scatter(d[:,0], d[:,1], 
                   s=40, alpha=0.7, 
                   color='grey',
                   label='Song2016',
                   marker='o', zorder=zorder,
                   **kwargs)
    return()
    
def plot_linear_relationship(ax, log_x_range, log_slope, labels = None):
    '''
    Add additional linear relations to plot. x values and slope must be given
    in log, can deal with multiple slopes at once. Labels can be added manually,
    must be list with same length as slopes.
    '''
    log_slope = make_list(log_slope)
    
    if labels:
        labels = make_list(labels)
        if len(labels)!=len(log_slope):
            raise ValueError('List of labels must have same length as list of '
                             ' slopes.')

    # calculate linear relations (in log_space)
    y_values = {}
    for s in log_slope:
        y_values[s] = s + log_x_range
    # plot linear relations
    for i, s in enumerate(log_slope):
        ax.plot(log_x_range, y_values[s], alpha = 0.4, linestyle='--', 
                color = 'grey')
        if labels:
            CurvedText(x    = log_x_range,
                       y    = y_values[s],
                       text = labels[i],
                       va   = 'bottom',
                       axes = ax)  
    return()
    
def plot_q1_q2_additional(ax, ModelResult1, ModelResult2, z, log_q1, sigma,
                          linewidth=5, legend=False):
    '''
    Plot additional relations for q1 - q2 relations if necessary.
    ''' 
    if (ModelResult1.quantity_name == 'Lbol' and
        ModelResult2.quantity_name == 'mbh'):
        
        # turn previous plotted lines grey grey
        for line in ax.get_lines():
            line.set_color('grey')
        for i, col in enumerate(ax.collections):
            sigma_color_alphas = get_sigma_color_alphas()
            col.set_color(blend_color(to_rgb('lightgrey'),
                                      sigma_color_alphas[sigma[i]]))
        
        # add new plot with conditional ERDF
        mbh_dict = calculate_expected_black_hole_mass_from_ERDF(ModelResult1,
                        log_q1, z, sigma=sigma)
        plot_data_with_confidence_intervals(ax, mbh_dict, 'C3',
                                            linewidth=linewidth)
        
        # put later plot in foreground
        ax.collections[-1].set_zorder(100)
        ax.get_lines()[-1].set_zorder(101)
        
        # legend text change
        ax.legend(ax.get_lines(), 
                      [r'$\langle \lambda \rangle$',
                       r'$\langle \lambda | L_\mathrm{bol} \rangle$'],
                      frameon=False,
                      loc='upper left',
                      fontsize=mpl.rcParams['font.size']*1.3)
    
    elif (ModelResult1.quantity_name == 'Muv' and
          ModelResult2.quantity_name == 'mstar'):     
        
        # calculate ranges from Song
        median, lower, upper = song_relation_ranges(z, log_q1)
        
        #plot ranges (4sigma) and median
        alpha_val = 0.5
        ax.fill_between(log_q1, lower, upper, 
                        color=blend_color(to_rgb('lightgrey'),alpha_val), 
                        edgecolor=blend_color(to_rgb('grey'),alpha_val),
                        zorder=0)
        ax.plot(log_q1, median, color='grey',alpha=alpha_val,
                linewidth=linewidth)
        
        # add redshift text
        ax.text(0.97, 0.94, r'$z \sim$ ' + str(z),
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes,
                fontsize=mpl.rcParams['font.size']*1.5)
        
    return()

def plot_feedback_regimes(axes, ModelResult, redshift=None, log_epsilon=-1,
                          vertical_lines=False, shaded=True, linewidth=2, 
                          linecolor='grey', alpha=0.2):
    '''
    Plot feedback regimes, where one of the feedback modes becomes dominant
    at redshift z. Relative strength is controlled using log_epsilon argument
    (see calculate_feedback_regimes in Model.physics_model for more details).
    Plot vertical lines if vertical_lines=True, shaded area if shaded=True.
    '''
    axes = make_list(axes)

    if redshift is None:
        redshift = ModelResult.redshift
    elif np.isscalar(redshift):
        redshift = make_list(redshift)
    
    if ModelResult.physics_name in ['stellar', 'stellar_blackhole', 'changing']:
        transition_quantity_value = {}
        for z in redshift:
            parameter = ModelResult.parameter.at_z(z)
            transition_quantity_value[z] = ModelResult.\
                                            calculate_feedback_regimes(
                                                 z, parameter, 
                                                 output='quantity',
                                                 log_epsilon=log_epsilon)
    
        for z in redshift:
            if len(axes) == 1:
                ax = axes[0]
            else:
                ax = axes[z]
            
            # plot vertical line where M_h = M_c
            if vertical_lines: # plot vertical lines for feedback dominated areas
                ax.axvline(transition_quantity_value[z][0], linewidth=linewidth,
                           color=linecolor)
                ax.axvline(transition_quantity_value[z][1], linewidth=linewidth,
                           color=linecolor)
                ax.axvline(transition_quantity_value[z][2], linewidth=linewidth,
                           color=linecolor)
            if shaded: # plot shaded aread areas of feedback dominated areas
                ax.axvspan(transition_quantity_value[z][1],
                            transition_quantity_value[z][2],
                            facecolor='C3', alpha=alpha, zorder= 0)
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
 
################ DATA ######################################################### 


def song_relation_parameter(z):
    ''' Parameter for linear relation in mstar-Muv relation from Song2016.'''
    # {redshift: array} where array is
    # [slope, slope_error, offset (at Muv=-21), offset_error]
    parameter = {4: [-0.54, 0.03, 9.70, 0.02],
                 5: [-0.50, 0.04, 9.59, 0.03],
                 6: [-0.50, 0.03, 9.53, 0.02],
                 7: [-0.50, 0,    9.36, 0.16],
                 8: [-0.50, 0,    9.00, 0.32]}
    return(parameter[z])

def song_relation_ranges(z, quantity_range, num=int(1e+6), sigma=4):
    ''' 95% range of linear relationship from Song 2016.'''
    parameter = song_relation_parameter(z)
    
    # draw from gaussian 
    slope_draw  = parameter[0] + parameter[1] * np.random.randn(num)  
    offset_draw = parameter[2] + parameter[3] * np.random.randn(num)
    # possible y values (offset is given at x=-21)
    y = slope_draw * (quantity_range[:,np.newaxis]+21) + offset_draw
    # calculate percentiles
    percentiles = calculate_percentiles(y, axis=1, sigma_equiv=sigma)
    return(percentiles)
    
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

def get_sigma_color_alphas():
    '''For confidence interval plot, get alpha values for blending.'''
    sigma_color_alphas = {1: 0.8, 2: 0.55, 3: 0.4, 4: 0.2, 5: 0.1}
    return(sigma_color_alphas)
  
################ TEXT #########################################################


def add_redshift_text(axes, redshifts):
    ''' Add current redshift as text to upper plot corner. '''
    for z in redshifts:
        axes[z].text(0.97, 0.94, r'$z \sim$ ' + str(z),
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

    
