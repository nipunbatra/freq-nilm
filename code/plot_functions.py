def autolabel(rects, ax):
    # Get y-axis height to calculate label position from.
    (y_bottom, y_top) = ax.get_ylim()
    y_height = y_top - y_bottom

    for rect in rects:
        height = rect.get_height()

        # Fraction of axis height taken up by this rectangle
        p_height = (height / y_height)

        # If we can fit the label above the column, do that;
        # otherwise, put it inside the column.
        if p_height > 0.95: # arbitrary; 95% looked good to me.
            label_position = height - (y_height * 0.05)
        else:
            label_position = height + (y_height * 0.01)

        ax.text(rect.get_x() + rect.get_width()/2., label_position,
                '%d' % height,
                ha='center', va='bottom')


"""Contains helper functions"""
from math import sqrt

import pandas as pd
import matplotlib as mpl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

SPINE_COLOR = 'gray'



_to_ordinalf_np_vectorized = np.vectorize(mdates._to_ordinalf)


def plot_series(series, **kwargs):
	"""Plot function for series which is about 5 times faster than
	pd.Series.plot().

	Parameters
	----------
	series : pd.Series
	ax : matplotlib Axes, optional
		If not provided then will generate our own axes.
	fig : matplotlib Figure
	date_format : str, optional, default='%d/%m/%y %H:%M:%S'
	tz_localize : boolean, optional, default is True
		if False then display UTC times.

	Can also use all **kwargs expected by `ax.plot`
	"""
	ax = kwargs.pop('ax', None)
	fig = kwargs.pop('fig', None)
	date_format = kwargs.pop('date_format', '%d/%m/%y %H:%M:%S')
	tz_localize = kwargs.pop('tz_localize', True)

	if ax is None:
		ax = plt.gca()

	if fig is None:
		fig = plt.gcf()

	x = _to_ordinalf_np_vectorized(series.index.to_pydatetime())
	ax.plot(x, series, **kwargs)
	tz = series.index.tzinfo if tz_localize else None
	ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format,
	                                                  tz=tz))
	ax.set_ylabel('watts')
	fig.autofmt_xdate()
	return ax


def latexify(fig_width=None, fig_height=None, columns=1):
	"""Set up matplotlib's RC params for LaTeX plotting.
	Call this before plotting a figure.

	Parameters
	----------
	fig_width : float, optional, inches
	fig_height : float,  optional, inches
	columns : {1, 2}
	"""

	# code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

	# Width and max height in inches for IEEE journals taken from
	# computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf


	assert (columns in [1, 2])

	if fig_width is None:
		fig_width = 3.39 if columns == 1 else 7.1  # width in inches

	if fig_height is None:
		golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
		fig_height = fig_width * golden_mean  # height in inches

	MAX_HEIGHT_INCHES = 8.0
	if fig_height > MAX_HEIGHT_INCHES:
		print("WARNING: fig_height too large:" + fig_height +
		      "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
		fig_height = MAX_HEIGHT_INCHES

	params = {'backend': 'ps',
	          'text.latex.preamble': ['\usepackage{gensymb}'],
	          'axes.labelsize': 10,  # fontsize for x and y labels (was 10)
	          'axes.titlesize': 10,
	          'text.fontsize': 10,  # was 10
	          'legend.fontsize': 10,  # was 10
	          'xtick.labelsize': 10,
	          'ytick.labelsize': 10,
	          'text.usetex': True,
	          'figure.figsize': [fig_width, fig_height],
	          'font.family': 'serif'
	          }

	matplotlib.rcParams.update(params)


def format_axes(ax):
	for spine in ['top', 'right']:
		ax.spines[spine].set_visible(False)

	for spine in ['left', 'bottom']:
		ax.spines[spine].set_color(SPINE_COLOR)
		ax.spines[spine].set_linewidth(0.5)

	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')

	for axis in [ax.xaxis, ax.yaxis]:
		axis.set_tick_params(direction='out', color=SPINE_COLOR)

	# matplotlib.pyplot.tight_layout()

	return ax


def pd_to_epoch(pd_time):
	temp = pd.DatetimeIndex([pd_time]).astype(np.int64) // 10 ** 6
	return temp[0]


def heatmap(df,
            edgecolors='w',
            cmap=mpl.cm.RdYlBu_r,
            log=False):
	width = len(df.columns) / 4
	height = len(df.index) / 4

	fig, ax = plt.subplots(figsize=(width, height))

	heatmap = ax.pcolor(df,
	                    edgecolors=edgecolors,  # put white lines between squares in heatmap
	                    cmap=cmap,
	                    norm=mpl.colors.LogNorm() if log else None)

	ax.autoscale(tight=True)  # get rid of whitespace in margins of heatmap
	ax.set_aspect('equal')  # ensure heatmap cells are square
	ax.xaxis.set_ticks_position('top')  # put column labels at the top
	ax.tick_params(bottom='off', top='off', left='off', right='off')  # turn off ticks

	plt.yticks(np.arange(len(df.index)) + 0.5, df.index)
	plt.xticks(np.arange(len(df.columns)) + 0.5, df.columns, rotation=90)

	# ugliness from http://matplotlib.org/users/tight_layout_guide.html
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", "3%", pad="1%")
	plt.colorbar(heatmap, cax=cax)