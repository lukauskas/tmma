from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from tmma import two_sample_tmm, ma_statistics, tmm_trim_mask, asymptotic_variance
import numpy as np

def plot_two_sample_tmm(obs,
                        ref,
                        kw_tmm=None,
                        min_scatter_size=4,
                        max_scatter_size=100,
                        color_tmm_points=None,
                        color_other_points='black',
                        color_sf_estimate=None,
                        xlabel=None,
                        ylabel=None,
                        title=None,
                        show_legend=True,
                        ax=None,
                        kw_scatter=None,
                        kw_sf_estimate=None):
    """
    Plots two sample TMM normalisation

    :param obs: counts observed
    :param ref: counts reference
    :param kw_tmm: Keyword arguments for tmm normalisation, see `two_sample_tmm`
    :param min_scatter_size: Minimum scatterplot point size
    :param max_scatter_size: Maximum scatterplot point size.
                             If TMM is weighted, points will be sized by their weights.
    :param color_tmm_points: Colour of points used for TMM, if none will default
                             to next coolur in matplotlib cycle
    :param color_other_points: Colour of points excluded from TMM
    :param color_sf_estimate: Colour of the line for scaling factor estimate
    :param xlabel: label of x axis
    :param ylabel: label of y axis
    :param title: title of the plot
    :param show_legend: Whether to show legend or not
    :param ax: Axis to plot on, defaults to `plt.gca()`
    :param kw_scatter: other scatterplot kwargs. Defaults to `dict(alpha=0.1, rasterized=True)`
    :param kw_sf_estimate: other kwargs for the scaling factor estimate line
    :return: axis with the plot
    """
    if kw_tmm is None:
        kw_tmm = {}

    lib_size_obs = kw_tmm.pop('lib_size_obs', None)
    lib_size_ref = kw_tmm.pop('lib_size_ref', None)
    weighted = kw_tmm.pop('weighted', True)

    sf = two_sample_tmm(obs, ref,
                        lib_size_obs=lib_size_obs,
                        lib_size_ref=lib_size_ref,
                        weighted=weighted,
                        **kw_tmm)

    m_values, a_values = ma_statistics(obs, ref,
                                       lib_size_obs=lib_size_obs,
                                       lib_size_ref=lib_size_ref)

    kept_after_trim = tmm_trim_mask(m_values, a_values, **kw_tmm)

    if weighted:
        weights = 1.0 / asymptotic_variance(obs, ref,
                                            lib_size_obs=lib_size_obs,
                                            lib_size_ref=lib_size_ref)

        size_function = interp1d([weights[kept_after_trim].min(),
                                  weights[kept_after_trim].max()],
                                 [min_scatter_size, max_scatter_size],
                                 bounds_error=False,
                                 fill_value='extrapolate'
                                 )

        sizes = size_function(weights)
        sizes[~kept_after_trim] = min_scatter_size
    else:
        sizes = np.repeat(min_scatter_size, len(m_values))

    # -- Plotting ---------------------
    if ax is None:
        ax = plt.gca()

    if kw_scatter is None:
        kw_scatter = {}
    if kw_sf_estimate is None:
        kw_sf_estimate = {}

    kw_scatter.setdefault('alpha', 0.1)
    kw_scatter.setdefault('rasterized', True)

    kw_sf_estimate.setdefault('linewidth', 1.0)
    kw_sf_estimate.setdefault('label', r"Estimated $sf \approx {:.3f}$".format(sf))

    if color_tmm_points is None:
        color_tmm_points = next(ax._get_lines.prop_cycler)['color']
    if color_sf_estimate is None:
        color_sf_estimate = next(ax._get_lines.prop_cycler)['color']

    try:
        obs_name = obs.name
    except AttributeError:
        obs_name = 'obs'

    try:
        ref_name = ref.name
    except AttributeError:
        ref_name = 'ref'

    if xlabel is None:
        xlabel = '\n'.join([
            r'A (average $\log_2$ signal):',
            r'$\frac{{1}}{{2}} \left( \log_2( \mathrm{{ {obs} }} ) + \log_2( \mathrm{{ {ref} }} ) \right)$'.format(
                obs=obs_name, ref=ref_name)
        ])

    if ylabel is None:
        ylabel = '\n'.join([
            r'M ($\log_2$ fold change):',
            r'$\log_2( \mathrm{{ {obs} }} ) - \log_2( \mathrm{{ {ref} }} )$'.format(obs=obs_name,
                                                                                    ref=ref_name)
        ])

    if title is None:
        title = "TMM normalisation {obs} vs. {ref}".format(obs=obs_name, ref=ref_name)

    ax.axhline(0, linestyle=':', color='black',
               label="$x=0$ line")

    ax.scatter(a_values[~kept_after_trim],
               m_values[~kept_after_trim],
               s=sizes[~kept_after_trim],
               color=color_other_points,
               label="Excluded points",
               **kw_scatter
               )

    ax.scatter(a_values[kept_after_trim],
               m_values[kept_after_trim],
               s=sizes[kept_after_trim],
               color=color_tmm_points,
               label="TMM inputs",
               **kw_scatter
               )

    ax.plot([a_values[kept_after_trim].min(),
             a_values[kept_after_trim].max()],
            [np.log2(sf), np.log2(sf)],
            color=color_sf_estimate,
            **kw_sf_estimate)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show_legend:
        ax.legend()

    return ax
