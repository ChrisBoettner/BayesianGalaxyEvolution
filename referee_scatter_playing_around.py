#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:41:12 2022

@author: chris
"""
# %%
from model.interface import load_model, run_model, save_model
from model.plotting.plotting import *
from model.analysis.parameter import tabulate_parameter

import warnings

warnings.filterwarnings("ignore")

# %%
mstar = load_model("mstar", "stellar_blackhole")
muv = load_model("Muv", "stellar_blackhole")
mbh = load_model("mbh", "quasar")
lbol = load_model("Lbol", "eddington")


# %%
mh_space = np.linspace(9, 15, 100)
plot_limits = {
    "top": 0.965,
    "bottom": 0.175,
    "left": 0.135,
    "right": 0.995,
    "hspace": 0.0,
    "wspace": 0.0,
}
scatter = 0.5

# %%
models = {}

for ref, redshift in zip([mstar, muv, mbh], [(0, 1, 2), (3, 4, 5), (0, 1, 2)]):
    model = run_model(
        ref.quantity_name,
        ref.physics_name,
        fitting_method="annealing",
        scatter_name="lognormal",
        scatter_parameter=scatter,
        fixed_m_c=True if ref.quantity_name == "mbh" else False,
        # redshift=list(redshift),
    )
    models[ref.quantity_name] = model

    with_scatter = []
    wo_scatter = []

    for z in redshift:
        with_scatter.append(
            model.physics_model.at_z(z).calculate_log_quantity(
                mh_space, *model.parameter.at_z(z)
            )
        )
        wo_scatter.append(
            ref.physics_model.at_z(z).calculate_log_quantity(
                mh_space, *ref.parameter.at_z(z)
            )
        )

    # general plotting configuration
    fig, ax = plt.subplots(1, 1)
    fig.subplots_adjust(**plot_limits)

    # add axes labels
    ax.set_xlabel(r"log $M_\mathrm{h}$ [$M_\odot$]")
    y_label = "log " + model.quantity_options["quantity_name_tex"]
    ax.set_ylabel(y_label, x=0.01)

    washed_out_color = blend_color("C3", 0.2)
    cm1 = LinearSegmentedColormap.from_list(
        "Custom", [washed_out_color, "C3"], N=len(redshift)
    )
    cm1 = [cm1(number) for number in np.linspace(0, 1, len(redshift))]
    cm2 = LinearSegmentedColormap.from_list(
        "Custom", ["lightgrey", "grey"], N=len(redshift)
    )
    cm2 = [cm2(number) for number in np.linspace(0, 1, len(redshift))]

    for i, z in enumerate(redshift):
        ax.plot(
            mh_space,
            wo_scatter[i],
            color=cm2[i],
            label=" " * (i + 1),
            linewidth=4,
            linestyle="--",
        )
    for i, z in enumerate(redshift):
        ax.plot(
            mh_space,
            with_scatter[i],
            color=cm1[i],
            label=f"z={z}",
            linewidth=4,
        )

    # add axis limits
    ax.set_xlim((mh_space[0], mh_space[-1]))

    # add legend and minor ticks
    ax.minorticks_on()
    # add legend
    add_legend(ax, 0, fontsize=32, loc="lower right", ncol=2)

    plt.savefig(f"scatter_comparison_{ref.quantity_name}_{scatter}.pdf")

# %%
stellar_table = tabulate_parameter(
    [models["mstar"], models["Muv"]],
    use_best_fit=True,
    caption="Parameter for stellar properties with a scatter of 0.2 dex.",
)
bh_table = tabulate_parameter(
    [models["mbh"]],
    use_best_fit=True,
    caption="Parameter for black hole properties with a scatter of 0.2 dex.",
)

# %%
