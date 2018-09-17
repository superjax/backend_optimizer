import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


if __name__ == "__main__":

    f = open('data1.txt', 'r')
    init_data = json.load(f)
    f.close()
    init_nodes = np.array(init_data['poses'])
    init_lc = init_data['loop_closures']

    g = open("optimization_data.txt", 'r')
    opt_data = json.load(g)
    g.close()
    reo_nodes = np.array(opt_data['REO_opt'])
    gpo_nodes = np.array(opt_data['GPO_opt'])


    # gs1 = gridspec.GridSpec(3, 1)
    # gs1.update(wspace = 0.005, hspace=0.005)
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, sharey=True, figsize=(6, 4))

    # ax1.plot(init_nodes[:, 1], init_nodes[:, 2], 'b')
    ax1.plot(init_nodes[:, 1], init_nodes[:, 2], 'b')
    for i, loop in enumerate(init_lc):
        ax1.plot(init_nodes[loop, 1], init_nodes[loop, 2], 'r')  # plot the loop closures
    ax1.set_title('Initial Data')
    ax1.axis('square')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    ax2.plot(reo_nodes[0, :], reo_nodes[1, :], label='REO', color='b')
    for i, loop in enumerate(init_lc):
        ax2.plot(reo_nodes[0, loop], reo_nodes[1, loop], 'r')  # plot the loop closures.
    ax2.set_title('REO')
    ax2.axis('square')
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.legend(['Path', 'Loop closures'], loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=False,
               shadow=False, ncol=2, prop={'size': 12})

    ax3.plot(gpo_nodes[0, :], gpo_nodes[1, :], label='GPO', color='b')
    for i, loop in enumerate(init_lc):
        ax3.plot(gpo_nodes[0, loop], gpo_nodes[1, loop], 'r')  # plot the loop closures.
    ax3.set_title('GPO')
    ax3.axis('square')
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])

    plt.subplots_adjust(wspace = 0, hspace=0)

    plt.savefig("tests/well_conditioned/plots_hw/combined_single.eps", bbox_inches='tight', format='eps', pad_inches=0)


    plt.show()

    # Now for multi-agent plots

    f2 = open('data2.txt', 'r')
    init_data = json.load(f2)
    f2.close()
    init_nodes2_1 = np.array(init_data['poses1'])  # will need to fix in the other script
    init_nodes2_2 = np.array(init_data['poses2'])
    init_lc2_1 = init_data['lc1']
    init_lc2_2 = init_data['lc2']
    init_lc2 = init_data['lc']

    t = len(init_nodes2_1)
    l = 460

    g2 = open("tests/well_conditioned/optimization_data2.txt", 'r')
    opt_data2 = json.load(g2)
    g2.close()
    reo_nodes2 = np.array(opt_data2['REO_opt'])
    gpo_nodes2 = np.array(opt_data2['GPO_opt'])

    fig2, ((ax4, ax5, ax6)) = plt.subplots(1, 3, sharex=True, figsize=(6, 4))
    # fig2, ((ax4)) = plt.subplots(1, 1, sharex=True, figsize=(6, 4))

    ax4.plot(init_nodes2_1[:, 1], init_nodes2_1[:, 2], 'b')
    ax4.plot(init_nodes2_2[:, 1], init_nodes2_2[:, 2], 'k')
    for i, loop in enumerate(init_lc2_1):
        ax4.plot(init_nodes2_1[loop, 1], init_nodes2_1[loop, 2], 'r')  # plot the loop closures

    for i, loop in enumerate(init_lc2_2):
        if loop[0] < t:
            loop[1] -= l
            ax4.plot([init_nodes2_1[loop[0], 1], init_nodes2_2[loop[1], 1]], [init_nodes2_1[loop[0], 2], init_nodes2_2[loop[1], 2]],
                     color='r')  # plot the loop closures
        elif loop[1] < t:
            loop[0] -= l
            ax4.plot([init_nodes2_2[loop[0], 1], init_nodes2_1[loop[1], 1]], [init_nodes2_2[loop[0], 2], init_nodes2_1[loop[1], 2]],
                     color='r')  # plot the loop closures
        else:
            ax4.plot(init_nodes2_2[loop, 1], init_nodes2_2[loop, 2], 'r')  # plot the loop closures

    ax4.set_title('Initial Data')
    ax4.set_xlim([-25, 25])
    ax4.axis('square')
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])

    ax5.plot(reo_nodes2[0, :], reo_nodes2[1, :], label='REO', color='b')
    for i, loop in enumerate(init_lc2):
        ax5.plot(reo_nodes2[0, loop], reo_nodes2[1, loop], 'r')  # plot the loop closures.
    ax5.set_title('REO')
    ax5.axis('square')
    ax5.set_xlim([-25, 25])
    ax5.set_xticklabels([])
    ax5.set_yticklabels([])
    ax5.legend(['Path', 'Loop closures'], loc='upper center', bbox_to_anchor=(0.5, -0.5), fancybox=False,
               shadow=False, ncol=2, prop={'size': 12})

    ax6.plot(gpo_nodes2[0, :], gpo_nodes2[1, :], label='GPO', color='b')
    for i, loop in enumerate(init_lc2):
        ax6.plot(gpo_nodes2[0, loop], gpo_nodes2[1, loop], 'r')  # plot the loop closures.
    ax6.set_title('GPO')
    ax6.axis('square')
    ax5.set_xlim([-25, 25])
    ax6.set_xticklabels([])
    ax6.set_yticklabels([])

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig("tests/well_conditioned/plots_hw/combined_multi.eps", bbox_inches='tight', format='eps', pad_inches=0)

    plt.show()
