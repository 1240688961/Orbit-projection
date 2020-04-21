# coding=utf-8
#######################################################
# Input file: EIGENVAL PROCAR CONTCAR KPOINTS
# Output file: band.eps
# Date : 2019
# Author : yan0746
####################################
# Explanation of each variable
# Startband, endband means which bands are needed in the ordinate range of the drawing
# Atom and orbit means which atom and orbit need to be drawn, the current program is inefficient,
# reading once PROCAR can only draw one band one orbit corresponding to one atom,
# maybe it will improve efficiency in the future, but I don’t have time to correct it now
# Bands means endband - startband
# Delline is  an array representing repeated k points in plot band,need to be delline
# Band_up,band_down means the ordinate when drawing the band, spin-up spin-down,respectively
# Compress_radio means for the K path of the band, the scaling factor of the abscissa of different paths
# Procar_axis_x  refers to the abscissa after compression, and then the coordinates after taking the difference
# to ensure that the point density of each k path is as consistent as possible.
########################################################
# !/usr/bin/python
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import os

################# Input the date
reference_level = -2.6880
kpath = 'GXSYG'
y_axis_min = -5
y_axis_max = 3


####################################
def get_band_range_up(ycoordinate_min, ycoordinate_max, ref_level):
    ycoor_max = float(ycoordinate_max + ref_level)
    ycoor_min = float(ycoordinate_min + ref_level)
    band_max = []
    band_min = []
    # Get the data from EIGENVAL
    rf = open('EIGENVAL', 'r')
    for i in range(0, 5):  # Skip the beginning of EIGENVAL 5 lines
        rf.readline()
    # Read the bands kpoints
    str1 = rf.readline()
    bands = int(str1.split()[2])
    kpoints = int(str1.split()[1])
    rf.readline()  # Skip the line
    # Read the band data
    for i in range(0, kpoints):
        rf.readline()  # Skip the line
        band_range = []
        for j in range(0, bands):
            str2 = rf.readline()
            band_data = float(str2.split()[1])
            if ycoor_max >= band_data >= ycoor_min:
                band_range.append(int(str2.split()[0]))
        rf.readline()  # Skip the line
        if band_range:              # Determine whether the array is empty, this is sometimes empty
            band_max.append(max(band_range))
            band_min.append(min(band_range))
    if band_max:              # Determine whether the array is empty, this is sometimes empty
        band_range_min = min(band_min)
        band_range_max = max(band_max)
    else:
        band_range_min = 1
        band_range_max = bands
    return band_range_min, band_range_max


def get_band_range_down(ycoordinate_min, ycoordinate_max, ref_level):
    ycoor_max = float(ycoordinate_max + ref_level)
    ycoor_min = float(ycoordinate_min + ref_level)
    band_max = []
    band_min = []
    # Get the data from EIGENVAL
    rf = open('EIGENVAL', 'r')
    for i in range(0, 5):  # Skip the beginning of EIGENVAL 5 lines
        rf.readline()
    # Read the bands kpoints
    str1 = rf.readline()
    bands = int(str1.split()[2])
    kpoints = int(str1.split()[1])
    rf.readline()  # Skip the line
    # Read the band data
    for i in range(0, kpoints):
        rf.readline()  # Skip the line
        band_range = []
        for j in range(0, bands):
            str2 = rf.readline()
            band_data = float(str2.split()[2])
            if ycoor_max >= band_data >= ycoor_min:
                band_range.append(int(str2.split()[0]))
        rf.readline()  # Skip the line
        if band_range:              # Determine whether the array is empty, this is sometimes empty
            band_max.append(max(band_range))
            band_min.append(min(band_range))
    if band_max:              # Determine whether the array is empty, this is sometimes empty
        band_range_min = min(band_min)
        band_range_max = max(band_max)
    else:
        band_range_min = 1
        band_range_max = bands
    return band_range_min, band_range_max


def get_small_PROCAR(startband, endband):
    # Open the file and read the cooresponding parameters
    rf = open('PROCAR', 'r')
    wf = open('newprocar', 'w')
    wf.write(rf.readline())
    str1 = rf.readline()
    kpoint = int(str1.split()[3])
    oldbands = int(str1.split()[7])
    ions = int(str1.split()[11])
    newbands = int(endband - startband + 1)
    wf.write(str1.replace(str(oldbands), str(newbands)))
    wf.write(rf.readline())  # Skip the black line
    # Start to officially do the file
    for j in range(0, kpoint):
        for i in range(0, 2):
            wf.write(rf.readline())
        # Skip the begining of the band
        for k in range(0, (startband - 1) * (ions + 5)):
            rf.readline()
        # Write the required band
        for l in range(0, newbands * (ions + 5)):
            wf.write(rf.readline())
        # Skip the ending of the band
        for m in range(0, (oldbands - (startband - 1) - newbands) * (ions + 5)):
            rf.readline()
        wf.write(rf.readline())
    # Start to write the next spin file
    wf.write(rf.readline())
    # Start do the k-point cycle
    for j in range(0, kpoint):
        for i in range(0, 2):
            wf.write(rf.readline())
        # Skip the begining of the band
        for k in range(0, (startband - 1) * (ions + 5)):
            rf.readline()
        # Write the required band
        for l in range(0, newbands * (ions + 5)):
            wf.write(rf.readline())
        # Skip the ending of the band
        for m in range(0, (oldbands - (startband - 1) - newbands) * (ions + 5)):
            rf.readline()
        wf.write(rf.readline())
    rf.close()
    wf.close()


def get_small_EIGENVAL(startband, endband):
    # Open the file and read thhe cooresponding parameters
    rf = open('EIGENVAL', 'r')
    wf = open('neweigenval', 'w')
    for i in range(0, 5):  # Skip the begin EIGENVAL
        wf.write(rf.readline())
    # Read the bands and kpints
    str3 = rf.readline()
    oldbands = int(str3.split()[2])
    newbands = int(endband - startband + 1)
    kpointse = int(str3.split()[1])
    wf.write(str3.replace(str(oldbands), str(newbands)))
    wf.write(rf.readline())  # Skip the  black line

    ######Start to officially do the file
    for j in range(0, kpointse):
        for i in range(0, oldbands):
            wf.write(rf.readline())  # Skip the start line
            # Skip the start no need line
            for m in range(0, startband - 1):
                rf.readline()
            # Write the middle need line
            for m in range(0, newbands):
                wf.write(rf.readline())
            # Skip the ending of the band
            for m in range(0, oldbands - (startband - 1) - newbands):
                rf.readline()
            wf.write(rf.readline())  # Skip the end line
    rf.close()
    wf.close()


def get_cbm_vbm(band_up, band_down):
    cbm_all = []
    vbm_all = []
    for i in range(band_up.shape[0]):
        for j in range(band_up.shape[1]):
            if band_up[i, j] > 0:
                cbm_all.append(band_up[i, j])
            else:
                vbm_all.append(band_up[i, j])

    for i in range(band_down.shape[0]):
        for j in range(band_down.shape[1]):
            if band_down[i, j] > 0:
                cbm_all.append(band_down[i, j])
            else:
                vbm_all.append(band_down[i, j])

    return min(cbm_all), max(vbm_all)


def read_EIGENVAL(ref_level, filename):
    # Get the data from EIGENVAl
    rf = open(filename, 'r')
    for i in range(0, 5):  # Skip the beginning of EIGENVAL 5 lines
        rf.readline()
    # Read the bands kpoints
    str1 = rf.readline()
    bands = int(str1.split()[2])
    kpoints = int(str1.split()[1])
    rf.readline()  # Skip the line
    # Read the band data
    axis_up = [[0 for i in range(kpoints)] for i in range(bands)]
    axis_down = [[0 for i in range(kpoints)] for i in range(bands)]
    for i in range(0, kpoints):
        rf.readline()  # Skip the blank line
        for j in range(0, bands):
            str2 = rf.readline()
            axis_up[j][i] = float(str2.split()[1]) - ref_level
            axis_down[j][i] = float(str2.split()[2]) - ref_level
        rf.readline()  # Skip the black line
    rf.close()
    axis_up = np.array(axis_up)
    axis_down = np.array(axis_down)
    # Delete duplicate data
    delline = []
    for i in range(0, kpoints - 1):
        aline = axis_up[0, i]
        bline = axis_up[0, i + 1]
        if aline == bline:
            delline.append(i)
    delline.reverse()
    for j in delline:
        axis_up = np.delete(axis_up, j, 1)
        axis_down = np.delete(axis_down, j, 1)
    kpoints = kpoints - len(delline)
    return bands, kpoints, axis_up, axis_down, delline


def read_PROCAR_up(atom, orbit):
    rf = open('newprocar', 'r')
    # Get the bands kpoints ions
    rf.readline()
    str1 = rf.readline()
    pro_kpoints = int(str1.split()[3])
    pro_bands = int(str1.split()[7])
    pro_ions = int(str1.split()[11])
    rf.readline()

    # Get the data
    pro_data = [[0 for i in range(pro_kpoints)] for i in range(pro_bands)]
    for i in range(0, pro_kpoints):
        for m in range(0, 2):  # Skip the line k-point
            rf.readline()
        for j in range(0, pro_bands):
            for m in range(0, atom + 2):  # Skip the line-start
                rf.readline()
            str2 = rf.readline()
            pro_data[j][i] = float(str2.split()[orbit])
            for n in range(0, (pro_ions + 5) - (atom + 2) - 1):  # Skip the line-end
                rf.readline()
        rf.readline()
    rf.close()
    pro_data = np.array(pro_data)
    # Delete duplicate data
    for j in delline:
        pro_data = np.delete(pro_data, j, 1)
    return pro_data


def read_PROCAR_down(atom, orbit):
    rf = open('newprocar', 'r')
    # Get the bands kpoints ions
    rf.readline()
    str1 = rf.readline()
    pro_kpoints = int(str1.split()[3])
    pro_bands = int(str1.split()[7])
    pro_ions = int(str1.split()[11])
    rf.readline()

    for i in range(0, pro_kpoints * (pro_bands * (pro_ions + 5) + 3) + 1):  # Skip the ispin-bands
        rf.readline()

    # Get the data
    pro_data = [[0 for i in range(pro_kpoints)] for i in range(pro_bands)]
    for i in range(0, pro_kpoints):
        for m in range(0, 2):  # Skip the line k-point
            rf.readline()
        for j in range(0, pro_bands):
            for m in range(0, atom + 2):  # Skip the line-start
                rf.readline()
            str2 = rf.readline()
            pro_data[j][i] = float(str2.split()[orbit])
            for n in range(0, (pro_ions + 5) - (atom + 2) - 1):  # Skip the line-end
                rf.readline()
        rf.readline()
    rf.close()
    pro_data = np.array(pro_data)
    # Delete duplicate data
    for j in delline:
        pro_data = np.delete(pro_data, j, 1)
    return pro_data


def get_new_axis_x(delline):
    # Need CONTCAR and KPOINTS
    rf = open('CONTCAR', 'r')
    lines = rf.readlines()
    rf.close()
    # Convert lattice constants in real space to inverted space
    real_lattice_x = np.array(
        [abs(float(lines[2].split()[0])), abs(float(lines[2].split()[1])), abs(float(lines[2].split()[2]))])
    real_lattice_y = np.array(
        [abs(float(lines[3].split()[0])), abs(float(lines[3].split()[1])), abs(float(lines[3].split()[2]))])
    real_lattice_z = np.array(
        [abs(float(lines[4].split()[0])), abs(float(lines[4].split()[1])), abs(float(lines[4].split()[2]))])
    volume_xyz = np.dot(np.cross(real_lattice_x, real_lattice_y), real_lattice_z)
    down_lattice_x = 2 * math.pi * np.cross(real_lattice_y, real_lattice_z) / volume_xyz
    down_lattice_y = 2 * math.pi * np.cross(real_lattice_x, real_lattice_z) / volume_xyz
    down_lattice_z = 2 * math.pi * np.cross(real_lattice_x, real_lattice_y) / volume_xyz
    rf = open('KPOINTS', 'r')
    kpt_lines = rf.readlines()
    rf.close()

    def down_distance_calculation_kpt(line_i, line_next_i):
        # Calculate distance of points in KPOINTS file
        dot_x = float(kpt_lines[line_i].split()[0]) - float(kpt_lines[line_next_i].split()[0])
        dot_y = float(kpt_lines[line_i].split()[1]) - float(kpt_lines[line_next_i].split()[1])
        dot_z = float(kpt_lines[line_i].split()[2]) - float(kpt_lines[line_next_i].split()[2])
        dot_i = np.array([dot_x, dot_y, dot_z])
        down_dot_x = np.dot(dot_i, [down_lattice_x[0], down_lattice_y[0], down_lattice_z[0]])
        down_dot_y = np.dot(dot_i, [down_lattice_x[1], down_lattice_y[1], down_lattice_z[1]])
        down_dot_z = np.dot(dot_i, [down_lattice_x[2], down_lattice_y[2], down_lattice_z[2]])
        down_dot_i = np.array([down_dot_x, down_dot_y, down_dot_z])
        distance = math.sqrt(np.dot(down_dot_i, down_dot_i))
        return distance

    # Making coordinate axes after proportional compression
    new_axis_x = []
    compression_radio = []
    insert_point = int(kpt_lines[1].split()[0]) - 1
    num_axis_x = float(0)
    new_axis_x.append(num_axis_x)
    for i in range(0, len(delline) + 1):
        k = i * 3 + 4
        compression_radio.append(down_distance_calculation_kpt(k, k + 1))
        for j in range(0, insert_point):
            num_axis_x = num_axis_x + down_distance_calculation_kpt(k, k + 1)
            new_axis_x.append(num_axis_x)
    # Get the K-path point density of each segment
    int_compress_radio = []
    for i in compression_radio:
        int_compress_radio.append(int(max(compression_radio) // i))

    return new_axis_x, int_compress_radio, insert_point


def get_plot_procar_axis_x(compress_radio, kpt_insert_point, list_axis_x, axis_x):
    # Get the abscissa after indenting when drawing PROCAR
    procar_axis_x = []
    k = int(0)
    for i in compress_radio:
        for j in range(0, kpt_insert_point):
            if j * i <= kpt_insert_point:
                procar_axis_x.append(list_axis_x[j * i + k * kpt_insert_point][1])
        k = k + 1
    procar_axis_x.append(axis_x[-1])

    return procar_axis_x


def get_plot_procar_axis_y_up(atom, orbit, procar_axis_x, bands, compress_radio, kpt_insert_point):
    old_procar_multiplied_up = read_PROCAR_up(atom, orbit)
    procar_multiplied_up = [[0 for i in range(len(procar_axis_x))] for i in range(bands)]
    procar_axis_y_up = [[0 for i in range(len(procar_axis_x))] for i in range(bands)]
    k = int(0)  # The k means how many segments there are
    n = int(0)  # The n means how many k points are there

    for i in compress_radio:
        # A certain segment kpoint
        for j in range(0, kpt_insert_point):
            if j * i <= kpt_insert_point:
                for m in range(0, bands):  # The m is the each band
                    procar_multiplied_up[m][n] = old_procar_multiplied_up[m, j * i + k * kpt_insert_point]
                    procar_axis_y_up[m][n] = axis_up[m, j * i + k * kpt_insert_point]
                n = n + 1
        k = k + 1

    # Add the data from the last row
    for m in range(0, bands):
        procar_multiplied_up[m][-1] = old_procar_multiplied_up[m, -1]
        procar_axis_y_up[m][-1] = axis_up[m, -1]

    procar_axis_y_up = np.array(procar_axis_y_up)
    procar_multiplied_up = np.array(procar_multiplied_up)

    return procar_axis_y_up, procar_multiplied_up


def get_plot_procar_axis_y_down(atom, orbit, procar_axis_x, bands, compress_radio, kpt_insert_point):
    old_procar_multiplied_down = read_PROCAR_down(atom, orbit)
    procar_multiplied_down = [[0 for i in range(len(procar_axis_x))] for i in range(bands)]
    procar_axis_y_down = [[0 for i in range(len(procar_axis_x))] for i in range(bands)]
    k = int(0)  # The k means how many segments there are
    n = int(0)  # The n means how many k points are there

    for i in compress_radio:
        # A certain segment kpoint
        for j in range(0, kpt_insert_point):
            if j * i <= kpt_insert_point:
                for m in range(0, bands):  # The m is the each band
                    procar_multiplied_down[m][n] = old_procar_multiplied_down[m, j * i + k * kpt_insert_point]
                    procar_axis_y_down[m][n] = axis_down[m, j * i + k * kpt_insert_point]
                n = n + 1
        k = k + 1

    # Add the data from the last row
    for m in range(0, bands):
        procar_multiplied_down[m][-1] = old_procar_multiplied_down[m, -1]
        procar_axis_y_down[m][-1] = axis_down[m, -1]

    procar_axis_y_down = np.array(procar_axis_y_down)
    procar_multiplied_down = np.array(procar_multiplied_down)

    return procar_axis_y_down, procar_multiplied_down


def band_plot_up(bands, axis_x, axis_y_up, delline):
    # Plot bands
    for i in range(0, bands):
        plt.plot(axis_x, axis_y_up[i, :], color='red', linewidth=1.0, linestyle="-")
    # Plot dished lines
    wdish = len(delline)
    for i in delline:
        wdish = wdish - 1
        ydish = np.linspace(-100, 100, 100)
        xdish = 0 * ydish + list_axis_x[i - wdish][1]
        plt.plot(xdish, ydish, color='black', linewidth=0.8, linestyle="--")
    # Plot the horizontal coordinate
    if len(delline) + 2 == len(kpath):
        xlocs = []
        xlocs.append(float(0))  # Get initial position
        # Get middle series position
        xloc_get = []
        xloc1 = len(delline)
        for i in delline:
            xloc1 = xloc1 - 1
            xloc_get.append(i - xloc1)
        xloc_get.reverse()
        for j in xloc_get:
            xlocs.append(list_axis_x[j][1])
        xlocs.append(max(axis_x))  # Get the last position
        plt.xticks(xlocs, kpath, size=25)


def band_plot_down(bands, axis_x, axis_y_down):
    # Plot bands
    for i in range(0, bands):
        plt.plot(axis_x, axis_y_down[i, :], color='black', linewidth=1.0, linestyle="-")


def procar_plt_up(ion, orb, mult, dotcolor):
    pro_axis_up, pro_data_up = get_plot_procar_axis_y_up(ion, orb, procar_axis_x, bands, compress_radio, kpt_insert_point)
    for i in range(0, len(procar_axis_x)):
        # Plot the kpoints
        for j in range(0, bands):
            # Plot the bands
            ax.scatter(procar_axis_x[i], pro_axis_up[j, i], s=pro_data_up[j, i] * mult, c=dotcolor, edgecolor='none',
                       alpha=1.0)


def procar_plt_down(ion, orb, mult, dotcolor):
    pro_axis_down, pro_data_down = get_plot_procar_axis_y_down(ion, orb, procar_axis_x, bands, compress_radio, kpt_insert_point)
    for i in range(0, len(procar_axis_x)):
        # Plot the kpoints
        for j in range(0, bands):
            # Plot the bands
            ax.scatter(procar_axis_x[i], pro_axis_down[j, i], s=pro_data_down[j, i] * mult, c=dotcolor,
                       edgecolor='none', alpha=1.0)


def blend_tow_images(figure1, figure2):
    img1 = Image.open(figure1)
    img1_rgba = img1.convert('RGBA')
    img2 = Image.open(figure2)
    img2_rgba = img2.convert('RGBA')
    img_merge = Image.blend(img1_rgba, img2_rgba, 0.5)
    img_merge.save("blend.png")
    return


if __name__ == '__main__':
    temp_bands, temp_kpoints, temp_axis_up, temp_axis_down, temp_delline = read_EIGENVAL(reference_level, 'EIGENVAL')  # Get temporary bands data
    cbm, vbm = get_cbm_vbm(temp_axis_up, temp_axis_down)  # Get the vbm and cbm
    startband_up, endband_up = get_band_range_up(y_axis_min, y_axis_max, reference_level)  # Get the required bands range from the drawing
    startband_down, endband_down = get_band_range_down(y_axis_min, y_axis_max, reference_level)
    startband = min(startband_up, startband_down)
    endband = max(endband_up, endband_down)
    get_small_EIGENVAL(startband, endband)  # Get minified EIGENVAL file
    get_small_PROCAR(startband, endband)  # Get minified PROCAR file
    bands, kpoints, axis_up, axis_down, delline = read_EIGENVAL(reference_level, 'neweigenval')  # Get bands data
    axis_x, compress_radio, kpt_insert_point = get_new_axis_x(delline)  # Get reduced abscissa data
    list_axis_x = list(enumerate(axis_x))
    ######### Plot the band
    fig = plt.figure(figsize=(8, 10))  # Canvas initialization and set size

    # Plot the procar_data
    ax = fig.add_subplot(111)
    procar_axis_x = get_plot_procar_axis_x(compress_radio, kpt_insert_point, list_axis_x, axis_x)

    band_plot_up(bands, axis_x, axis_up, delline)     # Plot the band up
    band_plot_down(bands, axis_x, axis_down)          # Plot the band down
    # blue c green black magenta red white yellow
    dict1 = {'s': 1, 'py': 2, 'pz': 3, 'px': 4, 'dxy': 5, 'dyz': 6, 'dz2': 7, 'dxz': 8, 'dx2-y2': 9, 'tot': 10}
    # Plot the need orbit and atoms
    i = 1
    procar_plt_up(i, dict1['dz2'], 1000, 'red')
    procar_plt_down(i, dict1['dz2'], 1000, 'skyblue')

    # Set the range of abscissa and ordinate
    plt.xlim(min(axis_x), max(axis_x))
    plt.ylim(y_axis_min, y_axis_max)
    plt.yticks(np.arange(y_axis_min, y_axis_max, 0.2), size=25)
    # Set the xlabel and xylabel  and title
    # plt.ylabel("E-E_VBM(eV)", size=25)

    # Set the legend
    ax.scatter(10000, 10000, s=240, c='red', edgecolor='none', alpha=1.0, label='Ni-dxz_up')
    ax.scatter(10000, 10000, s=240, c='skyblue', edgecolor='none', alpha=1.0, label='Ni-dxz_down')

    font1 = {'weight': 'normal', 'size': 20}
    legend = plt.legend(prop=font1)
    plt.savefig('band.eps')
    # os.system("convert -density 288 band.eps band.png")
    plt.show()

