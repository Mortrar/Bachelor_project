import sys
import math
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def noise_matrix_diff(filepath, module):

    files_noise = sorted(glob.glob(filepath+ "scan*/" + module + "_ECS_Scan_Trim0_1550_5_90_1of1_Noise_Mean_decimals.csv"))
    files_mask  = sorted(glob.glob(filepath+ "scan*/" + module + "_Matrix_Mask.csv"))
    #Load Data
    try:
      scan1 = np.loadtxt(files_noise[0], dtype=float, delimiter=',')
      scan2 = np.loadtxt(files_noise[1], dtype=float, delimiter=',')
      mask1 = np.loadtxt(files_mask[0], dtype=int, delimiter=',')
      mask2 = np.loadtxt(files_mask[1], dtype=int, delimiter=',')
      scan1 = np.nan_to_num(scan1)
      scan2 = np.nan_to_num(scan2)
    except:
      print('Failed to load data noise matrix')
      scan1 = np.ones((256, 256))
      scan2 = np.ones((256, 256))
      mask1  = np.ones((256, 256))   
      mask2  = np.ones((256, 256))

    noisemat1 = np.ma.masked_where(mask1>0, scan1)
    noisemat2 = np.ma.masked_where(mask2>0, scan2)

    ### Plot 1 ###
    Plot = plt.figure(figsize=(6,6), facecolor='white')

    cMap = plt.cm.get_cmap("viridis").copy()
    cMap.set_bad('red', 1.0)

    plt.imshow(noisemat1, aspect=1, cmap=cMap, origin='lower')
    plt.axis([-2,257,-2, 257]) # to better visualise the border
    plt.xticks(np.arange(0,257,16), fontsize=8)
    plt.yticks(np.arange(0,257,16), fontsize=8)
    plt.colorbar(fraction = 0.047)
    #plt.axes().set_aspect('equal')  was doing something weird to the axis

    plt.savefig(filepath + module + "/Plot_Matrix_NoiseRatio_scan1.png", bbox_inches='tight', format='png')
    plt.close()

    ### Plot 2 ###
    Plot = plt.figure(figsize=(6,6), facecolor='white')
    plt.imshow(noisemat2, aspect=1, cmap=cMap, origin='lower')
    plt.axis([-2,257,-2, 257]) # to better visualise the border
    plt.xticks(np.arange(0,257,16), fontsize=8)
    plt.yticks(np.arange(0,257,16), fontsize=8)
    plt.colorbar(fraction = 0.047)

    plt.savefig(filepath + module + "/Plot_Matrix_NoiseRatio_scan2.png", bbox_inches='tight', format='png')
    plt.close()

    ### Difference ###
    diffmat = noisemat2 - noisemat1

    mean_diff = np.mean(diffmat)
    std_diff = np.std(diffmat)
    print("Mean diff: ", mean_diff, "\nStandar deviation: ", std_diff)
    if ((i,j) == (0,0)):
      np.savetxt('matrix_diff_0_0.csv',diffmat, delimiter=',')
    


    ### Plot Difference ###
    Plot = plt.figure(figsize=(6,6), facecolor='white')
    plt.imshow(diffmat, aspect=1, cmap= "seismic", origin='lower')
    plt.colorbar(fraction = 0.047)

    plt.savefig(filepath + module + "/Plot_Matrix_NoiseRatio_diff.png", bbox_inches='tight', format='png')
    plt.close()
    
    noise_array = diffmat.flatten()
    Plot = plt.figure(figsize=(8,6), facecolor='white')
    plt.hist(noise_array, bins = 30, range = (-3,3), histtype='step',edgecolor='r',linewidth=3)
    plt.xlabel('Noise mean difference (DAC)')
    plt.ylabel('Number of pixels')
    mytext = "Mean =  %0.1f $\pm$ %0.1f DAC" % (mean_diff, std_diff)
    plt.text(-3, 8000, mytext, size=14, color='black')
    plt.savefig(filepath + module + "/matrix_diff_hist.png", bbox_inches='tight', format='png')
    plt.close()

def mask_matrix_diff(filepath, module):
  masks  = sorted(glob.glob(filepath + "scan*/" + module + "_Matrix_Mask.csv"))

  try:
    mask1 = np.loadtxt(masks[0], dtype=int, delimiter=',')
    mask2 = np.loadtxt(masks[1], dtype=int, delimiter=',')
  except:
    print('Failed to load data mask matrix')
    mask1  = np.ones((256, 256))   
    mask2  = np.ones((256, 256))

  mask1[mask1 > 0] = 1
  mask2[mask2 > 0] = 1
  diff = mask1 + mask2
  diff[diff == 2] = 0
  number = np.count_nonzero(diff)
  print("mask change: " + str(number))
  


  Plot = plt.figure(figsize=(6,6), facecolor='white')
  Plot.suptitle("mask 1", fontsize=20, horizontalalignment='center', verticalalignment='center', color='black')
  plt.imshow(mask1,cmap='Greens',aspect='auto')
  plt.colorbar()
  plt.axis([-2,257,-2, 257]) # to better visualise the border
  plt.xticks(np.arange(0,257,16), fontsize=8)
  plt.yticks(np.arange(0,257,16), fontsize=8) 
  plt.savefig(filepath + module + "/mask 1.png", bbox_inches='tight', format='png')
  plt.close()

  Plot = plt.figure(figsize=(6,6), facecolor='white')
  Plot.suptitle("mask 2", fontsize=20, horizontalalignment='center', verticalalignment='center', color='black') 
  plt.imshow(mask2,cmap='Blues',aspect='auto') 
  plt.colorbar()
  plt.axis([-2,257,-2, 257]) # to better visualise the border
  plt.xticks(np.arange(0,257,16), fontsize=8)
  plt.yticks(np.arange(0,257,16), fontsize=8) 
  plt.savefig(filepath + module + "/mask 2.png", bbox_inches='tight', format='png')
  plt.close()
  
  Plot = plt.figure(figsize=(6,6), facecolor='white')
  Plot.suptitle("Changed masks: " + str(number), fontsize=20, horizontalalignment='center', verticalalignment='center', color='black', y = 0.95) 
  cmap_bin = colors.ListedColormap(['white', 'red'])
  plt.imshow(diff, cmap_bin, aspect='auto')
  plt.axis([-2,257,-2, 257]) # to better visualise the border
  plt.xticks(np.arange(0,257,16), fontsize=8)
  plt.yticks(np.arange(0,257,16), fontsize=8) 
  plt.savefig(filepath + module + "/mask_diff.png", bbox_inches='tight', format='png')
  plt.close()



  
def trim_matrix_diff(filepath, module):

  trims  = sorted(glob.glob(filepath+ "scan*/" + module + "_Matrix_Trim.csv"))
  masks  = sorted(glob.glob(filepath+ "scan*/" + module +  "_Matrix_Mask.csv"))

  try:
    trim1 = np.loadtxt(trims[0], dtype=int, delimiter=',')
    trim2 = np.loadtxt(trims[1], dtype=int, delimiter=',')
    mask1 = np.loadtxt(masks[0], dtype=int, delimiter=',')
    mask2 = np.loadtxt(masks[1], dtype=int, delimiter=',')
  except:
    print('Failed to load data trim matrix')
    trim1  = np.ones((256, 256))   
    trim2  = np.ones((256, 256))
    mask1  = np.ones((256, 256))
    mask2  = np.ones((256, 256))  

  trim1 = np.ma.masked_where(mask1>0, trim1)
  trim2 = np.ma.masked_where(mask2>0, trim2)

  cMap = plt.cm.get_cmap("viridis").copy()
  cMap.set_bad('red', 1.0)
  
  Plot = plt.figure(figsize=(6,6), facecolor='white')
  Plot.suptitle("trim 1", fontsize=20, horizontalalignment='center', verticalalignment='center', color='black') 
  plt.imshow(trim1, aspect=1, cmap=cMap, origin='lower')
  plt.axis([-2,257,-2, 257]) # to better visualise the border
  plt.xticks(np.arange(0,257,16), fontsize=8)
  plt.yticks(np.arange(0,257,16), fontsize=8)
  plt.colorbar()
  plt.savefig(filepath + module + "/Plot_Matrix_Trim_scan1.png", bbox_inches='tight', format='png')
  plt.close()

  Plot = plt.figure(figsize=(6,6), facecolor='white')
  Plot.suptitle("trim 2", fontsize=20, horizontalalignment='center', verticalalignment='center', color='black') 
  plt.imshow(trim2, aspect=1, cmap=cMap, origin='lower')
  plt.axis([-2,257,-2, 257]) # to better visualise the border
  plt.xticks(np.arange(0,257,16), fontsize=8)
  plt.yticks(np.arange(0,257,16), fontsize=8)
  plt.colorbar()
  plt.savefig(filepath + module + "/Plot_Matrix_Trim_scan2.png", bbox_inches='tight', format='png')
  plt.close()

  trim_diff = trim2-trim1
  pos_change = np.argwhere(trim_diff > 0)
  neg_change = np.argwhere(trim_diff < 0)
  print(neg_change[0], " pixel changed\n")
  print(str(round((np.count_nonzero(trim_diff > 0))/(np.count_nonzero(trim_diff > 0) + np.count_nonzero(trim_diff < 0)) * 100,1)) + "positive change")
  print(str(round((np.count_nonzero(trim_diff < 0))/(np.count_nonzero(trim_diff > 0) + np.count_nonzero(trim_diff < 0)) * 100,1)) + "negative change")
  change = str(round((np.count_nonzero(trim_diff != 0))/trim_diff.size * 100,1))

  Plot = plt.figure(figsize=(6,6), facecolor='white')
  Plot.suptitle("changed trims: " + change + "%", fontsize=20, horizontalalignment='center', verticalalignment='bottom', color='black', y = 0.95)
  plt.imshow(trim_diff, aspect=1, cmap = "seismic", origin='lower') 
  plt.axis([-2,257,-2, 257]) # to better visualise the border
  plt.xticks(np.arange(0,257,16), fontsize=8)
  plt.yticks(np.arange(0,257,16), fontsize=8)
  Plot.tight_layout()
  plt.savefig(filepath + module + "/Plot_Matrix_Trim_diff.png", bbox_inches='tight', format='png')
  plt.close()

  #np.savetxt("trim_diff.csv", trim_diff, fmt='%i', delimiter = ",")
  print("%s%%"%"Trim change percentage: ", change)

  return pos_change, neg_change





def pixel_noise_diff(filepath, module):

    files = sorted(glob.glob(filepath+ "scan*/" + module + "_ECS_Scan_Trim0_1550_5_90_1of1_Pixel_1_1.csv"))

    #Load Data
    try:
      scan1 = np.loadtxt(files[0], dtype=float, delimiter=',')
      scan2 = np.loadtxt(files[1], dtype=float, delimiter=',')
      scan1 = np.nan_to_num(scan1)
      scan2 = np.nan_to_num(scan2)
      x_scan1 = scan1[:,0]
      y_scan1 = scan1[:,1]
      x_scan2 = scan2[:,0]
      y_scan2 = scan2[:,1]
    except:
      print('Failed to load data pixel noise')
      scan1 = np.ones((1,1))
      scan1 = np.ones((1,1))

    scan1_array = []
    for i in range(0, len(x_scan1)):
      for j in range(0, int(y_scan1[i])):
        scan1_array.append(x_scan1[i])   #list of noise values repeated by the hit number
    #scan1_hist = np.histogram(scan1_array, bins=90, range=(1100,1550))
    mean1 = np.mean(scan1_array)
    std1  = np.std(scan1_array)
    print("Scan 1:", mean1, std1)

    scan2_array = []
    for i in range(0, len(scan2)):
      for j in range(0, int(y_scan2[i])):
        scan2_array.append(x_scan2[i])
    #scan2_hist = np.histogram(scan2_array, bins=90, range=(1100,1550))
    mean2 = np.mean(scan2_array)
    std2  = np.std(scan2_array)
    print("Scan 2:", mean2, std2)
    print("Ratio_std:", std1/std2)
    print("Discrepancy:", np.abs(mean1-mean2)/mean1, "\n\n")

    ### Plot ###
    Plot = plt.figure(figsize=(8,6), facecolor='white')
    plt.plot(x_scan1, y_scan1, 'ro')
    plt.axis([1100,1550,0,64])
    plt.savefig(filepath + module + "/pixel_scan1.png", bbox_inches='tight', format='png')
    plt.close()
    Hist1 = plt.figure(figsize=(8,6), facecolor='white')
    plt.hist(scan1_array, bins = 90, range = (1100,1550), histtype='step',edgecolor='r',linewidth=3, density = 1)
    plt.savefig(filepath + module + "/pixel_scan1_hist.png", bbox_inches='tight', format='png')
    plt.close()


    Plot = plt.figure(figsize=(8,6), facecolor='white')
    plt.plot(x_scan2, y_scan2)
    plt.plot(x_scan2, y_scan2, 'bo')
    plt.axis([1250,1380,-4,64])
    plt.xlabel("DAC threshold", fontsize = 15)
    plt.ylabel("number of hits", fontsize = 15)
    plt.savefig(filepath + module + "/pixel_scan2.png", bbox_inches='tight', format='png')
    plt.close()

    Plot_comp = plt.figure(figsize=(8,6), facecolor='white')
    plt.plot(x_scan1, y_scan1, 'ro')
    plt.plot(x_scan2, y_scan2, 'bo')
    plt.axis([1100,1550,0,64])
    plt.savefig(filepath + module + "/pixel_scan_comparison.png", bbox_inches='tight', format='png')
    plt.close()
    

def scans_equalizations(filepath, module, pos_change, neg_change):

    dacMin = 1150
    dacMax = 1700

    noise_Trim0 = sorted(glob.glob(filepath+ "scan*/" + module + "_Trim0_Noise_Mean.csv"))
    noise_TrimF = sorted(glob.glob(filepath+ "scan*/" + module + "_TrimF_Noise_Mean.csv"))
    eq_noise    = sorted(glob.glob(filepath+ "scan*/" + module + "_TrimBest_Noise_Predict.csv"))
    files_mask  = sorted(glob.glob(filepath+ "scan*/" + module + "_Matrix_Mask.csv"))
    #Load Data
    try:
      scan1_Trim0 = np.loadtxt(noise_Trim0[0], dtype=float, delimiter=',')
      scan1_TrimF = np.loadtxt(noise_TrimF[0], dtype=float, delimiter=',')
      scan2_Trim0 = np.loadtxt(noise_Trim0[1], dtype=float, delimiter=',')
      scan2_TrimF = np.loadtxt(noise_TrimF[1], dtype=float, delimiter=',')
      scan1_equal = np.loadtxt(eq_noise[0], dtype=float, delimiter=',')
      scan2_equal = np.loadtxt(eq_noise[1], dtype=float, delimiter=',')

      mask1 = np.loadtxt(files_mask[0], dtype=int, delimiter=',')
      mask2 = np.loadtxt(files_mask[1], dtype=int, delimiter=',')
      scan1_Trim0 = np.nan_to_num(scan1_Trim0)
      scan1_TrimF = np.nan_to_num(scan1_TrimF)
      scan2_Trim0 = np.nan_to_num(scan2_Trim0)
      scan2_TrimF = np.nan_to_num(scan2_TrimF)
      
    except:
      print('Failed to load data noise equalization')
      scan1 = np.ones((256, 256))
      scan2 = np.ones((256, 256))
      mask1  = np.ones((256, 256))   
      mask2  = np.ones((256, 256))

    scan1_nMasked = np.count_nonzero(mask1)
    scan2_nMasked = np.count_nonzero(mask2)
    #noisemat1 = np.ma.masked_where(mask1>0, scan1)
    #noisemat2 = np.ma.masked_where(mask2>0, scan2)

    dac_bins = np.arange(dacMin, dacMax+1, 1)

    hist_scan1_Trim0 = np.histogram(scan1_Trim0, bins=dac_bins)
    hist_scan1_TrimF = np.histogram(scan1_TrimF, bins=dac_bins)
    hist_scan2_Trim0 = np.histogram(scan2_Trim0, bins=dac_bins)
    hist_scan2_TrimF = np.histogram(scan2_TrimF, bins=dac_bins)
    hist_scan1_equal = np.histogram(scan1_equal, bins=dac_bins)
    hist_scan2_equal = np.histogram(scan2_equal, bins=dac_bins)
    hist_scan2_changed_pos = np.histogram(scan2_equal[pos_change[:,0], pos_change[:,1]], bins=dac_bins)
    hist_scan2_changed_neg = np.histogram(scan2_equal[neg_change[:,0], neg_change[:,1]], bins=dac_bins)
    hist_scan1_changed_pos = np.histogram(scan1_equal[pos_change[:,0], pos_change[:,1]], bins=dac_bins)


    hist_scan1_max = math.ceil((np.max(hist_scan1_equal[0]) + 1000)/1000)*1000
    hist_scan2_max = math.ceil((np.max(hist_scan2_equal[0]) + 1000)/1000)*1000

    scan1_target = np.mean( 0.5*(scan1_Trim0[(scan1_Trim0>0) & (scan1_TrimF>0)]+scan1_TrimF[(scan1_Trim0>0) & (scan1_TrimF>0)]) )
    scan2_target = np.mean( 0.5*(scan2_Trim0[(scan2_Trim0>0) & (scan2_TrimF>0)]+scan2_TrimF[(scan2_Trim0>0) & (scan2_TrimF>0)]) )

    ### Plot1 ###
    Plot = plt.figure(figsize = (6,6), facecolor = 'white')
    Plot.suptitle("Scan 1 equalization", fontsize=20, horizontalalignment='center', verticalalignment='center', color='black') 

    eps = 0.01
    # convert zeros to small
    logs = hist_scan1_Trim0[0].astype(float)
    logs[logs==0] = eps
    plt.semilogy(hist_scan1_Trim0[1][:-1], logs, color='red', linestyle='-', linewidth=2)
    logs = hist_scan1_TrimF[0].astype(float)
    logs[logs==0] = eps
    plt.semilogy(hist_scan1_TrimF[1][:-1], logs, color='blue', linestyle='-', linewidth=2)
    logs = hist_scan1_equal[0].astype(float)
    logs[logs==0] = eps
    plt.semilogy(hist_scan1_equal[1][:-1], logs, color='black', linestyle='-', linewidth=2)
    # axes
    plt.axis([dacMin,dacMax,0.9,hist_scan1_max])
    plt.xticks(np.arange(dacMin,dacMax+1,50), fontsize=9)
    #plt.subplot(111).xaxis.set_ticks(np.arange(dacMin, dacMax+1,20))  #ERROR, deleted the True argument

    for tick in plt.subplot(111).yaxis.get_major_ticks():
      tick.label.set_fontsize(15)
    plt.subplot(111).axes.tick_params(direction='in', which='both', bottom=True, top=True, left=True, right=True)

    plt.xlabel("DAC Threshold", fontsize=15)
    plt.ylabel("Number of Pixels", fontsize=15)

    # Stats
    mytext = "0 Trim:\n%.1f +/- %.1f" % (np.mean(scan1_Trim0[scan1_Trim0>0]), np.std(scan1_Trim0[scan1_Trim0>0]))
    plt.text(dacMax + 20, math.exp(8), mytext, fontsize=12, horizontalalignment='left', verticalalignment='center', color='red')
    mytext = "F Trim:\n%.1f +/- %.1f" % (np.mean(scan1_TrimF[scan1_TrimF>0]), np.std(scan1_TrimF[scan1_TrimF>0]))
    plt.text(dacMax + 20, math.exp(7), mytext, fontsize=12, horizontalalignment='left', verticalalignment='center', color='blue')
    mytext = "Target:\n%.1f" % scan1_target
    plt.text(dacMax + 20, math.exp(6), mytext, fontsize=12, horizontalalignment='left', verticalalignment='center', color='black')
    mytext = "Predicted:\n%.1f +/- %.1f" % (np.mean(scan1_equal[scan1_equal>0]), np.std(scan1_equal[scan1_equal>0]))
    plt.text(dacMax + 20, math.exp(5), mytext, fontsize=12, horizontalalignment='left', verticalalignment='center', color='black')

    mytext = "Masked:\n%d" % (scan1_nMasked)
    plt.text(dacMax + 20, math.exp(2), mytext, fontsize=12, horizontalalignment='left', verticalalignment='center', color='black')

    ### Save ###
    plt.savefig(filepath + module + "/Plot_Summary_scan2.png", bbox_inches='tight', format='png')
    plt.close()

    ### Plot2 ###
    Plot = plt.figure(figsize = (6,6), facecolor = 'white')
    Plot.suptitle("Scan 1 equalization", fontsize=20, horizontalalignment='center', verticalalignment='center', color='black') 

    eps = 0.01
    # convert zeros to small
    logs = hist_scan2_Trim0[0].astype(float)
    logs[logs==0] = eps
    plt.semilogy(hist_scan2_Trim0[1][:-1], logs, color='red', linestyle='-', linewidth=2)
    logs = hist_scan2_TrimF[0].astype(float)
    logs[logs==0] = eps
    plt.semilogy(hist_scan2_TrimF[1][:-1], logs, color='blue', linestyle='-', linewidth=2)
    logs = hist_scan2_equal[0].astype(float)
    logs[logs==0] = eps
    plt.semilogy(hist_scan2_equal[1][:-1], logs, color='black', linestyle='-', linewidth=2)
    # axes
    plt.axis([dacMin,dacMax,0.9,hist_scan2_max])
    plt.xticks(np.arange(dacMin,dacMax+1,50), fontsize=9)
    #plt.subplot(111).xaxis.set_ticks(np.arange(dacMin, dacMax+1,20))  #ERROR, deleted the True argument

    for tick in plt.subplot(111).yaxis.get_major_ticks():
      tick.label.set_fontsize(15)
    plt.subplot(111).axes.tick_params(direction='in', which='both', bottom=True, top=True, left=True, right=True)

    plt.xlabel("DAC Threshold", fontsize=15)
    plt.ylabel("Number of Pixels", fontsize=15)

    # Stats
    mytext = "0 Trim:\n%.1f +/- %.1f" % (np.mean(scan2_Trim0[scan2_Trim0>0]), np.std(scan2_Trim0[scan2_Trim0>0]))
    plt.text(dacMax + 20, math.exp(8), mytext, fontsize=12, horizontalalignment='left', verticalalignment='center', color='red')
    mytext = "F Trim:\n%.1f +/- %.1f" % (np.mean(scan2_TrimF[scan2_TrimF>0]), np.std(scan2_TrimF[scan2_TrimF>0]))
    plt.text(dacMax + 20, math.exp(7), mytext, fontsize=12, horizontalalignment='left', verticalalignment='center', color='blue')
    mytext = "Target:\n%.1f" % scan2_target
    plt.text(dacMax + 20, math.exp(6), mytext, fontsize=12, horizontalalignment='left', verticalalignment='center', color='black')
    mytext = "Predicted:\n%.1f +/- %.1f" % (np.mean(scan2_equal[scan2_equal>0]), np.std(scan2_equal[scan2_equal>0]))
    plt.text(dacMax + 20, math.exp(5), mytext, fontsize=12, horizontalalignment='left', verticalalignment='center', color='black')

    mytext = "Masked:\n%d" % (scan2_nMasked)
    plt.text(dacMax + 20, math.exp(2), mytext, fontsize=12, horizontalalignment='left', verticalalignment='center', color='black')

    ### Save ###
    plt.savefig(filepath + module + "/Plot_Summary_scan2.png", bbox_inches='tight', format='png')
    plt.close()

    ### Comparison plot ###
    Plot = plt.figure(figsize = (6,6), facecolor = 'white')

    eps = 0.01
    # convert zeros to small
    logs = hist_scan1_equal[0].astype(float)
    logs[logs==0] = eps
    plt.semilogy(hist_scan1_equal[1][:-1], logs, color='blue', linestyle='-', linewidth=2)
    logs = hist_scan2_equal[0].astype(float)
    logs[logs==0] = eps
    plt.semilogy(hist_scan2_equal[1][:-1], logs, color='red', linestyle='-', linewidth=1)
    # axes
    plt.axis([dacMin+240,dacMax-275,0.9,hist_scan2_max])
    plt.xticks(np.arange(dacMin+240,dacMax-275+1,10), fontsize=9)
    #plt.subplot(111).xaxis.set_ticks(np.arange(dacMin, dacMax+1,20))  #ERROR, deleted the True argument

    for tick in plt.subplot(111).yaxis.get_major_ticks():
      tick.label.set_fontsize(15)
    plt.subplot(111).axes.tick_params(direction='in', which='both', bottom=True, top=True, left=True, right=True)

    plt.xlabel("DAC Threshold", fontsize=15)
    plt.ylabel("Number of Pixels", fontsize=15)

    ### Save ###
    plt.savefig(filepath + module + "/Plot_Summary_comparison.png", bbox_inches='tight', format='png')
    plt.close()


    ### Plot with changed pixels ###
    Plot = plt.figure(figsize = (6,6), facecolor = 'white')

    eps = 0.01
    # convert zeros to small

    logs = hist_scan1_equal[0].astype(float)
    logs[logs==0] = eps
    plt.semilogy(hist_scan1_equal[1][:-1], logs, color='green', linestyle='-', linewidth=2)
    logs = hist_scan2_equal[0].astype(float)
    logs[logs==0] = eps
    plt.semilogy(hist_scan2_equal[1][:-1], logs, color='black', linestyle='-', linewidth=2)
    logs = hist_scan2_changed_pos[0].astype(float)
    logs[logs==0] = eps
    plt.semilogy(hist_scan2_changed_pos[1][:-1], logs, color='blue', linestyle='-', linewidth=2)
    logs = hist_scan2_changed_neg[0].astype(float)
    logs[logs==0] = eps
    plt.semilogy(hist_scan2_changed_neg[1][:-1], logs, color='red', linestyle='-', linewidth=1)
    # axes
    plt.axis([dacMin+225,dacMax-275,0.9,hist_scan2_max])
    plt.xticks(np.arange(dacMin+225,dacMax-275+1,10), fontsize=9)
    if (module == "Module0_VP0-1"):
      plt.axis([dacMin+255,dacMax-245,0.9,hist_scan2_max])
      plt.xticks(np.arange(dacMin+255,dacMax-245+1,10), fontsize=9)
    elif (module == "Module0_VP3-1"):
      plt.axis([dacMin+245,dacMax-275,0.9,hist_scan2_max])
      plt.xticks(np.arange(dacMin+245,dacMax-275+1,10), fontsize=9)      
    else:
      mytext = "Target: %.1f DAC" % scan2_target
      plt.text(dacMin+225+1, math.exp(6), mytext, fontsize=12, horizontalalignment='left', verticalalignment='center', color='black')

    for tick in plt.subplot(111).yaxis.get_major_ticks():
      tick.label.set_fontsize(15)
    plt.subplot(111).axes.tick_params(direction='in', which='both', bottom=True, top=True, left=True, right=True)

    plt.xlabel("DAC Threshold", fontsize=15)
    plt.ylabel("Number of Pixels", fontsize=15)

    #mytext = "+1 Trim value"
    #plt.text(dacMax -250 + 10, math.exp(7), mytext, fontsize=12, horizontalalignment='left', verticalalignment='center', color='blue')
    #mytext = "-1 Trim value"
    #plt.text(dacMax -250 + 10, math.exp(6), mytext, fontsize=12, horizontalalignment='left', verticalalignment='center', color='red')

    ### Save ###
    plt.savefig(filepath + module + "/Plot_Summary_changed.png", bbox_inches='tight', format='png')
    plt.close()


    ### Plot with changed pixels vs old ones ###
    Plot = plt.figure(figsize = (6,6), facecolor = 'white')
    Plot.suptitle("Equalization of changed pixels", fontsize=20, horizontalalignment='center', verticalalignment='center', color='black') 

    eps = 0.01
    # convert zeros to small
    logs = hist_scan2_equal[0].astype(float)
    logs[logs==0] = eps
    plt.semilogy(hist_scan2_equal[1][:-1], logs, color='black', linestyle='-', linewidth=2)
    logs = hist_scan2_changed_pos[0].astype(float)
    logs[logs==0] = eps
    plt.semilogy(hist_scan2_changed_pos[1][:-1], logs, color='blue', linestyle='-', linewidth=2)
    logs = hist_scan1_changed_pos[0].astype(float)
    logs[logs==0] = eps
    plt.semilogy(hist_scan1_changed_pos[1][:-1], logs, color='red', linestyle='-', linewidth=1)
    # axes
    plt.axis([dacMin+200,dacMax-250,0.9,hist_scan2_max])
    plt.xticks(np.arange(dacMin+200,dacMax-250+1,50), fontsize=9)
    #plt.subplot(111).xaxis.set_ticks(np.arange(dacMin, dacMax+1,20))  #ERROR, deleted the True argument

    for tick in plt.subplot(111).yaxis.get_major_ticks():
      tick.label.set_fontsize(15)
    plt.subplot(111).axes.tick_params(direction='in', which='both', bottom=True, top=True, left=True, right=True)

    plt.xlabel("DAC Threshold", fontsize=15)
    plt.ylabel("Number of Pixels", fontsize=15)

    mytext = "+1 Scan 2"
    plt.text(dacMax -250 + 20, math.exp(7), mytext, fontsize=12, horizontalalignment='left', verticalalignment='center', color='blue')
    mytext = "Scan 1"
    plt.text(dacMax -250 + 20, math.exp(6), mytext, fontsize=12, horizontalalignment='left', verticalalignment='center', color='red')

    ### Save ###
    plt.savefig(filepath + module + "/Plot_Summary_changed_vs_old.png", bbox_inches='tight', format='png')
    plt.close()


    ## Scan2 with Scan1 equalization ##
    if (module == "Module0_VP0-0" or module == "Module0_VP3-1"):
      try:
        scan2_equal_scan1 = np.loadtxt(filepath+ "scan2_scan1/" + module + "_TrimBest_Noise_Predict.csv", dtype=float, delimiter=',')
      except:
        print('failed to load scan2 with scan1 trim')
      
      hist_scan2_equal_scan1 = np.histogram(scan2_equal_scan1, bins=dac_bins)


      Plot = plt.figure(figsize = (6,6), facecolor = 'white')

      eps = 0.01
      # convert zeros to small
      logs = hist_scan2_equal_scan1[0].astype(float)
      logs[logs==0] = eps
      plt.semilogy(hist_scan2_equal[1][:-1], logs, color='red', linestyle='-', linewidth=2)
      logs = hist_scan2_equal[0].astype(float)
      logs[logs==0] = eps
      plt.semilogy(hist_scan2_equal[1][:-1], logs, color='blue', linestyle='-', linewidth=1)
      # axes
      plt.axis([dacMin+240,dacMax-275,0.9,hist_scan2_max])
      plt.xticks(np.arange(dacMin+240,dacMax-275+1,10), fontsize=9)
      #plt.subplot(111).xaxis.set_ticks(np.arange(dacMin, dacMax+1,20))  #ERROR, deleted the True argument

      for tick in plt.subplot(111).yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
      plt.subplot(111).axes.tick_params(direction='in', which='both', bottom=True, top=True, left=True, right=True)

      mytext = "Target:\n%.1f" % scan2_target
      plt.text(dacMax -275 + 1, math.exp(7), mytext, fontsize=12, horizontalalignment='left', verticalalignment='center', color='black')
      mytext = "Achieved:\n%.1f +/- %.1f" % (np.mean(scan2_equal[scan2_equal>0]), np.std(scan2_equal[scan2_equal>0]))
      plt.text(dacMax -275 + 1, math.exp(6), mytext, fontsize=12, horizontalalignment='left', verticalalignment='center', color='blue')
      mytext = "Achieved:\n%.1f +/- %.1f" % (np.mean(scan2_equal_scan1[scan2_equal_scan1>0]), np.std(scan2_equal_scan1[scan2_equal_scan1>0]))
      plt.text(dacMax -275 + 1, math.exp(5), mytext, fontsize=12, horizontalalignment='left', verticalalignment='center', color='red')

      plt.xlabel("DAC Threshold", fontsize=15)
      plt.ylabel("Number of Pixels", fontsize=15)

      ### Save ###
      plt.savefig(filepath + module + "/Plot_Summary_scan2_scan1.png", bbox_inches='tight', format='png')
      plt.close()

### Main ###

if (len(sys.argv) < 2):
    print("Not enough imput arguments")
    exit

filepath = sys.argv[1]

for i in range(4):
  for j in range(3):
    module = "Module0_VP" + str(i) + "-" + str(j)
    print(module)
    noise_matrix_diff(filepath,module)
    mask_matrix_diff(filepath,module)
    pos_change, neg_change = trim_matrix_diff(filepath,module)
    pixel_noise_diff(filepath,module)
    scans_equalizations(filepath, module, pos_change, neg_change)


    